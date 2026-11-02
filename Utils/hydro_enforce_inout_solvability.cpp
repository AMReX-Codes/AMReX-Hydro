/** \addtogroup Utilities
 * @{
 */

#include <hydro_utils.H>

using namespace amrex;

namespace HydroUtils {

namespace {

void set_inout_masks(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    Array<iMultiFab, AMREX_SPACEDIM>& inout_masks,
    const Array<iMultiFab, AMREX_SPACEDIM>& level_masks,
    const BCRec* bc_type,
    const Box& domain,
    const bool corners)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        const int dir = ori.coordDir();
        const auto oriIsLow = ori.isLow();
        const auto oriIsHigh = ori.isHigh();

        // Multifab for normal velocity
        const auto& vel_mf = vels_vec[lev][dir];

        // mask iMF for the respective velocity direction
        auto& inout_mask = inout_masks[dir];

        // mask iMF for finest cells not covered
        const auto& level_mask = level_masks[dir];

        IndexType::CellIndex dir_index_type = (vel_mf->ixType()).ixType(dir);
        // domain extent indices for the velocities
        int dlo;
        if (dir_index_type == IndexType::CellIndex::CELL) {
            // lower boundary is at -1 for cell-centered velocity
            dlo = domain.smallEnd(dir) - 1;
        } else {
            // lower boundary is at  0 for face-centered velocity
            dlo = domain.smallEnd(dir);
        }
        int dhi = domain.bigEnd(dir) + 1;

        // get BCs for the normal velocity and set the boundary index
        // based on low or high side
        const BCRec ibcrec = bc_type[dir];
        int bc, bndry;
        if (oriIsLow) {
            bc = ibcrec.lo(dir);
            bndry = dlo;
        } else {
            bc = ibcrec.hi(dir);
            bndry = dhi;
        }

        // limit influx/outflux calculations to the in-out boundaries only
        if (bc == BCType::direction_dependent) {
            for (MFIter mfi(*vel_mf, TilingIfNotGPU()); mfi.isValid(); ++mfi) {

                Box box = mfi.tilebox();
                const IntVect ungrown_sm_end = box.smallEnd();
                const IntVect ungrown_bg_end = box.bigEnd();

                // include ghost cells along normal velocity for cell-centered
                // not for face-centered as boundary lies in valid region
                if (dir_index_type == IndexType::CellIndex::CELL) {
                    box.grow(dir, 1);
                }

                // include boundary corners if specified
                if (corners) {
                    int tang_dir_1 = (dir+1)%AMREX_SPACEDIM;
                    if (box.smallEnd(tang_dir_1) == domain.smallEnd(tang_dir_1)) {
                        box.growLo(tang_dir_1,1);
                    }
                    if (box.bigEnd(tang_dir_1) == domain.bigEnd(tang_dir_1)) {
                        box.growHi(tang_dir_1,1);
                    }
#if (AMREX_SPACEDIM == 3)
                    int tang_dir_2 = (dir+2)%AMREX_SPACEDIM;
                    if (box.smallEnd(tang_dir_2) == domain.smallEnd(tang_dir_2)) {
                        box.growLo(tang_dir_2,1);
                    }
                    if (box.bigEnd(tang_dir_2) == domain.bigEnd(tang_dir_2)) {
                        box.growHi(tang_dir_2,1);
                    }
#endif
                }

                // Enter further only if the box bndry is at the domain bndry
                if ((oriIsLow  && (box.smallEnd(dir) == dlo))
                 || (oriIsHigh && (box.bigEnd(dir)   == dhi))) {

                    // create a 2D box normal to dir at the low/high bndry
                    Box box2d(box); box2d.setRange(dir, bndry);

                    auto vel_arr = vel_mf->array(mfi);
                    auto inout_mask_arr = inout_mask.array(mfi);
                    const auto level_mask_arr = level_mask.const_array(mfi);

                    ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        // only evaluate cells that are not covered by finer mesh
                        const IntVect iv{AMREX_D_DECL(i,j,k)};
                        const IntVect lmask_idx = amrex::max(amrex::min(iv,ungrown_bg_end),ungrown_sm_end);
                        // consider nearest valid cell to determine if boundary cell is covered
                        if (level_mask_arr(lmask_idx) == 1) {
                            // tag cells as inflow or outflow by checking vel direction
                            if ((oriIsLow  && vel_arr(i,j,k) >= 0)
                             || (oriIsHigh && vel_arr(i,j,k) <= 0)) {
                                inout_mask_arr(i,j,k) = -1;
                            } else {
                                inout_mask_arr(i,j,k) = +1;
                            }
                        }
                    });
                }
            }
        }
    }
}

void compute_influx_outflux(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const Array<iMultiFab, AMREX_SPACEDIM>& inout_masks,
    const Geometry& geom,
    Real& influx,
    Real& outflux,
    Real& area_in,
    Real& area_out,
    const bool corners)
{
    influx = Real(0.0), outflux = Real(0.0);
    area_in = Real(0.0), area_out = Real(0.0);

    const Real* a_dx = geom.CellSize();
#if (AMREX_SPACEDIM == 2)
    // In RZ the face area is not constant along a face, so it has to go inside the
    // reduction: an r-face carries 2*pi*r_face*dz, and a z-face 2*pi*r_cell*dr. Both
    // are 2*pi*r times the Cartesian ds below, with r taken at the face location in
    // the radial direction.
    const bool is_rz = geom.IsRZ();
    const Real problo_x = geom.ProbLo(0);
    const Real dx_r = a_dx[0];
#endif

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++) {

        // normal face area
#if (AMREX_SPACEDIM == 2)
        const Real ds = a_dx[(idim+1) % AMREX_SPACEDIM];
#else
        const Real ds =
            a_dx[(idim+1) % AMREX_SPACEDIM] * a_dx[(idim+2) % AMREX_SPACEDIM];
#endif

        // Multifab for normal velocity
        const auto& vel_mf = vels_vec[lev][idim];

        // grow in the respective direction if vel is cell-centered
        IndexType index_type = vel_mf->ixType();
        index_type.flip(idim); IntVect ngrow = index_type.ixType();

        // grow in the transverse direction to include boundary corners
        if (corners) {
            ngrow[(idim+1)%AMREX_SPACEDIM] = 1;
#if (AMREX_SPACEDIM == 3)
            ngrow[(idim+2)%AMREX_SPACEDIM] = 1;
#endif
        }

        // mask iMF for the respective velocity direction
        const auto& inout_mask = inout_masks[idim];

#if (AMREX_SPACEDIM == 2)
        // Offset of this MultiFab's index from the radial location of the face: 0 for
        // the r-velocity, which is nodal in r, and 1/2 for the z-velocity, which is
        // cell-centered in r.
        const Real r_off = vel_mf->ixType().nodeCentered(0) ? Real(0.0) : Real(0.5);
#endif

        // define "multi-arrays" and perform reduction using the mask
        auto const& vel_ma = vel_mf->const_arrays();
        auto const& inout_mask_ma = inout_mask.const_arrays();

        // Accumulate the tagged boundary measure alongside the flux, so that the
        // caller can compare a mean normal speed, rather than an integrated flux,
        // against the velocity scale small_vel.
        const auto r_in =
            ParReduce(TypeList<ReduceOpSum, ReduceOpSum>{},
                      TypeList<Real, Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real,Real>
            {
                if (inout_mask_ma[box_no](i,j,k) == -1) {
#if (AMREX_SPACEDIM == 2)
                    const Real da = is_rz
                        ? Real(2.0) * amrex::Math::pi<Real>()
                              * (problo_x + (Real(i) + r_off) * dx_r) * ds
                        : ds;
#else
                    const Real da = ds;
#endif
                    return { da * std::abs(vel_ma[box_no](i,j,k)), da };
                } else {
                    return { Real(0.), Real(0.) };
                }
            });
        influx  += amrex::get<0>(r_in);
        area_in += amrex::get<1>(r_in);

        const auto r_out =
            ParReduce(TypeList<ReduceOpSum, ReduceOpSum>{},
                      TypeList<Real, Real>{},
                      *vel_mf, ngrow,
            [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
                noexcept -> GpuTuple<Real,Real>
            {
                if (inout_mask_ma[box_no](i,j,k) == 1) {
#if (AMREX_SPACEDIM == 2)
                    const Real da = is_rz
                        ? Real(2.0) * amrex::Math::pi<Real>()
                              * (problo_x + (Real(i) + r_off) * dx_r) * ds
                        : ds;
#else
                    const Real da = ds;
#endif
                    return { da * std::abs(vel_ma[box_no](i,j,k)), da };
                } else {
                    return { Real(0.), Real(0.) };
                }
            });
        outflux  += amrex::get<0>(r_out);
        area_out += amrex::get<1>(r_out);
    }
    ParallelDescriptor::ReduceRealSum(influx);
    ParallelDescriptor::ReduceRealSum(outflux);
    ParallelDescriptor::ReduceRealSum(area_in);
    ParallelDescriptor::ReduceRealSum(area_out);
}

void correct_outflow(
    const int lev,
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const BCRec* bc_type,
    const Box& domain,
    const Real alpha_fcf,
    const bool corners)
{
    for (OrientationIter oit; oit != nullptr; ++oit) {
        const auto ori = oit();
        const int dir = ori.coordDir();
        const auto oriIsLow = ori.isLow();
        const auto oriIsHigh = ori.isHigh();

        // Multifab for normal velocity
        const auto& vel_mf = vels_vec[lev][dir];

        IndexType::CellIndex dir_index_type = (vel_mf->ixType()).ixType(dir);
        // domain extent indices for the velocities
        int dlo;
        if (dir_index_type == IndexType::CellIndex::CELL) {
            dlo = domain.smallEnd(dir) - 1; // cell-centered boundary
        } else {
            dlo = domain.smallEnd(dir);     // face-centered boundary
        }
        int dhi = domain.bigEnd(dir) + 1;

        // get BCs for the normal velocity and set the boundary index
        const BCRec ibcrec = bc_type[dir];
        int bc, bndry;
        if (oriIsLow) {
            bc = ibcrec.lo(dir);
            bndry = dlo;
        } else {
            bc = ibcrec.hi(dir);
            bndry = dhi;
        }

        if (bc == BCType::direction_dependent) {
            for (MFIter mfi(*vel_mf, false); mfi.isValid(); ++mfi) {

                Box box = mfi.validbox();
                if (dir_index_type == IndexType::CellIndex::CELL) {
                    box.grow(dir, 1);
                }

                // Include tangential ghost copies so box-boundary duplicates
                // receive the same outflow correction.
                if (corners) {
                    int tang_dir_1 = (dir+1)%AMREX_SPACEDIM;
                    box.grow(tang_dir_1,1);
#if (AMREX_SPACEDIM == 3)
                    int tang_dir_2 = (dir+2)%AMREX_SPACEDIM;
                    box.grow(tang_dir_2,1);
#endif
                }

                // Enter further only if the box boundary is at the domain boundary
                if ((oriIsLow  && (box.smallEnd(dir) == dlo))
                 || (oriIsHigh && (box.bigEnd(dir)   == dhi))) {

                    // create a 2D box normal to dir at the low/high boundary
                    Box box2d(box); box2d.setRange(dir, bndry);

                    auto vel_arr = vel_mf->array(mfi);

                    ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        if ((oriIsLow  && vel_arr(i,j,k) < 0)
                         || (oriIsHigh && vel_arr(i,j,k) > 0)) {
                            vel_arr(i,j,k) *= alpha_fcf;
                        }
                    });
                }
            }
        }
    }
}

} // file-local namespace

void enforceInOutSolvability (
    const Vector<Array<MultiFab*, AMREX_SPACEDIM>>& vels_vec,
    const BCRec* bc_type,
    const Vector<Geometry>& geom,
    bool include_bndry_corners
)
{
    const auto nlevs = int(vels_vec.size());
    Real influx = Real(0.0), outflux = Real(0.0);
    Real area_in = Real(0.0), area_out = Real(0.0);
    for (int lev = 0; lev < nlevs; ++lev) {
        // Cartesian and 2D RZ face areas are handled below; spherical is not.
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(geom[lev].IsCartesian() || geom[lev].IsRZ(),
            "enforceInOutSolvability: only Cartesian and RZ coordinates are supported");

        const Box domain = geom[lev].Domain();

        // masks to tag inflow/outflow at the boundaries
        // separate iMultifab for each velocity direction
        //  0 for interior cells,
        // -1 for inflow bndry cells, +1 for outflow bndry cells
        Array<iMultiFab, AMREX_SPACEDIM> inout_masks;

        // Should not consider covered boundary cells in influx, outflux calculations
        Array<iMultiFab, AMREX_SPACEDIM> level_masks;
        IntVect rr; // refinement ratio
        if (lev < nlevs - 1) {
            for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
            {
                rr[idim] = static_cast<int>(std::round(geom[lev].CellSize(idim) / geom[lev+1].CellSize(idim)));
            }
        }

        // defining the mask iMultifabs in each direction
        for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
        {
            const auto& vel_mf = vels_vec[lev][idim];    // normal velocity multifab

            // grow in the respective direction if vel is cell-centered
            // to include the boundary cells
            IndexType index_type = vel_mf->ixType();
            index_type.flip(idim); IntVect ngrow = index_type.ixType();

            // grow in the transverse direction to include boundary corners
            if (include_bndry_corners) {
                ngrow[(idim+1)%AMREX_SPACEDIM] = 1;
#if (AMREX_SPACEDIM == 3)
                ngrow[(idim+2)%AMREX_SPACEDIM] = 1;
#endif
            }

            // Both the mask setup and the outflow correction index the velocity
            // over this same grown region, so the velocity must carry (filled)
            // ghost cells there.
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(vel_mf->nGrowVect().allGE(ngrow),
                "enforceInOutSolvability: velocity needs ghost cells -- one in the "
                "normal direction if it is cell-centered, and one in each tangential "
                "direction if include_bndry_corners is true");

            inout_masks[idim].define(vel_mf->boxArray(), vel_mf->DistributionMap(), 1, ngrow);
            inout_masks[idim].setVal(0);

            if (lev < nlevs - 1) {
                level_masks[idim] = makeFineMask(
                    vel_mf->boxArray(), vel_mf->DistributionMap(),
                    vels_vec[lev+1][idim]->boxArray(), rr, 1, 0);
            } else {
                level_masks[idim].define(
                    vel_mf->boxArray(), vel_mf->DistributionMap(), 1, 0,
                    amrex::MFInfo());
                level_masks[idim].setVal(1);
            }
        }

        set_inout_masks(lev, vels_vec, inout_masks, level_masks, bc_type, domain, include_bndry_corners);

        Real influx_lev = Real(0.0), outflux_lev = Real(0.0);
        Real area_in_lev = Real(0.0), area_out_lev = Real(0.0);
        compute_influx_outflux(lev, vels_vec, inout_masks, geom[lev], influx_lev, outflux_lev,
                               area_in_lev, area_out_lev, include_bndry_corners);
        influx += influx_lev;
        outflux += outflux_lev;
        area_in += area_in_lev;
        area_out += area_out_lev;
    }

    // influx and outflux are integrated fluxes, so they carry the units of the
    // domain as well as of the velocity. Compare the mean normal speed over the
    // tagged boundary faces against small_vel instead, which makes the test
    // independent of the length scale of the problem.
    const Real influx_tol  = small_vel * area_in;
    const Real outflux_tol = small_vel * area_out;

    if ((influx > influx_tol) && (outflux <= outflux_tol)) {
        Abort("Cannot enforce solvability, no outflow from the direction dependent boundaries");
    } else if ((influx <= influx_tol) && (outflux <= outflux_tol)) {
        return; // do nothing
    } else {
        for (int lev = 0; lev < nlevs; ++lev) {
            const Box domain = geom[lev].Domain();
            const Real alpha_fcf = influx/outflux;  // flux correction factor
            correct_outflow(lev, vels_vec, bc_type, domain, alpha_fcf, include_bndry_corners);
        }
    }
}

}
