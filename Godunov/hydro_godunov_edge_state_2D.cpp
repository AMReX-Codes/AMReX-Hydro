/**
 * \file hydro_godunov_edge_state_2D.cpp
 *
 * \addtogroup Godunov
 *  @{
 */

#include <hydro_godunov_plm.H>
#include <hydro_godunov_ppm.H>
#include <hydro_godunov.H>
#include <hydro_godunov_K.H>
#include <hydro_bcs_K.H>


using namespace amrex;

void
Godunov::ComputeEdgeState (Box const& bx, int ncomp,
                           Array4<Real const> const& q,
                           Array4<Real> const& xedge,
                           Array4<Real> const& yedge,
                           Array4<Real const> const& umac,
                           Array4<Real const> const& vmac,
                           Array4<Real const> const& divu,
                           Array4<Real const> const& fq,
                           Geometry geom,
                           Real l_dt,
                           BCRec const* pbc, int const* iconserv,
                           bool use_ppm,
                           bool use_forces_in_trans,
                           bool is_velocity)
{
    Box const& xbx = amrex::surroundingNodes(bx,0);
    Box const& ybx = amrex::surroundingNodes(bx,1);

    Box const& bxg1 = amrex::grow(bx,1);

    FArrayBox tmpfab(amrex::grow(bx,1),  (4*AMREX_SPACEDIM + 2)*ncomp);
    Elixir tmpeli = tmpfab.elixir();
    Real* p   = tmpfab.dataPtr();

    Box xebox = Box(xbx).grow(1,1);
    Box yebox = Box(ybx).grow(0,1);

    const Real dx = geom.CellSize(0);
    const Real dy = geom.CellSize(1);

    Real dtdx = l_dt/dx;
    Real dtdy = l_dt/dy;

    const bool is_rz = geom.IsRZ();
    Box const& domain = geom.Domain();
    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);

    Array4<Real> Imx = makeArray4(p, bxg1, ncomp);
    p +=         Imx.size();
    Array4<Real> Ipx = makeArray4(p, bxg1, ncomp);
    p +=         Ipx.size();
    Array4<Real> Imy = makeArray4(p, bxg1, ncomp);
    p +=         Imy.size();
    Array4<Real> Ipy = makeArray4(p, bxg1, ncomp);
    p +=         Ipy.size();
    Array4<Real> xlo = makeArray4(p, xebox, ncomp);
    p +=         xlo.size();
    Array4<Real> xhi = makeArray4(p, xebox, ncomp);
    p +=         xhi.size();
    Array4<Real> ylo = makeArray4(p, yebox, ncomp);
    p +=         ylo.size();
    Array4<Real> yhi = makeArray4(p, yebox, ncomp);
    p +=         yhi.size();
    Array4<Real> xyzlo = makeArray4(p, bxg1, ncomp);
    p +=         xyzlo.size();
    Array4<Real> xyzhi = makeArray4(p, bxg1, ncomp);
    p +=         xyzhi.size();

    // Use PPM to generate Im and Ip */
    if (use_ppm)
    {
        amrex::ParallelFor(bxg1, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            PPM::PredictStateOnXFace(i, j, k, n, l_dt, dx, Imx(i,j,k,n), Ipx(i,j,k,n),
                                   q, umac, pbc[n], dlo.x, dhi.x);
            PPM::PredictStateOnYFace(i, j, k, n, l_dt, dy, Imy(i,j,k,n), Ipy(i,j,k,n),
                                   q, vmac, pbc[n], dlo.y, dhi.y);
        });
    // Use PLM to generate Im and Ip */
    }
    else
    {

        amrex::ParallelFor(xebox, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            PLM::PredictStateOnXFace(i, j, k, n, l_dt, dx, Imx(i,j,k,n), Ipx(i-1,j,k,n),
                                     q, umac(i,j,k), pbc[n], dlo.x, dhi.x, is_velocity);
        });

        amrex::ParallelFor(yebox, ncomp,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            PLM::PredictStateOnYFace(i, j, k, n, l_dt, dy, Imy(i,j,k,n), Ipy(i,j-1,k,n),
                                     q, vmac(i,j,k), pbc[n], dlo.y, dhi.y, is_velocity);
        });
    }


    amrex::ParallelFor(
    xebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real uad = umac(i,j,k);
        Real fux = (amrex::Math::abs(uad) < small_vel)? 0. : 1.;
        Real lo = Ipx(i-1,j,k,n);
        Real hi = Imx(i  ,j,k,n);

        if (use_forces_in_trans && fq)
        {
            lo += 0.5*l_dt*fq(i-1,j,k,n);
            hi += 0.5*l_dt*fq(i  ,j,k,n);
        }

        auto bc = pbc[n];

        GodunovTransBC::SetTransTermXBCs(i, j, k, n, q, lo, hi, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);
        xlo(i,j,k,n) = lo;
        xhi(i,j,k,n) = hi;
        Real st = (uad >= 0.) ? lo : hi;
        Imx(i,j,k,n) = fux*st + (1. - fux)*0.5*(hi + lo);

    },
    yebox, ncomp, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real vad = vmac(i,j,k);
        Real fuy = (amrex::Math::abs(vad) < small_vel)? 0. : 1.;
        Real lo = Ipy(i,j-1,k,n);
        Real hi = Imy(i,j  ,k,n);

        if (use_forces_in_trans && fq)
        {
            lo += 0.5*l_dt*fq(i,j-1,k,n);
            hi += 0.5*l_dt*fq(i,j  ,k,n);
        }

        auto bc = pbc[n];

        GodunovTransBC::SetTransTermYBCs(i, j, k, n, q, lo, hi, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

        ylo(i,j,k,n) = lo;
        yhi(i,j,k,n) = hi;
        Real st = (vad >= 0.) ? lo : hi;
        Imy(i,j,k,n) = fuy*st + (1. - fuy)*0.5*(hi + lo);
    }
    );

    // We can reuse the space in Ipx, Ipy and Ipz.

    //
    // x-direction
    //
    Box const& xbxtmp = amrex::grow(bx,0,1);
    Array4<Real> yzlo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(xbxtmp,1), ncomp);
    amrex::ParallelFor(
    Box(yzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_yzlo, l_yzhi;

        l_yzlo = ylo(i,j,k,n);
        l_yzhi = yhi(i,j,k,n);
        Real vad = vmac(i,j,k);
        GodunovTransBC::SetTransTermYBCs(i, j, k, n, q, l_yzlo, l_yzhi, bc.lo(1), bc.hi(1), dlo.y, dhi.y, is_velocity);

        Real st = (vad >= 0.) ? l_yzlo : l_yzhi;
        Real fu = (amrex::Math::abs(vad) < small_vel) ? 0.0 : 1.0;
        yzlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_yzhi + l_yzlo);
    });

    //
    amrex::ParallelFor(xbx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        stl = xlo(i,j,k,n);
        sth = xhi(i,j,k,n);
        // To match EBGodunov
        // Here we add  dt/2 (-q u_x - (v q)_y) to the term that is already
        //     q + dx/2 q_x + dt/2 (-u q_x) to get
        //     q + dx/2 q_x - dt/2 (u q_x  + q u_x + (v q)_y) which is equivalent to
        // --> q + dx/2 q_x - dt/2 ( div (uvec q) )
        Real quxl = (umac(i,j,k) - umac(i-1,j,k)) * q(i-1,j,k,n);
        stl += ( - (0.5*dtdx) * quxl
                 - (0.5*dtdy) * (yzlo(i-1,j+1,k  ,n)*vmac(i-1,j+1,k  )
                                 -yzlo(i-1,j  ,k  ,n)*vmac(i-1,j  ,k  )) );

        // Here we adjust for non-conservative by removing the q divu contribution to get
        //     q + dx/2 q_x - dt/2 ( div (uvec q) - q divu ) which is equivalent to
        // --> q + dx/2 q_x - dt/2 ( uvec dot grad q)
        stl += (!iconserv[n])               ?  0.5*l_dt* q(i-1,j,k,n)*divu(i-1,j,k) : 0.;

        stl += (!use_forces_in_trans && fq) ?  0.5*l_dt*fq(i-1,j,k,n) : 0.;

        // Here we add uq/r for RZ
        stl += (is_rz) ? -0.25 * l_dt * q(i-1,j,k,n)*( umac(i,j,k) + umac(i-1,j,k) ) / ( dx*(amrex::Math::abs(Real(i)-0.5)) ) : 0.;

        // High side
        Real quxh = (umac(i+1,j,k) - umac(i,j,k)) * q(i,j,k,n);
        sth += ( - (0.5*dtdx) * quxh
                 - (0.5*dtdy)*(yzlo(i,j+1,k,n)*vmac(i,j+1,k)
                               -yzlo(i,j  ,k,n)*vmac(i,j  ,k)) );

        sth += (!iconserv[n])               ? 0.5*l_dt* q(i  ,j,k,n)*divu(i,j,k) : 0.;

        sth += (!use_forces_in_trans && fq) ? 0.5*l_dt*fq(i  ,j,k,n) : 0.;

        sth += (is_rz) ? -0.25 * l_dt * q(i,j,k,n)*( umac(i,j,k) + umac(i+1,j,k) ) / ( dx*(amrex::Math::abs(Real(i)+0.5)) ) : 0.;


        auto bc = pbc[n];
        HydroBC::SetXEdgeBCs(i, j, k, n, q, stl, sth, bc.lo(0), dlo.x, bc.hi(0), dhi.x, is_velocity);

        if ( (i==dlo.x) && (bc.lo(0) == BCType::foextrap || bc.lo(0) == BCType::hoextrap) )
        {
            if ( umac(i,j,k) >= 0. && n==XVEL && is_velocity )  sth = amrex::min(sth,0.0_rt);
            stl = sth;
        }
        if ( (i==dhi.x+1) && (bc.hi(0) == BCType::foextrap || bc.hi(0) == BCType::hoextrap) )
        {
            if ( umac(i,j,k) <= 0. && n==XVEL && is_velocity ) stl = amrex::max(stl,0.0_rt);
             sth = stl;
        }

        Real temp = (umac(i,j,k) >= 0.) ? stl : sth;
        temp = (amrex::Math::abs(umac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp;
        xedge(i,j,k,n) = temp;


	if ((i==0 || i==1) && j==28)
	{
	    std::cout<<"qx = "<<temp<<std::endl;
	}

    });

    //
    // y-direction
    //
    Box const& ybxtmp = amrex::grow(bx,1,1);
    Array4<Real> xzlo = makeArray4(xyzlo.dataPtr(), amrex::surroundingNodes(ybxtmp,0), ncomp);
    amrex::ParallelFor(
    Box(xzlo), ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        const auto bc = pbc[n];
        Real l_xzlo, l_xzhi;

        l_xzlo = xlo(i,j,k,n);
        l_xzhi = xhi(i,j,k,n);

        Real uad = umac(i,j,k);
        GodunovTransBC::SetTransTermXBCs(i, j, k, n, q, l_xzlo, l_xzhi, bc.lo(0), bc.hi(0), dlo.x, dhi.x, is_velocity);

        Real st = (uad >= 0.) ? l_xzlo : l_xzhi;
        Real fu = (amrex::Math::abs(uad) < small_vel) ? 0.0 : 1.0;
        xzlo(i,j,k,n) = fu*st + (1.0 - fu) * 0.5 * (l_xzhi + l_xzlo);
    });

    //
    amrex::ParallelFor(ybx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        Real stl, sth;

        stl = ylo(i,j,k,n);
        sth = yhi(i,j,k,n);

        // To match EBGodunov
        // Here we add  dt/2 (-q v_y - (u q)_x) to the term that is already
        //     q + dy/2 q_y + dt/2 (-v q_y) to get
        //     q + dy/2 q_y - dt/2 (v q_y  + q v_y + (u q)_x) which is equivalent to
        // --> q + dy/2 q_y - dt/2 ( div (uvec q) )
        Real qvyl = (vmac(i,j,k) - vmac(i,j-1,k)) * q(i,j-1,k,n);
        stl += ( - (0.5*dtdy)*qvyl
                 - (0.5*dtdx)*(xzlo(i+1,j-1,k  ,n)*umac(i+1,j-1,k  )
                               -xzlo(i  ,j-1,k  ,n)*umac(i  ,j-1,k  )) );

        // Here we adjust for non-conservative by removing the q divu contribution to get
        //     q + dy/2 q_y - dt/2 ( div (uvec q) - q divu ) which is equivalent to
        // --> q + dy/2 q_y - dt/2 ( uvec dot grad q)
        stl += (!iconserv[n])               ? 0.5*l_dt* q(i,j-1,k,n)*divu(i,j-1,k) : 0.;

        stl += (!use_forces_in_trans && fq) ? 0.5*l_dt*fq(i,j-1,k,n) : 0.;

        // Here we add uq/r for RZ
        stl += (is_rz) ? -0.25 * l_dt * q(i,j-1,k,n)*( umac(i,j-1,k) + umac(i+1,j-1,k) ) / ( dx*(amrex::Math::abs(Real(i)+0.5)) ) : 0.;

        // High side
        Real qvyh = (vmac(i,j+1,k) - vmac(i,j,k)) * q(i,j,k,n);
        sth += ( - (0.5*dtdy)*qvyh
                 - (0.5*dtdx)*(xzlo(i+1,j,k  ,n)*umac(i+1,j,k  )
                               -xzlo(i  ,j,k  ,n)*umac(i  ,j,k  )) );

        sth += (!iconserv[n])               ? 0.5*l_dt* q(i,j,k,n)*divu(i,j,k) : 0.;

        sth += (!use_forces_in_trans && fq) ? 0.5*l_dt*fq(i,j,k,n) : 0.;

        sth += (is_rz) ? -0.25 * l_dt * q(i,j,k,n)*( umac(i,j  ,k) + umac(i+1,j  ,k) ) / ( dx*(amrex::Math::abs(Real(i)+0.5)) ) : 0.;


        auto bc = pbc[n];
        HydroBC::SetYEdgeBCs(i, j, k, n, q, stl, sth, bc.lo(1), dlo.y, bc.hi(1), dhi.y, is_velocity);

        if ( (j==dlo.y) && (bc.lo(1) == BCType::foextrap || bc.lo(1) == BCType::hoextrap) )
        {
            if ( vmac(i,j,k) >= 0. && n==YVEL && is_velocity ) sth = amrex::min(sth,0.0_rt);
            stl = sth;
        }
        if ( (j==dhi.y+1) && (bc.hi(1) == BCType::foextrap || bc.hi(1) == BCType::hoextrap) )
        {
            if ( vmac(i,j,k) <= 0. && n==YVEL && is_velocity ) stl = amrex::max(stl,0.0_rt);
            sth = stl;
        }

        Real temp = (vmac(i,j,k) >= 0.) ? stl : sth;
        temp = (amrex::Math::abs(vmac(i,j,k)) < small_vel) ? 0.5*(stl + sth) : temp;
        yedge(i,j,k,n) = temp;

	if (i==0 && (j==28 || j==29) )
	{
	    std::cout<<"qy = "<<temp<<std::endl;
	}

    });

}
/** @} */
