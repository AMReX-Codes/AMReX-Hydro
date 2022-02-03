/**
 * \file hydro_bds_edge_state_2D.cpp
 *
 * \addtogroup BDS
 *  @{
 */

#include <hydro_bds.H>




using namespace amrex;

constexpr amrex::Real eps = 1.0e-8;


/**
 * Uses the Bell-Dawson-Shubin (BDS) algorithm, a higher order Godunov
 * method for scalar conservation laws in two dimensions, to compute
 * edge states.
 *
 * \param [in]     s_mf       MultiFab of state
 * \param [in]     state_comp The component of the state MultiFab.
 * \param [in]     geom       Box geometry.
 * \param [in,out] xedge      MultiFab containing x-edges.
 * \param [in,out] yedge      MultiFab containing y-edges.
 * \param [in]     edge_comp  The component of the edge MultiFabs.
 * \param [in]     umac       Face velocities.
 * \param [in]     vmac       Face velocities.
 * \param [in]     fq         Multifab for forces.
 * \param [in]     fq_comp    Component for Multifab for forces.
 * \param [in]     iconserv   Indicates conservative dimensions.
 * \param [in]     dt         Time step.
 *
 */

void
BDS::ComputeEdgeState ( const MultiFab& s_mf,
                               const int state_comp,
                               const Geometry& geom,
                               MultiFab& xedge,
                               MultiFab& yedge,
                               const int edge_comp,
                               MultiFab const& umac,
                               MultiFab const& vmac,
                               MultiFab const& fq,
                               const int fq_comp,
                               const int is_conservative,
                               const Real dt)
{

    if(!is_conservative){
        Abort("For 2D, BDS algorithm currently only supports conservative computations");
    }



    BoxArray ba = s_mf.boxArray();
    DistributionMapping dmap = s_mf.DistributionMap();

    MultiFab slope_mf(ba,dmap,3,1);

    BDS::ComputeSlopes(s_mf,geom,slope_mf,state_comp);

    BDS::ComputeConc(s_mf, state_comp,
                         geom,
                         xedge, yedge, edge_comp,
                         slope_mf,
                         umac,
                         vmac,
                         fq, fq_comp,
                         dt);

}


/**
 * Compute bilinear slopes for BDS algorithm.
 *
 * \param [in]  s_mf MultiFab of state.
 * \param [in]  geom Box geometry.
 * \param [out] slope_mf MuliFab to store slope information.
 * \param [in]  comp The component of the MultiFab.
 *
 */

void
BDS::ComputeSlopes (MultiFab const& s_mf,
                        const Geometry& geom,
                        MultiFab& slope_mf,
                        const int state_comp)
{
    constexpr bool limit_slopes = true;


    BoxArray ba = s_mf.boxArray();
    DistributionMapping dmap = s_mf.DistributionMap();
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // local variables
    MultiFab sint_mf(convert(ba,IntVect(AMREX_D_DECL(1,1,1))), dmap, 1, 1);

    Real hx = dx[0];
    Real hy = dx[1];

    for ( MFIter mfi(sint_mf); mfi.isValid(); ++mfi){

        const Box& bx = mfi.growntilebox(1);
        Array4<const Real> const& s    = s_mf.array(mfi, state_comp);
        Array4<      Real> const& sint = sint_mf.array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // bicubic interpolation to corner points
            // (i,j,k) refers to lower corner of cell
            // Added k index -- placeholder for 2d
            sint(i,j,k) = (s(i-2,j-2,k) + s(i-2,j+1,k) + s(i+1,j-2,k) + s(i+1,j+1,k)
                    - 7.0*(s(i-2,j-1,k) + s(i-2,j  ,k) + s(i-1,j-2,k) + s(i  ,j-2,k) +
                           s(i-1,j+1,k) + s(i  ,j+1,k) + s(i+1,j-1,k) + s(i+1,j  ,k))
                   + 49.0*(s(i-1,j-1,k) + s(i  ,j-1,k) + s(i-1,j  ,k) + s(i  ,j  ,k)) ) / 144.0;
        });

    }

    for ( MFIter mfi(s_mf); mfi.isValid(); ++mfi){

        const Box& bx = mfi.growntilebox(1);

        Array4<const Real> const& s     = s_mf.array(mfi, state_comp);
        Array4<const Real> const& sint  = sint_mf.array(mfi);
        Array4<      Real> const& slope = slope_mf.array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            // compute initial estimates of slopes from unlimited corner points

            // local variables
            Real sumloc, redfac, redmax, div, kdp, sumdif, sgndif;

            Array1D<Real, 1, 4> diff;
            Array1D<Real, 1, 4> smin;
            Array1D<Real, 1, 4> smax;
            Array1D<Real, 1, 4> sc;

            // sx
            slope(i,j,k,0) = 0.5*(sint(i+1,j+1,k) + sint(i+1,j,k) - sint(i,j+1,k) - sint(i,j,k)) / hx;
            // sy
            slope(i,j,k,1) = 0.5*(sint(i+1,j+1,k) - sint(i+1,j,k) + sint(i,j+1,k) - sint(i,j,k)) / hy;
            // sxy
            slope(i,j,k,2) =     (sint(i+1,j+1,k) - sint(i+1,j,k) - sint(i,j+1,k) + sint(i,j,k)) / (hx*hy);

            if (limit_slopes) {

                // ++ / sint(i+1,j+1)
                sc(4) = s(i,j,k) + 0.5*(hx*slope(i,j,k,0) + hy*slope(i,j,k,1)) + 0.25*hx*hy*slope(i,j,k,2);

                // +- / sint(i+1,j  )
                sc(3) = s(i,j,k) + 0.5*(hx*slope(i,j,k,0) - hy*slope(i,j,k,1)) - 0.25*hx*hy*slope(i,j,k,2);

                // -+ / sint(i  ,j+1)
                sc(2) = s(i,j,k) - 0.5*(hx*slope(i,j,k,0) - hy*slope(i,j,k,1)) - 0.25*hx*hy*slope(i,j,k,2);

                // -- / sint(i  ,j  )
                sc(1) = s(i,j,k) - 0.5*(hx*slope(i,j,k,0) + hy*slope(i,j,k,1)) + 0.25*hx*hy*slope(i,j,k,2);

                // enforce max/min bounds
                smin(4) = min(s(i,j,k), s(i+1,j,k), s(i,j+1,k), s(i+1,j+1,k));
                smax(4) = max(s(i,j,k), s(i+1,j,k), s(i,j+1,k), s(i+1,j+1,k));

                smin(3) = min(s(i,j,k), s(i+1,j,k), s(i,j-1,k), s(i+1,j-1,k));
                smax(3) = max(s(i,j,k), s(i+1,j,k), s(i,j-1,k), s(i+1,j-1,k));

                smin(2) = min(s(i,j,k), s(i-1,j,k), s(i,j+1,k), s(i-1,j+1,k));
                smax(2) = max(s(i,j,k), s(i-1,j,k), s(i,j+1,k), s(i-1,j+1,k));

                smin(1) = min(s(i,j,k), s(i-1,j,k), s(i,j-1,k), s(i-1,j-1,k));
                smax(1) = max(s(i,j,k), s(i-1,j,k), s(i,j-1,k), s(i-1,j-1,k));

                for(int mm=1; mm<=4; ++mm){
                   sc(mm) = max(min(sc(mm), smax(mm)), smin(mm));
                }

                // iterative loop
                for(int ll=1; ll<=3; ++ll){
                   sumloc = 0.25*(sc(4) + sc(3) + sc(2) + sc(1));
                   sumdif = (sumloc - s(i,j,k))*4.0;
                   sgndif = std::copysign(1.0,sumdif);

                   for(int mm=1; mm<=4; ++mm){
                      diff(mm) = (sc(mm) - s(i,j,k))*sgndif;
                   }

                   kdp = 0;

                   for(int mm=1; mm<=4; ++mm){
                      if (diff(mm) > eps) {
                         kdp = kdp+1;
                      }
                   }

                   for(int mm=1; mm<=4; ++mm){
                      if (kdp<1) {
                         div = 1.0;
                      } else {
                         div = kdp;
                      }

                      if (diff(mm)>eps) {
                         redfac = sumdif*sgndif/div;
                         kdp = kdp-1;
                      } else {
                         redfac = 0.0;
                      }

                      if (sgndif > 0.0) {
                         redmax = sc(mm) - smin(mm);
                      } else {
                         redmax = smax(mm) - sc(mm);
                      }

                      redfac = min(redfac,redmax);
                      sumdif = sumdif - redfac*sgndif;
                      sc(mm) = sc(mm) - redfac*sgndif;
                   }
                }

                // final slopes
                // sx
                slope(i,j,k,0) = 0.5*( sc(4) + sc(3) -sc(1) - sc(2) )/hx;
                // sy
                slope(i,j,k,1) = 0.5*( sc(4) + sc(2) -sc(1) - sc(3) )/hy;
                // sxy
                slope(i,j,k,2) =     ( sc(1) + sc(4) -sc(2) - sc(3) )/(hx*hy);

            }
        });
    }
}


/**
 * Compute Conc for BDS algorithm.
 *
 * \param [in] s_mf MultiFab of state.
 * \param [in] state_comp Component of the MultiFab of state.
 * \param [in] geom Box geometry.
 * \param [in,out] xedge MuliFab containing x-edges.
 * \param [in,out] yedge MuliFab containing y-edges.
 * \param [in] edge_comp The component of the edge MultiFab.
 * \param [in] slope_mf MuliFab containing slope information.
 * \param [in] umac MuliFab for u-face velocity.
 * \param [in] vmac MuliFab for v-face velocity.
 * \param [in] fq Multifab for forces.
 * \param [in] fq_comp Component for Multifab for forces.
 * \param [in] dt Time step.
 *
 *
 */

void
BDS::ComputeConc (const MultiFab& s_mf,
                      const int state_comp,
                      const Geometry& geom,
                      MultiFab& xedge,
                      MultiFab& yedge,
                      const int edge_comp,
                      const MultiFab& slope_mf,
                      MultiFab const& umac,
                      MultiFab const& vmac,
                      MultiFab const& fq,
                      const int fq_comp,
                      const Real dt)
{

    BoxArray ba = s_mf.boxArray();
    DistributionMapping dmap = s_mf.DistributionMap();
    GpuArray<Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // local variables
    Real hx = dx[0];
    Real hy = dx[1];

    // calculate Gamma plus for flux F
    for ( MFIter mfi(umac); mfi.isValid(); ++mfi){

        const Box& bx = mfi.tilebox();

        Array4<const Real> const& s      = s_mf.array(mfi, state_comp);
        Array4<const Real> const& slope  = slope_mf.array(mfi);
        Array4<const Real> const& uadv   = umac.array(mfi);
        Array4<const Real> const& vadv   = vmac.array(mfi);
        Array4<const Real> const& force  = fq.array(mfi,fq_comp);

        Array4<      Real> const& siphj = xedge.array(mfi,edge_comp);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            //local variables
            Real hxs,hys;
            Real gamp,gamm;
            Real vtrans,stem,vaddif,vdif;
            Real u1,u2;
            Real vv;

            int iup,jup;
            Real isign, jsign;
            Real divu;


            if (uadv(i,j,k) > 0.0) {
               iup   = i-1;
               isign = 1.0;
            } else {
               iup   = i;
               isign = -1.0;
            }

            vtrans = vadv(iup,j+1,k);
            u1 = uadv(i,j,k);
            if (vtrans > 0.0) {
               jup   = j;
               jsign = 1.0;
               u2 = uadv(i,j,k);
            } else {
               jup   = j+1;
               jsign = -1.0;
               u2 = 0.0;
               if (uadv(i,j,k)*uadv(i,j+1,k) > 0.0) {
                  u2 = uadv(i,j+1,k);
               }
            }

            vv = vadv(iup,j+1,k);

            hxs = hx*isign;
            hys = hy*jsign;

            gamp = s(iup,jup,k)+
                 (hxs*.5 - (u1+u2)*dt/3.0)*slope(iup,jup,k,0) +
                 (hys*.5 -      vv*dt/3.0)*slope(iup,jup,k,1) +
                 (3.*hxs*hys-2.*(u1+u2)*dt*hys-2.*vv*hxs*dt+
                 vv*(2.*u2+u1)*dt*dt)     *slope(iup,jup,k,2)/12.0;

            // end of calculation of Gamma plus for flux F
            // ****************************************

            // *****************************************
            // calculate Gamma minus for flux F

            if (uadv(i,j,k) > 0.0) {
               iup   = i-1;
               isign = 1.0;
            } else {
               iup   = i;
               isign = -1.0;
            }

            vtrans = vadv(iup,j,k);
            u1 = uadv(i,j,k);
            if (vtrans > 0.0) {
               jup   = j-1;
               jsign = 1.0;
               u2 = 0.0;
               if (uadv(i,j,k)*uadv(i,j-1,k) > 0.0) {
                  u2 = uadv(i,j-1,k);
               }
            } else {
               jup   = j;
               jsign = -1.0;
               u2 = uadv(i,j,k);
            }

            vv = vadv(iup,j,k);

            hxs = hx*isign;
            hys = hy*jsign;

            gamm = s(iup,jup,k)+
                 (hxs*0.5 - (u1+u2)*dt/3.0)*slope(iup,jup,k,0) +
                 (hys*0.5 -      vv*dt/3.0)*slope(iup,jup,k,1) +
                 (3.0*hxs*hys-2.0*(u1+u2)*dt*hys-2.0*vv*hxs*dt +
                      vv*(2.0*u2+u1)*dt*dt)*slope(iup,jup,k,2)/12.0;

            // end of calculation of Gamma minus for flux F
            // ****************************************

            // *********************************
            // calculate siphj

            if (uadv(i,j,k) > 0.0) {
               iup   = i-1;
               isign = 1.0;
            } else {
               iup   = i;
               isign = -1.0;
            }

            vdif = 0.5*dt*(vadv(iup,j+1,k)*gamp -
                 vadv(iup,j,k)*gamm ) / hy;
            stem = s(iup,j,k) + (isign*hx - uadv(i,j,k)*dt)*0.5*slope(iup,j,k,0);
            vaddif = stem*0.5*dt*(
                    uadv(iup+1,j,k)-uadv(iup,j,k))/hx;
            divu = (uadv(iup+1,j,k)-uadv(iup,j,k))/hx +
                   (vadv(iup,j+1,k)-vadv(iup,j,k))/hy;
            siphj(i,j,k) = stem - vdif - vaddif + 0.5*dt*stem*divu + (dt/2.)*force(iup,j,k);

        });
    } // end of calculation of siphj}


    // calculate Gamma plus for flux G
    for ( MFIter mfi(vmac); mfi.isValid(); ++mfi){

        const Box& bx = mfi.tilebox();

        Array4<const Real> const& s      = s_mf.array(mfi, state_comp);
        Array4<const Real> const& slope  = slope_mf.array(mfi);
        Array4<const Real> const& uadv  = umac.array(mfi);
        Array4<const Real> const& vadv  = vmac.array(mfi);
        Array4<const Real> const& force  = fq.array(mfi,fq_comp);

        Array4<      Real> const& sijph = yedge.array(mfi, edge_comp);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k){

            //local variables
            Real hxs,hys;
            Real gamp,gamm;
            Real vtrans,stem,vaddif,vdif;
            Real v1,v2;
            Real uu;

            int iup,jup;
            Real isign, jsign;
            Real divu;

            if (vadv(i,j,k) > 0.0) {
               jup   = j-1;
               jsign = 1.0;
            } else {
               jup   = j;
               jsign = -1.0;
            }


            vtrans = uadv(i+1,jup,k);
            v1 = vadv(i,j,k);
            if (vtrans > 0.0) {
               iup   = i;
               isign = 1.0;
               v2 = vadv(i,j,k);
            } else {
               iup   = i+1;
               isign = -1.0;
               v2 = 0.0;
               if (vadv(i,j,k)*vadv(i+1,j,k) > 0.0) {
                  v2 = vadv(i+1,j,k);
               }
            }

            uu = uadv(i+1,jup,k);

            hxs = hx*isign;
            hys = hy*jsign;

            gamp = s(iup,jup,k)+
                 (hys*0.5 - (v1+v2)*dt/3.0)*slope(iup,jup,k,1) +
                 (hxs*0.5 - uu*dt/3.0)     *slope(iup,jup,k,0) +
                 (3.0*hxs*hys-2.0*(v1+v2)*dt*hxs-2.*uu*hys*dt+
                 (2.0*v2+v1)*uu*dt*dt)     *slope(iup,jup,k,2)/12.0;

            // end of calculation of Gamma plus for flux G
            // ****************************************

            // *****************************************
            // calculate Gamma minus for flux G

            if (vadv(i,j,k) > 0.0) {
               jup   = j-1;
               jsign = 1.0;
            } else {
               jup   = j;
               jsign = -1.0;
            }

            vtrans = uadv(i,jup,k);
            v1 = vadv(i,j,k);
            if (vtrans > 0.0) {
               iup   = i-1;
               isign = 1.0;
               v2 = 0.0;
               if (vadv(i,j,k)*vadv(i-1,j,k) > 0) {
                  v2 = vadv(i-1,j,k);
               }
            } else {
               iup   = i;
               isign = -1.0;
               v2 = vadv(i,j,k);
            }

            uu = uadv(i,jup,k);

            hxs = hx*isign;
            hys = hy*jsign;

            gamm = s(iup,jup,k) +
                 (hys*.5 - (v1+v2)*dt/3.)*slope(iup,jup,k,1) +
                 (hxs*.5 - uu*dt/3.)     *slope(iup,jup,k,0) +
                 (3.*hxs*hys-2.*(v1+v2)*dt*hxs-2.*uu*hys*dt+
                 (2.*v2+v1)*uu*dt*dt)    *slope(iup,jup,k,2)/12.0;

            // end of calculation of Gamma minus for flux G
            // ****************************************

            // *********************************
            // calculate sijph

            if (vadv(i,j,k) > 0) {
               jup   = j-1;
               jsign = 1.0;
            } else {
               jup   = j;
               jsign = -1.0;
            }

            vdif = 0.5*dt*
                 (uadv(i+1,jup,k)*gamp-uadv(i,jup,k)*gamm)/hx;
            stem = s(i,jup,k) + (jsign*hy - vadv(i,j,k)*dt)*0.5*slope(i,jup,k,1);
            vaddif = stem*0.5*dt*(vadv(i,jup+1,k) - vadv(i,jup,k))/hy;
            divu =  (uadv(i+1,jup,k)-uadv(i,jup,k))/hx +
                 (vadv(i,jup+1,k)-vadv(i,jup,k))/hy;
            sijph(i,j,k) = stem - vdif - vaddif + 0.5*dt*stem*divu + (dt/2.)*force(i,jup,k);

            // end of calculation of sijph
            // *************************************
        });
    }
}

/** @} */
