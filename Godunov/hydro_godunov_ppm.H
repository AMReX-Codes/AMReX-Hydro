/*
 * \file hydro_godunov_ppm.H
 *
 * \addtogroup Godunov
 *  @{
 */

#ifndef HYDRO_GODUNOV_PPM_H
#define HYDRO_GODUNOV_PPM_H

#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_BC_TYPES.H>
#include <hydro_bcs_K.H>
#include <hydro_constants.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
minmod_fn (const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           amrex::Real& dsm,
           amrex::Real& dsp)
{
    // Calculate gradients on both sides
    dsp = sp1 - s0;
    dsm = s0 - sm1;

    if (!(dsp * dsm < amrex::Real(0.0))) {
        // Select the smaller slope if same sign
        if (std::abs(dsp) < std::abs(dsm)) {
            dsm = dsp;
        } else {
            dsp = dsm;
        }
    } else {
        // Set to zero if opposite sign
        dsp = amrex::Real(0.0);
        dsm = amrex::Real(0.0);
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
Godunov_minmod_bc_lo (const int n,
                      const amrex::Real sm1,
                      const amrex::Real s0,
                      const amrex::Real vel_edge,
                      amrex::Real& dsm,
                      const int bclo,
                      const int domlo)
{
    // Ensure that left-side slope is used unchanged
    if (n == domlo) {
        if ( bclo == amrex::BCType::ext_dir || bclo == amrex::BCType::hoextrap ||
            (bclo == amrex::BCType::direction_dependent && vel_edge >= amrex::Real(0.0)) )
        {
            dsm = s0 - sm1;
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void
Godunov_minmod_bc_hi (const int n,
                      const amrex::Real s0,
                      const amrex::Real sp1,
                      const amrex::Real vel_edge,
                      amrex::Real& dsp,
                      const int bchi,
                      const int domhi)
{
    // Ensure that the right-side slope is used unchanged
    if (n == domhi) {
        if ( bchi == amrex::BCType::ext_dir || bchi == amrex::BCType::hoextrap ||
            (bchi == amrex::BCType::direction_dependent && vel_edge <= amrex::Real(0.0)) )
        {
            dsp = sp1 - s0;
        }
    }
}

namespace PPM {

enum limiters {VanLeer, WENOZ, WENO_JS, NoLimiter, UPWIND, MINMOD};

static constexpr int default_limiter = VanLeer;

struct nolimiter {

    static constexpr bool do_limiting = false;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge1(const amrex::Real sm2,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real /*sp2*/)
    {

        amrex::Real d1 = amrex::Real(0.5) * (sp1 - sm1);
        amrex::Real d2 = amrex::Real(0.5) * (s0 - sm2);

        amrex::Real sedge = amrex::Real(0.5)*(s0 + sm1) - (amrex::Real(1)/amrex::Real(6))*(d1 - d2);
        return sedge;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge2(const amrex::Real /*sm2*/,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real sp2)
    {
        amrex::Real d1 = amrex::Real(0.5) * (sp2 - s0);
        amrex::Real d2 = amrex::Real(0.5) * (sp1 - sm1);

        amrex::Real sedge = amrex::Real(0.5)*(sp1 + s0) - (amrex::Real(1)/amrex::Real(6))*(d1 - d2);
        return sedge;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::GpuTuple<amrex::Real,amrex::Real>
    sm_sp(const amrex::Real /*s0*/,
         const amrex::Real sedge1,
         const amrex::Real sedge2)
    {
        return amrex::makeTuple(sedge1, sedge2);
    }
}; // nolimiter

struct vanleer {

    static constexpr bool do_limiting = true;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real vanLeer(const amrex::Real a, const amrex::Real b, const amrex::Real c)
    {
        constexpr auto small_qty_sq = amrex::Real(1.e-10) * amrex::Real(1.e-10);

        amrex::Real dsc = amrex::Real(0.5)*(b - c);
        amrex::Real dsl = amrex::Real(2.0)*(a - c);
        amrex::Real dsr = amrex::Real(2.0)*(b - a);
        return (dsl*dsr > small_qty_sq) ?
            std::copysign(amrex::Real(1.0), dsc)*amrex::min(amrex::Math::abs(dsc),amrex::min(amrex::Math::abs(dsl),
                                                                                             amrex::Math::abs(dsr))) : amrex::Real(0.0);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge1(const amrex::Real sm2,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real /*sp2*/)
    {

        amrex::Real d1 = vanLeer(s0,sp1,sm1);
        amrex::Real d2 = vanLeer(sm1,s0,sm2);

        amrex::Real sedge = amrex::Real(0.5)*(s0 + sm1) - (amrex::Real(1)/amrex::Real(6))*(d1 - d2);
        return amrex::min(amrex::max(sedge, amrex::min(s0, sm1)),amrex::max(s0,sm1));

    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge2(const amrex::Real /*sm2*/,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real sp2)
    {

        amrex::Real d1 = vanLeer(sp1,sp2,s0);
        amrex::Real d2 = vanLeer(s0,sp1,sm1);

        amrex::Real sedge = amrex::Real(0.5)*(sp1 + s0) - (amrex::Real(1)/amrex::Real(6))*(d1 - d2);
        return amrex::min(amrex::max(sedge, amrex::min(s0, sp1)),amrex::max(s0,sp1));


    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::GpuTuple<amrex::Real,amrex::Real>
    sm_sp(const amrex::Real s0,
         const amrex::Real sedge1,
         const amrex::Real sedge2)
    {
        amrex::Real sm_ = sedge1;
        amrex::Real sp_ = sedge2;

        if ((sedge2-s0)*(s0-sedge1) < amrex::Real(0.0)) {
            sm_ = s0;
            sp_ = s0;
        } else if (amrex::Math::abs(sedge2-s0) >= amrex::Real(2.0)*amrex::Math::abs(sedge1-s0)) {
            sp_ = amrex::Real(3)*s0 - amrex::Real(2)*sedge1;
        } else if (amrex::Math::abs(sedge1-s0) >=  amrex::Real(2.0)*amrex::Math::abs(sedge2-s0)) {
            sm_ = amrex::Real(3)*s0 - amrex::Real(2)*sedge2;
        }
        return amrex::makeTuple(sm_,sp_);

        // this might be the correct way to do this in case the else if are both == 0.0
        // could also change the >= to just > for each
//        if ((sedge2-s0)*(s0-sedge1) < amrex::Real(0.0)) {
//            return amrex::makeTuple(s0,s0);
//        }
//        amrex::Real sm_ = sedge1;
//        amrex::Real sp_ = sedge2;
//        if (amrex::Math::abs(sedge2-s0) >= amrex::Real(2.0)*amrex::Math::abs(sedge1-s0)) {
//            sp_ = amrex::Real(3)*s0 - amrex::Real(2)*sedge1;
//        }
//        if (amrex::Math::abs(sedge1-s0) >=  amrex::Real(2.0)*amrex::Math::abs(sedge2-s0)) {
//            sm_ = amrex::Real(3)*s0 - amrex::Real(2)*sedge2;
//        }
//        return amrex::makeTuple(sm_,sp_);

    }
}; // vanleer

struct wenoz {

    static constexpr bool do_limiting = false;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge2(const amrex::Real sm2,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real sp2)
    {
        constexpr auto eps = amrex::Real(1.0e-6);

        const amrex::Real beta1 =
            amrex::Real(13) / amrex::Real(12) * (sm2 - amrex::Real(2) * sm1 + s0) * (sm2 - amrex::Real(2) * sm1 + s0) +
            amrex::Real(0.25) * (sm2 - amrex::Real(4) * sm1 + amrex::Real(3) * s0) * (sm2 - amrex::Real(4) * sm1 + amrex::Real(3) * s0);
        const amrex::Real beta2 =
            amrex::Real(13) / amrex::Real(12) * (sm1 - amrex::Real(2) * s0 + sp1) * (sm1 - amrex::Real(2) * s0 + sp1) +
            amrex::Real(0.25) * (sm1 - sp1) * (sm1 - sp1);
        const amrex::Real beta3 =
            amrex::Real(13) / amrex::Real(12) * (s0 - amrex::Real(2) * sp1 + sp2) * (s0 - amrex::Real(2) * sp1 + sp2) +
            amrex::Real(0.25) * (amrex::Real(3) * s0 - amrex::Real(4) * sp1 + sp2) * (amrex::Real(3) * s0 - amrex::Real(4) * sp1 + sp2);

        const amrex::Real t5 = amrex::Math::abs(beta3 - beta1);
        const amrex::Real omega1 = amrex::Real(0.1) * (amrex::Real(1) + t5 / (eps + beta1));
        const amrex::Real omega2 = amrex::Real(0.6) * (amrex::Real(1) + t5 / (eps + beta2));
        const amrex::Real omega3 = amrex::Real(0.3) * (amrex::Real(1) + t5 / (eps + beta3));

        const amrex::Real omega = omega1 + omega2 + omega3;

        const amrex::Real v_1 = amrex::Real(2) * sm2 - amrex::Real(7) * sm1 + amrex::Real(11) * s0;
        const amrex::Real v_2 = -sm1 + amrex::Real(5) * s0 + amrex::Real(2) * sp1;
        const amrex::Real v_3 = amrex::Real(2) * s0 + amrex::Real(5) * sp1 - sp2;

        return (omega1 * v_1 + omega2 * v_2 + omega3 * v_3) / (amrex::Real(6) * omega);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge1(const amrex::Real sm2,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real sp2)
    {
        return sedge2(sp2,sp1,s0,sm1,sm2); // NOLINT(readability-suspicious-call-argument)
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::GpuTuple<amrex::Real,amrex::Real>
    sm_sp(const amrex::Real /*s0*/,
          const amrex::Real sedge1,
          const amrex::Real sedge2)
    {
        return amrex::makeTuple(sedge1, sedge2);
    }
}; // wenoz

struct weno_js {

    static constexpr bool do_limiting = false;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge2(const amrex::Real sm2,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real sp2)
    {
        constexpr auto eps = amrex::Real(1.0e-6);

        const amrex::Real beta1 =
            amrex::Real(13) / amrex::Real(12) * (sm2 - amrex::Real(2) * sm1 + s0) *
                                                (sm2 - amrex::Real(2) * sm1 + s0) +
            amrex::Real(0.25) * (sm2 - amrex::Real(4) * sm1 + amrex::Real(3) * s0) *
                                (sm2 - amrex::Real(4) * sm1 + amrex::Real(3) * s0);
        const amrex::Real beta2 =
            amrex::Real(13) / amrex::Real(12) * (sm1 - amrex::Real(2) * s0 + sp1) *
                                                (sm1 - amrex::Real(2) * s0 + sp1) +
            amrex::Real(0.25) * (sm1 - sp1) * (sm1 - sp1);
        const amrex::Real beta3 =
            amrex::Real(13) / amrex::Real(12) * (s0 - amrex::Real(2) * sp1 + sp2) *
                                                (s0 - amrex::Real(2) * sp1 + sp2) +
            amrex::Real(0.25) * (amrex::Real(3) * s0 - amrex::Real(4) * sp1 + sp2) *
                                (amrex::Real(3) * s0 - amrex::Real(4) * sp1 + sp2);

        const amrex::Real omega1 = amrex::Real(0.1) / (eps + beta1);
        const amrex::Real omega2 = amrex::Real(0.6) / (eps + beta2);
        const amrex::Real omega3 = amrex::Real(0.3) / (eps + beta3);

        const amrex::Real omega = omega1 + omega2 + omega3;

        const amrex::Real v_1 = amrex::Real(2) * sm2 - amrex::Real(7) * sm1 + amrex::Real(11) * s0;
        const amrex::Real v_2 = -sm1 + amrex::Real(5) * s0 + amrex::Real(2) * sp1;
        const amrex::Real v_3 = amrex::Real(2) * s0 + amrex::Real(5) * sp1 - sp2;

        return (omega1 * v_1 + omega2 * v_2 + omega3 * v_3) / (amrex::Real(6) * omega);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge1(const amrex::Real sm2,
           const amrex::Real sm1,
           const amrex::Real s0,
           const amrex::Real sp1,
           const amrex::Real sp2)
    {
        return sedge2(sp2,sp1,s0,sm1,sm2); // NOLINT(readability-suspicious-call-argument)
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::GpuTuple<amrex::Real,amrex::Real>
    sm_sp(const amrex::Real /*s0*/,
          const amrex::Real sedge1,
          const amrex::Real sedge2)
    {
        return amrex::makeTuple(sedge1, sedge2);
    }
}; // weno_js

struct upwind {

    static constexpr bool do_limiting = false;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge1(const amrex::Real /*sm2*/,
           const amrex::Real /*sm1*/,
           const amrex::Real /*s0*/,
           const amrex::Real /*sp1*/,
           const amrex::Real /*sp2*/)
    {
        return 0;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge2(const amrex::Real /*sm2*/,
           const amrex::Real /*sm1*/,
           const amrex::Real /*s0*/,
           const amrex::Real /*sp1*/,
           const amrex::Real /*sp2*/)
    {
        return 0;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::GpuTuple<amrex::Real,amrex::Real>
    sm_sp(const amrex::Real /*s0*/,
         const amrex::Real sedge1,
         const amrex::Real sedge2)
    {
        return amrex::makeTuple(sedge1, sedge2);
    }
}; // upwind

struct minmod {

    static constexpr bool do_limiting = false;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge1(const amrex::Real /*sm2*/,
           const amrex::Real /*sm1*/,
           const amrex::Real /*s0*/,
           const amrex::Real /*sp1*/,
           const amrex::Real /*sp2*/)
    {
        return 0;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::Real
    sedge2(const amrex::Real /*sm2*/,
           const amrex::Real /*sm1*/,
           const amrex::Real /*s0*/,
           const amrex::Real /*sp1*/,
           const amrex::Real /*sp2*/)
    {
        return 0;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static amrex::GpuTuple<amrex::Real,amrex::Real>
    sm_sp(const amrex::Real /*s0*/,
         const amrex::Real sedge1,
         const amrex::Real sedge2)
    {
        return amrex::makeTuple(sedge1, sedge2);
    }
}; // minmod

template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetXBCs ( const int i, const int j, const int k, const int n,
               amrex::Real &sm, amrex::Real &sp,
               amrex::Real &sedge1, amrex::Real &sedge2,
               const amrex::Array4<const amrex::Real> &s,
               const amrex::Real vel_dlo, const amrex::Real vel_dhi,
               const int bclo, const int bchi,
               const int domlo, const int domhi)
{
    using namespace amrex;

    // Both the domlo and the domlo+1 branch below use s(domlo-1) as a face value.
    // For direction_dependent that only holds when the domain face is inflow, so the
    // gate is the domain-boundary velocity vel_dlo, not the velocity of the face
    // bounding the cell being filled.
    if ( (bclo == BCType::ext_dir) || (bclo == BCType::hoextrap) ||
         (bclo == BCType::direction_dependent && vel_dlo >= Real(0)) )
    {
        if (i == domlo)
        {
            sedge2 = -amrex::Real(0.2)*s(domlo-1,j,k,n) + amrex::Real(0.75)*s(domlo,j, k, n)
                     +amrex::Real(0.5)*s(domlo+1,j,k,n) - amrex::Real(0.05)*s(domlo+2,j,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge2 = amrex::max(sedge2, amrex::min(s(domlo+1,j,k,n), s(domlo,j,k,n)));
                sedge2 = amrex::min(sedge2, amrex::max(s(domlo+1,j,k,n), s(domlo,j,k,n)));
            }

            sm = s(domlo-1,j,k,n);
            sp = sedge2;

        } else if (i == domlo+1) {

            sedge1 = -amrex::Real(0.2)*s(domlo-1,j,k,n) + amrex::Real(0.75)*s(domlo  ,j,k,n)
                   +  amrex::Real(0.5)*s(domlo+1,j,k,n) - amrex::Real(0.05)*s(domlo+2,j,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge1 = amrex::max(sedge1, amrex::min(s(domlo+1,j,k,n), s(domlo,j,k,n)));
                sedge1 = amrex::min(sedge1, amrex::max(s(domlo+1,j,k,n), s(domlo,j,k,n)));
            }

            sp = sedge2;
            sm = sedge1;

            if constexpr (Limiter::do_limiting) {
                if ( (sp - s(domlo+1,j,k,n))*(s(domlo+1,j,k,n) - sm) <= amrex::Real(0))
                {
                    sp = s(domlo+1,j,k,n);
                    sm = s(domlo+1,j,k,n);
                } else if(amrex::Math::abs(sp - s(domlo+1,j,k,n)) >= amrex::Real(2)*amrex::Math::abs(sm - s(domlo+1,j,k,n))) {
                    sp = amrex::Real(3)*s(domlo+1,j,k,n) - amrex::Real(2)*sm;
                } else if(amrex::Math::abs(sm - s(domlo+1,j,k,n)) >= amrex::Real(2)*amrex::Math::abs(sp - s(domlo+1,j,k,n))) {
                    sm = amrex::Real(3)*s(domlo+1,j,k,n) - amrex::Real(2)*sp;
                }
            }
        }
    }

    // Likewise the domhi and domhi-1 branches are gated on the hi domain face.
    if ( (bchi == BCType::ext_dir) || (bchi == BCType::hoextrap) ||
         (bchi == BCType::direction_dependent && vel_dhi <= Real(0)) )
    {
        if (i == domhi)
        {
            sedge1 = -amrex::Real(0.2)*s(domhi+1,j,k,n) + amrex::Real(0.75)*s(domhi,j,k, n)
                     +amrex::Real(0.5)*s(domhi-1,j,k,n) - amrex::Real(0.05)*s(domhi-2,j,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge1 = amrex::max(sedge1, amrex::min(s(domhi-1,j,k,n), s(domhi,j,k,n)));
                sedge1 = amrex::min(sedge1, amrex::max(s(domhi-1,j,k,n), s(domhi,j,k,n)));
            }

            sp = s(domhi+1,j,k,n);
            sm = sedge1;

        } else if (i == domhi-1) {

            sedge2 = -amrex::Real(0.2)*s(domhi+1,j,k,n) + amrex::Real(0.75)*s(domhi  ,j,k,n)
                     +amrex::Real(0.5)*s(domhi-1,j,k,n) - amrex::Real(0.05)*s(domhi-2,j,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge2 = amrex::max(sedge2, amrex::min(s(domhi-1,j,k,n), s(domhi,j,k,n)));
                sedge2 = amrex::min(sedge2, amrex::max(s(domhi-1,j,k,n), s(domhi,j,k,n)));
            }

            sp = sedge2;
            sm = sedge1;

            if constexpr (Limiter::do_limiting) {
                if( (sp - s(domhi-1,j,k,n))*(s(domhi-1,j,k,n) - sm) <= amrex::Real(0))
                {
                    sp = s(domhi-1,j,k,n);
                    sm = s(domhi-1,j,k,n);
                } else if(amrex::Math::abs(sp - s(domhi-1,j,k,n)) >= amrex::Real(2.)*amrex::Math::abs(sm - s(domhi-1,j,k,n))) {
                    sp = amrex::Real(3)*s(domhi-1,j,k,n) - amrex::Real(2)*sm;
                } else if(amrex::Math::abs(sm - s(domhi-1,j,k,n)) >= amrex::Real(2.)*amrex::Math::abs(sp - s(domhi-1,j,k,n))) {
                    sm = amrex::Real(3)*s(domhi-1,j,k,n) - amrex::Real(2)*sp;
                }
            }
        }
    }
}

template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetYBCs ( const int i, const int j, const int k, const int n,
               amrex::Real &sm, amrex::Real &sp,
               amrex::Real &sedge1, amrex::Real &sedge2,
               const amrex::Array4<const amrex::Real> &s,
               const amrex::Real vel_dlo, const amrex::Real vel_dhi,
               const int bclo, const int bchi,
               const int domlo, const int domhi)
{
    using namespace amrex;

    // Both the domlo and the domlo+1 branch below use s(domlo-1) as a face value.
    // For direction_dependent that only holds when the domain face is inflow, so the
    // gate is the domain-boundary velocity vel_dlo, not the velocity of the face
    // bounding the cell being filled.
    if ( (bclo == BCType::ext_dir) || (bclo == BCType::hoextrap) ||
         (bclo == BCType::direction_dependent && vel_dlo >= Real(0)) )
    {
        if (j == domlo)
        {
            sedge2 = -amrex::Real(0.2)*s(i,domlo-1,k,n) + amrex::Real(0.75)*s(i,domlo  ,k,n)
                     +amrex::Real(0.5)*s(i,domlo+1,k,n) - amrex::Real(0.05)*s(i,domlo+2,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge2 = amrex::max(sedge2, amrex::min(s(i,domlo+1,k,n), s(i,domlo,k,n)));
                sedge2 = amrex::min(sedge2, amrex::max(s(i,domlo+1,k,n), s(i,domlo,k,n)));
            }

            sm = s(i,domlo-1,k,n);
            sp = sedge2;

        } else if (j == domlo+1) {

            sedge1 = -amrex::Real(0.2)*s(i,domlo-1,k,n) + amrex::Real(0.75)*s(i,domlo  ,k,n)
                     +amrex::Real(0.5)*s(i,domlo+1,k,n) - amrex::Real(0.05)*s(i,domlo+2,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge1 = amrex::max(sedge1, amrex::min(s(i,domlo+1,k,n), s(i,domlo,k,n)));
                sedge1 = amrex::min(sedge1, amrex::max(s(i,domlo+1,k,n), s(i,domlo,k,n)));
            }

            sp = sedge2;
            sm = sedge1;

            if constexpr (Limiter::do_limiting) {
                if ( (sp - s(i,domlo+1,k,n))*(s(i,domlo+1,k,n) - sm) <= amrex::Real(0))
                {
                    sp = s(i,domlo+1,k,n);
                    sm = s(i,domlo+1,k,n);
                } else if(amrex::Math::abs(sp - s(i,domlo+1,k,n)) >= amrex::Real(2)*amrex::Math::abs(sm - s(i,domlo+1,k,n))) {
                    sp = amrex::Real(3)*s(i,domlo+1,k,n) - amrex::Real(2)*sm;
                } else if(amrex::Math::abs(sm - s(i,domlo+1,k,n)) >= amrex::Real(2)*amrex::Math::abs(sp - s(i,domlo+1,k,n))) {
                    sm = amrex::Real(3)*s(i,domlo+1,k,n) - amrex::Real(2)*sp;
                }
            }
        }
    }

    // Likewise the domhi and domhi-1 branches are gated on the hi domain face.
    if ( (bchi == BCType::ext_dir) || (bchi == BCType::hoextrap) ||
         (bchi == BCType::direction_dependent && vel_dhi <= Real(0)) )
    {
        if (j == domhi)
        {
            sedge1 = -amrex::Real(0.2)*s(i,domhi+1,k,n) + amrex::Real(0.75)*s(i,domhi  ,k,n)
                     +amrex::Real(0.5)*s(i,domhi-1,k,n) - amrex::Real(0.05)*s(i,domhi-2,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge1 = amrex::max(sedge1, amrex::min(s(i,domhi-1,k,n), s(i,domhi,k,n)));
                sedge1 = amrex::min(sedge1, amrex::max(s(i,domhi-1,k,n), s(i,domhi,k,n)));
            }

            sp = s(i,domhi+1,k, n);
            sm = sedge1;

        } else if (j == domhi-1) {

            sedge2 = -amrex::Real(0.2)*s(i,domhi+1,k,n) + amrex::Real(0.75)*s(i,domhi  ,k,n)
                     +amrex::Real(0.5)*s(i,domhi-1,k,n) - amrex::Real(0.05)*s(i,domhi-2,k,n);
            if constexpr (Limiter::do_limiting) {
                sedge2 = amrex::max(sedge2, amrex::min(s(i,domhi-1,k,n), s(i,domhi,k,n)));
                sedge2 = amrex::min(sedge2, amrex::max(s(i,domhi-1,k,n), s(i,domhi,k,n)));
            }

            sp = sedge2;
            sm = sedge1;

            if constexpr (Limiter::do_limiting) {
                if( (sp - s(i,domhi-1,k,n))*(s(i,domhi-1,k,n) - sm) <= amrex::Real(0)){
                    sp = s(i,domhi-1,k,n);
                    sm = s(i,domhi-1,k,n);
                } else if(amrex::Math::abs(sp - s(i,domhi-1,k,n)) >= amrex::Real(2.)*amrex::Math::abs(sm - s(i,domhi-1,k,n))) {
                    sp = amrex::Real(3)*s(i,domhi-1,k,n) - amrex::Real(2)*sm;
                } else if(amrex::Math::abs(sm - s(i,domhi-1,k,n)) >= amrex::Real(2.)*amrex::Math::abs(sp - s(i,domhi-1,k,n))) {
                    sm = amrex::Real(3)*s(i,domhi-1,k,n) - amrex::Real(2)*sp;
                }
            }
        }
    }
}

#if (AMREX_SPACEDIM==3)
template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void SetZBCs ( const int i, const int j, const int k, const int n,
               amrex::Real &sm, amrex::Real &sp,
               amrex::Real &sedge1, amrex::Real &sedge2,
               const amrex::Array4<const amrex::Real> &s,
               const amrex::Real vel_dlo, const amrex::Real vel_dhi,
               const int bclo, const int bchi,
               const int domlo, const int domhi)
{
    using namespace amrex;

    // Both the domlo and the domlo+1 branch below use s(domlo-1) as a face value.
    // For direction_dependent that only holds when the domain face is inflow, so the
    // gate is the domain-boundary velocity vel_dlo, not the velocity of the face
    // bounding the cell being filled.
    if ( (bclo == BCType::ext_dir) || (bclo == BCType::hoextrap) ||
         (bclo == BCType::direction_dependent && vel_dlo >= Real(0)) )
    {
        if (k == domlo)
        {
            sedge2 = -amrex::Real(0.2)*s(i,j,domlo-1,n) + amrex::Real(0.75)*s(i,j,domlo  ,n)
                     +amrex::Real(0.5)*s(i,j,domlo+1,n) - amrex::Real(0.05)*s(i,j,domlo+2,n);
            if constexpr (Limiter::do_limiting) {
                sedge2 = amrex::max(sedge2, amrex::min(s(i,j,domlo+1,n), s(i,j,domlo,n)));
                sedge2 = amrex::min(sedge2, amrex::max(s(i,j,domlo+1,n), s(i,j,domlo,n)));
            }

            sm = s(i,j,domlo-1,n);
            sp = sedge2;

        } else if (k == domlo+1) {

            sedge1 = -amrex::Real(0.2)*s(i,j,domlo-1,n) + amrex::Real(0.75)*s(i,j,domlo  ,n)
                     +amrex::Real(0.5)*s(i,j,domlo+1,n) - amrex::Real(0.05)*s(i,j,domlo+2,n);
            if constexpr (Limiter::do_limiting) {
                sedge1 = amrex::max(sedge1, amrex::min(s(i,j,domlo+1,n), s(i,j,domlo,n)));
                sedge1 = amrex::min(sedge1, amrex::max(s(i,j,domlo+1,n), s(i,j,domlo,n)));
            }

            sp = sedge2;
            sm = sedge1;

            if constexpr (Limiter::do_limiting) {
                if ( (sp - s(i,j,domlo+1,n))*(s(i,j,domlo+1,n) - sm) <= amrex::Real(0.) )
                {
                    sp = s(i,j,domlo+1,n);
                    sm = s(i,j,domlo+1,n);
                } else if(amrex::Math::abs(sp - s(i,j,domlo+1,n)) >= amrex::Real(2.)*amrex::Math::abs(sm - s(i,j,domlo+1,n))) {
                    sp = amrex::Real(3)*s(i,j,domlo+1,n) - amrex::Real(2)*sm;
                } else if(amrex::Math::abs(sm - s(i,j,domlo+1,n)) >= amrex::Real(2.)*amrex::Math::abs(sp - s(i,j,domlo+1,n))) {
                    sm = amrex::Real(3)*s(i,j,domlo+1,n) - amrex::Real(2)*sp;
                }
            }
        }
    }

    // Likewise the domhi and domhi-1 branches are gated on the hi domain face.
    if ( (bchi == BCType::ext_dir) || (bchi == BCType::hoextrap) ||
         (bchi == BCType::direction_dependent && vel_dhi <= Real(0)) )
    {
        if (k == domhi)
        {
            sedge1 =  -amrex::Real(0.2)*s(i,j,domhi+1,n) + amrex::Real(0.75)*s(i,j,domhi  ,n)
                      +amrex::Real(0.5)*s(i,j,domhi-1,n) - amrex::Real(0.05)*s(i,j,domhi-2,n);
            if constexpr (Limiter::do_limiting) {
                sedge1 = amrex::max(sedge1, amrex::min(s(i,j,domhi-1,n), s(i,j,domhi,n)));
                sedge1 = amrex::min(sedge1, amrex::max(s(i,j,domhi-1,n), s(i,j,domhi,n)));
            }

            sp = s(i,j,domhi+1,n);
            sm = sedge1;

        } else if (k == domhi-1) {

            sedge2 = -amrex::Real(0.2)*s(i,j,domhi+1,n) + amrex::Real(0.75)*s(i,j,domhi  ,n)
                     +amrex::Real(0.5)*s(i,j,domhi-1,n) - amrex::Real(0.05)*s(i,j,domhi-2,n);
            if constexpr (Limiter::do_limiting) {
                sedge2 = amrex::max(sedge2, amrex::min(s(i,j,domhi-1,n), s(i,j,domhi,n)));
                sedge2 = amrex::min(sedge2, amrex::max(s(i,j,domhi-1,n), s(i,j,domhi,n)));
            }

            sp = sedge2;
            sm = sedge1;

            if constexpr (Limiter::do_limiting) {
                if ( (sp - s(i,j,domhi-1,n))*(s(i,j,domhi-1,n) - sm) <= amrex::Real(0.) )
                {
                    sp = s(i,j,domhi-1,n);
                    sm = s(i,j,domhi-1,n);
                } else if(amrex::Math::abs(sp - s(i,j,domhi-1,n)) >= amrex::Real(2.)*amrex::Math::abs(sm - s(i,j,domhi-1,n))) {
                    sp = amrex::Real(3)*s(i,j,domhi-1,n) - amrex::Real(2)*sm;
                } else if(amrex::Math::abs(sm - s(i,j,domhi-1,n)) >= amrex::Real(2.)*amrex::Math::abs(sp - s(i,j,domhi-1,n))) {
                    sm = amrex::Real(3)*s(i,j,domhi-1,n) - amrex::Real(2)*sp;
                }
            }

        }
    }
}
#endif


// Right now only ppm type 1 is supported on GPU
// This version is called before the MAC projection, when we use the cell-centered velocity
//      for upwinding
template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictVelOnXFace ( const int i, const int j, const int k, const int n,
                         const amrex::Real dtdx, const amrex::Real v_ad,
                         const amrex::Real vel_dlo, const amrex::Real vel_dhi,
                         const amrex::Array4<const amrex::Real> &S,
                         const amrex::Array4<amrex::Real> &Im,
                         const amrex::Array4<amrex::Real> &Ip,
                         const amrex::BCRec bc, const int domlo, const int domhi,
                         const Limiter& /*limiter*/)
{
    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1)};
    constexpr auto two3rds{amrex::Real(2.0)/amrex::Real(3.0)};

    const amrex::Real sm2 = S(i-2,j,k,n);
    const amrex::Real sm1 = S(i-1,j,k,n);
    const amrex::Real s0  = S(i  ,j,k,n);
    const amrex::Real sp1 = S(i+1,j,k,n);
    const amrex::Real sp2 = S(i+2,j,k,n);

    amrex::Real sedge1 = Limiter::sedge1(sm2,sm1,s0,sp1,sp2);
    amrex::Real sedge2 = Limiter::sedge2(sm2,sm1,s0,sp1,sp2);

    auto [sm, sp] = Limiter::sm_sp(s0, sedge1, sedge2);

    SetXBCs<Limiter>(i, j, k, n, sm, sp, sedge1, sedge2, S, vel_dlo, vel_dhi,
                     bc.lo(0), bc.hi(0), domlo, domhi);

    const amrex::Real s6 = amrex::Real(6)*s0 - amrex::Real(3)*(sm + sp);

    const amrex::Real sigma = amrex::Math::abs(v_ad)*dtdx;

    if (v_ad > small_vel)
    {
        Ip(i,j,k,n) = sp - (half*sigma)*((sp-sm) - (one - two3rds*sigma)*s6);
        Im(i,j,k,n) = S(i,j,k,n);
    }
    else if (v_ad < -small_vel)
    {
        Ip(i,j,k,n) = S(i,j,k,n);
        Im(i,j,k,n) = sm + (half*sigma)*((sp-sm) + (one - two3rds*sigma)*s6);
    } else
    {
        Ip(i,j,k,n) = S(i,j,k,n);
        Im(i,j,k,n) = S(i,j,k,n);
    }
}

template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictVelOnYFace ( const int i, const int j, const int k, const int n,
                         const amrex::Real dtdy, const amrex::Real v_ad,
                         const amrex::Real vel_dlo, const amrex::Real vel_dhi,
                         const amrex::Array4<const amrex::Real> &S,
                         const amrex::Array4<amrex::Real> &Im,
                         const amrex::Array4<amrex::Real> &Ip,
                         const amrex::BCRec bc, const int domlo, const int domhi,
                         const Limiter& /*limiter*/)
{
    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1)};
    constexpr auto two3rds{amrex::Real(2.0)/amrex::Real(3.0)};

    const amrex::Real sm2 = S(i,j-2,k,n);
    const amrex::Real sm1 = S(i,j-1,k,n);
    const amrex::Real s0  = S(i,j  ,k,n);
    const amrex::Real sp1 = S(i,j+1,k,n);
    const amrex::Real sp2 = S(i,j+2,k,n);

    amrex::Real sedge1 = Limiter::sedge1(sm2,sm1,s0,sp1,sp2);
    amrex::Real sedge2 = Limiter::sedge2(sm2,sm1,s0,sp1,sp2);

    auto [sm, sp] = Limiter::sm_sp(s0, sedge1, sedge2);

    SetYBCs<Limiter>(i, j, k, n, sm, sp, sedge1, sedge2, S, vel_dlo, vel_dhi,
                     bc.lo(1), bc.hi(1), domlo, domhi);

    const amrex::Real s6 = amrex::Real(6)*s0 - amrex::Real(3)*(sm + sp);

    const amrex::Real sigma = amrex::Math::abs(v_ad)*dtdy;

    if (v_ad > small_vel)
    {
        Ip(i,j,k,n) = sp - (half*sigma)*((sp-sm) - (one - two3rds*sigma)*s6);
        Im(i,j,k,n) = S(i,j,k,n);
    }
    else if (v_ad < -small_vel)
    {
        Ip(i,j,k,n) = S(i,j,k,n);
        Im(i,j,k,n) = sm + (half*sigma)*((sp-sm) + (one - two3rds*sigma)*s6);
    } else
    {
        Ip(i,j,k,n) = S(i,j,k,n);
        Im(i,j,k,n) = S(i,j,k,n);
    }
}

#if (AMREX_SPACEDIM==3)
template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictVelOnZFace ( const int i, const int j, const int k, const int n,
                         const amrex::Real dtdz, const amrex::Real v_ad,
                         const amrex::Real vel_dlo, const amrex::Real vel_dhi,
                         const amrex::Array4<const amrex::Real> &S,
                         const amrex::Array4<amrex::Real> &Im,
                         const amrex::Array4<amrex::Real> &Ip,
                         const amrex::BCRec bc, const int domlo, const int domhi,
                         const Limiter& /*limiter*/)
{
    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1)};
    constexpr auto two3rds{amrex::Real(2.0)/amrex::Real(3.0)};

    const amrex::Real sm2 = S(i,j,k-2,n);
    const amrex::Real sm1 = S(i,j,k-1,n);
    const amrex::Real s0  = S(i,j,k  ,n);
    const amrex::Real sp1 = S(i,j,k+1,n);
    const amrex::Real sp2 = S(i,j,k+2,n);

    amrex::Real sedge1 = Limiter::sedge1(sm2,sm1,s0,sp1,sp2);
    amrex::Real sedge2 = Limiter::sedge2(sm2,sm1,s0,sp1,sp2);

    auto [sm, sp] = Limiter::sm_sp(s0, sedge1, sedge2);

    SetZBCs<Limiter>(i, j, k, n, sm, sp, sedge1, sedge2, S, vel_dlo, vel_dhi,
                     bc.lo(2), bc.hi(2), domlo, domhi);

    const amrex::Real s6 = amrex::Real(6)*s0 - amrex::Real(3)*(sm + sp);

    const amrex::Real sigma = amrex::Math::abs(v_ad)*dtdz;

    if (v_ad > small_vel)
    {
        Ip(i,j,k,n) = sp - (half*sigma)*((sp-sm) - (one - two3rds*sigma)*s6);
        Im(i,j,k,n) = S(i,j,k,n);
    }
    else if (v_ad < -small_vel)
    {
        Ip(i,j,k,n) = S(i,j,k,n);
        Im(i,j,k,n) = sm + (half*sigma)*((sp-sm) + (one - two3rds*sigma)*s6);
    } else
    {
        Ip(i,j,k,n) = S(i,j,k,n);
        Im(i,j,k,n) = S(i,j,k,n);
    }
}

#endif

// Right now only ppm type 1 is supported on GPU
// This version is called after the MAC projection, when we use the MAC-projected velocity
//      for upwinding
template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnXFace ( const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> &vel_edge,
                           const amrex::BCRec bc,
                           const int domlo, const int domhi,
                           const Limiter& /*limiter*/,
                           int /*unused*/)
{
    const amrex::Real sm1 = S(i-1,j,k,n);
    const amrex::Real s0  = S(i  ,j,k,n);
    const amrex::Real sp1 = S(i+1,j,k,n);

    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1)};

    const amrex::Real sigmap = amrex::Math::abs(vel_edge(i+1,j,k))*dt/dx;
    const amrex::Real sigmam = amrex::Math::abs(vel_edge(i  ,j,k))*dt/dx;

    constexpr auto two3rds{amrex::Real(2.0)/amrex::Real(3.0)};

    const amrex::Real sm2 = S(i-2,j,k,n);
    const amrex::Real sp2 = S(i+2,j,k,n);

    amrex::Real sedge1 = Limiter::sedge1(sm2,sm1,s0,sp1,sp2);
    amrex::Real sedge2 = Limiter::sedge2(sm2,sm1,s0,sp1,sp2);

    auto [sm, sp] = Limiter::sm_sp(s0, sedge1, sedge2);

    // SetXBCs applies its boundary stencil to both domlo and domlo+1 (domhi-1 and
    // domhi), so for direction_dependent the gate must be the domain face, not the
    // face bounding this cell. vel_edge carries one ghost face, so the domain face is
    // out of reach only for a box whose valid region starts at domlo+2 (ends at
    // domhi-2); there we fall back to this cell's face.
    const amrex::Real vel_dlo = (i <= domlo+1 && amrex::lbound(vel_edge).x <= domlo)
                              ? vel_edge(domlo  ,j,k) : vel_edge(i,j,k);
    const amrex::Real vel_dhi = (i >= domhi-1 && amrex::ubound(vel_edge).x >= domhi+1)
                              ? vel_edge(domhi+1,j,k) : vel_edge(i+1,j,k);
    SetXBCs<Limiter>(i, j, k, n, sm, sp, sedge1, sedge2, S, vel_dlo, vel_dhi,
                     bc.lo(0), bc.hi(0), domlo, domhi);

    const amrex::Real s6 = amrex::Real(6)*s0 - amrex::Real(3)*(sm + sp);

    if (vel_edge(i+1,j,k) > small_vel) {
        Ip = sp - (half*sigmap)*((sp - sm) - (one -two3rds*sigmap)*s6);
    } else if (i == domhi && bc.hi(0) == amrex::BCType::ext_dir) {
        // Unless the flow leaves through it, the state on a boundary with a Dirichlet value
        // is that value, which the ghost cell holds. Where |u| < small_vel, as on a wall, the
        // edge-state kernels average the states from the two sides of the face, so the cell
        // average here would put the state on the wall O(dx) off. Fluxes do not see that,
        // since u = 0 there, but the convective form adds q div(u) with q the mean of the
        // states on the faces of the cell (HydroUtils::ComputeConvectiveTerm), which would
        // then be O(dx) wrong wherever div(u) != 0 next to the wall.
        // PredictVelOn{X,Y,Z}Face does not need this: the velocity extrapolation has no
        // q div(u) term, and its transverse terms multiply the state on a wall by the normal
        // velocity of the cell next to it, which is O(dx), so there the cell average only
        // adds an O(dt dx) error. That is within the accuracy of the method itself: the PPM
        // predicted states are second order, with O(dx^2), O(dt dx) and O(dt^2) errors of
        // their own.
        Ip = S(i+1,j,k,n);
    } else {
        Ip = S(i,j,k,n);
    }

    if(vel_edge(i,j,k) < -small_vel) {
        Im = sm + (half*sigmam)*((sp-sm) + (one - two3rds*sigmam)*s6);
    } else if (i == domlo && bc.lo(0) == amrex::BCType::ext_dir) {
        Im = S(i-1,j,k,n);
    } else {
        Im = S(i,j,k,n);
    }
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnXFace<PPM::upwind> ( const int i, const int j, const int k, const int n,
                           const amrex::Real /*unused*/, const amrex::Real /*unused*/,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> & /*unused*/,
                           const amrex::BCRec /*unused*/,
                           const int /*unused*/, const int /*unused*/,
                           const PPM::upwind& /*limiter*/,
                           int /*unused*/)
{
    Im = S(i,j,k,n);
    Ip = S(i,j,k,n);
}


template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnXFace<PPM::minmod> ( const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> &vel_edge,
                           const amrex::BCRec bc,
                           const int domlo, const int domhi,
                           const PPM::minmod& /*limiter*/,
                           int /*unused*/)
{
    const amrex::Real sm1 = S(i-1,j,k,n);
    const amrex::Real s0  = S(i  ,j,k,n);
    const amrex::Real sp1 = S(i+1,j,k,n);

    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1.0)};

    const amrex::Real sigmap = amrex::Math::abs(vel_edge(i+1,j,k))*dt/dx;
    const amrex::Real sigmam = amrex::Math::abs(vel_edge(i  ,j,k))*dt/dx;

    auto dsp = amrex::Real(0.0);
    auto dsm = amrex::Real(0.0);
    minmod_fn(sm1, s0, sp1, dsm, dsp);
    Godunov_minmod_bc_lo(i, sm1, s0, vel_edge(i  ,j,k), dsm, bc.lo(0), domlo);
    Godunov_minmod_bc_hi(i, s0, sp1, vel_edge(i+1,j,k), dsp, bc.hi(0), domhi);

    if (vel_edge(i + 1, j, k) > small_vel) {
        Ip = s0 + half * (one - sigmap) * dsp;
    } else if (i == domhi && bc.hi(0) == amrex::BCType::ext_dir) {
        // Unless the flow leaves through it, the state on a boundary with a Dirichlet value
        // is that value, which the ghost cell holds (see PredictStateOnXFace)
        Ip = S(i+1,j,k,n);
    } else {
        Ip = S(i,j,k,n);
    }

    if (vel_edge(i, j, k) < -small_vel) {
        Im = s0 - half * (one - sigmam) * dsm;
    } else if (i == domlo && bc.lo(0) == amrex::BCType::ext_dir) {
        Im = S(i-1,j,k,n);
    } else {
        Im = S(i,j,k,n);
    }
}

template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnYFace ( const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> &vel_edge,
                           const amrex::BCRec bc,
                           const int domlo, const int domhi,
                           const Limiter& /*limiter*/,
                           int /*unused*/)
{
    const amrex::Real sm1 = S(i,j-1,k,n);
    const amrex::Real s0  = S(i,j  ,k,n);
    const amrex::Real sp1 = S(i,j+1,k,n);

    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1.0)};

    const amrex::Real sigmap = amrex::Math::abs(vel_edge(i,j+1,k))*dt/dx;
    const amrex::Real sigmam = amrex::Math::abs(vel_edge(i,j  ,k))*dt/dx;

    constexpr auto two3rds{amrex::Real(2.0)/amrex::Real(3.0)};

    const amrex::Real sm2 = S(i,j-2,k,n);
    const amrex::Real sp2 = S(i,j+2,k,n);

    amrex::Real sedge1 = Limiter::sedge1(sm2,sm1,s0,sp1,sp2);
    amrex::Real sedge2 = Limiter::sedge2(sm2,sm1,s0,sp1,sp2);

    auto [sm, sp] = Limiter::sm_sp(s0, sedge1, sedge2);

    // SetYBCs applies its boundary stencil to both domlo and domlo+1 (domhi-1 and
    // domhi), so for direction_dependent the gate must be the domain face, not the
    // face bounding this cell. vel_edge carries one ghost face, so the domain face is
    // out of reach only for a box whose valid region starts at domlo+2 (ends at
    // domhi-2); there we fall back to this cell's face.
    const amrex::Real vel_dlo = (j <= domlo+1 && amrex::lbound(vel_edge).y <= domlo)
                              ? vel_edge(i,domlo  ,k) : vel_edge(i,j,k);
    const amrex::Real vel_dhi = (j >= domhi-1 && amrex::ubound(vel_edge).y >= domhi+1)
                              ? vel_edge(i,domhi+1,k) : vel_edge(i,j+1,k);
    SetYBCs<Limiter>(i, j, k, n, sm, sp, sedge1, sedge2, S, vel_dlo, vel_dhi,
                     bc.lo(1), bc.hi(1), domlo, domhi);

    const amrex::Real s6 = amrex::Real(6)*s0- amrex::Real(3)*(sm + sp);

    if (vel_edge(i,j+1,k) > small_vel) {
        Ip = sp - (half*sigmap)*((sp - sm) - (one -two3rds*sigmap)*s6);
    } else if (j == domhi && bc.hi(1) == amrex::BCType::ext_dir) {
        // Unless the flow leaves through it, the state on a boundary with a Dirichlet value
        // is that value, which the ghost cell holds (see PredictStateOnXFace)
        Ip = S(i,j+1,k,n);
    } else {
        Ip = S(i,j,k,n);
    }

    if (vel_edge(i,j,k) < -small_vel) {
        Im = sm + (half*sigmam)*((sp-sm) + (one - two3rds*sigmam)*s6);
    } else if (j == domlo && bc.lo(1) == amrex::BCType::ext_dir) {
        Im = S(i,j-1,k,n);
    } else {
        Im = S(i,j,k,n);
    }
}


template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnYFace<PPM::upwind> ( const int i, const int j, const int k, const int n,
                           const amrex::Real /*unused*/, const amrex::Real /*unused*/,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> & /*unused*/,
                           const amrex::BCRec /*unused*/,
                           const int /*unused*/, const int /*unused*/,
                           const PPM::upwind& /*limiter*/,
                           int /*unused*/)
{
    Im = S(i,j,k,n);
    Ip = S(i,j,k,n);
}


template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnYFace<PPM::minmod> ( const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> &vel_edge,
                           const amrex::BCRec bc,
                           const int domlo, const int domhi,
                           const PPM::minmod& /*limiter*/,
                           int /*unused*/)
{
    const amrex::Real sm1 = S(i,j-1,k,n);
    const amrex::Real s0  = S(i,j  ,k,n);
    const amrex::Real sp1 = S(i,j+1,k,n);

    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1.0)};

    const amrex::Real sigmap = amrex::Math::abs(vel_edge(i,j+1,k))*dt/dx;
    const amrex::Real sigmam = amrex::Math::abs(vel_edge(i,j  ,k))*dt/dx;

    auto dsp = amrex::Real(0.0);
    auto dsm = amrex::Real(0.0);
    minmod_fn(sm1, s0, sp1, dsm, dsp);
    Godunov_minmod_bc_lo(j, sm1, s0, vel_edge(i,j  ,k), dsm, bc.lo(1), domlo);
    Godunov_minmod_bc_hi(j, s0, sp1, vel_edge(i,j+1,k), dsp, bc.hi(1), domhi);

    if (vel_edge(i,j+1,k) > small_vel) {
        Ip = s0 + half * (one - sigmap) * dsp;
    } else if (j == domhi && bc.hi(1) == amrex::BCType::ext_dir) {
        // Unless the flow leaves through it, the state on a boundary with a Dirichlet value
        // is that value, which the ghost cell holds (see PredictStateOnXFace)
        Ip = S(i,j+1,k,n);
    } else {
        Ip = S(i,j,k,n);
    }
    if (vel_edge(i,j,k) < -small_vel) {
        Im = s0 - half * (one - sigmam) * dsm;
    } else if (j == domlo && bc.lo(1) == amrex::BCType::ext_dir) {
        Im = S(i,j-1,k,n);
    } else {
        Im = S(i,j,k,n);
    }
}


#if (AMREX_SPACEDIM==3)
template <typename Limiter>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnZFace ( const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> &vel_edge,
                           const amrex::BCRec bc,
                           const int domlo, const int domhi,
                           const Limiter& /*limiter*/,
                           int /*unused*/)
{
    const amrex::Real sm1 = S(i,j,k-1,n);
    const amrex::Real s0  = S(i,j,k  ,n);
    const amrex::Real sp1 = S(i,j,k+1,n);

    constexpr auto half{amrex::Real(0.5)};

    const amrex::Real sigmap = amrex::Math::abs(vel_edge(i,j,k+1))*dt/dx;
    const amrex::Real sigmam = amrex::Math::abs(vel_edge(i,j,k  ))*dt/dx;

    constexpr auto two3rds{amrex::Real(2.0)/amrex::Real(3.0)};

    const amrex::Real sm2 = S(i,j,k-2,n);
    const amrex::Real sp2 = S(i,j,k+2,n);

    amrex::Real sedge1 = Limiter::sedge1(sm2,sm1,s0,sp1,sp2);
    amrex::Real sedge2 = Limiter::sedge2(sm2,sm1,s0,sp1,sp2);

    auto [sm, sp] = Limiter::sm_sp(s0, sedge1, sedge2);

    // SetZBCs applies its boundary stencil to both domlo and domlo+1 (domhi-1 and
    // domhi), so for direction_dependent the gate must be the domain face, not the
    // face bounding this cell. vel_edge carries one ghost face, so the domain face is
    // out of reach only for a box whose valid region starts at domlo+2 (ends at
    // domhi-2); there we fall back to this cell's face.
    const amrex::Real vel_dlo = (k <= domlo+1 && amrex::lbound(vel_edge).z <= domlo)
                              ? vel_edge(i,j,domlo  ) : vel_edge(i,j,k);
    const amrex::Real vel_dhi = (k >= domhi-1 && amrex::ubound(vel_edge).z >= domhi+1)
                              ? vel_edge(i,j,domhi+1) : vel_edge(i,j,k+1);
    SetZBCs<Limiter>(i, j, k, n, sm, sp, sedge1, sedge2, S, vel_dlo, vel_dhi,
                     bc.lo(2), bc.hi(2), domlo, domhi);

    const amrex::Real s6 = amrex::Real(6)*s0- amrex::Real(3)*(sm + sp);

    if(vel_edge(i,j,k+1) > small_vel) {
        Ip = sp - (half*sigmap)*((sp-sm) - (amrex::Real(1.0) -two3rds*sigmap)*s6);
    } else if (k == domhi && bc.hi(2) == amrex::BCType::ext_dir) {
        // Unless the flow leaves through it, the state on a boundary with a Dirichlet value
        // is that value, which the ghost cell holds (see PredictStateOnXFace)
        Ip = S(i,j,k+1,n);
    } else {
        Ip = S(i,j,k,n);
    }

    if(vel_edge(i,j,k) < -small_vel) {
        Im = sm + (half*sigmam)*((sp-sm) + (amrex::Real(1.0) - two3rds*sigmam)*s6);
    } else if (k == domlo && bc.lo(2) == amrex::BCType::ext_dir) {
        Im = S(i,j,k-1,n);
    } else {
        Im = S(i,j,k,n);
    }
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnZFace<PPM::upwind> ( const int i, const int j, const int k, const int n,
                           const amrex::Real /*unused*/, const amrex::Real /*unused*/,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> & /*unused*/,
                           const amrex::BCRec /*unused*/,
                           const int /*unused*/, const int /*unused*/,
                           const PPM::upwind& /*limiter*/,
                           int /*unused*/)
{
    Im = S(i,j,k,n);
    Ip = S(i,j,k,n);
}

template <>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void PredictStateOnZFace<PPM::minmod> ( const int i, const int j, const int k, const int n,
                           const amrex::Real dt, const amrex::Real dx,
                           amrex::Real& Im, amrex::Real& Ip,
                           const amrex::Array4<const amrex::Real> &S,
                           const amrex::Array4<const amrex::Real> &vel_edge,
                           const amrex::BCRec bc,
                           const int domlo, const int domhi,
                           const PPM::minmod& /*limiter*/,
                           int /*unused*/)
{
    const amrex::Real sm1 = S(i,j,k-1,n);
    const amrex::Real s0  = S(i,j,k  ,n);
    const amrex::Real sp1 = S(i,j,k+1,n);

    constexpr auto half{amrex::Real(0.5)};
    constexpr auto  one{amrex::Real(1.0)};

    const amrex::Real sigmap = amrex::Math::abs(vel_edge(i,j,k+1))*dt/dx;
    const amrex::Real sigmam = amrex::Math::abs(vel_edge(i,j,k  ))*dt/dx;

    auto dsp = amrex::Real(0.0);
    auto dsm = amrex::Real(0.0);
    minmod_fn(sm1, s0, sp1, dsm, dsp);
    Godunov_minmod_bc_lo(k, sm1, s0, vel_edge(i,j,k  ), dsm, bc.lo(2), domlo);
    Godunov_minmod_bc_hi(k, s0, sp1, vel_edge(i,j,k+1), dsp, bc.hi(2), domhi);

    if (vel_edge(i, j, k + 1) > small_vel) {
        Ip = s0 + half * (one - sigmap) * dsp;
    } else if (k == domhi && bc.hi(2) == amrex::BCType::ext_dir) {
        // Unless the flow leaves through it, the state on a boundary with a Dirichlet value
        // is that value, which the ghost cell holds (see PredictStateOnXFace)
        Ip = S(i,j,k+1,n);
    } else {
        Ip = S(i,j,k,n);
    }

    if (vel_edge(i, j, k) < -small_vel) {
        Im = s0 - half * (one - sigmam) * dsm;
    } else if (k == domlo && bc.lo(2) == amrex::BCType::ext_dir) {
        Im = S(i,j,k-1,n);
    } else {
        Im = S(i,j,k,n);
    }
}
#endif

template <typename Limiter>
void PredictVelOnFaces (amrex::Box const& bx,
                        AMREX_D_DECL(amrex::Array4<amrex::Real> const& Imx,
                                     amrex::Array4<amrex::Real> const& Imy,
                                     amrex::Array4<amrex::Real> const& Imz),
                        AMREX_D_DECL(amrex::Array4<amrex::Real> const& Ipx,
                                     amrex::Array4<amrex::Real> const& Ipy,
                                     amrex::Array4<amrex::Real> const& Ipz),
                        amrex::Array4<amrex::Real const> const& q,
                        amrex::Array4<amrex::Real const> const& vel,
                        amrex::Geometry geom,
                        amrex::Real dt,
                        amrex::BCRec const* pbc,
                        const Limiter& limiter,
                        amrex::Array4<int const> const& bc_arr = {})
{
    const amrex::Box& domain = geom.Domain();
    const amrex::Dim3 dlo = amrex::lbound(domain);
    const amrex::Dim3 dhi = amrex::ubound(domain);

    const auto dx = geom.CellSizeArray();
    AMREX_D_TERM( const amrex::Real l_dtdx = dt / dx[0];,
                  const amrex::Real l_dtdy = dt / dx[1];,
                  const amrex::Real l_dtdz = dt / dx[2];);

    amrex::ParallelFor(bx, AMREX_SPACEDIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        // The boundary stencils in SetXBCs/SetYBCs/SetZBCs cover the first two cell
        // layers, so for direction_dependent they must be gated on the velocity at the
        // domain boundary rather than on the velocity of the cell being filled. Here
        // that is the normal velocity of the boundary cell itself. The reads are
        // guarded so we never touch the boundary cell from a box that is nowhere near
        // it; vel carries 3 ghost cells, so it is in reach wherever the gate is used.
        const amrex::Real xlo_v = (i <= dlo.x+1) ? vel(dlo.x,j,k,0) : amrex::Real(0.0);
        const amrex::Real xhi_v = (i >= dhi.x-1) ? vel(dhi.x,j,k,0) : amrex::Real(0.0);
        const amrex::Real ylo_v = (j <= dlo.y+1) ? vel(i,dlo.y,k,1) : amrex::Real(0.0);
        const amrex::Real yhi_v = (j >= dhi.y-1) ? vel(i,dhi.y,k,1) : amrex::Real(0.0);
        // Use the position-dependent boundary condition where one was given, as the
        // PLM path does; getBC falls back to pbc[n] when bc_arr is null.
        const amrex::BCRec bc = HydroBC::getBC(i, j, k, n, domain, pbc, bc_arr);
        PredictVelOnXFace(i,j,k,n,l_dtdx,vel(i,j,k,0),xlo_v,xhi_v,q,Imx,Ipx,bc,dlo.x,dhi.x,limiter);
        PredictVelOnYFace(i,j,k,n,l_dtdy,vel(i,j,k,1),ylo_v,yhi_v,q,Imy,Ipy,bc,dlo.y,dhi.y,limiter);
#if (AMREX_SPACEDIM==3)
        const amrex::Real zlo_v = (k <= dlo.z+1) ? vel(i,j,dlo.z,2) : amrex::Real(0.0);
        const amrex::Real zhi_v = (k >= dhi.z-1) ? vel(i,j,dhi.z,2) : amrex::Real(0.0);
        PredictVelOnZFace(i,j,k,n,l_dtdz,vel(i,j,k,2),zlo_v,zhi_v,q,Imz,Ipz,bc,dlo.z,dhi.z,limiter);
#endif
    });
}

template <typename Limiter>
void PredictStateOnFaces (amrex::Box const& bx,
                          AMREX_D_DECL(amrex::Array4<amrex::Real> const& Imx,
                                       amrex::Array4<amrex::Real> const& Imy,
                                       amrex::Array4<amrex::Real> const& Imz),
                          AMREX_D_DECL(amrex::Array4<amrex::Real> const& Ipx,
                                       amrex::Array4<amrex::Real> const& Ipy,
                                       amrex::Array4<amrex::Real> const& Ipz),
                          AMREX_D_DECL(amrex::Array4<amrex::Real const> const& umac,
                                       amrex::Array4<amrex::Real const> const& vmac,
                                       amrex::Array4<amrex::Real const> const& wmac),
                          amrex::Array4<amrex::Real const> const& q,
                          amrex::Geometry geom,
                          amrex::Real l_dt,
                          amrex::BCRec const* pbc,
                          const int ncomp,
                          const Limiter& limiter,
                          int limiter_type,
                          amrex::Array4<int const> const& bc_arr = {})
{
    const amrex::Box& domain = geom.Domain();
    const amrex::Dim3 dlo = amrex::lbound(domain);
    const amrex::Dim3 dhi = amrex::ubound(domain);

    AMREX_D_TERM (const auto dx = geom.CellSize(0);,
                  const auto dy = geom.CellSize(1);,
                  const auto dz = geom.CellSize(2););

    amrex::ParallelFor(bx, ncomp,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        // Use the position-dependent boundary condition where one was given, as the
        // PLM path does; getBC falls back to pbc[n] when bc_arr is null.
        const amrex::BCRec bc = HydroBC::getBC(i, j, k, n, domain, pbc, bc_arr);
        PPM::PredictStateOnXFace(i, j, k, n, l_dt, dx, Imx(i,j,k,n), Ipx(i,j,k,n),
                                 q, umac, bc, dlo.x, dhi.x, limiter, limiter_type);
        PPM::PredictStateOnYFace(i, j, k, n, l_dt, dy, Imy(i,j,k,n), Ipy(i,j,k,n),
                                 q, vmac, bc, dlo.y, dhi.y, limiter, limiter_type);
#if (AMREX_SPACEDIM==3)
        PPM::PredictStateOnZFace(i, j, k, n, l_dt, dz, Imz(i,j,k,n), Ipz(i,j,k,n),
                                 q, wmac, bc, dlo.z, dhi.z, limiter, limiter_type);
#endif
    });
}
} // namespace

#endif
/** @} */
