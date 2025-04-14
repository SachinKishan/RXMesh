#pragma once
#include "rxmesh/rxmesh_static.h"

namespace rxmesh {

/**
 * @brief abstract class for iterative solvers
 */
template <typename T, typename StructureT>
struct IterativeSolver
{
    using Type = T;

    IterativeSolver(int max_iter,
                    T   abs_tol             = 1e-6,
                    T   rel_tol             = 1e-6,
                    int reset_residual_freq = std::numeric_limits<int>::max())
        : m_max_iter(max_iter),
          m_abs_tol(abs_tol),
          m_rel_tol(rel_tol),
          m_iter_taken(0),
          m_reset_residual_freq(reset_residual_freq),
          m_start_residual(0),
          m_final_residual(0)
    {
    }

    virtual void pre_solve(StructureT*       X,
                           const StructureT* B,
                           cudaStream_t      stream) = 0;

    virtual void solve(StructureT*       X,
                       const StructureT* B,
                       cudaStream_t      stream) = 0;

    virtual std::string name() = 0;

    virtual int iter_taken() const
    {
        return m_iter_taken;
    }

    virtual T final_residual() const
    {
        return m_final_residual;
    }

    virtual T start_residual() const
    {
        return m_start_residual;
    }

    virtual ~IterativeSolver()
    {
    }

    virtual bool is_converged(T init_res, T current_res)
    {
        bool abs_ok = current_res < m_abs_tol;
        bool rel_ok = current_res / init_res < m_rel_tol;

        return abs_ok || rel_ok;
    }

   protected:
    int m_max_iter;
    T   m_abs_tol, m_rel_tol;
    int m_iter_taken;
    int m_reset_residual_freq;
    T   m_start_residual;
    T   m_final_residual;
};

}  // namespace rxmesh