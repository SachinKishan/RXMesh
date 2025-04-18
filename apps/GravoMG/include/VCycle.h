#pragma once

#include "GMGCSR.h"

#include "VectorCSR3D.h"


#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform.h>


__global__ void csr_spmv_3d(const int*   row_ptr,
                            const int*   col_idx,
                            const float* values,
                            const float* v,
                            float*       y,
                            int          m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m)
        return;

    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;

    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    for (int j = row_start; j < row_end; j++) {
        int   col = col_idx[j];
        float val = values[j];

        sum_x += val * v[col * 3];
        sum_y += val * v[col * 3 + 1];
        sum_z += val * v[col * 3 + 2];
    }

    y[row * 3]     = sum_x;
    y[row * 3 + 1] = sum_y;
    y[row * 3 + 2] = sum_z;
}

template <typename T>
void spmm(rxmesh::SparseMatrix<T>& A,  // input
          rxmesh::DenseMatrix<T>&  V,  // input
          rxmesh::DenseMatrix<T>&  Y)   // output
{
    A.multiply(V, Y);
}


/**
 * @brief  Wrapper for cuda implementation of Matrix Vector multiplication
 * @param  Row_ptr Row pointer for CSR matrix
 * @param  col_idx Column pointer for CSR matrix
 * @param  values Value pointer for CSR matrix
 * @param  v vector v
 * @param  y new resulting vector y
 * @param  m number of rows in CSR matrix
 */
void SpMV_CSR_3D(const int*   row_ptr,
                 const int*   col_idx,
                 const float* values,
                 const float* v,
                 float*       y,
                 int          m)
{
    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;
    csr_spmv_3d<<<grid_size, block_size>>>(row_ptr, col_idx, values, v, y, m);
    cudaDeviceSynchronize();
}

//** DONE
__global__ void vec_subtract_3d(const float* b,
                                const float* Av,
                                float*       R,
                                int          m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m)
        return;

    int idx    = i * 3;
    R[idx]     = b[idx] - Av[idx];
    R[idx + 1] = b[idx + 1] - Av[idx + 1];
    R[idx + 2] = b[idx + 2] - Av[idx + 2];
}

//** DONE
void Compute_R_3D(const CSR& A, const float* v, const float* b, float* R, int m)
{
    float* Av;
    cudaMalloc(&Av, m * 3 * sizeof(float));

    SpMV_CSR_3D(A.row_ptr, A.value_ptr, A.data_ptr, v, Av, m);

    int block_size = 256;
    int grid_size  = (m + block_size - 1) / block_size;
    vec_subtract_3d<<<grid_size, block_size>>>(b, Av, R, m);

    cudaFree(Av);
    cudaDeviceSynchronize();
}

//** DONE
struct GaussJacobiUpdate3D
{
    const int*   row_ptr;
    const int*   value_ptr;
    const float* data_ptr;
    const float* b;
    const float* x_old;
    float*       x_new;

    GaussJacobiUpdate3D(const int*   row_ptr,
                        const int*   value_ptr,
                        const float* data_ptr,
                        const float* b,
                        const float* x_old,
                        float*       x_new)
        : row_ptr(row_ptr),
          value_ptr(value_ptr),
          data_ptr(data_ptr),
          b(b),
          x_old(x_old),
          x_new(x_new)
    {
    }

    __device__ void operator()(int i)
    {
        float sum_x    = 0.0f;
        float sum_y    = 0.0f;
        float sum_z    = 0.0f;
        float diag     = 0.0f;
        bool  has_diag = false;

        // Loop over non-zero elements in the row
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            int   col = value_ptr[j];  // Column index of non-zero element
            float val = data_ptr[j];   // Value of the non-zero element

            if (col == i) {
                // If it's a diagonal element, store its value
                diag     = val;
                has_diag = true;
            } else {
                // Sum non-diagonal elements
                sum_x += val * x_old[col * 3];
                sum_y += val * x_old[col * 3 + 1];
                sum_z += val * x_old[col * 3 + 2];
            }
        }

        assert(has_diag);
        if (has_diag && abs(diag) > 10e-8f) {
            x_new[i * 3]     = (b[i * 3] - sum_x) / diag;
            x_new[i * 3 + 1] = (b[i * 3 + 1] - sum_y) / diag;
            x_new[i * 3 + 2] = (b[i * 3 + 2] - sum_z) / diag;


        } else {
            x_new[i * 3]     = x_old[i * 3];
            x_new[i * 3 + 1] = x_old[i * 3 + 1];
            x_new[i * 3 + 2] = x_old[i * 3 + 2];
        }
    }
};

/**
 * \brief Parallel gauss jacobi implementation to solve Ax=b
 * \param A
 * \param vec_x
 * \param vec_b
 * \param max_iter number of iterations
 */
//** DONE
void gauss_jacobi_CSR_3D(const CSR& A, float* vec_x, float* vec_b, int max_iter)
{
    int                       N = A.num_rows;
    thrust::device_ptr<float> x(vec_x);
    thrust::device_ptr<float> b(vec_b);
    float*                    x_new_raw;
    cudaMalloc(&x_new_raw, N * 3 * sizeof(float));
    thrust::device_ptr<float> x_new(x_new_raw);

    for (int iter = 0; iter < max_iter; ++iter) {
        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(N),
                         GaussJacobiUpdate3D(A.row_ptr,
                                             A.value_ptr,
                                             A.data_ptr,
                                             thrust::raw_pointer_cast(b),
                                             thrust::raw_pointer_cast(x),
                                             thrust::raw_pointer_cast(x_new)));

        // Swap the raw pointers
        float* temp = thrust::raw_pointer_cast(x);
        x     = thrust::device_ptr<float>(thrust::raw_pointer_cast(x_new));
        x_new = thrust::device_ptr<float>(temp);
    }

    if (thrust::raw_pointer_cast(x) != vec_x) {
        cudaMemcpy(vec_x,
                   thrust::raw_pointer_cast(x),
                   N * 3 * sizeof(float),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(vec_x,
                   thrust::raw_pointer_cast(x),
                   N * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    cudaFree(x_new_raw);
    cudaDeviceSynchronize();
}


/**
 * @brief After computing prolongation operators, this class is used to solve
 * the given Lx=b equation, given certain parameters for the V cycle
 */
class GMGVCycle
{
   public:
    int              pre_relax_iterations  = 2;
    int              post_relax_iterations = 2;
    int              max_number_of_levels;
    std::vector<int> numberOfSamplesPerLevel;
    float            omega                 = 0.5;
    int              directSolveIterations = 5;
    float            ratio                 = 8;
    int              numberOfCycles        = 2;

    std::vector<CSR> prolongationOperators;
    std::vector<CSR> prolongationOperatorsTransposed;
    std::vector<CSR> LHS;
    VectorCSR3D      RHS;
    VectorCSR3D      X;  // final solution


    /**
     * @brief Solve the equation Av=f using V Cycle
     * @param A the LHS
     * @param f The RHS
     * @param v The variables we are solving for
     * @param currentLevel The current level of the multigrid V cycle, this
     * usually starts from 0
     */
    void VCycle(CSR& A, VectorCSR3D& f, VectorCSR3D& v, int currentLevel)
    {
        // Pre-smoothing $$
        gauss_jacobi_CSR_3D(A, v.vector, f.vector, pre_relax_iterations);

        // Calculate residual on current grid $$
        VectorCSR3D R(A.num_rows);
        Compute_R_3D(A, v.vector, f.vector, R.vector, A.num_rows);

        // Create coarse grid vectors $$
        int coarse_size = prolongationOperators[currentLevel].num_rows / ratio;
        VectorCSR3D restricted_residual(coarse_size);
        VectorCSR3D coarse_correction(coarse_size);

        // Restrict the residual $$
        CSR transposeProlongation =
            prolongationOperatorsTransposed[currentLevel];

        SpMV_CSR_3D(transposeProlongation.row_ptr,
                    transposeProlongation.value_ptr,
                    transposeProlongation.data_ptr,
                    R.vector,
                    restricted_residual.vector,
                    transposeProlongation.num_rows);

        if (currentLevel < max_number_of_levels - 1) {
            VCycle(LHS[currentLevel + 1],
                   restricted_residual,
                   coarse_correction,
                   currentLevel + 1);
        } else {
            //  Initialize coarse correction to zero
            for (int i = 0; i < coarse_size * 3; i++) {
                coarse_correction.vector[i] = 0.0f;
            }

            assert(restricted_residual.n == coarse_correction.n);

            gauss_jacobi_CSR_3D(LHS[currentLevel + 1],
                                coarse_correction.vector,
                                restricted_residual.vector,
                                directSolveIterations);
        }

        // Prolongate
        VectorCSR3D fine_correction(A.num_rows);
        SpMV_CSR_3D(prolongationOperators[currentLevel].row_ptr,
                    prolongationOperators[currentLevel].value_ptr,
                    prolongationOperators[currentLevel].data_ptr,
                    coarse_correction.vector,
                    fine_correction.vector,
                    prolongationOperators[currentLevel].num_rows);

        for (int i = 0; i < v.n * 3; i++) {
            v.vector[i] += fine_correction.vector[i];
        }

        // Post-smoothing $$
        gauss_jacobi_CSR_3D(A, v.vector, f.vector, post_relax_iterations);
    }

    /**
     * @brief Solve the system using GMG and Vcycle
     */
    void solve()
    {
        // X.reset();
        for (int i = 0; i < numberOfCycles; i++)
            VCycle(LHS[0], RHS, X, 0);
    }

    GMGVCycle()
    {
    }
    GMGVCycle(int initialNumberOfRows) : X(initialNumberOfRows)
    {
    }

    ~GMGVCycle()
    {
    }
};


void constructLHS(CSR               A_csr,
                  std::vector<CSR>  prolongationOperatorCSR,
                  std::vector<CSR>  prolongationOperatorCSRTranspose,
                  std::vector<CSR>& equationsPerLevel,
                  int               numberOfLevels,
                  int               numberOfSamples,
                  float             ratio)
{
    int currentNumberOfSamples = numberOfSamples;

    CSR result = A_csr;

    // make all the equations for each level

    for (int i = 0; i < numberOfLevels - 1; i++) {

        result = multiplyCSR(result.num_rows,
                             result.num_rows,
                             currentNumberOfSamples,
                             result.row_ptr,
                             result.value_ptr,
                             result.data_ptr,
                             result.non_zeros,
                             prolongationOperatorCSR[i].row_ptr,
                             prolongationOperatorCSR[i].value_ptr,
                             prolongationOperatorCSR[i].data_ptr,
                             prolongationOperatorCSR[i].non_zeros);

        /*result = multiplyCSR(result.num_rows,
                             result.num_rows,
                             currentNumberOfSamples,
                             result.row_ptr,
                             result.value_ptr,
                             result.data_ptr,
                             result.non_zeros,
                             prolongationOperatorCSR[i].row_ptr,
                             prolongationOperatorCSR[i].value_ptr,
                             prolongationOperatorCSR[i].data_ptr,
                             prolongationOperatorCSR[i].non_zeros,
                            1);*/


        CSR transposeOperator = prolongationOperatorCSRTranspose[i];
        // transposeCSR(prolongationOperatorCSR[i]);

        result = multiplyCSR(transposeOperator.num_rows,
                             prolongationOperatorCSR[i].num_rows,
                             numberOfSamples,
                             transposeOperator.row_ptr,
                             transposeOperator.value_ptr,
                             transposeOperator.data_ptr,
                             transposeOperator.non_zeros,
                             result.row_ptr,
                             result.value_ptr,
                             result.data_ptr,
                             result.non_zeros);

        equationsPerLevel.push_back(result);

        currentNumberOfSamples /= ratio;
        // std::cout << "Equation level " << i << "\n\n";
        // prolongationOperatorCSR[i].printCSR();
        // std::cout << "\n TRANSPOSE : \n";
        // transposeOperator.printCSR();
        // result.printCSR();
    }
    // prolongationOperatorCSR[numberOfLevels-2].printCSR();
}
