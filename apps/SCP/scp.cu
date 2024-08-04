#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;


template <typename T, uint32_t blockThreads>
__global__ static void compute_area_matrix(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<int> boundaryVertices,
    rxmesh::SparseMatrix<cuComplex>    AreaMatrix)
{

    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& vv) {
        if (boundaryVertices(vv[0], 0) == 1 && boundaryVertices(vv[1], 0) == 1) 
        {
            AreaMatrix(vv[0], vv[1]) = make_cuComplex(0, -0.25);  
            AreaMatrix(vv[1], vv[0]) = make_cuComplex(0, 0.25);
            //printf("\nfirst edge: %f", AreaMatrix(vv[0], vv[1]).y);

        }
        else if (boundaryVertices(vv[1], 0) == 1 &&
            boundaryVertices(vv[2], 0) == 1) {
            AreaMatrix(vv[1], vv[2]) = make_cuComplex(0, -0.25);
            AreaMatrix(vv[2], vv[1]) = make_cuComplex(0, 0.25);
            //printf("\nsecond edge: %f", AreaMatrix(vv[1], vv[2]).y);

            }
        else if (boundaryVertices(vv[2], 0) == 1 &&
            boundaryVertices(vv[0], 0) == 1) {
            AreaMatrix(vv[2], vv[0]) = make_cuComplex(0, -0.25);
            AreaMatrix(vv[0], vv[2]) = make_cuComplex(0, 0.25);
            //printf("\nthird edge: %f", AreaMatrix(vv[2], vv[0]).y);

        }
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, vn_lambda);
}

template <typename T>
__device__ __forceinline__ T
edge_cotan_weight(const rxmesh::VertexHandle&       p_id,
                  const rxmesh::VertexHandle&       r_id,
                  const rxmesh::VertexHandle&       q_id,
                  //const rxmesh::VertexHandle&       s_id,
                  const rxmesh::VertexAttribute<T>& X)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r

    const vec3<T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const vec3<T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));
    const vec3<T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    //const vec3<T> s(X(s_id, 0), X(s_id, 1), X(s_id, 2));

    // cotans[(v1, v2)] =np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))

    float weight = 0;
    if (q_id.is_valid())
        weight += dot((p - q), (r - q)) / length(cross(p - q, r - q));
    
    //weight /= 2;
    return std::max(0.f, weight);
}




template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_weights_evd(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,
    rxmesh::SparseMatrix<T>    A_mat)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& vv) {
        T e_weight = 0;

        if (vv[1].is_valid())
            e_weight += edge_cotan_weight(vv[0], vv[2], vv[1], coords);
        if (vv[3].is_valid())
            e_weight += edge_cotan_weight(vv[0], vv[2], vv[3], coords);
        //if (vv[1].is_valid() && vv[3].is_valid())
        e_weight /= 4;
        A_mat(vv[0], vv[2]) = 1;
        A_mat(vv[2], vv[0]) = 1;

    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void calculate_Ld_matrix(
    const rxmesh::Context   context,
    rxmesh::SparseMatrix<float> weight_mat,  // [num_coord, num_coord]
    rxmesh::SparseMatrix<cuComplex> Ld           // [num_coord, num_coord]
)
{
    auto init_lambda = [&](VertexHandle v_id, VertexIterator& vv) {
        Ld(v_id, v_id) = make_cuComplex(0, 0);
        for (int nei_index = 0; nei_index < vv.size(); nei_index++)
            Ld(v_id, vv[nei_index]) = make_cuComplex(0, 0);

        for (int nei_index = 0; nei_index < vv.size(); nei_index++) {

            Ld(v_id, v_id) = //make_cuComplex(5, 0);
                cuCaddf(Ld(v_id, v_id),
                        make_cuComplex(1,0));//weight_mat(v_id, vv[nei_index]),0));
            //                                       weight_mat(v_id, vv[nei_index])));

            Ld(v_id, vv[nei_index]) =
                cuCsubf(Ld(v_id, vv[nei_index]),
                    make_cuComplex(weight_mat(v_id, vv[nei_index]), 0));



        }
        //printf("\nOwner vertex: %f", Ld(v_id, v_id).x);

        for (int nei_index = 0; nei_index < vv.size(); nei_index++) {
            //printf("\n%d: %f", nei_index, Ld(v_id, vv[nei_index]).x);

        }


    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void subtract_matrix(const rxmesh::Context   context,
                                       rxmesh::SparseMatrix<T> A_mat,
                                       rxmesh::SparseMatrix<T> B_mat,
                                       rxmesh::SparseMatrix<T> C_mat)
{

    auto subtract = [&](VertexHandle v_id, VertexIterator& vv) {
        for (int i = 0; i < vv.size(); ++i) {

            //printf("\nBr:%f", B_mat(v_id, vv[i]).x);
            //printf("\nBc:%f", B_mat(v_id, vv[i]).y);


            A_mat(v_id, vv[i]) = //B_mat(v_id, vv[i]);
                cuCsubf(B_mat(v_id, vv[i]), C_mat(v_id, vv[i]));
            
            //printf("\nAr:%f", A_mat(v_id, vv[i]).x);
            //printf("\nAc:%f", A_mat(v_id, vv[i]).y);

        }
        A_mat(v_id, v_id) = cuCsubf(B_mat(v_id, v_id), C_mat(v_id, v_id));
        //printf("\nAdiagr:%f", A_mat(v_id, v_id).x);
        //printf("\nAdiagc:%f", A_mat(v_id, v_id).y);
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, subtract);
}

int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "plane.obj");

    auto boundaryVertices =
        *rx.add_vertex_attribute<int>("boundaryVertices", 1);

    rx.get_boundary_vertices(
        boundaryVertices); 

    //for matrix calls
    constexpr uint32_t CUDABlockSize = 256;

    SparseMatrix<cuComplex> Ld(rx);  // complex V x V

    SparseMatrix<cuComplex> A(rx);  // 2V x 2V

    auto coords = *rx.get_input_vertex_coordinates();

    rxmesh::LaunchBox<CUDABlockSize> launch_box_area;
    rx.prepare_launch_box({rxmesh::Op:: FV},
                          launch_box_area,
                          (void*)compute_area_matrix<float, CUDABlockSize>);

    compute_area_matrix<cuComplex, CUDABlockSize>
        <<<launch_box_area.blocks,
           launch_box_area.num_threads,
           launch_box_area.smem_bytes_dyn>>>(
        rx.get_context(), boundaryVertices, A);

 SparseMatrix<float> weight_matrix(rx);

    // obtain cotangent weight matrix
    rxmesh::LaunchBox<CUDABlockSize> launch_box;
    rx.prepare_launch_box(
        {rxmesh::Op::EVDiamond},
        launch_box,
        (void*)compute_edge_weights_evd<float, CUDABlockSize>);

    compute_edge_weights_evd<float, CUDABlockSize>
        <<<launch_box.blocks,
           launch_box.num_threads,
           launch_box.smem_bytes_dyn>>>(
            rx.get_context(), coords, weight_matrix);

    rxmesh::LaunchBox<CUDABlockSize> launch_box_ld;
    rx.prepare_launch_box(
        {rxmesh::Op::VV},
        launch_box_ld,
        (void*)calculate_Ld_matrix<float, CUDABlockSize>);

    calculate_Ld_matrix<float, CUDABlockSize>
        <<<launch_box_ld.blocks,
           launch_box_ld.num_threads,
           launch_box_ld.smem_bytes_dyn>>>(
            rx.get_context(), weight_matrix, Ld);


    SparseMatrix<cuComplex> Lc(rx);
    rxmesh::LaunchBox<CUDABlockSize> launch_box_lc;
    rx.prepare_launch_box({rxmesh::Op::VV},
                          launch_box_lc,
                          (void*)subtract_matrix<cuComplex, CUDABlockSize>);

    subtract_matrix<cuComplex, CUDABlockSize>
        <<<launch_box_lc.blocks,
           launch_box_lc.num_threads,
           launch_box_lc.smem_bytes_dyn>>>
    (rx.get_context(), Lc, Ld, A);

    int number_of_vertices = rx.get_num_vertices();


    DenseMatrix<cuComplex> eb(rx, number_of_vertices, 1);
    DenseMatrix<cuComplex> u(rx, number_of_vertices, 1);
    DenseMatrix<cuComplex> T1(rx, number_of_vertices, 1);

    DenseMatrix<cuComplex> y(rx, number_of_vertices, 1);


    SparseMatrix<cuComplex> B(rx);

    uint32_t num_bd_vertices = 0;


    rx.for_each_vertex(
        HOST,
        [&](const VertexHandle& vh) {
            if (boundaryVertices(vh)) {
                num_bd_vertices++;
            }
        },
        NULL,
        false);


    rx.for_each_vertex(rxmesh::DEVICE,
                       [B, eb, boundaryVertices, num_bd_vertices] __device__(
                           const rxmesh::VertexHandle vh) mutable {
                           eb(vh, 0) = make_cuComplex(
                               (float)boundaryVertices(vh, 0) / num_bd_vertices, 0.0f);
                           B(vh, vh) =
                               make_cuComplex((float)boundaryVertices(vh, 0), 0.0f);

                            
                       });

    //B.move(rxmesh::DEVICE, rxmesh::HOST);
    eb.move(rxmesh::DEVICE, rxmesh::HOST);

    //
    // S = [B- (1/Vb) * ebebT];
    u.fill_random();
    Lc.pre_solve(PermuteMethod::NSTDIS);  // can be outside the loop

    int iterations=8;

    //std::cout << std::endl << u(0, 0).x;
    //std::cout << std::endl << u(0, 0).y;
    //std::cout << eb(0, 0).x;

    

    
    for (int i = 0; i < iterations; i++) {

        cuComplex T2 = eb.dot(u);
        //std::cout << std::endl << T2.x;
        //std::cout << std::endl << T2.y;

        B.multiply(u, T1);



        //eb.multiply(T2);

        rx.for_each_vertex(
            rxmesh::DEVICE,
            [eb, T2, T1] __device__(const rxmesh::VertexHandle vh) mutable {

            
            T1(vh, 0) = cuCsubf(
                    T1(vh, 0), 
                    cuCmulf(eb(vh, 0),T2)

                );
                
                


            });

        Lc.solve(T1, y);                      // Ly=T1

        y.move(DEVICE, HOST);

        //Lc.solve(T1, y, Solver::QR, PermuteMethod::NSTDIS);

        float norm = y.norm2();
        rx.for_each_vertex(
            rxmesh::DEVICE,
            [y] __device__(const rxmesh::VertexHandle vh) mutable {
                printf("\nx:%f", y(vh, 0).x);
                printf("\ny:%f", y(vh, 0).y);
            });

        y.multiply(1.0f / norm);
        



        u.copy_from(y);
    }
    // conversion step


    auto parametric_coordinates = *rx.add_vertex_attribute<float>("pCoords", 2);

    rx.for_each_vertex(rxmesh::DEVICE,
                      [u,parametric_coordinates] __device__(
                          const rxmesh::VertexHandle vh) mutable {
                            parametric_coordinates(vh, 0) = u(vh, 0).x;
                            parametric_coordinates(vh, 1) = u(vh, 0).y;
                      });

    parametric_coordinates.move(DEVICE, HOST);
    
    //calculate cntre, shift mesh by centre (translate back)
    //divide maximum value irrespective of axis (abs max value verte)- divide by all coordinates
    //u,v is always [(0,0),(1,1)]





    rx.get_polyscope_mesh()->addVertexParameterizationQuantity(
        "pCoords", parametric_coordinates);

    rx.get_polyscope_mesh()->addVertexVectorQuantity2D("vq",
                                                       parametric_coordinates);

    rx.get_polyscope_mesh()->addVertexScalarQuantity("vBoundary",
                                                     boundaryVertices);




#if USE_POLYSCOPE
    polyscope::show();
#endif
}