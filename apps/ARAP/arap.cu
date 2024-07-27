#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

#include "Eigen/Dense"


using namespace rxmesh;

template <typename T>
__device__ __forceinline__ T
edge_cotan_weight(const rxmesh::VertexHandle&       p_id,
                  const rxmesh::VertexHandle&       r_id,
                  const rxmesh::VertexHandle&       q_id,
                  const rxmesh::VertexHandle&       s_id,
                  const rxmesh::VertexAttribute<T>& X)
{
    // Get the edge weight between the two vertices p-r where
    // q and s composes the diamond around p-r

    const vec3<T> p(X(p_id, 0), X(p_id, 1), X(p_id, 2));
    const vec3<T> r(X(r_id, 0), X(r_id, 1), X(r_id, 2));
    const vec3<T> q(X(q_id, 0), X(q_id, 1), X(q_id, 2));
    const vec3<T> s(X(s_id, 0), X(s_id, 1), X(s_id, 2));

    //cotans[(v1, v2)] =np.dot(e1, e2) / np.linalg.norm(np.cross(e1, e2))

    float weight = 0;
    if (q_id.is_valid())
        weight   += dot((p - q), (r - q)) / length(cross(p - q, r - q));
    if (s_id.is_valid())
        weight   += dot((p - s), (r - s)) / length(cross(p - s, r - s));
    weight /= 2;
    return weight;
}



template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_weights(const rxmesh::Context      context,
                                             rxmesh::VertexAttribute<T> coords,
                                            rxmesh::SparseMatrix<T> A_mat)
{

    auto vn_lambda = [&](VertexHandle vertex_id, VertexIterator& vv)
    {
        VertexHandle q_id = vv.back();

        for (uint32_t v = 0; v < vv.size(); ++v) 
        {
            VertexHandle r_id = vv[v];
            T e_weight = 0;
            VertexHandle s_id = (v == vv.size() - 1) ? vv[0] : vv[v + 1];
            e_weight = edge_cotan_weight(vertex_id, r_id, q_id, s_id, coords);
            A_mat(vertex_id, vv[v]) = e_weight;
        }

    };

    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_edge_weights_evd(const rxmesh::Context      context,
                                            rxmesh::VertexAttribute<T> coords,
                                            rxmesh::SparseMatrix<T>    A_mat)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& vv) {
            T e_weight = 0;
            e_weight = edge_cotan_weight(vv[0], vv[2], vv[1], vv[3], coords);
            A_mat(vv[0], vv[2]) = e_weight;
        
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EVDiamond>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void edge_weight_values(
    const rxmesh::Context      context,
    rxmesh::EdgeAttribute<T> edge_weights,
    rxmesh::SparseMatrix<T>    A_mat)
{

    auto vn_lambda = [&](EdgeHandle edge_id, VertexIterator& ev) {
        edge_weights(edge_id, 0) = A_mat(ev[0], ev[1]);
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::EV>(block, shrd_alloc, vn_lambda);
}



////////

__host__ __device__ Eigen::Matrix3f calculateSVD(Eigen::Matrix3f S)
{
    Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(S);


    Eigen::MatrixXf V = svd.matrixV();
    //TODO: it should be transpose
    Eigen::MatrixXf U = svd.matrixU().eval(); 

    float smallest_singular_value = svd.singularValues().minCoeff();

    U.col(smallest_singular_value) = U.col(smallest_singular_value) * -1;

    Eigen::MatrixXf R = V * U;

    return R;
}

template <typename T, uint32_t blockThreads>
__global__ static void calculate_rotation_matrix(const rxmesh::Context    context,
                                          rxmesh::VertexAttribute<T> ref_coords,
                                          rxmesh::VertexAttribute<T> current_coords,
                                          rxmesh::VertexAttribute<T>  rotationVector,
                                          rxmesh::SparseMatrix<T> weight_mat)
{

    auto vn_lambda = [&](VertexHandle v_id, VertexIterator& vv) {
        // pi
        
        Eigen::MatrixXf pi = Eigen::MatrixXf::Identity(3, vv.size());
        for (int j = 0; j < vv.size(); j++) 
        {
            pi(0, j) = ref_coords(v_id, 0) - ref_coords(vv[j], 0);
            pi(1, j) = ref_coords(v_id, 1) - ref_coords(vv[j], 1);
            pi(2, j) = ref_coords(v_id, 2) - ref_coords(vv[j], 2);
        }
        
        // Di
        Eigen::VectorXf weight_vector;
        weight_vector.resize(vv.size());

        for (int v = 0; v < vv.size();v++) 
        {
            weight_vector(v) = weight_mat(v_id, vv[v]);
        }
        Eigen::MatrixXf diagonal_mat = weight_vector.asDiagonal();
        
        // pi'T
        Eigen::MatrixXf pi_dash = Eigen::MatrixXf::Identity(3, vv.size());
        for (int j = 0; j < vv.size(); j++) {
            pi_dash(0, j) = current_coords(v_id, 0) - current_coords(vv[j], 0);
            pi_dash(1, j) = current_coords(v_id, 1) - current_coords(vv[j], 1);
            pi_dash(2, j) = current_coords(v_id, 2) - current_coords(vv[j], 2);
        }

        // calculate covariance matrix S = piDiPiTdash
        
        Eigen::Matrix3f S = pi * diagonal_mat * pi_dash.transpose();

        // perform svd on S (eigen)
        
        
        // R =VU


        Eigen::JacobiSVD<Eigen::Matrix3f, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(S);


        /*
        Eigen::MatrixXf V = S.jacobiSvd().matrixV();
        Eigen::MatrixXf U = S.jacobiSvd().matrixU().eval();

        float smallest_singular_value =
            S.jacobiSvd().singularValues().minCoeff();

       U.col(smallest_singular_value)= U.col(smallest_singular_value) * -1;

        Eigen::MatrixXf R = V * U;
        // Matrix R to vector attribute R
        for (int i=0;i<3;i++) {
            for (int j = 0; j < 3; j++)
                rotationVector(v_id, i * 3 + j) = R(i, j);
        }
        */
    };

    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, vn_lambda);
}



/* compute all entries of bMatrix parallely */
template <typename T, uint32_t blockThreads>
__global__ static void calculate_b(const rxmesh::Context    context,
                                          rxmesh::VertexAttribute<T> original_coords, // [num_coord, 3]
                                          rxmesh::DenseMatrix<T>  rot_mat, // [num_coord, 9]
                                          rxmesh::SparseMatrix<T> weight_mat, // [num_coord, num_coord]
                                          rxmesh::MatrixXf<T> bMatrix) // [num_coord, 3]
{
    auto init_lambda = [&](VertexHandle v_id, VertexIterator& vv) {
        // variable to store ith entry of bMatrix
        Eigen::Vector3d bi(0.0f, 0.0f, 0.0f);

        // get rotation matrix for ith vertex
        Eigen::Matrix3f Ri = Eigen::Matrix3f::Zero(3,3)

        for (int i=0;i<3;i++) {
            for (int j = 0; j < 3; j++)
                Ri(i,j) = rot_mat(v_id, i * 3 + j);
        }

        for (int nei_index = 0; nei_index < vv.size();nei_index++) 
        {
            // get weight vector
            Eigen::VectorXf w = weight_mat(v_id, vv[nei_index]); 

            // get rotation matrix for neightbor j
            Eigen::Matrix3f Rj = Eigen::Matrix3f::Zero(3,3)
            for (int i = 0; i < 3; i++) 
                for (int j = 0; j < 3; j++)
                    Rj(i,j) = rot_mat(vv[nei_index], i * 3 + j);
            
            
            Eigen::Matrix3f rot_add = Ri + Rj
            
            // find coord difference
            Eigen::Vector3f vert_diff = original_coords(vid) - original_coords(vv[nei_index])
            
            bi += 0.5 * w * rot_add * vert_diff
        }
        bMatrix[vid] = bi
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VV>(block, shrd_alloc, init_lambda);
}



int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    //RXMeshStatic rx(STRINGIFY(INPUT_DIR) "sphere3.obj");
    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    //compute wij
    auto weights = rx.add_edge_attribute<float>("edgeWeights", 1);


    auto ref_vertex_pos = *rx.get_input_vertex_coordinates();  // stays same across computation
    auto changed_vertex_pos = rx.add_vertex_attribute<float>("P", 3);  // changes per iteration

    SparseMatrix<float> weight_matrix(rx);

    //obtain cotangent weight matrix
    constexpr uint32_t               CUDABlockSize = 256;
    rxmesh::LaunchBox<CUDABlockSize> launch_box;
    rx.prepare_launch_box({rxmesh::Op::EVDiamond},
                          launch_box,
                          (void*)compute_edge_weights_evd<float, CUDABlockSize>);

     compute_edge_weights_evd<float, CUDABlockSize>
        <<<launch_box.blocks,
                                                  launch_box.num_threads,
                                                  launch_box.smem_bytes_dyn>>>(
                                                  rx.get_context(), ref_vertex_pos, weight_matrix);
    
    //visualise edge weights
     rxmesh::LaunchBox<CUDABlockSize> launch_box2;
     rx.prepare_launch_box(
         {rxmesh::Op::EV},
         launch_box2,
         (void*)edge_weight_values<float, CUDABlockSize>);

     edge_weight_values<float, CUDABlockSize>
         <<<launch_box2.blocks,
            launch_box2.num_threads,
            launch_box2.smem_bytes_dyn>>>(rx.get_context(), *weights, weight_matrix );

     weights->move(DEVICE, HOST);


     //pi and p'i

     //rx.get_polyscope_mesh()->addEdgeScalarQuantity("edgeWeights", *weights);
     //

     //calculate rotation matrix
     auto rot_mat = *rx.add_vertex_attribute<float>("RotationMatrix", 9);

    rxmesh::LaunchBox<CUDABlockSize> rotation_launch_box;

    
    rx.prepare_launch_box({rxmesh::Op::VV},
                           rotation_launch_box,
                           (void*)calculate_rotation_matrix<float, CUDABlockSize>);
    /*
    calculate_rotation_matrix<float, CUDABlockSize>
        <<<rotation_launch_box.blocks,
           rotation_launch_box.num_threads,
           rotation_launch_box.smem_bytes_dyn>>>(rx.get_context(),
                                                 ref_vertex_pos,
                                                 *changed_vertex_pos,
                                                 rot_mat,
                                                 weight_matrix);
                                                 */
                                                 

    



#if USE_POLYSCOPE
    polyscope::show();
#endif
}