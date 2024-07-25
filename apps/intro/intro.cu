#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;
/*
 *calculating face normals
 *
 *get the normal
 *normalise
 *allocate as the color for the current vertex
 *
 *
 */
template <typename T, uint32_t blockThreads>
__global__ static void compute_face_normal(
    const rxmesh::Context      context,
    rxmesh::VertexAttribute<T> coords,   // in
    rxmesh::FaceAttribute<T> normals)  // out
{
    
    auto vn_lambda = [&](FaceHandle face_id, VertexIterator& fv) {
        // get the face's three vertices coordinates
        
        glm::fvec3 c0(coords(fv[0], 0), coords(fv[0], 1), coords(fv[0], 2));
        glm::fvec3 c1(coords(fv[1], 0), coords(fv[1], 1), coords(fv[1], 2));
        glm::fvec3 c2(coords(fv[2], 0), coords(fv[2], 1), coords(fv[2], 2));

        // compute the face normal
        glm::fvec3 n = cross(c1 - c0, c2 - c0);
        normals(face_id, 0) = n[0];
        normals(face_id, 1) = n[1];
        normals(face_id, 2) = n[2];
    };
    auto block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator shrd_alloc;
    query.dispatch<Op::FV>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_face_color(
    const rxmesh::Context      context,
    rxmesh::FaceAttribute<T>   color)   // out
{

    auto vn_lambda = [&](FaceHandle face_id, FaceIterator &ff)
    {
        auto number_of_faces = context.m_num_faces;



        color(face_id, 0) = face_id.unique_id() / *number_of_faces;
        color(face_id, 1) = 40 / *number_of_faces;
        color(face_id, 2) = 50 / *number_of_faces;
    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::FF>(block, shrd_alloc, vn_lambda);
}

template <typename T, uint32_t blockThreads>
__global__ static void compute_vertex_color(
    const rxmesh::Context    context,
    rxmesh::VertexAttribute<T> color)  // out
{
    auto vn_lambda = [&](VertexHandle vertex_id, FaceIterator& vf) {

        

        color(vertex_id, 0) = vf.size() / 5;
        color(vertex_id, 1) = vf.size() / 5;
        color(vertex_id, 2) = vf.size() / 5;

    };
    auto                block = cooperative_groups::this_thread_block();
    Query<blockThreads> query(context);
    ShmemAllocator      shrd_alloc;
    query.dispatch<Op::VF>(block, shrd_alloc, vn_lambda);
}




int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    rxmesh::RXMeshStatic rx(STRINGIFY(INPUT_DIR) "dragon.obj");

    //auto vertex_color = *rx.add_vertex_attribute<float>("vColor", 3);

    /*
    rx.for_each_vertex(  // for_each operation defined as lambda function
        DEVICE,
        [vertex_color,number_of_vertices] __device__(const VertexHandle vh) {
            vertex_color(vh, 0) = vh.local_id()/number_of_vertices;
            vertex_color(vh, 1) = vh.local_id() / number_of_vertices;
            vertex_color(vh, 2) = vh.local_id() / number_of_vertices;
        });

    vertex_color.move(DEVICE, HOST);  // move from device to host

    rx.get_polyscope_mesh()->addVertexColorQuantity("vColor", vertex_color);
    rx.get_polyscope_mesh()->quantities["vColor"]->setEnabled(true);
    */

    //create attribute

    auto vertex_pos = *rx.get_input_vertex_coordinates();
    auto face_normals = rx.add_face_attribute<float>("fNorm", 3);
    //face_normals.reset(0, LOCATION_ALL); bad line
    

    constexpr uint32_t CUDABlockSize = 256;
    LaunchBox<CUDABlockSize> launch_box;
    LaunchBox<CUDABlockSize> launch_box2;

    
    rx.prepare_launch_box({rxmesh::Op::FV},
    launch_box,
    (void*)compute_face_normal<float, CUDABlockSize>
    );
    


    compute_face_normal<float, CUDABlockSize>
    <<<launch_box.blocks,
    launch_box.num_threads,
    launch_box.smem_bytes_dyn>>>
    (rx.get_context(), vertex_pos, *face_normals);
    
    face_normals->move(DEVICE, HOST);

    rx.get_polyscope_mesh()->addFaceVectorQuantity("fNorm", *face_normals);

    //face color
    auto face_color = rx.add_face_attribute<float>("fColor", 3);
    
    rx.prepare_launch_box({rxmesh::Op::FF},
                          launch_box2,
                          (void*)compute_face_color<float, CUDABlockSize>);

    
    compute_face_color<float, CUDABlockSize><<<launch_box2.blocks,
                                                launch_box2.num_threads,
                                                launch_box2.smem_bytes_dyn>>>(
        rx.get_context(), *face_color);

    face_color->move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addFaceScalarQuantity("fColor", *face_color);

    //vertex color
    LaunchBox<CUDABlockSize> launch_box3;
    
    auto vertex_color = rx.add_vertex_attribute<float>("vColor", 3);

    rx.prepare_launch_box({rxmesh::Op::VF},
                          launch_box3,
                          (void*)compute_vertex_color<float, CUDABlockSize>);
    
    compute_vertex_color<float, CUDABlockSize>
        <<<launch_box3.blocks,
           launch_box3.num_threads,
           launch_box3.smem_bytes_dyn>>>(rx.get_context(), *vertex_color);

    vertex_color->move(DEVICE, HOST);
    rx.get_polyscope_mesh()->addVertexScalarQuantity("vColor", *vertex_color);
    
#if USE_POLYSCOPE
    polyscope::show();
#endif
}