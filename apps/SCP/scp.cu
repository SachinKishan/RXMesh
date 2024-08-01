#include "rxmesh/query.cuh"
#include "rxmesh/rxmesh_static.h"

#include "rxmesh/matrix/sparse_matrix.cuh"

using namespace rxmesh;





int main(int argc, char** argv)
{
    Log::init();

    const uint32_t device_id = 0;
    cuda_query(device_id);

    RXMeshStatic rx(STRINGIFY(INPUT_DIR) "bunnyhead.obj");

    auto boundaryVertices = rx.add_vertex_attribute<int>("boundaryVertices", 1);
    
    rx.get_boundary_vertices(*boundaryVertices);

    DenseMatrix<cuComplex> eb(rx, rx.get_num_vertices(), 1);

    rx.for_each_vertex
    (rxmesh::DEVICE,[eb] __device__(const rxmesh::VertexHandle vh) 
    {
        eb(vh, 0) = make_cuComplex(1.0f, 1.0f);

    });

    //vertex_color.move(rxmesh::DEVICE, rxmesh::HOST);

    rx.get_polyscope_mesh()->addVertexScalarQuantity("vBoundary", *boundaryVertices);



#if USE_POLYSCOPE
    polyscope::show();
#endif
}