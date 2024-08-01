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

    int  number_of_vertices = rx.get_num_vertices();
    auto boundaryVertices = *rx.add_vertex_attribute<int>("boundaryVertices", 1);

//    auto parameter_coords = *rx.add_vertex_attribute();

    rx.get_boundary_vertices(boundaryVertices);

    DenseMatrix<cuComplex> eb(rx, number_of_vertices, 1);
    DenseMatrix<cuComplex> u(rx, number_of_vertices, 1);
    DenseMatrix<cuComplex> T3(rx, number_of_vertices, 1);
    SparseMatrix<cuComplex> B(rx);

    rx.for_each_vertex
    (rxmesh::DEVICE,[B, eb, boundaryVertices] __device__(const rxmesh::VertexHandle vh) mutable
    {
        eb(vh, 0) = make_cuComplex(boundaryVertices(vh, 0) , 0.0f);
        B(vh, vh) = make_cuComplex(boundaryVertices(vh, 0), 0.0f);

    });

    B.move(rxmesh::DEVICE, rxmesh::HOST);
    eb.move(rxmesh::DEVICE, rxmesh::HOST);

    //
    // S = [B- (1/Vb) * ebebT];

    cuComplex T2 = eb.dot(u);
    
    rx.for_each_vertex(
        rxmesh::DEVICE,
        [u, eb, B,T3,T2] __device__(
            const rxmesh::VertexHandle vh) mutable 
        {
            cuComplex T1 = make_cuComplex(B(vh, vh).x * u(vh, 0).x,
                                          B(vh, vh).y * u(vh, 0).y);

            
            T3(vh,0)= cuCsubf(T1,cuCmulf(T2, eb(vh, 0)));

        });

    

    



    rx.get_polyscope_mesh()->addVertexScalarQuantity("vBoundary", boundaryVertices);



#if USE_POLYSCOPE
    polyscope::show();
#endif
}