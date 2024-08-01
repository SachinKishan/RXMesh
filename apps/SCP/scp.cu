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
    DenseMatrix<cuComplex> T1(rx, number_of_vertices, 1);

    DenseMatrix<cuComplex> y(rx, number_of_vertices, 1);


    SparseMatrix<cuComplex> B(rx);
    SparseMatrix<cuComplex> L(rx);

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



    rx.for_each_vertex
    (rxmesh::DEVICE,[B, eb, boundaryVertices, num_bd_vertices] __device__(const rxmesh::VertexHandle vh) mutable
    {
        //we can add a divide for each entry by number of boundary vertices- TODO: Calculate vb
        eb(vh, 0) = make_cuComplex(boundaryVertices(vh, 0)/num_bd_vertices , 0.0f);
        B(vh, vh) = make_cuComplex(boundaryVertices(vh, 0), 0.0f);

    });

    B.move(rxmesh::DEVICE, rxmesh::HOST);
    eb.move(rxmesh::DEVICE, rxmesh::HOST);

    //
    // S = [B- (1/Vb) * ebebT];

    cuComplex T2 = eb.dot(u);

    B.multiply(u, T1);

    eb.multiply(T2);

    rx.for_each_vertex(
        rxmesh::DEVICE,
        [eb, B,T2,T1] __device__(
            const rxmesh::VertexHandle vh) mutable 
        {
            T1(vh, 0) = cuCsubf(T1(vh,0), eb(vh,0));

        });

    L.pre_solve(PermuteMethod::NSTDIS);//can be outside the loop
    L.solve(T1, y); //Ly=T1

    y.multiply(1/y.norm2());
    u.copy_from(y);

    //conversion step

    rx.get_polyscope_mesh()->addVertexScalarQuantity("vBoundary", boundaryVertices);



#if USE_POLYSCOPE
    polyscope::show();
#endif
}