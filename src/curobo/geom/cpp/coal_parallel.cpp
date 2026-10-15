#include <coal/collision_object.h>
#include <coal/distance.h>
#include <coal/shape/convex.h>
#include <coal/shape/geometric_shapes.h>
#include <coal/BVH/BVH_model.h>
#include <coal/mesh_loader/loader.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <memory>
#include <iostream>
#define COAL_NUM_THREADS 16

namespace py = pybind11;

std::shared_ptr<const coal::CollisionGeometry> loadConvexMeshCpp(const std::string& file_name, const py::array_t<double> scale) {
    coal::NODE_TYPE bv_type = coal::BV_AABB;
    coal::MeshLoader loader(bv_type);
    auto scale_ptr = scale.data();
    Eigen::Vector3d eigen_scale(scale_ptr[0], scale_ptr[1], scale_ptr[2]);
    coal::BVHModelPtr_t bvh = loader.load(file_name, eigen_scale);
    bvh->buildConvexHull(true, "Qt");
    return bvh->convex; 
}

void batched_coal_distance(
    const std::vector<std::shared_ptr<const coal::CollisionGeometry>> shape1_lst,  // Use CollisionGeometry here
    const py::array_t<int> shape1_idx_lst, 
    const py::array_t<double> rot1_lst, 
    const py::array_t<double> trans1_lst,
    const std::vector<std::shared_ptr<const coal::CollisionGeometry>> shape2_lst,  // Use CollisionGeometry here
    const py::array_t<int> shape2_idx_lst, 
    const py::array_t<double> rot2_lst, 
    const py::array_t<double> trans2_lst, 
    const py::array_t<int> select_idx_lst, 
    const int n,
    py::array_t<double> dist_result, 
    py::array_t<double> normal_result, 
    py::array_t<double> cp1_result, 
    py::array_t<double> cp2_result) {
    
    auto shape1_idx_lst_ptr = shape1_idx_lst.data();
    auto rot1_lst_ptr = rot1_lst.data();
    auto trans1_lst_ptr = trans1_lst.data();
    auto shape2_idx_lst_ptr = shape2_idx_lst.data();
    auto rot2_lst_ptr = rot2_lst.data();
    auto trans2_lst_ptr = trans2_lst.data();
    auto select_idx_lst_ptr = select_idx_lst.data();
    auto dist_result_ptr = dist_result.mutable_data();
    auto normal_result_ptr = normal_result.mutable_data();
    auto cp1_result_ptr = cp1_result.mutable_data();
    auto cp2_result_ptr = cp2_result.mutable_data();

    omp_set_num_threads(COAL_NUM_THREADS);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int idx = select_idx_lst_ptr[i];
        Eigen::Matrix3d R1;
        R1 << rot1_lst_ptr[9*idx], rot1_lst_ptr[9*idx + 1], rot1_lst_ptr[9*idx + 2],
              rot1_lst_ptr[9*idx + 3], rot1_lst_ptr[9*idx + 4], rot1_lst_ptr[9*idx + 5],
              rot1_lst_ptr[9*idx + 6], rot1_lst_ptr[9*idx + 7], rot1_lst_ptr[9*idx + 8];
        Eigen::Vector3d T1;
        T1 << trans1_lst_ptr[3*idx], trans1_lst_ptr[3*idx + 1], trans1_lst_ptr[3*idx + 2];

        coal::Transform3s transform1(R1, T1);

        Eigen::Matrix3d R2;
        R2 << rot2_lst_ptr[9*idx], rot2_lst_ptr[9*idx + 1], rot2_lst_ptr[9*idx + 2],
              rot2_lst_ptr[9*idx + 3], rot2_lst_ptr[9*idx + 4], rot2_lst_ptr[9*idx + 5],
              rot2_lst_ptr[9*idx + 6], rot2_lst_ptr[9*idx + 7], rot2_lst_ptr[9*idx + 8];
        Eigen::Vector3d T2;
        T2 << trans2_lst_ptr[3*idx], trans2_lst_ptr[3*idx + 1], trans2_lst_ptr[3*idx + 2];

        coal::Transform3s transform2(R2, T2);

        coal::DistanceRequest dist_req;
        coal::DistanceResult dist_res;
        
        coal::distance(shape1_lst[shape1_idx_lst_ptr[idx]].get(), transform1, shape2_lst[shape2_idx_lst_ptr[idx]].get(), transform2, dist_req, dist_res);

        // Assigning distances directly to numpy arrays
        dist_result_ptr[idx] = dist_res.min_distance;

        // Assigning normals
        if (dist_res.min_distance > 0.0) {
            normal_result_ptr[3 * idx] = dist_res.normal[0];
            normal_result_ptr[3 * idx + 1] = dist_res.normal[1];
            normal_result_ptr[3 * idx + 2] = dist_res.normal[2];
        }
        else {
            normal_result_ptr[3 * idx] = - dist_res.normal[0];
            normal_result_ptr[3 * idx + 1] = - dist_res.normal[1];
            normal_result_ptr[3 * idx + 2] = - dist_res.normal[2];
        }

        // Assigning closest points
        cp1_result_ptr[3 * idx] = dist_res.nearest_points[0][0];
        cp1_result_ptr[3 * idx + 1] = dist_res.nearest_points[0][1];
        cp1_result_ptr[3 * idx + 2] = dist_res.nearest_points[0][2];

        cp2_result_ptr[3 * idx] = dist_res.nearest_points[1][0];
        cp2_result_ptr[3 * idx + 1] = dist_res.nearest_points[1][1];
        cp2_result_ptr[3 * idx + 2] = dist_res.nearest_points[1][2];
    }
}

// Python bindings using pybind11
PYBIND11_MODULE(coal_openmp_wrapper, m) {
    py::class_<coal::CollisionGeometry, std::shared_ptr<coal::CollisionGeometry>>(m, "CollisionGeometry");

    // Register the loadConvexMeshCpp function
    m.def("loadConvexMeshCpp", &loadConvexMeshCpp, "Load a convex mesh from file and return as CollisionGeometry pointer",
          py::arg("file_name"), py::arg("scale"));

    // Register the batched_coal_distance function
    m.def("batched_coal_distance", &batched_coal_distance, "Compute batched distances using FCL",
          py::arg("shape1_lst"), py::arg("shape1_idx_lst"), py::arg("rot1_lst"), py::arg("trans1_lst"),
          py::arg("shape2_lst"), py::arg("shape2_idx_lst"), py::arg("rot2_lst"), py::arg("trans2_lst"),
          py::arg("select_idx_lst"), py::arg("n"), py::arg("dist_result").noconvert(),
          py::arg("normal_result").noconvert(), py::arg("cp1_result").noconvert(), py::arg("cp2_result").noconvert());
}