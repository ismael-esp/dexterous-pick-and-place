import numpy as np
import coal_openmp_wrapper
import time 
import coal 

def loadConvexMesh(file_name: str, scale: np.ndarray):
    loader = coal.MeshLoader()
    bvh: coal.BVHModelBase = loader.load(file_name, scale)
    bvh.buildConvexHull(True, "Qt")
    return bvh.convex

def coal_gjk_worker(shape1, rot1, trans1, shape2, rot2, trans2):
    dist_req = coal.DistanceRequest()
    dist_res = coal.DistanceResult()
    T1 = coal.Transform3f()
    T1.setTranslation(trans1)
    T1.setRotation(rot1.reshape(3,3))
    T2 = coal.Transform3f()
    T2.setTranslation(trans2)
    T2.setRotation(rot2.reshape(3,3))
    coal.distance(shape1, T1, shape2, T2, dist_req, dist_res)
    dist = dist_res.min_distance
    normal = dist_res.normal * (float(dist > 0) * 2 - 1)
    cp1 = dist_res.getNearestPoint1()
    cp2 = dist_res.getNearestPoint2()
    return dist, normal, cp1, cp2
    

data_dict = np.load('idx_pose_dict.npy', allow_pickle=True).item()
file_dict = np.load('mesh_file_dict.npy', allow_pickle=True).item()

num_query = 1000

obj_idx_cpu = data_dict['obj_idx'].numpy()[:num_query]
obj_rot_cpu = data_dict['obj_rot'].reshape(-1, 9).cpu().numpy()[:num_query]
obj_trans_cpu = data_dict['obj_trans'].reshape(-1, 3).cpu().numpy()[:num_query]
robot_idx_cpu = data_dict['robot_idx'].numpy()[:num_query]
robot_rot_cpu = data_dict['robot_rot'].reshape(-1, 9).cpu().numpy()[:num_query]
robot_trans_cpu = data_dict['robot_trans'].reshape(-1, 3).cpu().numpy()[:num_query]

obj_mesh_list = []
obj_mesh_list_debug = []
obj_scale = 0.1
for op in file_dict['obj']:
    om_debug = loadConvexMesh(op, np.array([obj_scale]*3))
    om = coal_openmp_wrapper.loadConvexMeshCpp(op, np.array([obj_scale]*3))
    obj_mesh_list.append(om)
    obj_mesh_list_debug.append(om_debug)

robot_mesh_list = []
robot_mesh_list_debug = []
robot_scale = 1.0
for rp in file_dict['robot']:
    rm_debug = loadConvexMesh(rp, np.array([robot_scale]*3))
    rm = coal_openmp_wrapper.loadConvexMeshCpp(rp, np.array([robot_scale]*3))
    robot_mesh_list.append(rm)
    robot_mesh_list_debug.append(rm_debug)

for i in range(100):
    dist_result = np.zeros(num_query, dtype=np.float64)
    normal_result = np.zeros((num_query, 3), dtype=np.float64)
    cp1_result = np.zeros((num_query, 3), dtype=np.float64)
    cp2_result = np.zeros((num_query, 3), dtype=np.float64)
    st = time.time()

    coal_openmp_wrapper.batched_coal_distance(
        obj_mesh_list, obj_idx_cpu, obj_rot_cpu, obj_trans_cpu,
        robot_mesh_list, robot_idx_cpu, robot_rot_cpu, robot_trans_cpu,
        np.arange(num_query), num_query, dist_result, normal_result, cp1_result, cp2_result)
    
    st2 = time.time()
    results = [coal_gjk_worker(obj_mesh_list_debug[obj_idx_cpu[i]], obj_rot_cpu[i], obj_trans_cpu[i], robot_mesh_list_debug[robot_idx_cpu[i]], robot_rot_cpu[i], robot_trans_cpu[i]) for i in range(num_query)]
    st3 = time.time()


    print('error', np.abs(dist_result-np.array([result[0] for result in results])).max())
    print('error', np.abs(normal_result-np.stack([result[1] for result in results])).max())
    print('error', np.abs(cp1_result-np.stack([result[2] for result in results])).max())
    print('error', np.abs(cp2_result-np.stack([result[3] for result in results])).max())
    print('time', st2-st, st3-st2)
        
