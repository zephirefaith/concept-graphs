import numpy as np
import open3d as o3d

from gradslam.structures.pointclouds import Pointclouds

PCD_PATH = "/home/priyamp/hitl/habitat-llm/data/trajectories/epidx_508_scene_104348361_171513414/main_agent/rgb_cloud"

gradslam_pcd = Pointclouds.load_pointcloud_from_h5(PCD_PATH)
o3d_pcd = gradslam_pcd.open3d(0)
points = np.asarray(o3d_pcd.points)
o3d_pcd = o3d_pcd.select_by_index(np.where(points[:, 1] < 2.0)[0])

o3d.visualization.draw_geometries([o3d_pcd])
