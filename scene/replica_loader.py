import os
import numpy as np
from PIL import Image
import trimesh

# IMG_WIDTH=512
# IMG_HEIGHT=512

DEPTH_SCALE=4000.0

def read_extrinsic_pose_replica(filepath: str) -> np.array:
    
    T_c2w = np.eye(4)
    # tx ty tz qx qy qz qw
    with open(filepath, 'r') as f:
        lines = f.readline().strip().split(' ')
        # print(f'lines: {lines} ')
        
        cam_center = np.array([float(lines[0]), float(lines[1]), float(lines[2])])
        q_c2w = np.array([float(lines[3]), float(lines[4]), float(lines[5]), float(lines[6])])
        R = trimesh.transformations.quaternion_matrix([q_c2w[3], q_c2w[0], q_c2w[1], q_c2w[2]])[:3,:3]
        # matrix = Rotation.from_quat(q_c2w).as_matrix()
        # if np.allclose(R, matrix):
        #     print('R is equal to matrix')
        
        T_c2w[:3, :3] = R
        T_c2w[:3, 3] = cam_center
    
    return T_c2w        