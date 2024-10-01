# %%
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import numpy as np
from rotation_transformer import RotationTransformer
from rotation_utils import rot6d_to_mat, mat_to_rot6d, normalize, quat_from_rot_m

# %%
def test():
    N = 100
    d6 = np.random.normal(size=(N,6))
    rt = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')
    gt_mat = rt.forward(d6)
    mat = rot6d_to_mat(d6)
    assert np.allclose(gt_mat, mat)
    
    to_d6 = mat_to_rot6d(mat)
    to_d6_gt = rt.inverse(mat)
    assert np.allclose(to_d6, to_d6_gt)
    gt_mat = rt.forward(d6[1])
    mat = rot6d_to_mat(d6[1])
    assert np.allclose(gt_mat, mat)
    print(mat)
    norm_mat = normalize(mat)
    print(norm_mat)
    to_d6 = mat_to_rot6d(norm_mat)
    to_d6_gt = rt.inverse(norm_mat)
    print(np.sqrt(to_d6[0]**2+to_d6[1]**2+to_d6[2]**2))
    print(np.sqrt(to_d6[3]**2+to_d6[4]**2+to_d6[5]**2))
    print(quat_from_rot_m(norm_mat))
    assert np.allclose(to_d6, to_d6_gt)
    
    
if __name__ == "__main__":
    test()