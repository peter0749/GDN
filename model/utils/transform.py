import numpy as np
import numba as nb


@nb.njit
def rotation_euler(x, y, z):
    rx = np.array([[1., 0., 0.], [0., np.cos(x), -np.sin(x)], [0., np.sin(x), np.cos(x)]], dtype=np.float32)
    ry = np.array([[np.cos(y), 0., np.sin(y)], [0., 1., 0.], [-np.sin(y), 0., np.cos(y)]], dtype=np.float32)
    rz = np.array([[np.cos(z), -np.sin(z), 0.], [np.sin(z), np.cos(z), 0.], [0., 0., 1.]], dtype=np.float32)
    return np.dot(rz, np.dot(ry, rx).astype(np.float32)).astype(np.float32)


def random_rotation_matrix_xyz():
    x = np.random.uniform(-np.pi/4.0, np.pi/4.0)  # rotation about x-axis:  +/-   45 degree
    y = np.random.uniform(-np.pi/4.0, np.pi/4.0)  # rotation about y-axis:  +/-   45 degree
    z = np.random.uniform(-np.pi+1e-3, np.pi-1e-3)  # rotation about z-axis +/- ~180 degree
    return rotation_euler(float(x), float(y), float(z))


@nb.njit
def generate_gripper_edge(width, hand_height, gripper_pose_wrt_mass_center, thickness, backward=0.0):
    pad = np.array([[0.,0.,0.,1.]]*4, dtype=np.float32)
    gripper_r = np.array([0.,1.,0.], dtype=np.float32)*width/2.
    gripper_l = -gripper_r
    gripper_l_t = gripper_l + np.array([hand_height,0.,0.], dtype=np.float32)
    gripper_r_t = gripper_r + np.array([hand_height,0.,0.], dtype=np.float32)
    gripper_r[0] -= backward
    gripper_l[0] -= backward
    thickness_d = np.array([0., 0., thickness/2.], dtype=np.float32).reshape(1, 3)

    gripper = np.stack( (gripper_l, gripper_r, gripper_l_t, gripper_r_t) ) # (4, 3)
    gripper_outer1 = gripper - thickness_d
    gripper_outer2 = gripper + thickness_d

    # gripper = np.pad(gripper, ((0,0), (0,1)), mode='constant', constant_values=1)
    pad[:,:3] = gripper
    gripper = np.dot(gripper_pose_wrt_mass_center, pad.T).T

    pad[:,:3] = gripper_outer1
    gripper_outer1 = np.dot(gripper_pose_wrt_mass_center, pad.T).T

    pad[:,:3] = gripper_outer2
    gripper_outer2 = np.dot(gripper_pose_wrt_mass_center, pad.T).T

    return gripper, gripper_outer1, gripper_outer2


@nb.njit
def get_plane(a, b):
    # a on plane
    n = b-a  # (a, b, c)
    D = (a*n).sum()
    return np.array([n[0], n[1], n[2], D], dtype=np.float32)  # aX+bY+cZ-D=0


def crop_index(pc, gripper_outer1, gripper_outer2, search_idx=None):
    # cross sign > 0 : in area
    (glb,
     grb,
     gltb,
     grtb) = gripper_outer1

    (glu,
     gru,
     gltu,
     grtu) = gripper_outer2

    p1 = get_plane(glb, grb)  # (4,)
    p2 = get_plane(glb, gltb)
    p3 = get_plane(glb, glu)

    p4 = get_plane(grtu, grtb)
    p5 = get_plane(grtu, gru)
    p6 = get_plane(grtu, gltu)

    planes = np.stack((p1,p2,p3,p4,p5,p6)).T  # (4, 6)
    if search_idx is None:
        in_area = np.where(np.min(np.dot(pc, planes[:3])>planes[-1:], axis=1))[0]
        return in_area
    else:
        in_area = np.where(np.min(np.dot(pc[search_idx], planes[:3])>planes[-1:], axis=1))[0]
        return search_idx[in_area]


def read_registration_file(path):
    with open(path, 'r') as fp:
        registration = np.zeros((3,4), dtype=np.float32)
        registration_list = list(map(float, next(fp).split(',')))
        registration[:3, 0] = registration_list[:3]
        registration[:3, 1] = registration_list[3:6]
        registration[:3, 2] = registration_list[6:9]
        registration[:3, 3] = registration_list[9:12]
    return registration

#from numba.typed import Set # Not implemented in numba. Use built-in set for now
@nb.jit(nopython=True, nogil=True)
def index_to_new_index(old_index, reverse_lookup):
    new_index = []
    for ind, v in enumerate(reverse_lookup):
        if v in old_index:
            new_index.append(ind)
    return new_index
