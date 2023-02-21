"""
copied from DISN dataset:
https://github.com/Xharlie/ShapenetRender_more_variation/blob/master/cam_read.py
"""

import numpy as np
import os
import sys
import h5py
import cv2

rot90y = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]], dtype=np.float32)

params = [[
[13.609081057113638,44.08659091162214,0,0.7250000039172821,35,32,1.75,0.011900090612471104,0.18190337717533112,0.006238838657736778],
[18.771401768548415,18.1609811873455,0,0.8679759320460956,35,32,1.75,0.09003015607595444,-0.14270350337028503,0.05223459377884865],
[25.878850162901006,41.81812383397336,0,0.9751217424184879,35,32,1.75,0.18383225798606873,-0.009411958046257496,0.09324256330728531],
[35.99688020783701,44.16026916688504,0,0.7531933092574246,35,32,1.75,0.19104833900928497,0.04526694118976593,0.18508180975914001],
[48.19587966587669,5.634790095730125,0,0.8690428772309136,35,32,1.75,-0.1652826964855194,0.08896388858556747,0.16622555255889893],
[63.86510514171296,38.54127457742132,0,0.7137306097298087,35,32,1.75,-0.040847986936569214,0.10500576347112656,0.06099827215075493],
[71.03331485170746,31.445728064763074,0,0.7631809542510639,35,32,1.75,0.05915249139070511,-0.19200047850608826,0.1869116872549057],
[78.93149731392128,8.373853466957998,0,0.9278049610816022,35,32,1.75,0.04910890385508537,0.18967053294181824,0.09642542898654938],
[92.93504577985884,30.47282082362938,0,0.9460081651709269,35,32,1.75,0.11576724052429199,-0.19369271397590637,-0.10284043103456497],
[104.06258690756752,34.89268826144493,0,0.9681601773368812,35,32,1.75,0.04314812645316124,-0.15145060420036316,-0.1740894317626953],
[108.95573152324525,23.11176700395444,0,0.770639013064592,35,32,1.75,0.15823353826999664,0.13090209662914276,-0.05024176836013794],
[116.6008502422741,0.2166350071844464,0,0.9161140203446375,35,32,1.75,0.04646913707256317,-0.1679994910955429,0.14490677416324615],
[130.58465088703826,5.184621942865357,0,0.8602119283710821,35,32,1.75,-0.0523235946893692,0.08109834790229797,0.041353490203619],
[140.0814191209536,30.7580598376864,0,0.8225658059383021,35,32,1.75,0.02494741417467594,-0.15723608434200287,0.1662987470626831],
[151.6945424628218,40.30938294045265,0,0.8300863497152311,35,32,1.75,0.04902523383498192,0.12987209856510162,-0.14116427302360535],
[155.10293763254376,36.00196909180084,0,0.9385488975504941,35,32,1.75,-0.060411594808101654,0.11220726370811462,0.14445669949054718],
[172.66686856029312,23.401623371118287,0,0.6077178503384195,35,32,1.75,0.18273067474365234,0.024926139041781425,0.07727616280317307],
[184.39893514490126,26.50639191519731,0,0.7616966307025058,35,32,1.75,0.11640718579292297,-0.15759973227977753,0.08047189563512802],
[186.31522987643604,22.34199368701631,0,0.781413880723633,35,32,1.75,0.09689559042453766,-0.18857409060001373,0.10417064279317856],
[200.3669351695706,30.966916192283943,0,0.9726330816884996,35,32,1.75,-0.10276813805103302,-0.03201678767800331,0.1582796275615692],
[212.81609745138903,5.242959119303972,0,0.9610687351783843,35,32,1.75,-0.03486974909901619,-0.07885606586933136,0.06740577518939972],
[222.69074555755464,34.97655081294485,0,0.7449820536270011,35,32,1.75,0.15827082097530365,-0.15252475440502167,-0.02471393346786499],
[226.36156985649166,24.12025052159019,0,0.6084750359327343,35,32,1.75,0.16977459192276,-0.1849607676267624,0.16296547651290894],
[238.13127042168853,38.96101470087499,0,0.9527401793315309,35,32,1.75,0.03490840271115303,-0.14361941814422607,0.0456993505358696],
[246.6971544140491,42.96785554909819,0,0.7632392246742773,35,32,1.75,0.07354340702295303,-0.19146820902824402,0.029128391295671463],
[263.188201153171,43.58194878329786,0,0.9971558337468894,35,32,1.75,-0.07030873000621796,0.015776721760630608,-0.15853647887706757],
[273.5738581640022,21.771445368627766,0,0.9574869361642284,35,32,1.75,0.1489466279745102,-0.0993940532207489,0.06929264217615128],
[277.83357477482525,44.22157811849666,0,0.7603451527181114,35,32,1.75,0.11418163776397705,-0.14371995627880096,-0.09554994851350784],
[286.95374625476126,14.978816297828445,0,0.8355745390330079,35,32,1.75,0.029406240209937096,-0.09835749864578247,0.1129160076379776],
[301.2761431690078,2.2205958119809477,0,0.9640588118493543,35,32,1.75,-0.039721615612506866,-0.1531001478433609,-0.06817512214183807],
[305.53574837592873,30.577959276347084,0,0.9563756397852338,35,32,1.75,-0.1973000019788742,-0.07772478461265564,-0.09494104981422424],
[324.94101052194753,25.490943074788202,0,0.8470615554015754,35,32,1.75,-0.009690315462648869,-0.18609729409217834,-0.028144175186753273],
[331.9528475015708,27.953736988053663,0,0.9882868522848163,35,32,1.75,-0.03387337923049927,0.004485097248107195,-0.05609232187271118],
[335.20370027377317,27.595871390765616,0,0.6746722947065639,35,32,1.75,-0.09185966849327087,-0.02847306989133358,0.09793012589216232],
[348.3053274377389,5.936488500371313,0,0.9645940583148378,35,32,1.75,-0.09842805564403534,0.009305895306169987,0.194208025932312],
[357.5995457541137,22.899753896755964,0,0.7912261645286426,35,32,1.75,-0.00983845442533493,0.18935218453407288,-0.007812697440385818]
]]


def getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""
    F_MM = 35.  # Focal length
    SENSOR_SIZE_MM = 32.
    PIXEL_ASPECT_RATIO = 1.  # pixel_aspect_x / pixel_aspect_y
    RESOLUTION_PCT = 100.
    SKEW = 0.
    CAM_MAX_DIST = 1.75
    CAM_ROT = np.asarray([[1.910685676922942e-15, 4.371138828673793e-08, 1.0],
                          [1.0, -4.371138828673793e-08, -0.0],
                          [4.371138828673793e-08, 1.0, -4.371138828673793e-08]])

    # Calculate intrinsic matrix.
    # 2 atan(35 / 2*32)
    scale = RESOLUTION_PCT / 100
    f_u = F_MM * img_w * scale / SENSOR_SIZE_MM
    f_v = F_MM * img_h * scale * PIXEL_ASPECT_RATIO / SENSOR_SIZE_MM
    u_0 = img_w * scale / 2
    v_0 = img_h * scale / 2
    K = np.matrix(((f_u, SKEW, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    # Step 1: World coordinate to object coordinate.
    sa = np.sin(np.radians(-az))
    ca = np.cos(np.radians(-az))
    se = np.sin(np.radians(-el))
    ce = np.cos(np.radians(-el))
    R_world2obj = np.transpose(np.matrix(((ca * ce, -sa, ca * se),
                                          (sa * ce, ca, sa * se),
                                          (-se, 0, ce))))

    # Step 2: Object coordinate to camera coordinate.
    R_obj2cam = np.transpose(np.matrix(CAM_ROT))
    R_world2cam = R_obj2cam * R_world2obj
    cam_location = np.transpose(np.matrix((distance_ratio * CAM_MAX_DIST,
                                           0,
                                           0)))
    # print('distance', distance_ratio * CAM_MAX_DIST)
    T_world2cam = -1 * R_obj2cam * cam_location

    # Step 3: Fix blender camera's y and z axis direction.
    R_camfix = np.matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_world2cam = R_camfix * R_world2cam
    T_world2cam = R_camfix * T_world2cam

    RT = np.hstack((R_world2cam, T_world2cam))
    return K, RT


def get_rotate_matrix(rotation_angle1):
    cosval = np.cos(rotation_angle1)
    sinval = np.sin(rotation_angle1)

    rotation_matrix_x = np.array([[1, 0, 0, 0],
                                  [0, cosval, -sinval, 0],
                                  [0, sinval, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_y = np.array([[cosval, 0, sinval, 0],
                                  [0, 1, 0, 0],
                                  [-sinval, 0, cosval, 0],
                                  [0, 0, 0, 1]])
    rotation_matrix_z = np.array([[cosval, -sinval, 0, 0],
                                  [sinval, cosval, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    scale_y_neg = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    neg = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    # y,z swap = x rotate -90, scale y -1
    # new_pts0[:, 1] = new_pts[:, 2]
    # new_pts0[:, 2] = new_pts[:, 1]
    #
    # x y swap + negative = z rotate -90, scale y -1
    # new_pts0[:, 0] = - new_pts0[:, 1] = - new_pts[:, 2]
    # new_pts0[:, 1] = - new_pts[:, 0]

    # return np.linalg.multi_dot([rotation_matrix_z, rotation_matrix_y, rotation_matrix_y, scale_y_neg, rotation_matrix_z, scale_y_neg, rotation_matrix_x])
    return np.linalg.multi_dot([neg, rotation_matrix_z, rotation_matrix_z, scale_y_neg, rotation_matrix_x])


def get_norm_matrix(sdf_h5_file):
    with h5py.File(sdf_h5_file, 'r') as h5_f:
        norm_params = h5_f['norm_params'][:]
        center, m, = norm_params[:3], norm_params[3]
        x, y, z = center[0], center[1], center[2]
        M_inv = np.asarray(
            [[m, 0., 0., 0.],
             [0., m, 0., 0.],
             [0., 0., m, 0.],
             [0., 0., 0., 1.]]
        )
        T_inv = np.asarray(
            [[1.0, 0., 0., x],
             [0., 1.0, 0., y],
             [0., 0., 1.0, z],
             [0., 0., 0., 1.]]
        )
    return np.matmul(T_inv, M_inv)


def get_W2O_mat(shift):
    T_inv = np.asarray(
        [[1.0, 0., 0., shift[0]],
         [0., 1.0, 0., shift[1]],
         [0., 0., 1.0, shift[2]],
         [0., 0., 0., 1.]]
    )
    return T_inv

def gen_obj_img_h5():
    #img_dir = "./test_render/image/03001627/17e916fc863540ee3def89b32cef8e45/hard/"
    img_dir = "D:/GeoTextureExp/image/04530566/1b00e4c41b4195807e1c97634acf0214/easy/"
    sample_pc = np.asarray([[0, 0, 0]])
    colors = np.asarray([[0, 0, 255, 255]])
    # norm_mat = get_norm_matrix("/ssd1/datasets/ShapeNet/SDF_v2/03001627/17e916fc863540ee3def89b32cef8e45/ori_sample.h5")
    rot_mat = get_rotate_matrix(-np.pi / 2)
    for i in range(len(params[0])):
        param = params[0][i]
        camR, _ = get_img_cam(param)
        obj_rot_mat = np.dot(rot90y, camR)
        img_file = os.path.join(img_dir, '{0:02d}.png'.format(i))
        out_img_file = os.path.join(img_dir, '{0:02d}_out.png'.format(i))
        print("img_file", img_file)
        img_arr = cv2.imread(img_file, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        az, el, distance_ratio = param[0], param[1], param[3]
        K, RT = getBlenderProj(az, el, distance_ratio, img_w=224, img_h=224)
        print((-param[-1], -param[-2], param[-3]))
        W2O_mat = get_W2O_mat((param[-3], param[-1], -param[-2]))
        # trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat, norm_mat])
        trans_mat = np.linalg.multi_dot([K, RT, rot_mat, W2O_mat])
        trans_mat_right = np.transpose(trans_mat)
        # regress_mat = np.transpose(np.linalg.multi_dot([RT, rot_mat, W2O_mat, norm_mat]))
        pc_xy = get_img_points(sample_pc, trans_mat_right)  # sample_pc - camloc
        print("trans_mat_right", trans_mat_right)
        for j in range(pc_xy.shape[0]):
            y = int(pc_xy[j, 1])
            x = int(pc_xy[j, 0])
            print(x,y)
            cv2.circle(img_arr, (x, y), 10, tuple([int(x) for x in colors[j]]), -2)
        cv2.imwrite(out_img_file, img_arr)

def get_img_points(sample_pc, trans_mat_right):
    sample_pc = sample_pc.reshape((-1, 3))
    homo_pc = np.concatenate((sample_pc, np.ones((sample_pc.shape[0], 1), dtype=np.float32)), axis=-1)
    pc_xyz = np.dot(homo_pc, trans_mat_right).reshape((-1, 3))
    # pc_xyz = np.transpose(np.matmul(trans_mat, np.transpose(homo_pc))).reshape((-1,3))
    # print("pc_xyz",pc_xyz)
    print("pc_xyz shape: ", pc_xyz.shape)
    pc_xy = pc_xyz[:, :2] / pc_xyz[:, 2]
    return pc_xy.astype(np.int32)

def get_img_cam(param):
    cam_mat, cam_pos = camera_info(degree2rad(param))
    return cam_mat, cam_pos

def camera_info(param):
    az_mat = get_az(param[0])
    el_mat = get_el(param[1])
    inl_mat = get_inl(param[2])
    cam_mat = np.transpose(np.matmul(np.matmul(inl_mat, el_mat), az_mat))
    cam_pos = get_cam_pos(param)
    return cam_mat, cam_pos

def get_cam_pos(param):
    camX = 0
    camY = 0
    camZ = param[3]
    cam_pos = np.array([camX, camY, camZ])
    return -1 * cam_pos

def get_az(az):
    cos = np.cos(az)
    sin = np.sin(az)
    mat = np.asarray([cos, 0.0, sin, 0.0, 1.0, 0.0, -1.0 * sin, 0.0, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_el(el):
    cos = np.cos(el)
    sin = np.sin(el)
    mat = np.asarray([1.0, 0.0, 0.0, 0.0, cos, -1.0 * sin, 0.0, sin, cos], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def get_inl(inl):
    cos = np.cos(inl)
    sin = np.sin(inl)
    # zeros = np.zeros_like(inl)
    # ones = np.ones_like(inl)
    mat = np.asarray([cos, -1.0 * sin, 0.0, sin, cos, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    mat = np.reshape(mat, [3, 3])
    return mat

def degree2rad(params):
    params_new = np.zeros_like(params)
    params_new[0] = np.deg2rad(params[0] + 180.0)
    params_new[1] = np.deg2rad(params[1])
    params_new[2] = np.deg2rad(params[2])
    return params_new
