import numpy as np
from raisimGymTorch.helper import rotations

def euler_noise_to_quat(quats, palm_pose, noise):
    eulers_palm_mats = np.array([rotations.euler2mat(pose) for pose in palm_pose]).copy()
    eulers_mats =  np.array([rotations.quat2mat(quat) for quat in quats])

    rotmats_list = np.array([rotations.euler2mat(noise) for noise in noise])

    eulers_new = np.matmul(rotmats_list,eulers_mats)
    # print("---------------------")
    # print("eulers_new.shape = ", eulers_new.shape)
    # print("rotmats_list.shape = ", rotmats_list.shape)
    # print("eulers_mats.shape = ", eulers_mats.shape)
    # print("eulers_palm_mats.shape = ", eulers_palm_mats.shape)

    eulers_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_new])

    eulers_palm_new = np.matmul(rotmats_list,eulers_palm_mats)
    eulers_palm_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_palm_new])

    quat_list = [rotations.euler2quat(noise) for noise in eulers_rotmated]

    return np.array(quat_list), eulers_new, eulers_palm_rotmated

def generate_reset_pose(final_obj_pos_random, obj_pose_reset, final_pose_robot):
    finalobj_mat = np.array([rotations.quat2mat(quat) for quat in final_obj_pos_random[:,3:]]).copy()
    resetobj_mat = np.array([rotations.quat2mat(quat) for quat in obj_pose_reset[:,3:]])

    for i in range(finalobj_mat.shape[0]):
        rotmat_single = np.matmul(resetobj_mat[i,:], np.linalg.inv(finalobj_mat[i,:]))
        # rotmat_single, _ = np.linalg.qr(rotmat_single)

        # if np.linalg.det(rotmat_single) < 0:
        #     rotmat_single[:,-1] *= -1
        
        # print("rotmat_single.shape = ", rotmat_single.shape)
        rotmat_single = np.expand_dims(rotmat_single, axis=0)
        # print("rotmat_single.shape after expand_dims= ", rotmat_single.shape)

        if i==0:
            rotmat = rotmat_single.copy()
        else:
            rotmat = np.vstack((rotmat, rotmat_single))

    eulers_wrist_mats = np.array([rotations.euler2mat(pose) for pose in final_pose_robot[:,:3]]).copy()
    eulers_wrist_new = np.matmul(rotmat, eulers_wrist_mats)
    # print("eulers_wrist_new.shape = ", eulers_wrist_new.shape)
    # print("eulers_wrist_new[0].shape[-2:] = ", eulers_wrist_new[0].shape[-2:])
    # print("rotmat.shape = ", rotmat.shape)
    # print("eulers_wrist_mats.shape = ", eulers_wrist_mats.shape)

    eulers_wrist_rotmated = np.array([rotations.mat2euler(mat) for mat in eulers_wrist_new])
    final_pose_robot[:,:3] = eulers_wrist_rotmated.copy()

    return final_pose_robot, rotmat

def final_ee_from_world_to_relative_obj(final_ee_robot_world,final_obj_pos_world,transformGlobalxyz=False):
    if transformGlobalxyz == False:
        num_bodyparts = 17
    else:
        num_bodyparts = 1
    final_ee_rel_robot = final_ee_robot_world.copy()
    for j in range(num_bodyparts):
        final_ee_rel_robot[:,j*3] -= final_obj_pos_world[:,0]
        final_ee_rel_robot[:,(j*3+1)] -= final_obj_pos_world[:,1]
        final_ee_rel_robot[:,(j*3+2)] -= final_obj_pos_world[:,2]
    final_ee_rel_robot_expand = np.expand_dims(final_ee_rel_robot, axis=2)  # (n,num_bodyparts*3,1)
    final_obj_pos_mat = np.array([rotations.quat2mat(quat) for quat in final_obj_pos_world[:,3:]])  #  (:, 3, 3)
    final_obj_pos_mat_T = np.array([mat.T for mat in final_obj_pos_mat])

    for j in range(num_bodyparts):
        final_ee_rel_robot_expand_relative_obj_single = np.matmul(final_obj_pos_mat_T, final_ee_rel_robot_expand[:,j*3:((j+1)*3),:])
        final_ee_rel_robot_expand_relative_obj_single = np.squeeze(final_ee_rel_robot_expand_relative_obj_single)
        if j==0:
            final_ee_rel_robot_expand_relative_obj = final_ee_rel_robot_expand_relative_obj_single.copy()
        else:
            final_ee_rel_robot_expand_relative_obj = np.hstack((final_ee_rel_robot_expand_relative_obj, final_ee_rel_robot_expand_relative_obj_single))
        
    return final_ee_rel_robot_expand_relative_obj

def get_point_online_tip(p1,p2,posrobot,resize=1):
    return posrobot + resize*(p1-p2)

def update_hand_reset_globalxyz(final_ee_random_or_world_resetobj,obj_pose_reset_seq,resizeNum=1.3):
    for i in range(obj_pose_reset_seq.shape[0]):
        final_ee_xyz = final_ee_random_or_world_resetobj.copy()
        final_ee_xyz = np.array(final_ee_xyz[i:(i+1),:]) 
        obj_pose_reset_xyz = np.array(obj_pose_reset_seq[i:(i+1),:3])
        # print("final_ee_xyz.shape = ", final_ee_xyz.shape)
        # print("obj_pose_reset_xyz.shape = ", obj_pose_reset_xyz.shape)
        qpos_reset_xyz_single = get_point_online_tip(final_ee_xyz, obj_pose_reset_xyz, obj_pose_reset_xyz, resize=resizeNum)  # 这里需要改大小，reset和final之间的距离, 上一次是1.3
        # print("qpos_reset_xyz_single.shape = ", qpos_reset_xyz_single.shape)
        if i==0:
            qpos_reset_xyz = qpos_reset_xyz_single.copy()
        else:
            qpos_reset_xyz = np.vstack((qpos_reset_xyz,qpos_reset_xyz_single))
        
    return qpos_reset_xyz

def final_ee2Reset_ee(final_ee_robot, obj_pose_reset_seq, final_pose_robot, final_obj_pos, finger_bending=0.75, resize=1.3):
    final_ee_robot_rel = final_ee_from_world_to_relative_obj(final_ee_robot, final_obj_pos, transformGlobalxyz=True)
    final_ee_robot_rel = np.expand_dims(final_ee_robot_rel,axis=2)
    obj_pose_reset_seq_mat = np.array([rotations.quat2mat(quat) for quat in obj_pose_reset_seq[:,3:]])
    final_ee_robot_rel_resetobj = np.matmul(obj_pose_reset_seq_mat, final_ee_robot_rel)
    final_ee_robot_world_resetobj = np.squeeze(final_ee_robot_rel_resetobj) + obj_pose_reset_seq[:,:3]

    qpos_reset_xyz = update_hand_reset_globalxyz(final_ee_robot_world_resetobj,obj_pose_reset_seq,resizeNum=resize)
    qpos_reset_new, noise_guess = generate_reset_pose(final_obj_pos.copy(), obj_pose_reset_seq.copy(), final_pose_robot.copy())
    qpos_reset_robot_new = np.hstack((qpos_reset_xyz, qpos_reset_new))
    qpos_reset_robot_new[:,6:] *= finger_bending

    return qpos_reset_robot_new.astype('float32')