from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import Allegro_Teacher_Student as mano
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.Allegro_Teacher_Student import NormalSampler
import os
import os.path
import time
import raisimGymTorch.algo.ppo_with_dagger.module as ppo_module
from raisimGymTorch.algo.ppo_with_dagger.dagger import DaggerExpert, DaggerAgent, DaggerTrainer
import torch.nn as nn
import numpy as np
import torch
import datetime
import argparse
import joblib
from raisimGymTorch.helper import utils, rotations

GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

translation_names = ['WRJTx', 'WRJTy', 'WRJTz']
rot_names = ['WRJRx', 'WRJRy', 'WRJRz']
joint_names = [
    'joint_0.0', 'joint_1.0', 'joint_2.0', 'joint_3.0', 
    'joint_4.0', 'joint_5.0', 'joint_6.0', 'joint_7.0', 
    'joint_8.0', 'joint_9.0', 'joint_10.0', 'joint_11.0', 
    'joint_12.0', 'joint_13.0', 'joint_14.0', 'joint_15.0'
]

IDX_TO_OBJ = {
    1: ['002_master_chef_can',0.414, 0, [0.051,0.139,0.0]],
    2: ['003_cracker_box', 0.453, 1, [0.06, 0.158, 0.21]],
    3: ['004_sugar_box', 0.514, 1, [0.038, 0.089, 0.175]],
    4: ['005_tomato_soup_can', 0.349, 0, [0.033, 0.101,0.0]],
    5: ['006_mustard_bottle', 0.431,2, [0.0,0.0,0.0]],
    6: ['007_tuna_fish_can', 0.171, 0, [0.0425, 0.033,0.0]],
    7: ['008_pudding_box', 0.187, 3, [0.21, 0.089, 0.035]],
    8: ['009_gelatin_box', 0.097, 3, [0.028, 0.085, 0.073]],
    9: ['010_potted_meat_can', 0.37, 3, [0.05, 0.097, 0.089]],
    10: ['011_banana', 0.066,2, [0.028, 0.085, 0.073]],
    11: ['019_pitcher_base', 0.178,2, [0.0,0.0,0.0]],
    12: ['021_bleach_cleanser', 0.302,2, [0.0,0.0,0.0]],
    13: ['024_bowl', 0.147,2, [0.0,0.0,0.0]],
    14: ['025_mug', 0.118,2, [0.0,0.0,0.0]],
    15: ['035_power_drill', 0.895,2, [0.0,0.0,0.0]],
    16: ['036_wood_block', 0.729, 3, [0.085, 0.085, 0.2]],
    17: ['037_scissors', 0.082,2, [0.0,0.0,0.0]],
    18: ['040_large_marker', 0.01, 3, [0.009,0.121,0.0]],
    19: ['051_large_clamp', 0.125,2, [0.0,0.0,0.0]],
    20: ['052_extra_large_clamp', 0.102,2, [0.0,0.0,0.0]],
    21: ['061_foam_brick', 0.028, 1, [0.05, 0.075, 0.05]],
}

### configuration of command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='config file', type=str, default='cfg.yaml')
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-d', '--logdir', help='set dir for storing data', type=str, default=None)
parser.add_argument('-e', '--exp_name', help='exp_name', type=str, default="grasping")
parser.add_argument('-w', '--weight', type=str, default='2021-09-29-18-20-07/full_400.pt')
parser.add_argument('-sd', '--storedir', type=str, default='data_all')
parser.add_argument('-pr','--prior', action="store_true")
parser.add_argument('-o', '--obj_id',type=int, default=-1)
parser.add_argument('-t','--test', action="store_true")
parser.add_argument('-mc','--mesh_collision', action="store_true")
parser.add_argument('-ao','--all_objects', action="store_true")
parser.add_argument('-ev','--evaluate', action="store_true")
parser.add_argument('-sv','--store_video', action="store_true")
parser.add_argument('-to','--test_object_set', type=int, default=-1)
parser.add_argument('-ac','--all_contact', action="store_true")
parser.add_argument('-seed','--seed', type=int, default=1)
parser.add_argument('-itr','--num_iterations', type=int, default=3001)
parser.add_argument('-nr','--num_repeats', type=int, default=1)
parser.add_argument('--motion_synthesis_extra_steps', type=int, default=50)
parser.add_argument('-log','--exp_log', action="store_true")
parser.add_argument('-vis_ev','--vis_evaluate', action="store_true")
parser.add_argument('-idx', '--rand_6D_idx',type=int, default=0)
parser.add_argument('-d_vis', '--debug_single_vis',type=int, default=-1)
parser.add_argument('-ok','--ok_example', action="store_true")
parser.add_argument("--loadpth", type = str, default = "/home/ubuntu/raisim/dgrasp/raisimGymTorch/AllegroHand_Checkpoint/AllegroHand_Policy")
parser.add_argument("--prop_enc_pth", type = str, default = "/home/ubuntu/raisim/dgrasp/raisimGymTorch/AllegroHand_Checkpoint/AllegroHand_LSTMEncoder") 
parser.add_argument("--loadid", type = str, default = "3450")
parser.add_argument("--prop_enc_id", type = str, default = "750")
parser.add_argument("--ext_act", type = str, default='leakyRelu')
parser.add_argument('-GT','--GT_info', action="store_true")
parser.add_argument('-tactile_ev','--evaluate_encoder', action="store_true")
parser.add_argument('-ait','--record_aitviewer', action="store_true")


args = parser.parse_args()
mode = args.mode
weight_path = args.weight

print(f"Configuration file: \"{args.cfg}\"")
print(f"Experiment name: \"{args.exp_name}\"")

### task specification
task_name = args.exp_name
### check if gpu is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#torch.set_default_dtype(torch.double)
### directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."

if args.logdir is None:
    exp_path = home_path
else:
    exp_path = args.logdir

### load config
cfg = YAML().load(open(task_path+'/cfgs/' + args.cfg, 'r'))
print(task_path+'/cfgs/' + args.cfg)

### set seed
if args.seed != 1:
    cfg['seed']=args.seed

if args.exp_log:
    cfg['exp_log']=True

### get experiment parameters
num_envs = cfg['environment']['num_envs']
pre_grasp_steps = cfg['environment']['pre_grasp_steps']
trail_steps = cfg['environment']['trail_steps']
test_inference = args.test
train_obj_id = args.obj_id
all_obj_train = True if args.all_objects else False
meta_info_dim = 4

num_repeats= args.num_repeats
activations = nn.LeakyReLU
output_activation = nn.Tanh

### Load data labels
if not args.test:
    dict_labels=joblib.load("raisimGymTorch/data/dexycb_train_labels_processed_with_retarget.pkl")
else:
    dict_labels=joblib.load("raisimGymTorch/data/dexycb_test_labels_processed_with_retarget.pkl")



### Load data labels for all objects
if all_obj_train:
    obj_w_list, obj_pose_reset_list, qpos_reset_list, obj_dim_list, obj_type_list, obj_idx_list_list = [], [], [], [], [], []
    final_qpos_list, final_obj_pos_list, final_pose_list,final_ee_list, final_ee_rel_list, final_contact_pos_list, final_contacts_list = [], [], [], [], [], [], []
    final_pose_robot_list, final_ee_robot_list, final_ee_rel_robot_list, final_contacts_robot_list, qpos_reset_robot_list = [], [], [], [], []
    ### Iterate through all objects and add to single dict
    if not args.test:
        for obj_key in dict_labels.keys():

            final_qpos_list.append(dict_labels[obj_key]['final_qpos'])
            final_obj_pos_list.append(dict_labels[obj_key]['final_obj_pos'])
            final_pose_list.append(dict_labels[obj_key]['final_pose'])
            final_ee_list.append(dict_labels[obj_key]['final_ee'])
            final_ee_rel_list.append(dict_labels[obj_key]['final_ee_rel'])
            final_contact_pos_list.append(dict_labels[obj_key]['final_contact_pos'])
            final_contacts_list.append(dict_labels[obj_key]['final_contacts'])

            final_pose_robot_list.append(dict_labels[obj_key]['final_pose_robot'])
            final_ee_robot_list.append(dict_labels[obj_key]['final_ee_robot'])
            final_ee_rel_robot_list.append(dict_labels[obj_key]['final_ee_rel_robot'])
            final_contacts_robot_list.append(dict_labels[obj_key]['final_contacts_robot'])
            qpos_reset_robot_list.append(dict_labels[obj_key]['qpos_reset_robot'])
            
            obj_w_list.append(dict_labels[obj_key]['obj_w_stacked'])
            obj_dim_list.append(dict_labels[obj_key]['obj_dim_stacked'])
            obj_type_list.append(dict_labels[obj_key]['obj_type_stacked'])
            obj_idx_list_list.append(dict_labels[obj_key]['obj_idx_stacked'])
            obj_pose_reset_list.append(dict_labels[obj_key]['obj_pose_reset'])
            qpos_reset_list.append(dict_labels[obj_key]['qpos_reset'])

    else:
        for obj_key in dict_labels.keys():

            final_qpos_list.append(dict_labels[obj_key]['final_qpos'])
            final_obj_pos_list.append(dict_labels[obj_key]['final_obj_pos'])
            final_pose_list.append(dict_labels[obj_key]['final_pose'])
            final_ee_list.append(dict_labels[obj_key]['final_ee'])
            final_ee_rel_list.append(dict_labels[obj_key]['final_ee_rel'])
            final_contact_pos_list.append(dict_labels[obj_key]['final_contact_pos'])
            final_contacts_list.append(dict_labels[obj_key]['final_contacts'])

            final_pose_robot_list.append(dict_labels[obj_key]['final_pose_robot'])
            final_ee_robot_list.append(dict_labels[obj_key]['final_ee_robot'])
            final_ee_rel_robot_list.append(dict_labels[obj_key]['final_ee_rel_robot'])
            final_contacts_robot_list.append(dict_labels[obj_key]['final_contacts_robot'])
            qpos_reset_robot_list.append(dict_labels[obj_key]['qpos_reset_robot'])

            obj_w_list.append(dict_labels[obj_key]['obj_w_stacked'])
            obj_dim_list.append(dict_labels[obj_key]['obj_dim_stacked'])
            obj_type_list.append(dict_labels[obj_key]['obj_type_stacked'])
            obj_idx_list_list.append(dict_labels[obj_key]['obj_idx_stacked'])
            obj_pose_reset_list.append(dict_labels[obj_key]['obj_pose_reset'])
            qpos_reset_list.append(dict_labels[obj_key]['qpos_reset'])


    final_qpos = np.vstack(final_qpos_list).astype('float32')
    final_obj_pos = np.vstack(final_obj_pos_list).astype('float32')
    final_pose = np.vstack(final_pose_list).astype('float32')
    final_ee = np.vstack(final_ee_list).astype('float32')
    final_ee_rel =  np.vstack(final_ee_rel_list).astype('float32')
    final_contact_pos = np.vstack(final_contact_pos_list).astype('float32')
    final_contacts = np.vstack(final_contacts_list).astype('float32')

    final_pose_robot = np.vstack(final_pose_robot_list).astype('float32')
    final_ee_robot = np.vstack(final_ee_robot_list).astype('float32')
    final_ee_rel_robot = np.vstack(final_ee_rel_robot_list).astype('float32')
    final_contacts_robot = np.vstack(final_contacts_robot_list).astype('float32')
    qpos_reset_robot = np.vstack(qpos_reset_robot_list).astype('float32')

    obj_w_stacked = np.hstack(obj_w_list).astype('float32')
    obj_dim_stacked = np.vstack(obj_dim_list).astype('float32')
    obj_type_stacked = np.hstack(obj_type_list)
    obj_idx_stacked = np.hstack(obj_idx_list_list)
    obj_pose_reset = np.vstack(obj_pose_reset_list).astype('float32')
    qpos_reset = np.vstack(qpos_reset_list).astype('float32')
### Load labels for single object
else:
    final_qpos = np.array(dict_labels[train_obj_id]['final_qpos']).astype('float32')
    final_obj_pos = np.array(dict_labels[train_obj_id]['final_obj_pos']).astype('float32')
    final_pose = np.array(dict_labels[train_obj_id]['final_pose']).astype('float32')
    final_ee = np.array(dict_labels[train_obj_id]['final_ee']).astype('float32')
    final_ee_rel = np.array(dict_labels[train_obj_id]['final_ee_rel']).astype('float32')
    final_contact_pos = np.array(dict_labels[train_obj_id]['final_contact_pos']).astype('float32')
    final_contacts = np.array(dict_labels[train_obj_id]['final_contacts']).astype('float32')

    final_pose_robot = np.array(dict_labels[obj_key]['final_pose_robot']).astype('float32')
    final_ee_robot = np.array(dict_labels[obj_key]['final_ee_robot']).astype('float32')
    final_ee_rel_robot = np.array(dict_labels[obj_key]['final_ee_rel_robot']).astype('float32')
    final_contacts_robot = np.array(dict_labels[obj_key]['final_contacts_robot']).astype('float32')
    qpos_reset_robot = np.array(dict_labels[obj_key]['qpos_reset_robot']).astype('float32')

    obj_w_stacked = np.array(dict_labels[train_obj_id]['obj_w_stacked']).astype('float32')
    obj_dim_stacked = np.array(dict_labels[train_obj_id]['obj_dim_stacked']).astype('float32')
    obj_type_stacked = np.array(dict_labels[train_obj_id]['obj_type_stacked'])
    obj_idx_stacked = np.array(dict_labels[train_obj_id]['obj_idx_stacked'])
    obj_pose_reset = np.array(dict_labels[train_obj_id]['obj_pose_reset']).astype('float32')
    qpos_reset = np.array(dict_labels[train_obj_id]['qpos_reset']).astype('float32')

num_envs = 1 if args.vis_evaluate else final_qpos.shape[0]
cfg['environment']['hand_model'] = "mano_mean_meshcoll.urdf" if args.mesh_collision else "AllegroHand.urdf"
cfg['environment']['num_envs'] = num_envs
cfg["testing"] = True if test_inference else False
print('num envs', final_qpos.shape[0])

env = VecEnv(mano.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

# Setting dimensions from environments
ob_dim = env.num_obs
act_dim = env.num_acts

### Set training step parameters
grasp_steps = pre_grasp_steps
n_steps = grasp_steps  + trail_steps + args.motion_synthesis_extra_steps
total_steps = n_steps * env.num_envs

avg_rewards = []

log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name
saver = ConfigurationSaver(log_dir = exp_path + "/raisimGymTorch/" + args.storedir + "/" + task_name,
                           save_items=[task_path + "/cfgs/" + args.cfg, task_path + "/Environment.hpp", task_path + "/runner.py"], test_dir=True)

module_type = ppo_module.MLPEncode_wrap
activation_fn_map = {'none': None, 'tanh': nn.Tanh, 'leakyRelu': nn.LeakyReLU}
output_activation_fn = None
small_init_flag = cfg['architecture']['small_init']
init_var = 0.3
baseDim = 113
hand_dim = 95
obj_dim = 18
hand_obj_dim = 26
tobeEncode_dim = 60
obDim_double = 159
obDim_double += tobeEncode_dim

geomDim = 9
n_futures = 1
t_steps = 10
ext_activation_map = activation_fn_map[args.ext_act]
prop_latent_dim=26
geom_latent_dim=0
priv_dim = ob_dim - tobeEncode_dim * t_steps - (tobeEncode_dim+obj_dim)

actor_expert = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], activations, obDim_double-meta_info_dim-tobeEncode_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim, num_envs, 1.0, NormalSampler(act_dim)),device='cpu')

PPOPolicy_path = '/'.join([args.loadpth, 'full_' + args.loadid + '.pt'])
PPOPolicy = torch.load(PPOPolicy_path,map_location={'cpu':'cuda:0'})
actor_expert.architecture.load_state_dict(PPOPolicy['actor_architecture_state_dict'])

prop_enc_pth = '/'.join([args.prop_enc_pth, 'prop_encoder_' + args.prop_enc_id + '.pt'])
prop_loaded_encoder_expert = torch.jit.load(prop_enc_pth,map_location="cpu")

hidden_size=64
num_layers=3
prop_loaded_encoder = ppo_module.LSTM_StateHistoryEncoder(tobeEncode_dim, prop_latent_dim, hidden_size,num_layers,t_steps)                                   
prop_loaded_encoder.load_state_dict(prop_loaded_encoder_expert.state_dict())

### Loading a pretrained model
fname = log_dir+'/success_idxs.npy'

if args.ok_example and (os.path.isfile(fname)):
    success_idxs = np.load(fname)
else:
    success_idxs = np.arange(final_qpos.shape[0])

print(GREEN + "Finish to load_param" + END)

if all_obj_train and train_obj_id != -1:
    obj_success_idx = np.where(obj_idx_stacked[success_idxs] == train_obj_id-1)[0]
    success_idxs=success_idxs[obj_success_idx] if obj_success_idx.shape[0] > 0 else np.where(obj_idx_stacked == train_obj_id-1)[0]

### Load dictionary of object target 6D poses (displacements to the initial position)
eval_dict = joblib.load('raisimGymTorch/data/motion_eval_dict_easy.pkl')
obj_target_pos = eval_dict['target_pos']
obj_target_ang_noise = eval_dict['ang_noise']

### Initialize the environment
final_obj_pos_random = final_obj_pos.copy()
rand_6D_idx = args.rand_6D_idx
final_obj_pos_random[:,:3] += obj_target_pos[rand_6D_idx]

final_ee_random = final_ee_robot.copy()
for i in range(17):
    final_ee_random[:, i*3] += obj_target_pos[rand_6D_idx,0]
    final_ee_random[:, i*3+1] += obj_target_pos[rand_6D_idx,1]
    final_ee_random[:, i*3+2] += obj_target_pos[rand_6D_idx,2]


env.set_goals(final_obj_pos_random,final_ee_random ,final_pose_robot ,final_contact_pos,final_contacts_robot)

env.reset_state(qpos_reset_robot, np.zeros((num_envs,22),'float32'), obj_pose_reset)


print(GREEN + "Finish to Initialize the environment" + END)

### Evaluate trained model visually (note always the first environment gets visualized)


last_idx = -1
cc = 0

total_num = 0
success_num = 0
reward = -20
success_rate = 0
i=0

hlen = tobeEncode_dim * t_steps
tail_size_impulse_begin = 39
tail_size_contact_end = 13

if args.vis_evaluate:
    while True:
        if all_obj_train:
            i = success_idxs[np.random.randint(success_idxs.shape[0])]
        else:
            if i == final_qpos.shape[0]:
                i = 0
        
        if args.debug_single_vis != -1:
            i = args.debug_single_vis
    
        if args.rand_6D_idx != 0:
            rand_6D_idx = args.rand_6D_idx
        else:
            rand_6D_idx = np.random.randint(obj_target_pos.shape[0])
            
        ### Set labels and load objects for current episode and target 6D pose
        qpos_reset_seq = qpos_reset_robot.copy()
        qpos_reset_seq[0] = qpos_reset_seq[i]
        final_qpos_seq = final_qpos.copy()
        final_qpos_seq[0] = final_qpos_seq[i].copy()

        obj_pose_reset_seq = obj_pose_reset.copy()
        obj_pose_reset_seq[0] = obj_pose_reset_seq[i].copy()

        final_contact_pos_seq = final_contact_pos.copy()
        final_contact_pos_seq[0] = final_contact_pos[i].copy()

        final_contacts_seq = final_contacts_robot.copy()
        final_contacts_seq[0] = final_contacts_robot[i].copy()

        final_obj_pos_seq = final_obj_pos.copy()
        final_ee_seq = final_ee_robot.copy()
        final_pose_seq = final_pose_robot.copy()
        final_ee_rel_seq = final_ee_rel_robot.copy()

        final_obj_pos_seq[0] = final_obj_pos[i].copy()
        final_ee_seq[0] = final_ee_seq[i].copy()
        final_ee_rel_seq[0] = final_ee_rel_seq[i].copy()
        final_pose_seq[0] = final_pose_seq[i].copy()

        obj_idx_stacked[0] =  obj_idx_stacked[i].copy()
        obj_w_stacked[0] =  obj_w_stacked[i].copy()
        obj_dim_stacked[0] =  obj_dim_stacked[i].copy()
        obj_type_stacked[0] =  obj_type_stacked[i].copy()



        ### Add position displacement to object position
        final_obj_pos_random = final_obj_pos_seq.copy()
        final_obj_pos_random[:,:3] += obj_target_pos[rand_6D_idx]

        ### Add angle displacement to object pose
        ang_noise =  np.repeat(obj_target_ang_noise[rand_6D_idx].reshape(1,3),final_obj_pos_random.shape[0],0)
        perturbed_obj_pose, rotmats, eulers_palm_new = utils.euler_noise_to_quat(final_obj_pos_random[:,3:].copy(), final_pose_seq[:,:3].copy(), ang_noise)

        final_obj_pos_random[:,3:] = perturbed_obj_pose
        final_pose_perturbed = final_pose_seq.copy()
        final_pose_perturbed[:,:3] = eulers_palm_new

        rotmats_neutral = np.tile(np.eye(3),(num_envs,1,1))

        ### Convert position goal features to the new, distorted object pose
        final_ee_random_or = np.squeeze(np.matmul(np.expand_dims(rotmats,1),np.expand_dims(final_ee_rel_seq.copy().reshape(num_envs,-1,3),-1)),-1)
        final_ee_random_or += np.expand_dims(final_obj_pos_random[:,:3],1)

        ### Reload env if new object
        if last_idx==-1 or obj_idx_stacked[last_idx]!=obj_idx_stacked[i]:
            env.load_object(obj_idx_stacked,obj_w_stacked, obj_dim_stacked, obj_type_stacked)

        last_idx = i
        env.set_goals(final_obj_pos_random,final_ee_random_or.reshape(num_envs,-1).astype('float32'),final_pose_perturbed,final_contact_pos_seq,final_contacts_seq)
        
        final_ee_random_or_rel = final_ee_random_or.copy()
        final_ee_random_or_rel = final_ee_random_or_rel.reshape((obj_pose_reset_seq.shape[0]),obj_pose_reset_seq.shape[0],17,-1).astype('float32')
        for j in range(obj_pose_reset_seq.shape[0]):
            final_ee_random_or_rel_single = final_ee_random_or_rel[i,i,0,:]
            if j==0:
                final_ee_random_or_rel_all = final_ee_random_or_rel_single.copy()
            else:
                final_ee_random_or_rel_all = np.vstack((final_ee_random_or_rel_all,final_ee_random_or_rel_single))

        qpos_reset_robot_new = utils.final_ee2Reset_ee(final_ee_random_or_rel_all,obj_pose_reset_seq,final_pose_perturbed,final_obj_pos_random, finger_bending=0.75, resize=1.0)
        
        env.reset_state(qpos_reset_seq, np.zeros((num_envs,22),'float32'), obj_pose_reset_seq)

        time.sleep(2)

        set_guide=False
        obj_pose_pos_list = []
        hand_pose_list = []
        joint_pos_list = []

        ### Start recording
        if args.store_video:
            env.turn_on_visualization()
            env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "_"+task_name+'.mp4')
            print(GREEN + "Start to record the video" + END)

        total_num += 1

        mse_all = []
        Impulse_mse_all = []
        Contact_mse_all = []
        
        if args.record_aitviewer:
            trans_r = np.zeros((n_steps, 3))
            rot_r = np.zeros((n_steps, 3))
            pose_r = np.zeros((n_steps, 22))
            trans_obj = np.zeros((n_steps, 3))
            rot_obj = np.zeros((n_steps, 3))
            trans_r_dim = 15
            rot_r_dim = 12
            rot_obj_dim = 8
            trans_obj_dim = 4
            
            
        for step in range(n_steps):
            obs = env.observe(False)
            
            if args.record_aitviewer:
                trans_r[step,:] = obs[0,-trans_r_dim:(-trans_r_dim+3)]
                axis,angle = rotations.quat2axisangle(obs[0,-rot_r_dim:(-rot_r_dim+4)].reshape(1,4))
                rot_r[step,:] = axis * angle
                pose_r[step,:] = obs[0,hlen:(hlen+22)]
                trans_obj[step,:] = obs[0,-trans_obj_dim:-trans_obj_dim+3]
                axis,angle = rotations.quat2axisangle(obs[0,-rot_obj_dim:(-rot_obj_dim+4)].reshape(1,4))
                rot_obj[step,:] = axis * angle
                
                np.save(IDX_TO_OBJ.get(args.obj_id)[0]+str(args.debug_single_vis)+".npy",data)

            ### Get action from policy
            raw_obs = obs[:, : hlen]
            with torch.no_grad():
                prop_latent = prop_loaded_encoder(torch.from_numpy(raw_obs).cpu())
                
                if args.GT_info:
                    output = torch.from_numpy(obs[:, hlen + (tobeEncode_dim) : -4]).cpu()
                else:
                    output = torch.cat([torch.from_numpy(obs[:, hlen + (tobeEncode_dim) : -tail_size_impulse_begin]).cpu(), prop_latent], 1)
                    output = torch.cat([output, torch.from_numpy(obs[:, -tail_size_contact_end : -4]).cpu()], 1)
                if args.evaluate_encoder:
                    GT_tactile = obs[:,-tail_size_impulse_begin:-tail_size_contact_end]
                    mse_per_example_per_step = np.mean((GT_tactile-prop_latent.detach().cpu().numpy())**2,axis=1)
                    mse_per_step = np.mean(mse_per_example_per_step)
                    mse_all.append(mse_per_step)
                    
                    Impulse_mse_per_example_per_step = np.mean((GT_tactile[:,:13]-prop_latent.detach().cpu().numpy()[:,:13])**2,axis=1)
                    Impulse_mse_per_step = np.mean(Impulse_mse_per_example_per_step)
                    Impulse_mse_all.append(Impulse_mse_per_step)
                    
                    Contact_mse_per_example_per_step = np.mean((GT_tactile[:,13:]-prop_latent.detach().cpu().numpy()[:,13:])**2,axis=1)
                    Contact_mse_per_step = np.mean(Contact_mse_per_example_per_step)
                    Contact_mse_all.append(Contact_mse_per_step)
                    
                    
                
                action_ll = actor_expert.architecture.architecture(output)
            
            frame_start = time.time()

            action_ll = action_ll.cpu().detach().numpy()

            ### After grasp is established (set to motion synthesis mode)
            if step>grasp_steps:
                if not set_guide:
                    env.set_root_control()
                    set_guide=True


            reward_ll, dones = env.step(action_ll.astype('float32'))

            reward = reward_ll
            
            ### early exit if object not picked up successfully

            frame_end = time.time()
            wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
            
            if wait_time > 0.:
                time.sleep(wait_time)
        
        if args.record_aitviewer:
                data = {}
                data['right_hand'] = {}
                data[IDX_TO_OBJ.get(args.obj_id)[0]] = {}
                data['right_hand']['trans_r'] = np.float32(trans_r[:])
                data['right_hand']['rot_r'] = np.float32(rot_r[:])
                data['right_hand']['pose_r'] = np.float32(pose_r[:])
                data[IDX_TO_OBJ.get(args.obj_id)[0]]['trans'] = np.float32(trans_obj[:])
                data[IDX_TO_OBJ.get(args.obj_id)[0]]['rot'] = np.float32(rot_obj[:])
                
                np.save(IDX_TO_OBJ.get(args.obj_id)[0]+str(args.debug_single_vis)+".npy",data)
        
        if args.evaluate_encoder:
            print("mse_all = ", mse_all)
            print("Impulse_mse_all = ", Impulse_mse_all)
            print("Contact_mse_all = ", Contact_mse_all)
            print("np.mean(mse_all) = ", np.mean(mse_all))
            print("np.mean(Impulse_mse_all) = ", np.mean(Impulse_mse_all))
            print("np.mean(Contact_mse_all) = ", np.mean(Contact_mse_all))
        
        print(GREEN + "--------------------------------" + END)
        print("Reward in one episode = ", reward)
        if reward > 0:
            success_num += 1

        if obs[0,-1] == 1:
            success_num += 0
        
        success_rate = success_num / total_num
        print("success_rate = ", success_rate)
        print(GREEN + "--------------------------------" + END)

        cc += 1

        ### Store recording
        if args.store_video:
            env.stop_video_recording()
            env.turn_off_visualization()
            print('stored video')

        if not all_obj_train:
            i += 1

else:
    disp_list, slipped_list, contact_ratio_list = [], [], []
    qpos_list, joint_pos_list, obj_pose_list = [], [], []

    set_guide=False

    mse_all = []
    Impulse_mse_all = []
    Contact_mse_all = []
    print()
    print(GREEN + "processing ..." + END)
    for step in range(n_steps):
        obs = env.observe(False)

        raw_obs = obs[:, : hlen]
        with torch.no_grad():
            prop_latent = prop_loaded_encoder(torch.from_numpy(raw_obs)).to("cpu")    # .to("cuda:2")
            
            if args.evaluate_encoder:
                GT_tactile = obs[:,-tail_size_impulse_begin:-tail_size_contact_end]
                mse_per_example_per_step = np.mean((GT_tactile-prop_latent.detach().cpu().numpy())**2,axis=1)
                mse_per_step = np.mean(mse_per_example_per_step)
                mse_all.append(mse_per_step)
                
                Impulse_mse_per_example_per_step = np.mean((GT_tactile[:,:13]-prop_latent.detach().cpu().numpy()[:,:13])**2,axis=1)
                Impulse_mse_per_step = np.mean(Impulse_mse_per_example_per_step)
                Impulse_mse_all.append(Impulse_mse_per_step)
                
                Contact_mse_per_example_per_step = np.mean((GT_tactile[:,13:]-prop_latent.detach().cpu().numpy()[:,13:])**2,axis=1)
                Contact_mse_per_step = np.mean(Contact_mse_per_example_per_step)
                Contact_mse_all.append(Contact_mse_per_step)
            
            if args.GT_info:
                output = torch.from_numpy(obs[:, hlen + (tobeEncode_dim) : -4]).cpu()
            else:
                output = torch.cat([torch.from_numpy(obs[:, hlen + (tobeEncode_dim) : -tail_size_impulse_begin]).cpu(), prop_latent], 1)
                output = torch.cat([output, torch.from_numpy(obs[:, -tail_size_contact_end : -4]).cpu()], 1)
            action_ll = actor_expert.architecture.architecture(output)

        frame_start = time.time()

        ### After grasp is established remove surface and test stability
        if step>grasp_steps and not set_guide:
            obj_pos_fixed = obs[:,-4:-1].copy()
            env.set_root_control()
            set_guide=True
        
        if step>(grasp_steps+1):
            obj_disp = np.linalg.norm(obj_pos_fixed-obs[:,-4:-1],axis=-1)
            disp_list.append(obj_disp)
            obj_pos_fixed = obs[:,-4:-1].copy()

        ### Record slipping and displacement
        if step>(grasp_steps+80):   # grasp_steps+1     grasp_steps+80
            slipped_list.append(obs[:,-1].copy())

        reward_ll, dones = env.step(action_ll.cpu().detach().numpy())

        frame_end = time.time()
        wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
        if wait_time > 0.:
            time.sleep(wait_time)

    if args.evaluate_encoder:
        print("mse_all = ", mse_all)
        print("Impulse_mse_all = ", Impulse_mse_all)
        print("Contact_mse_all = ", Contact_mse_all)
        print("np.mean(mse_all) = ", np.mean(mse_all))
        print("np.mean(Impulse_mse_all) = ", np.mean(Impulse_mse_all))
        print("np.mean(Contact_mse_all) = ", np.mean(Contact_mse_all))
    
    ### Log quantiative results
    for obj_id in np.unique(obj_idx_stacked):
        train_obj_id = obj_id + 1

        ### compute testing window
        sim_dt = cfg['environment']['simulation_dt']
        control_dt = cfg['environment']['control_dt']
        control_steps = int(control_dt / sim_dt)
        sim_to_real_steps = 1/(control_steps * sim_dt)
        window_5s = int(5*sim_to_real_steps)

        obj_idx_array = np.where(obj_idx_stacked == obj_id)[0]

        slipped_array = np.array(slipped_list)[:].transpose()[obj_idx_array]
        disp_array = np.array(disp_list)[:].transpose()[obj_idx_array]

        slips, success_idx, disps = [], [], []
        ### evaluate slipping and sim dist
        for idx in range(slipped_array.shape[0]):
            if slipped_array[idx,:window_5s].any():
                slips.append(True)
            else:
                success_idx.append(idx)

            if slipped_array[idx].any():
                slip_step_5s =  np.clip(np.where(slipped_array[idx])[0][0]-1,1, window_5s)
                disps.append(disp_array[idx,:slip_step_5s].copy().mean())
            else:
                disps.append(disp_array[idx,:window_5s].copy().mean())

        avg_slip = 1-np.array(slips).sum()/slipped_array.shape[0]
        avg_disp =  np.array(disps).mean()*1000
        std_disp =  np.array(disps).std()*1000

        print('----------------------------------------------------')
        print('{:<40} {:>6}'.format("object: ", obj_id+1))
        print('{:<40} {:>6}'.format("success: ", '{:0.3f}'.format(avg_slip)))
        print('{:<40} {:>6}'.format("success num: ", '{}/{}'.format(slipped_array.shape[0]-np.array(slips).sum(),slipped_array.shape[0])))
        print('{:<40} {:>6}'.format("disp mean: ", '{:0.3f}'.format(avg_disp)))
        print('{:<40} {:>6}'.format("disp std: ", '{:0.3f}'.format(std_disp)))
        print('----------------------------------------------------\n')

        if not all_obj_train:
            np.save(log_dir+'/success_idxs',success_idx)


    ### Log average success rate over all objects
    if all_obj_train:
        slipped_array = np.array(slipped_list)[:].transpose()
        disp_array = np.array(disp_list)[:].transpose()

        slips, success_idx, disps = [], [], []
        ### evaluate slipping and sim dist
        for idx in range(slipped_array.shape[0]):
            if slipped_array[idx,:window_5s].any():
                slips.append(True)
            else:
                success_idx.append(idx)

            if slipped_array[idx].any():
                slip_step_5s =  np.clip(np.where(slipped_array[idx])[0][0]-1,1, window_5s)
                disps.append(disp_array[idx,:slip_step_5s].copy().mean())
            else:
                disps.append(disp_array[idx,:window_5s].copy().mean())

        avg_slip = 1-np.array(slips).sum()/slipped_array.shape[0]
        avg_disp =  np.array(disps).mean()*1000
        std_disp =  np.array(disps).std()*1000

        if len(success_idx) > 0:
            np.save(log_dir+'/success_idxs',success_idx)

        print('----------------------------------------------------')
        print('{:<40}'.format("all objects"))
        print('{:<40} {:>6}'.format("total success rate: ", '{:0.3f}'.format(avg_slip)))
        print('{:<40} {:>6}'.format("success num: ", '{}/{}'.format(slipped_array.shape[0]-np.array(slips).sum(),slipped_array.shape[0])))
        print('{:<40} {:>6}'.format("disp mean: ", '{:0.3f}'.format(avg_disp)))
        print('{:<40} {:>6}'.format("disp std: ", '{:0.3f}'.format(std_disp)))
        print('----------------------------------------------------\n')
