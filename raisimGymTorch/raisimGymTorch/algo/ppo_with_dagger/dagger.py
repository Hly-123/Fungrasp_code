import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# “.” means the module storage is in the same directory as the importing script
from .storage import ObsStorage 


# computes and returns the latent from the expert 【RETURN：Encoder output】
class DaggerExpert(nn.Module):
    def __init__(self, loadpth, runid, total_obs_size, T, base_obs_size, priv_dim, tobeEncode_dim, nenvs,actor_expert, geomDim = 4, n_futures = 3):
        super(DaggerExpert, self).__init__()
        path = '/'.join([loadpth, 'full_' + runid + '.pt'])
        self.policy = torch.load(path,map_location=torch.device('cuda:0'))
        self.actor_expert = actor_expert

        self.actor_expert.architecture.load_state_dict(self.policy['actor_architecture_state_dict'])

        self.geomDim = geomDim
        self.n_futures = n_futures
        mean_pth = loadpth + "/mean" + runid + ".csv"
        var_pth = loadpth + "/var" + runid + ".csv"
        obs_mean = np.loadtxt(mean_pth, dtype=np.float32)
        obs_var = np.loadtxt(var_pth, dtype=np.float32)
        self.mean = self.get_tiled_scales(obs_mean, nenvs, total_obs_size, tobeEncode_dim, T)
        self.var = self.get_tiled_scales(obs_var, nenvs, total_obs_size, tobeEncode_dim, T)
        self.tail_size_impulse_begin = 39
        self.tail_size_contact_end = 13

        prop_latent_dim = 13
        self.prop_latent_dim = prop_latent_dim

    def get_tiled_scales(self, invec, nenvs, total_obs_size, base_obs_size, T):
        outvec = np.zeros([nenvs, total_obs_size], dtype = np.float32)
        outvec[:, :base_obs_size * (T+1)] = np.tile(invec[0, :base_obs_size], [1, (T+1)])
        outvec[:, base_obs_size * (T+1):] = invec[0]
        return outvec

    def forward(self, obs):
        expert_latent = obs[:,-self.tail_size_impulse_begin:-self.tail_size_contact_end]
        return expert_latent

class DaggerAgent:
    def __init__(self, expert_policy,
                 prop_latent_encoder,
                 T, base_obs_size, tobeEncode_dim, hand_dim, device, n_futures=3):
        expert_policy.to(device)
        prop_latent_encoder.to(device)
        self.expert_policy = expert_policy
        self.prop_latent_encoder = prop_latent_encoder
        self.base_obs_size = base_obs_size
        self.tobeEncode_dim = tobeEncode_dim
        self.hand_dim = hand_dim

        self.T = T
        self.device = device
        self.mean = expert_policy.mean
        self.var = expert_policy.var
        self.n_futures = n_futures
        self.itr = 0
        self.current_prob = 0
        self.student_mlp = self.expert_policy.actor_expert.architecture.architecture


        for net_i in self.student_mlp:
            for param in net_i.parameters():
                param.requires_grad = False

    def set_itr(self, itr):
        self.itr = itr
        if (itr+1) % 100 == 0:
            self.current_prob += 0.1
            print(f"Probability set to {self.current_prob}")

    def get_history_encoding(self, obs):
        hlen = self.tobeEncode_dim * self.T
        raw_obs = obs[:, : hlen]
        prop_latent = self.prop_latent_encoder(raw_obs)
        return prop_latent

    def evaluate(self, obs):
        hlen = self.tobeEncode_dim * self.T
        obdim = self.base_obs_size
        prop_latent = self.get_history_encoding(obs)

        output = torch.cat([obs[:, hlen + (self.tobeEncode_dim) : -39], prop_latent], 1)
        output = torch.cat([output, obs[:, -13 : -4]], 1)
        output = self.student_mlp(output)
        return output

    def get_expert_action(self, obs):
        hlen = self.tobeEncode_dim * self.T
        obdim = self.base_obs_size
        output = obs[:, hlen + (self.tobeEncode_dim) : -4]
        output = self.student_mlp(output)
        return output

    def get_student_action(self, obs):
        return self.evaluate(obs)

    def get_expert_latent(self, obs):
        with torch.no_grad():
            latent = self.expert_policy(obs).detach()
            return latent

    def save_deterministic_graph(self, fname_prop_encoder,
                                 fname_mlp, example_input, device='cpu'):
        hlen = self.tobeEncode_dim * self.T

        prop_encoder_graph = torch.jit.script(self.prop_latent_encoder.to(device))
        torch.jit.save(prop_encoder_graph, fname_prop_encoder)

        self.prop_latent_encoder.to(self.device)
        self.student_mlp.to(self.device)

class DaggerTrainer:
    def __init__(self,
            actor,
            num_envs, 
            num_transitions_per_env,
            obs_shape, latent_shape,
            num_learning_epochs=4,
            num_mini_batches=4,
            device=None,
            learning_rate=5e-4):

        self.actor = actor
        self.storage = ObsStorage(num_envs, num_transitions_per_env, [obs_shape], [latent_shape], device)
        self.optimizer = optim.Adam([*self.actor.prop_latent_encoder.parameters()],
                                    lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.1)
        self.device = device
        self.itr = 0

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.loss_fn = nn.MSELoss()

    def observe_student(self, obs):
        with torch.no_grad():
            actions = self.actor.get_student_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def observe_teacher(self, obs):
        with torch.no_grad():
            actions = self.actor.get_expert_action(torch.from_numpy(obs).to(self.device))
        return actions.detach().cpu().numpy()

    def step(self, obs):
        expert_latent = self.actor.get_expert_latent(torch.from_numpy(obs).to(self.device))
        self.storage.add_obs(obs, expert_latent)

    def update(self):
        # Learning step
        mse_loss = self._train_step()
        self.storage.clear()
        return mse_loss

    def _train_step(self):
        self.itr += 1
        self.actor.set_itr(self.itr)
        for epoch in range(self.num_learning_epochs):
            # return loss in the last epoch
            prop_mse = 0
            loss_counter = 0
            for obs_batch, expert_action_batch in self.storage.mini_batch_generator_inorder(self.num_mini_batches):

                predicted_prop_latent = self.actor.get_history_encoding(obs_batch)

                loss_prop = self.loss_fn(predicted_prop_latent, expert_action_batch)

                loss = loss_prop

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                prop_mse += loss_prop.item()
                loss_counter += 1

            avg_prop_loss = prop_mse / loss_counter

        self.scheduler.step()
        return avg_prop_loss