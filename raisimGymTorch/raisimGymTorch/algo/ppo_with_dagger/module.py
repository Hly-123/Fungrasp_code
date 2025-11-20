import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal

# ly modified
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
END = '\033[0m'

class Expert:
    def __init__(self, policy, device='cpu', baseDim=46):
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.baseDim = baseDim 
        # 19 is size of priv info for the cvpr policy
        self.end_idx = baseDim + 19 #-self.geomDim*(self.n_futures+1) - 1

    def evaluate(self, obs):
        with torch.no_grad():
            resized_obs = obs[:, :self.end_idx]
            latent = self.policy.info_encoder(resized_obs[:,self.baseDim:])
            input_t = torch.cat((resized_obs[:,:self.baseDim], latent), dim=1)
            action = self.policy.action_mlp(input_t)
            return action

class Steps_Expert:
    def __init__(self, policy, device='cpu', baseDim=42, geomDim=2, n_futures=1, num_g1=8):
        self.policy = policy
        self.policy.to(device)
        self.device = device
        self.baseDim = baseDim 
        self.geom_dim = geomDim
        self.n_futures = n_futures
        # 23 is size of priv info for the cvpr policy
        self.privy_dim = 23
        self.end_prop_idx = baseDim + self.privy_dim
        self.g1_position = 0
        self.g1_slice = slice(self.end_prop_idx + self.g1_position * geomDim,
                              self.end_prop_idx + (self.g1_position+1) * geomDim)
        self.g2_slice = slice(-geomDim-1,-1)

    def evaluate(self, obs):
        with torch.no_grad():
            action = self.forward(obs)
            return action

    def forward(self, x):
        # get only the x related to the control policy
        x = x[:,:x.shape[1]//2]
        prop_latent = self.policy.prop_encoder(x[:,self.baseDim:self.end_prop_idx])
        geom_latents = []
        g1 = self.policy.geom_encoder(x[:,self.g1_slice])
        g2 = self.policy.geom_encoder(x[:,self.g2_slice])
        geom_latents = torch.hstack([g1,g2])
        return self.policy.action_mlp(torch.cat([x[:,:self.baseDim], prop_latent, geom_latents], 1))

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        # self.architecture.to(device)
        try:
            self.architecture.to(device)
        except:
            print("If you're not in ARMA mode you have a problem")
        self.distribution.to(device)
        self.device = device
        self.action_mean = None

    def sample(self, obs):
        self.action_mean = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(self.action_mean)
        self.action_mean = self.action_mean.cpu().numpy()
        return actions, log_prob

    def evaluate(self, obs, actions):
        self.action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, self.action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input, check_trace=False)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        # print("!!!obs.shape in crtic predict = ",obs.shape)
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Action_MLP_Encode(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = 45, geom_dim = 0,
                 n_futures = 3):
        super(Action_MLP_Encode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        regular_obs_dim = base_obdim
        self.regular_obs_dim = regular_obs_dim
        self.geom_dim = geom_dim

        self.geom_latent_dim = 1
        self.prop_latent_dim = 8
        self.n_futures = n_futures
        
        # creating the action encoder
        modules = [nn.Linear(regular_obs_dim + self.prop_latent_dim + (self.n_futures + 1)*self.geom_latent_dim, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.action_mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.action_mlp, scale)

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self, x):
        # get only the x related to the control policy
        geom_latent = x[:,-self.geom_dim:]
        prop_latent = x[:,-(self.geom_dim+self.prop_latent_dim):-self.geom_dim]
        return self.action_mlp(torch.cat([x[:,:self.regular_obs_dim], prop_latent, geom_latent], 1))

class Action_MLP_Encode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, base_obdim = 45, geom_dim = 0, n_futures = 3):
        super(Action_MLP_Encode_wrap, self).__init__()
        self.architecture = Action_MLP_Encode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, base_obdim, geom_dim, n_futures)
        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

class MLPEncode(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = 45, geom_dim = 0,
                 n_futures = 3):
        super(MLPEncode, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        regular_obs_dim = base_obdim
        self.regular_obs_dim = regular_obs_dim
        # self.geom_dim = geom_dim
        
        ## Encoder Architecture
        prop_latent_dim = 13    # origin: 8
        # geom_latent_dim = 3     # origin: 1
        self.n_futures = n_futures
        self.prop_latent_dim = prop_latent_dim
        self.prop_encoder =  nn.Sequential(*[
                                    nn.Linear(input_size - regular_obs_dim -4, 16), self.activation_fn(),
                                    nn.Linear(16, 64), self.activation_fn(),
                                    # nn.Linear(256, 128, 128), self.activation_fn(),  # ly encode yuan
                                    nn.Linear(64, prop_latent_dim), self.activation_fn(),
                                    ]) 

        scale_encoder = [np.sqrt(2), np.sqrt(2), np.sqrt(2)]

        # creating the action encoder
        modules = [nn.Linear(regular_obs_dim + prop_latent_dim, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
            print("Add output_activation_fn!!!")

        self.action_mlp = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.action_mlp, scale)
        self.init_weights(self.prop_encoder, scale_encoder)
        # self.init_weights(self.geom_encoder, scale_encoder)
        if small_init: action_output_layer.weight.data *= 1e-6

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

        #for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear)):
        #    module.weight.data *= 1e-6

    def forward(self, x):
        prop_latent = self.prop_encoder(x[:,self.regular_obs_dim:-4])

        return self.action_mlp(torch.cat([x[:,:self.regular_obs_dim], prop_latent], 1))

class MLPEncode_wrap(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None,
                 small_init= False, base_obdim = 45, geom_dim = 0, n_futures = 3):
        super(MLPEncode_wrap, self).__init__()
        self.architecture = MLPEncode(shape, actionvation_fn, input_size, output_size, output_activation_fn, small_init, base_obdim, geom_dim, n_futures)
        self.input_shape = self.architecture.input_shape
        self.output_shape = self.architecture.output_shape

# ly TODO:重点关注forward函数里的变换过程：从obs到projection到output
class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_shape = input_size*tsteps
        self.output_shape = output_size

        if tsteps == 50:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 8, stride = 4), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(),
                    nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 5, stride = 1), nn.LeakyReLU(), nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 10:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 1), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
            nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        elif tsteps == 20:
            self.encoder = nn.Sequential(
            nn.Linear(input_size, 32), self.activation_fn()
            )
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 6, stride = 2), nn.LeakyReLU(), 
                nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 4, stride = 2), nn.LeakyReLU(), 
                nn.Flatten())
            self.linear_output = nn.Sequential(
                nn.Linear(32 * 3, output_size), self.activation_fn()
            )
        else:
            raise NotImplementedError()
    def forward(self, obs):
        bs = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([bs * T, -1]))
        output = self.conv_layers(projection.reshape([bs, -1, T]))
        output = self.linear_output(output)
        return output

# LSTM-based StateHistoryEncoder
class LSTM_StateHistoryEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, tsteps):
        super(LSTM_StateHistoryEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1 # 单向LSTM
        self.tsteps = tsteps
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, obs):
        # input_seq.shape = (batch_size, seq_len, input_size)
        batch_size = obs.shape[0]
        T = self.tsteps
        obs = obs.reshape([batch_size, T, -1])
        device = 'cpu' # when training keep same with runner, cuda:2

        h_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, _ = self.lstm(obs, (h_0, c_0)) # output(batch_size, 10, num_directions * hidden_size)
        pred = self.linear(output)  # (batch_size, 10, output_size)
        pred = pred[:, -1, :]  # (batch_size, output_size)
        # print("pred.shape = ", pred.shape)
        return pred


# ly replace: 将原本raisim自带的ppo中的MLP整合到这里
class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]    

class MLP_1(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, output_activation_fn = None, small_init= False, base_obdim = None):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn
        self.output_activation_fn = output_activation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        action_output_layer = modules[-1]
        if self.output_activation_fn is not None:
            modules.append(self.output_activation_fn())
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        if small_init: action_output_layer.weight.data *= 1e-6

        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

# ly replace: 将原本raisim自带的ppo中的MultivariateGaussianDiagonalCovariance整合到这里
class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, size, init_std, fast_sampler, seed=0):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.fast_sampler = fast_sampler
        self.fast_sampler.seed(seed)
        self.samples = np.zeros([size, dim], dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.std_np = self.std.detach().cpu().numpy()

    def update(self):
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.fast_sampler.sample(logits, self.std_np, self.samples, self.logprob)
        return self.samples.copy(), self.logprob.copy()

    def evaluate(self, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)
        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

class MultivariateGaussianDiagonalCovariance_1(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        self.std = nn.Parameter(init_std * torch.ones(dim))
        self.distribution = None
        self.std_np = self.std.detach().cpu().numpy()

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std.data = new_std

class MultivariateGaussianDiagonalCovariance2(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance2, self).__init__()
        assert(dim == 12)
        self.dim = dim
        self.std_param = nn.Parameter(init_std * torch.ones(dim // 2))
        self.distribution = None

    def sample(self, logits):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        self.std = torch.cat([self.std_param[:3], self.std_param[:3], self.std_param[3:], self.std_param[3:]], dim=0)
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std_param.detach()
        new_std = torch.max(current_std, min_std.detach()).detach()
        self.std_param.data = new_std
