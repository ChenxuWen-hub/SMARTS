"""
reference: @author: wangmeng
https://github.com/mengwanglalala/RL-algorithms/blob/main/Continuous_action/PPO.py
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        # del语句作用在变量上，而不是数据对象上。删除的是变量，而不是数据。
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # 方差
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        # 手动设置异常
        raise NotImplementedError

    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        # torch.diag_embed(input, offset=0, dim1=-2, dim2=-1) → Tensor
        # Creates a tensor whose diagonals of certain 2D planes (specified by dim1 and dim2) are filled by input
        cov_mat = torch.diag_embed(action_var).to(device)
        # 生成一个多元高斯分布矩阵
        dist = MultivariateNormal(action_mean, cov_mat)
        # 我们的目的是要用这个随机的去逼近真正的选择动作action的高斯分布
        action_logprobs = dist.log_prob(action)
        # log_prob 是action在前面那个正太分布的概率的log ，我们相信action是对的 ，
        # 那么我们要求的正态分布曲线中点应该在action这里，所以最大化正太分布的概率的log， 改变mu,sigma得出一条中心点更加在a的正太分布。
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(
            self, state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip,
            # policy: Union[str, Type[ActorCriticPolicy]],
            # env: Union[GymEnv, str],
            # policy_kwargs: Optional[Dict[str, Any]] = None,
            # target_kl: Optional[float] = None,
            # tensorboard_log: Optional[str] = None,
            # verbose: int = 0,
            policy,
            env,
            policy_kwargs,
            target_kl,
            tensorboard_log,
            verbose: int = 0,
            render=False,
            solved_reward = 300,  # stop training if avg_reward > solved_reward
            log_interval = 20,  # print avg reward in the interval
            max_episodes = 10000,  # max training episodes
            max_timesteps = 1500,  # max timesteps in one episode

            update_timestep = 4000,  # update policy every n timesteps
            action_std = 0.5,  # constant std for action distribution (Multivariate Normal)
            K_epochs = 80,  # update policy for K epochs
            eps_clip = 0.2,  # clip parameter for PPO
            gamma = 0.99,  # discount factor

            lr = 0.0003,  # parameters for Adam optimizer
            betas = (0.9, 0.999),

            random_seed = None
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        policy,
        env,
        tensorboard_log = tensorboard_log,
        policy_kwargs = policy_kwargs,
        verbose = verbose,
        self.target_kl = target_kl

        render = False,
        self.solved_reward = solved_reward,  # stop training if avg_reward > solved_reward
        self.log_interval = log_interval,  # print avg reward in the interval
        self.max_episodes = max_episodes,  # max training episodes
        self.max_timesteps = max_timesteps,  # max timesteps in one episode

        self.update_timestep = update_timestep,  # update policy every n timesteps
        self.action_std = action_std,  # constant std for action distribution (Multivariate Normal)
        self.K_epochs = K_epochs,  # update policy for K epochs
        self.eps_clip = eps_clip,  # clip parameter for PPO
        self.gamma = gamma,  # discount factor

        self.lr =lr,  # parameters for Adam optimizer
        self.betas = betas,

        self.random_seed = random_seed



    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        # 使用stack可以保留两个信息：[1. 序列] 和 [2. 张量矩阵] 信息，属于【扩张再拼接】的函数；
        old_states = torch.squeeze(torch.stack(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def set_env(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

    # def save(
    #         self,
    #         # path: Union[str, pathlib.Path, io.BufferedIOBase],
    #         # exclude: Optional[Iterable[str]] = None,
    #         # include: Optional[Iterable[str]] = None,
    #         path,
    #         exclude,
    #         include
    # ) -> None:
    #     """
    #     Save all the attributes of the object and the model parameters in a zip-file.
    #
    #     :param path: path to the file where the rl agent should be saved
    #     :param exclude: name of parameters that should be excluded in addition to the default ones
    #     :param include: name of parameters that might be excluded but should be included anyway
    #     """
    #     # Copy parameter list so we don't mutate the original dict
    #     data = self.__dict__.copy()
    #
    #     # Exclude is union of specified parameters (if any) and standard exclusions
    #     if exclude is None:
    #         exclude = []
    #     exclude = set(exclude).union(self._excluded_save_params())
    #
    #     # Do not exclude params if they are specifically included
    #     if include is not None:
    #         exclude = exclude.difference(include)
    #
    #     state_dicts_names, torch_variable_names = self._get_torch_save_params()
    #     all_pytorch_variables = state_dicts_names + torch_variable_names
    #     for torch_var in all_pytorch_variables:
    #         # We need to get only the name of the top most module as we'll remove that
    #         var_name = torch_var.split(".")[0]
    #         # Any params that are in the save vars must not be saved by data
    #         exclude.add(var_name)
    #
    #     # Remove parameter entries of parameters which are to be excluded
    #     for param_name in exclude:
    #         data.pop(param_name, None)
    #
    #     # Build dict of torch variables
    #     pytorch_variables = None
    #     if torch_variable_names is not None:
    #         pytorch_variables = {}
    #         for name in torch_variable_names:
    #             attr = recursive_getattr(self, name)
    #             pytorch_variables[name] = attr
    #
    #     # Build dict of state_dicts
    #     params_to_save = self.get_parameters()
    #
    #     save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    def learn(
            self,
            total_timesteps,
            callback,
    ):

        memory = Memory()
        print(self.lr, self.betas)

        # logging variables
        running_reward = 0
        avg_length = 0
        time_step = 0

        # training loop
        for i_episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            for t in range(self.max_timesteps):
                time_step += 1
                # Running policy_old:
                action = self.select_action(state, memory)
                state, reward, done, _ = self.env.step(action)

                # Saving reward and is_terminals:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # update if its time
                if time_step % self.update_timestep == 0:
                    self.update(memory)
                    memory.clear_memory()
                    time_step = 0
                running_reward += reward
                if self.render:
                    self.env.render()
                if done:
                    break

            avg_length += t

            # stop training if avg_reward > solved_reward
            if running_reward > (self.log_interval * self.solved_reward):
                print("########## Solved! ##########")
                torch.save(self.policy.state_dict(), './PPO_continuous_solved_{}.pth'.format(self.env_name))
                break

            # save every 500 episodes
            if i_episode % 500 == 0:
                torch.save(self.policy.state_dict(), './PPO_continuous_{}.pth'.format(self.env_name))

            # logging
            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
                running_reward = 0
                avg_length = 0