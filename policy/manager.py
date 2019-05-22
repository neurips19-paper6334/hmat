import torch
import numpy as np
from policy.policy_base import PolicyBase
from misc.replay_buffer import ReplayBuffer
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Manager(PolicyBase):
    def __init__(self, env, tb_writer, log, args, name, i_agent):
        super(Manager, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args,
            name=name, i_agent=i_agent)

        self.set_dim()
        self.set_policy()
        self.memory = ReplayBuffer()

        assert "manager" in self.name

    def set_dim(self):
        self.actor_input_dim = self.env.observation_space[0].shape[0]
        if self.args.manager_done:
            self.actor_input_dim += 1  # +1 for remaining time in current episode
        self.actor_output_dim = self.env.action_space[0].shape[0] 
        self.critic_input_dim = (self.actor_input_dim + self.actor_output_dim) * self.args.n_manager
        self.max_action = float(self.env.action_space[0].high[0])
        self.n_hidden = self.args.manager_n_hidden

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))
        self.log[self.args.log_name].info("[{}] Max action: {}".format(
            self.name, self.max_action))

    def add_memory(self, obs, new_obs, action, reward, done):
        self.memory.add((obs, new_obs, action, reward, done))

    def select_stochastic_action(self, obs, session_timesteps):
        """Return stochastic action with added noise
        As in TD3, purely random noise is applied followed by Gaussian noise
        Empirically, we found that adding the purely random noise improves 
        stability of the algorithm
        """
        if session_timesteps < self.args.manager_start_timesteps:
            action = self.env.action_space[0].sample()
            assert not np.isnan(action).any()
        else:
            action = self.policy.select_action(obs)
            assert not np.isnan(action).any()
            if self.args.expl_noise != 0:
                noise = np.random.normal(0, self.args.expl_noise, size=self.env.action_space[0].shape[0])
                action = (action + noise).clip(
                    self.env.action_space[0].low, self.env.action_space[0].high)

        return action

    def update_policy(self, agents, iterations, batch_size, total_timesteps):
        debug = self.policy.centralized_train(
            agents=agents,
            replay_buffer=self.memory,
            iterations=iterations,
            batch_size=batch_size, 
            discount=self.args.manager_discount, 
            tau=self.args.tau, 
            policy_noise=self.args.policy_noise, 
            noise_clip=self.args.noise_clip, 
            policy_freq=self.args.policy_freq)

        self.tb_writer.add_scalars(
            "loss/actor", {self.name: debug["actor_loss"]}, total_timesteps)
        self.tb_writer.add_scalars(
            "loss/critic", {self.name: debug["critic_loss"]}, total_timesteps)

        return debug

    def fix_name(self, weight):
        weight_fixed = OrderedDict()
        for k, v in weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            weight_fixed[name_fixed] = v

        return weight_fixed

    def sync(self, target_agent):
        actor = self.fix_name(target_agent.policy.actor.state_dict())
        self.policy.actor.load_state_dict(actor)

        actor_target = self.fix_name(target_agent.policy.actor_target.state_dict())
        self.policy.actor_target.load_state_dict(actor_target)

        critic = self.fix_name(target_agent.policy.critic.state_dict())
        self.policy.critic.load_state_dict(critic)

        critic_target = self.fix_name(target_agent.policy.critic_target.state_dict())
        self.policy.critic_target.load_state_dict(critic_target)
