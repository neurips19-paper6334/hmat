import numpy as np
import gym.spaces as spaces
from policy.policy_base import PolicyBase
from misc.replay_buffer import ReplayBuffer
from trainer.utils import get_q_values, get_avg_reward, concat_in_order
from misc.utils import normalize


class Teacher(PolicyBase):
    def __init__(self, env, tb_writer, log, args, name, i_agent):
        super(Teacher, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args,
            name=name, i_agent=i_agent)

        self.set_dim()
        self.set_policy()
        self.memory = ReplayBuffer()
        self.tmp_memory = []
        self.n_advice = args.session / float(1 + args.n_eval)

        assert "teacher" in self.name

    def set_actor_input_dim(self):
        input_dim = 0

        # Add state (teacher obs, student obs)
        input_dim += self.env.observation_space[0].shape[0] * 2
        if self.args.manager_done:
            input_dim += 1 * 2

        # Add action (teacher action, student action, teacher action at)
        input_dim += self.env.action_space[0].shape[0] * 3

        # Add Q-values (teacher joint Q-value, student joint Q-value)
        input_dim += 2 * 2

        # Add reward mean
        input_dim += 2

        # Add teacher remain time within session
        input_dim += 1  

        return input_dim

    def set_dim(self):
        self.actor_input_dim = self.set_actor_input_dim()
        self.actor_output_dim = self.env.action_space[0].shape[0] + 2  # +2 for when to advise (one-hot encoding)
        self.critic_input_dim = (self.actor_input_dim + self.actor_output_dim) * self.args.n_teacher
        self.max_action = float(self.env.action_space[0].high[0])
        self.n_hidden = self.args.teacher_n_hidden
        self.action_space = spaces.Box(low=-1, high=+1, shape=(self.env.action_space[0].shape[0],), dtype=np.float32)

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))
        self.log[self.args.log_name].info("[{}] Max action: {}".format(
            self.name, self.max_action))

    def select_stochastic_action(self, obs, total_timesteps):
        """Return stochastic action with added noise
        As in TD3, purely random noise is applied followed by Gaussian noise
        """
        if total_timesteps < self.args.teacher_start_timesteps:
            action_what = self.action_space.sample()
            action_when = np.zeros((2,), dtype=np.float32)
            action_when[np.random.randint(low=0, high=2, size=(1,))] = 1
        else:
            action_what, action_when = self.policy.select_action(obs)
            if self.args.expl_noise != 0:
                noise = np.random.normal(0, self.args.expl_noise, size=self.action_space.shape[0])
                action_what = (action_what + noise).clip(self.action_space.low, self.action_space.high)
                
                if np.random.uniform() < 0.03:
                    action_when = np.zeros((2,), dtype=np.float32)
                    action_when[np.random.randint(low=0, high=2, size=(1,))] = 1

        action = np.concatenate([action_what, action_when])
        assert not np.isnan(action).any()

        return action

    def update_memory(self, teacher_reward, temp_managers, train_rewards, teacher_rewards):
        """Update memory
        The next observation is updated by replacing student Q-values with its updated temporary policy.
        Average rewards and remaining timestep are also updated.
        The measured teacher_reward is also updated.
        """
        self.corrected_memory = [[] for _ in range(5)]  # 5: obs, new_obs, action, reward, done

        i_student = 1
        for i_exp, exp in enumerate(self.tmp_memory):
            # Update student_action
            obs_dict = exp[-1]

            # Update q-value that measured using updated student_critic
            q_values = get_q_values(
                temp_managers, obs_dict["manager_observations"],
                [obs_dict["manager_actions"][0], obs_dict["student_action"]])
            q_values = np.clip(q_values, a_min=self.args.q_min, a_max=self.args.q_max)

            obs_dict["q_with_student_critic"] = np.array([normalize(
                value=q_values[i_student], min_value=self.args.q_min, max_value=self.args.q_max)])

            q_values = get_q_values(
                temp_managers, obs_dict["manager_observations"],
                [obs_dict["manager_actions"][0], obs_dict["teacher_action_at"]])
            q_values = np.clip(q_values, a_min=self.args.q_min, a_max=self.args.q_max)

            obs_dict["q_at_with_student_critic"] = np.array([normalize(
                value=q_values[i_student], min_value=self.args.q_min, max_value=self.args.q_max)])

            # Update avg_reward
            # Note that avg_train_reward = R_{Phase I}
            # Note that avg_teacher_reward = R_{Phase II}
            avg_train_reward, avg_teacher_reward = get_avg_reward(
                train_rewards=train_rewards, teacher_rewards=teacher_rewards, args=self.args)
            obs_dict["avg_train_reward"] = np.array([avg_train_reward])
            obs_dict["avg_teacher_reward"] = np.array([avg_teacher_reward])

            # Update teacher remain timestep
            obs_dict["remain_time"] = np.array([normalize(
                value=(self.n_advice - (obs_dict["session_advices"] + 1)),
                min_value=0., max_value=float(self.n_advice))])

            new_obs = concat_in_order(obs_dict, self.args)
            self.corrected_memory[0].append(exp[0])
            self.corrected_memory[1].append(new_obs)
            self.corrected_memory[2].append(exp[2])
            self.corrected_memory[3].append(teacher_reward)
            self.corrected_memory[4].append(exp[4])

        self.add_memory()
        self.clear_tmp_memory()

    def add_memory(self):
        self.memory.add(self.corrected_memory)

    def keep_memory(self, obs, new_obs, action, reward, done, obs_dict):
        self.tmp_memory.append([obs, new_obs, action, reward, done, obs_dict])

    def clear_tmp_memory(self):
        self.tmp_memory.clear()

    def update_policy(self, batch_size, total_timesteps):
        if len(self.memory) > self.args.ep_max_timesteps:
            debug = self.policy.train_teacher(
                replay_buffer=self.memory,
                iterations=self.args.ep_max_timesteps,
                batch_size=batch_size, 
                discount=self.args.teacher_discount, 
                tau=self.args.tau, 
                policy_noise=self.args.policy_noise, 
                noise_clip=self.args.noise_clip, 
                policy_freq=self.args.policy_freq)

            self.log[self.args.log_name].info("[{0}] Teacher actor loss {1} at {2}".format(
                self.name, debug["actor_loss"], total_timesteps))
            self.tb_writer.add_scalars(
                "loss/actor", {self.name: debug["actor_loss"]}, total_timesteps)

            self.log[self.args.log_name].info("[{0}] Teacher critic loss {1} at {2}".format(
                self.name, debug["critic_loss"], total_timesteps))
            self.tb_writer.add_scalars(
                "loss/critic", {self.name: debug["critic_loss"]}, total_timesteps)
