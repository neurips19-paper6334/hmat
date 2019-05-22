import torch
from policy.policy_base import PolicyBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Worker(PolicyBase):
    def __init__(self, env, tb_writer, log, args, name, i_agent):
        super(Worker, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args,
            name=name, i_agent=i_agent)

        self.set_dim()
        self.set_policy()

        assert "worker" in self.name

    def set_dim(self):
        self.actor_input_dim = 2 + 2  # NOTE Current location + (relative) Goal location
        self.actor_output_dim = self.env.action_space[0].shape[0] 
        self.critic_input_dim = self.actor_input_dim + self.actor_output_dim
        self.max_action = float(self.env.action_space[0].high[0])
        self.n_hidden = self.args.worker_n_hidden

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Actor output dim: {}".format(
            self.name, self.actor_output_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))
        self.log[self.args.log_name].info("[{}] Max action: {}".format(
            self.name, self.max_action))
