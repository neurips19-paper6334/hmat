"""Modified Twin Delayed Deep Deterministic Policy Gradients (TD3)
TD3 Ref: https://github.com/sfujim/TD3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.utils import onehot_from_logits, gumbel_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, actor_input_dim, actor_output_dim, n_hidden, max_action, name):
        super(Actor, self).__init__()

        setattr(self, name + "_l1", nn.Linear(actor_input_dim, n_hidden))
        setattr(self, name + "_l2", nn.Linear(n_hidden, n_hidden))

        if "teacher" in name:
            setattr(self, name + "_l3_what", nn.Linear(n_hidden, 2))  # What to advise
            setattr(self, name + "_l3_when", nn.Linear(n_hidden, 2))  # When to advise
        else:
            setattr(self, name + "_l3", nn.Linear(n_hidden, actor_output_dim))

        self.max_action = max_action
        self.name = name

    def forward(self, x):
        x = F.relu(getattr(self, self.name + "_l1")(x))
        x = F.relu(getattr(self, self.name + "_l2")(x))

        if "teacher" in self.name:
            x_what = self.max_action * torch.tanh(getattr(self, self.name + "_l3_what")(x)) 
            x_when = getattr(self, self.name + "_l3_when")(x)
            return x_what, x_when
        else:
            x = self.max_action * torch.tanh(getattr(self, self.name + "_l3")(x)) 
            return x


class Critic(nn.Module):
    def __init__(self, critic_input_dim, n_hidden, name):
        super(Critic, self).__init__()

        # Q1 architecture
        setattr(self, name + "_l1", nn.Linear(critic_input_dim, n_hidden))
        setattr(self, name + "_l2", nn.Linear(n_hidden, n_hidden))
        setattr(self, name + "_l3", nn.Linear(n_hidden, 1))

        # Q2 architecture
        setattr(self, name + "_l4", nn.Linear(critic_input_dim, n_hidden))
        setattr(self, name + "_l5", nn.Linear(n_hidden, n_hidden))
        setattr(self, name + "_l6", nn.Linear(n_hidden, 1))

        self.name = name

    def forward(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(getattr(self, self.name + "_l1")(xu))
        x1 = F.relu(getattr(self, self.name + "_l2")(x1))
        x1 = getattr(self, self.name + "_l3")(x1)

        x2 = F.relu(getattr(self, self.name + "_l4")(xu))
        x2 = F.relu(getattr(self, self.name + "_l5")(x2))
        x2 = getattr(self, self.name + "_l6")(x2)

        return x1, x2

    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)

        x1 = F.relu(getattr(self, self.name + "_l1")(xu))
        x1 = F.relu(getattr(self, self.name + "_l2")(x1))
        x1 = getattr(self, self.name + "_l3")(x1)

        return x1 


class TD3(object):
    def __init__(self, actor_input_dim, actor_output_dim, critic_input_dim, n_hidden, max_action, name, args, i_agent):
        self.actor = Actor(actor_input_dim, actor_output_dim, n_hidden, max_action, name=name + "_actor").to(device)

        if "worker" not in name:
            self.actor_target = Actor(
                actor_input_dim, actor_output_dim, n_hidden, max_action, name=name + "_actor").to(device)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

            self.critic = Critic(critic_input_dim, n_hidden, name=name + "_critic").to(device)
            self.critic_target = Critic(critic_input_dim, n_hidden, name=name + "_critic").to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.max_action = max_action
        self.name = name
        self.args = args
        self.i_agent = i_agent

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)

        if "teacher" in self.name:
            action_what, action_when = self.actor(state)
            action_what = action_what.cpu().data.numpy().flatten()

            action_when = onehot_from_logits(logits=action_when)
            action_when = action_when.cpu().data.numpy()
            action_when = np.squeeze(action_when, axis=0)

            return action_what, action_when
        else:
            return self.actor(state).cpu().data.numpy().flatten()

    def centralized_train(self, agents, replay_buffer, iterations, batch_size, discount, tau, 
                          policy_noise, noise_clip, policy_freq):
        if "manager" in self.name: 
            n_agent = self.args.n_manager
        else:
            raise ValueError("Invalid name")

        debug = {}
        debug["critic_loss"] = 0.
        debug["actor_loss"] = 0.

        for it in range(iterations):
            # Sample replay buffer 
            x_n, y_n, u_n, r_n, d_n = replay_buffer.centralized_sample(
                batch_size=batch_size, n_agent=n_agent)

            states = [
                torch.FloatTensor(x_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            actions = [
                torch.FloatTensor(u_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            next_states = [
                torch.FloatTensor(y_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            dones = [
                torch.FloatTensor(1 - d_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]
            rewards = [
                torch.FloatTensor(r_n[i_agent_]).to(device)
                for i_agent_ in range(n_agent)]

            # Select action according to policy and add clipped noise 
            next_actions = []
            for i_agent_ in range(n_agent):
                with torch.no_grad():
                    noise = torch.FloatTensor(u_n[i_agent_]).data.normal_(0, policy_noise).to(device)
                    noise = noise.clamp(-noise_clip, noise_clip)

                    next_action = agents[i_agent_].policy.actor_target(next_states[i_agent_]) + noise
                    next_action = next_action.clamp(-self.max_action, self.max_action)
                    next_actions.append(next_action)
            next_actions = torch.cat(next_actions, dim=1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(torch.cat(next_states, dim=1), next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards[self.i_agent] + (dones[self.i_agent] * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(
                torch.cat(states, dim=1), torch.cat(actions, dim=1))

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            debug["critic_loss"] += critic_loss.cpu().data.numpy().flatten()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                actions_ = []
                for i_agent_ in range(n_agent):
                    if i_agent_ == self.i_agent:
                        with torch.enable_grad():
                            action = agents[i_agent_].policy.actor(states[i_agent_])
                            actions_.append(action)
                    else:
                        with torch.no_grad():
                            action = agents[i_agent_].policy.actor(states[i_agent_])
                            actions_.append(action)

                actor_loss = -self.critic.Q1(
                    torch.cat(states, dim=1), torch.cat(actions_, dim=1)).mean()
                
                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                debug["actor_loss"] += actor_loss.cpu().data.numpy().flatten()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return debug

    def train_teacher(self, replay_buffer, iterations, batch_size, discount, tau, policy_noise, noise_clip, policy_freq):
        debug = {}
        debug["critic_loss"] = 0.
        debug["actor_loss"] = 0.

        for it in range(iterations):
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample_teacher(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1 - d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select action according to policy and add clipped noise 
            noise = torch.FloatTensor(u).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)[:, 0:2]

            next_action_what, next_action_when = self.actor_target(next_state)
            next_action_what = (next_action_what + noise).clamp(-self.max_action, self.max_action)
            next_action_what = next_action_what.clamp(-self.max_action, self.max_action)
            next_action_when = onehot_from_logits(next_action_when)
            next_action = torch.cat((next_action_what, next_action_when), dim=1)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (done * discount * target_Q).detach()

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) 

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            debug["critic_loss"] += critic_loss.cpu().data.numpy().flatten()

            # Delayed policy updates
            if it % policy_freq == 0:
                # Compute actor loss
                action_what, action_when = self.actor(state)
                action_when = gumbel_softmax(action_when, hard=True)
                action = torch.cat((action_what, action_when), dim=1)
                actor_loss = -self.critic.Q1(state, action).mean()

                # Optimize the actor 
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                debug["actor_loss"] += actor_loss.cpu().data.numpy().flatten()

                # Update the frozen target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        return debug

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        if "worker" not in self.name:
            torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        from collections import OrderedDict

        actor_weight = torch.load('%s/%s_actor.pth' % (directory, filename), map_location='cpu')

        actor_weight_fixed = OrderedDict()
        for k, v in actor_weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            actor_weight_fixed[name_fixed] = v

        self.actor.load_state_dict(actor_weight_fixed)

        if "worker" not in self.name:
            critic_weight = torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu')

            critic_weight_fixed = OrderedDict()
            for k, v in critic_weight.items():
                name_fixed = self.name
                for i_name, name in enumerate(k.split("_")):
                    if i_name > 0:
                        name_fixed += "_" + name
                critic_weight_fixed[name_fixed] = v

            self.critic.load_state_dict(critic_weight_fixed)

            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
