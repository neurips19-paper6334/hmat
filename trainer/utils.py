import copy
import numpy as np
import torch
from misc.utils import normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########################################################################################
# MISC
def load_trained_managers(managers):
    assert len(managers) == 2, "Only two managers are supported"

    # Manager0 corresponds to Agent $i$, which already knows about the domain
    managers[0].load_weight(
        filename="manager0",
        directory="./pytorch_models/one_direction")

    # Manager1 corresponds to Agent $k$, which has no knowledge about the domain
    managers[1].load_weight(
        filename="manager1",
        directory="./pytorch_models/one_direction")


def get_q_values(agents, agent_observations, agent_actions):
    assert len(agent_actions) == 2

    agent_observations = np.concatenate([agent_observations[0], agent_observations[1]]).reshape(1, -1)
    agent_observations = torch.FloatTensor(agent_observations).to(device)

    agent_actions = np.concatenate([agent_actions[0], agent_actions[1]]).reshape(1, -1)
    agent_actions = torch.FloatTensor(agent_actions).to(device)

    q_values = []
    for agent in agents:
        q_value = agent.policy.critic.Q1(agent_observations, agent_actions)
        q_value = q_value.cpu().data.numpy().flatten()[0]
        q_values.append(q_value)

    return q_values


def get_avg_reward(train_rewards, teacher_rewards, args):
    # Process train rewards
    if len(train_rewards) == 0:
        avg_train_reward = args.reward_min
    else:
        if len(train_rewards) > args.window_size:
            avg_train_reward = np.average(train_rewards[-args.window_size:-1])
        else:
            avg_train_reward = np.average(train_rewards)
    avg_train_reward = np.clip(avg_train_reward, a_min=args.reward_min, a_max=args.reward_max)
    avg_train_reward = normalize(avg_train_reward, min_value=args.reward_min, max_value=args.reward_max)

    # Process teacher rewards
    if len(teacher_rewards) == 0:
        avg_teacher_reward = args.reward_min
    else:
        if len(teacher_rewards) > args.window_size:
            avg_teacher_reward = np.average(teacher_rewards[-args.window_size:-1])
        else:
            avg_teacher_reward = np.average(teacher_rewards)
    avg_teacher_reward = np.clip(avg_teacher_reward, a_min=args.reward_min, a_max=args.reward_max)
    avg_teacher_reward = normalize(avg_teacher_reward, min_value=args.reward_min, max_value=args.reward_max)

    return avg_train_reward, avg_teacher_reward


########################################################################################
# MANAGER
def get_manager_obs(env_observations, ep_timesteps, args):
    if args.manager_done:
        manager_observations = []

        remaining_timesteps = normalize(
            value=(args.ep_max_timesteps - ep_timesteps), 
            min_value=0., max_value=float(args.ep_max_timesteps))
        remaining_timesteps = np.array([remaining_timesteps])
        
        for env_obs in env_observations:
            manager_obs = np.concatenate([env_obs, remaining_timesteps])
            manager_observations.append(manager_obs)
    else:
        manager_observations = env_observations

    return manager_observations


########################################################################################
# TEACHER
def concat_in_order(obs, args):
    return np.concatenate([
        obs["teacher_obs"], obs["student_obs"],
        obs["teacher_action"], obs["student_action"], obs["teacher_action_at"],
        obs["q_with_teacher_critic"], obs["q_with_student_critic"],
        obs["q_at_with_teacher_critic"], obs["q_at_with_student_critic"],
        obs["avg_teacher_reward"], obs["avg_train_reward"], obs["remain_time"]])


def teacher_input_process(managers, teacher, manager_observations, manager_actions, train_rewards, 
                          teacher_rewards, i_teacher, i_student, session_advices, args):
    obs = {}
    obs["manager_observations"] = manager_observations
    obs["manager_actions"] = manager_actions
    obs["teacher_obs"] = manager_observations[i_teacher]
    obs["teacher_action"] = manager_actions[i_teacher]
    obs["student_obs"] = manager_observations[i_student]
    obs["student_action"] = manager_actions[i_student]
    obs["session_advices"] = session_advices

    # Get Q-values for manager_actions
    q_values = get_q_values(managers, manager_observations, manager_actions)
    q_values = np.clip(q_values, a_min=args.q_min, a_max=args.q_max)

    obs["q_with_teacher_critic"] = np.array([normalize(
        value=q_values[i_teacher], min_value=args.q_min, max_value=args.q_max)])
    obs["q_with_student_critic"] = np.array([normalize(
        value=q_values[i_student], min_value=args.q_min, max_value=args.q_max)])

    # Get Q-values for manager_actions_at
    obs["teacher_action_at"] = managers[i_teacher].select_deterministic_action(np.array(manager_observations[i_student]))

    q_values = get_q_values(managers, manager_observations, [manager_actions[0], obs["teacher_action_at"]])
    q_values = np.clip(q_values, a_min=args.q_min, a_max=args.q_max)

    obs["q_at_with_teacher_critic"] = np.array([normalize(
        value=q_values[i_teacher], min_value=args.q_min, max_value=args.q_max)])
    obs["q_at_with_student_critic"] = np.array([normalize(
        value=q_values[i_student], min_value=args.q_min, max_value=args.q_max)])

    # Get avg reward
    avg_train_reward, avg_teacher_reward = get_avg_reward(
        train_rewards=train_rewards, teacher_rewards=teacher_rewards, args=args)
    obs["avg_train_reward"] = np.array([avg_train_reward])
    obs["avg_teacher_reward"] = np.array([avg_teacher_reward])

    # Get teacher remain timestep
    obs["remain_time"] = np.array([normalize(
        value=(teacher.n_advice - session_advices),
        min_value=0., max_value=float(teacher.n_advice))])

    return concat_in_order(obs, args), obs


def get_teacher_obs(managers, teacher, manager_observations, manager_actions, 
                    train_rewards, teacher_rewards, session_advices, args):
    i_teacher, i_student = 0, 1
    teacher_obs, teacher_obs_dict = teacher_input_process(
        managers=managers, teacher=teacher, 
        manager_observations=manager_observations, manager_actions=manager_actions, 
        train_rewards=train_rewards, teacher_rewards=teacher_rewards,
        i_teacher=i_teacher, i_student=i_student, 
        session_advices=session_advices, args=args)

    return teacher_obs, copy.deepcopy(teacher_obs_dict)


########################################################################################
# WORKER
def get_worker_loc(env, i_agent):
    return env.world.agents[i_agent].state.p_pos


def get_worker_obs(manager_action, env, i_agent):
    return np.concatenate([get_worker_loc(env, i_agent), manager_action])
