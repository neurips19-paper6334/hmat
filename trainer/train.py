import copy
import numpy as np 
from trainer.utils import *

total_timesteps = 0
total_eps = 0

session_advices = 0
session_timesteps = 0

train_rewards = []  # For R_{Phase I}
teacher_rewards = []  # For R_{Phase II}


def eval_progress(env, workers, managers, n_eval, log, tb_writer, args, is_advice_eval):
    global total_eps, session_timesteps
    eval_reward = 0.
    self_prac_traj = []

    for i_eval in range(n_eval):
        ep_timesteps = 0.
        env_observations = env.reset()

        while True:
            # Managers select goals
            if ep_timesteps % args.sub_goal_freq == 0:
                manager_observations = get_manager_obs(env_observations, ep_timesteps, args)
                manager_actions = []
                manager_reward = 0.
                for manager, manager_obs in zip(managers, manager_observations):
                    manager_action = manager.select_deterministic_action(np.array(manager_obs))
                    manager_actions.append(manager_action)
                    env.world.goals[manager.i_agent].state.p_pos = manager_action  # For visualization only

            # Workers take actions based on manager goals
            worker_actions = []
            for worker, manager_action, env_obs in zip(workers, manager_actions, env_observations):
                worker_obs = get_worker_obs(manager_action, env, worker.i_agent)
                worker_action = worker.select_deterministic_action(np.array(worker_obs))
                worker_actions.append(worker_action)

            new_env_observations, env_rewards, _, _ = env.step(copy.deepcopy(worker_actions))
            terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False
            manager_reward += env_rewards[0]  # Joint reward

            if (ep_timesteps + 1) % args.sub_goal_freq == 0 or terminal:
                # Add self_prac_traj
                new_manager_observations = get_manager_obs(new_env_observations, ep_timesteps + 1, args)
                manager_done = terminal if args.manager_done else False

                self_prac_traj.append((
                    manager_observations, 
                    new_manager_observations, 
                    manager_actions, 
                    [manager_reward for _ in range(args.n_manager)],
                    [float(manager_done) for _ in range(args.n_manager)]))

            # For next timestep
            env_observations = new_env_observations
            eval_reward += env_rewards[0]
            ep_timesteps += 1
            if is_advice_eval:
                session_timesteps += 1

            if terminal:
                if is_advice_eval:
                    total_eps += 1
                break
    eval_reward /= float(n_eval)

    if is_advice_eval is False:
        log[args.log_name].info("Evaluation Reward {} at episode {}".format(eval_reward, total_eps))
        tb_writer.add_scalars("reward", {"eval_reward": eval_reward}, total_eps)

    return eval_reward, self_prac_traj


def collect_one_traj(workers, managers, teacher, env, log, args, tb_writer):
    global total_timesteps, total_eps, session_timesteps

    ep_reward = 0
    ep_timesteps = 0
    env_observations = env.reset()

    while True:
        if ep_timesteps % args.sub_goal_freq == 0:
            # Managers select goals
            manager_observations = get_manager_obs(env_observations, ep_timesteps, args)
            manager_actions = []
            manager_reward = 0.
            for manager, manager_obs in zip(managers, manager_observations):
                manager_action = manager.select_stochastic_action(np.array(manager_obs), session_timesteps)
                manager_actions.append(manager_action)

            # Get teacher action
            teacher_obs, teacher_obs_dict = get_teacher_obs(
                managers=managers, teacher=teacher, 
                manager_observations=manager_observations, manager_actions=manager_actions,
                train_rewards=train_rewards, teacher_rewards=teacher_rewards,
                session_advices=session_advices, args=args)

            teacher_action = teacher.select_stochastic_action(np.array(teacher_obs), total_timesteps)

            # If teacher decides to advice, then overwrite manager_action with teacher advice
            if teacher_action[-1] > 0:
                manager_actions[1] = teacher_action[0:2]

        # Worker takes action based on manager goal
        worker_actions = []
        for worker, manager_action, env_obs in zip(workers, manager_actions, env_observations):
            worker_obs = get_worker_obs(manager_action, env, worker.i_agent)
            worker_action = worker.select_deterministic_action(np.array(worker_obs))
            worker_actions.append(worker_action)

        # Perform action
        new_env_observations, env_rewards, _, _ = env.step(copy.deepcopy(worker_actions))
        terminal = True if ep_timesteps + 1 == args.ep_max_timesteps else False
        manager_reward += env_rewards[0]  # Joint reward

        if (ep_timesteps + 1) % args.sub_goal_freq == 0 or terminal:
            # Add manager memory
            new_manager_observations = get_manager_obs(new_env_observations, ep_timesteps + 1, args)
            manager_done = terminal if args.manager_done else False

            for i_manager, manager in enumerate(managers):
                manager.add_memory(
                    obs=manager_observations,
                    new_obs=new_manager_observations,
                    action=manager_actions,
                    reward=[manager_reward for _ in range(args.n_manager)],
                    done=[float(manager_done) for _ in range(args.n_manager)])

            # Keep teacher memory
            # NOTE After the temp policy update, teacher reward and the next teacher observation
            # will be updated
            teacher_done = True if (session_advices + 1) == teacher.n_advice else False

            teacher.keep_memory(
                obs=teacher_obs,
                new_obs=teacher_obs,
                action=teacher_action,
                reward=0.,
                done=float(teacher_done),
                obs_dict=teacher_obs_dict)

        # For next timestep
        env_observations = new_env_observations
        ep_timesteps += 1
        total_timesteps += 1
        session_timesteps += 1
        ep_reward += env_rewards[0]

        if terminal: 
            total_eps += 1
            log[args.log_name].info("Train episode reward {} at episode {}".format(ep_reward, total_eps))
            tb_writer.add_scalars("reward", {"train_reward": ep_reward}, total_eps)

            return ep_reward


def train(workers, managers, temp_managers, teacher, env, log, tb_writer, args):
    global session_advices, session_timesteps

    while True:
        # Load trained managers for encouraging heterogeneous knowledge 
        # Refer to Section 6.1
        load_trained_managers(managers)

        while True:
            # Measure task-level performance for reporting results in paper
            eval_progress(
                env=env, workers=workers, managers=managers, n_eval=10, 
                log=log, tb_writer=tb_writer, args=args, is_advice_eval=False)

            # Collect advice_traj w/ teacher advice (Phase I)
            # Note that we copy temporary task-level policies here
            # to make the code easy. Whether sync before/after Phase I should not
            # make any difference
            for temp_manager, manager in zip(temp_managers, managers):
                temp_manager.sync(target_agent=manager)
                temp_manager.clear_memory()

            train_reward = collect_one_traj(
                workers=workers, managers=temp_managers, teacher=teacher,
                env=env, log=log, args=args, tb_writer=tb_writer)
            session_advices += 1
            train_rewards.append(train_reward)

            # Update temp manager policy
            # Note that the manager0 (Agent $i$) is already expert, so 
            # we only train manager1 (Agent $k$). This is not a case for the
            # two box push domain
            for i_temp_manager, temp_manager in enumerate(temp_managers):
                if i_temp_manager == 1:
                    temp_manager.update_policy(
                        agents=temp_managers, iterations=len(temp_manager.memory) * 5,
                        batch_size=len(temp_manager.memory), total_timesteps=total_timesteps)

            # Measure performance after manager update and measure teacher reward
            teacher_reward, self_prac_traj = eval_progress(
                env=env, workers=workers, managers=temp_managers, n_eval=args.n_eval,
                log=log, tb_writer=tb_writer, args=args, is_advice_eval=True)
            teacher_rewards.append(teacher_reward)

            log[args.log_name].info("Teacher Reward {:.5f} at episode {}".format(teacher_reward, total_eps))
            tb_writer.add_scalars("reward", {"teacher_reward": teacher_reward}, total_eps)

            # Add advice_traj and self_prac_traj to manager memory
            # Refer to Algorithm 1 Line 14
            for i_manager in range(args.n_manager):
                for exp in temp_managers[i_manager].memory.storage:
                    managers[i_manager].memory.add(data=exp)
                for exp in self_prac_traj:
                    managers[i_manager].memory.add(data=exp)

            # Update manager policy
            # Note that the manager0 (Agent $i$) is already expert, so 
            # we only train manager1 (Agent $k$). This is not a case for the
            # two box push domain
            for i_manager, manager in enumerate(managers):
                if i_manager == 1:
                    manager.update_policy(
                        agents=managers, iterations=args.ep_max_timesteps * (1 + args.n_eval),
                        batch_size=args.batch_size, total_timesteps=total_timesteps)

            # Update teacher memory with the measured teacher reward
            # Also update next teacher observation
            teacher.update_memory(
                teacher_reward=teacher_reward, temp_managers=temp_managers,
                train_rewards=train_rewards, teacher_rewards=teacher_rewards)

            # Teacher update
            if total_eps % 45 == 0:
                teacher.update_policy(batch_size=64, total_timesteps=total_timesteps)

            # Session reset
            if total_eps % args.session == 0:
                session_advices = 0
                session_timesteps = 0
                train_rewards.clear()
                teacher_rewards.clear()
                teacher.save_weight(
                    filename=teacher.name + "_" + str(total_eps), directory="./pytorch_models")
                break
