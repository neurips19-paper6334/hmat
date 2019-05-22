import torch
import argparse
import os
import numpy as np
from misc.utils import set_log, make_env
from tensorboardX import SummaryWriter
from trainer.train import train


def set_policy(env, tb_writer, log, args, name, i_agent):
    if name == "worker":
        from policy.worker import Worker
        policy = Worker(env=env, tb_writer=tb_writer, log=log, name=name, args=args, i_agent=i_agent)
        policy.load_weight(filename="worker", directory="./pytorch_models/worker")
        policy.set_eval_mode()
    elif name == "manager" or name == "temp_manager":
        from policy.manager import Manager
        policy = Manager(env=env, tb_writer=tb_writer, log=log, name=name, args=args, i_agent=i_agent)
    elif name == "teacher":
        from policy.teacher import Teacher
        policy = Teacher(env=env, tb_writer=tb_writer, log=log, name=name, args=args, i_agent=i_agent)
    else:
        raise ValueError("Invalid name")

    return policy


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))
    log = set_log(args)

    # Create env
    env = make_env(args)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize policy
    # Note that only one teacher is considered in the one box push domain 
    # to transfer knowledge from agent $i$ to agent $k$ (Section 6.1)
    workers = [
        set_policy(env, tb_writer, log, args, name="worker", i_agent=i_agent)
        for i_agent in range(args.n_worker)]
    managers = [
        set_policy(env, tb_writer, log, args, name="manager", i_agent=i_agent)
        for i_agent in range(args.n_manager)]
    temp_managers = [
        set_policy(env, tb_writer, log, args, name="temp_manager", i_agent=i_agent)
        for i_agent in range(args.n_manager)]
    teacher = set_policy(env, tb_writer, log, args, name="teacher", i_agent=0)

    assert len(workers) == len(managers), "The two number must be same"
    assert len(managers) == len(temp_managers), "The two number must be same"
    
    # Start train
    train(
        workers=workers, managers=managers, 
        temp_managers=temp_managers, teacher=teacher,
        env=env, log=log, tb_writer=tb_writer, args=args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--tau", default=0.01, type=float, 
        help="Target network update rate")
    parser.add_argument(
        "--expl-noise", default=0.1, type=float, 
        help="Std of Gaussian exploration noise")
    parser.add_argument(
        "--batch-size", default=50, type=int, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--policy-noise", default=0.2, type=float, 
        help="Noise added to target policy during critic update")
    parser.add_argument(
        "--noise-clip", default=0.5, type=float, 
        help="Range to clip target policy noise")
    parser.add_argument(
        "--policy-freq", default=2, type=int,
        help="Frequency of delayed policy updates")
    parser.add_argument(
        "--actor-lr", default=0.0001, type=float,
        help="Learning rate for actor")
    parser.add_argument(
        "--critic-lr", default=0.001, type=float,
        help="Learning rate for critic")

    # Worker
    parser.add_argument(
        "--n-worker", default=2, type=int,
        help="Number of workers")
    parser.add_argument(
        "--worker-n-hidden", default=128, type=int,
        help="Number of hidden units")

    # Manager
    parser.add_argument(
        "--n-manager", default=2, type=int,
        help="Number of managers")
    parser.add_argument(
        "--sub-goal-freq", default=5, type=int,
        help="Frequency of giving sub-goal to worker policy")
    parser.add_argument(
        "--manager-n-hidden", default=400, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--manager-done", action="store_false",
        help="Set manager done or not")
    parser.add_argument(
        "--manager-discount", default=0.99, type=float, 
        help="Discount factor for manager")
    parser.add_argument(
        "--manager-start-timesteps", default=2000, type=int, 
        help="How many time steps purely random policy is run for")

    # Teacher
    parser.add_argument(
        "--n-teacher", default=1, type=int,
        help="Number of teachers")
    parser.add_argument(
        "--session", required=True, type=int,
        help="Student reset every session")
    parser.add_argument(
        "--teacher-start-timesteps", default=30000, type=int, 
        help="How many time steps purely random policy is run for")
    parser.add_argument(
        "--teacher-n-hidden", default=400, type=int,
        help="Number of hidden units")
    parser.add_argument(
        "--teacher-discount", default=0.99, type=float, 
        help="Discount factor")
    parser.add_argument(
        "--n-eval", default=2, type=int,
        help="# of evaluation episodes to measure teacher reward (Phase II)")
    parser.add_argument(
        "--q-min", default=-20., type=float,
        help="Minimum Q-value used for clipping")
    parser.add_argument(
        "--q-max", default=0., type=float,
        help="Maximum Q-value used for clipping")
    parser.add_argument(
        "--reward-min", default=-20., type=float,
        help="Minimum reward used for clipping")
    parser.add_argument(
        "--reward-max", default=-10., type=float,
        help="Maximum reward used for clipping")
    parser.add_argument(
        "--window-size", default=30, type=int, 
        help="Window size used to get average rewards for teacher obs")

    # Env
    parser.add_argument(
        "--env-name", type=str, required=True,
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", type=int, required=True,
        help="Episode is terminated when max timestep is reached.")

    # Misc
    parser.add_argument(
        "--prefix", default="", type=str,
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", default=5, type=int, 
        help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()

    # Set log name
    args.log_name = "env::%s_prefix::%s_log" % (args.env_name, args.prefix)

    main(args=args)
