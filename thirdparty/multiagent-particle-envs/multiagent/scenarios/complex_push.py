import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        """Define two agents, one box, and two targets.
        Note that world.goals are used only for hierarchical RL visualization only
        Note that this domain should have one target.
        But, we empirically found that having the additional right target does not affect
        performance much as the reward is based on the left target only
        """
        world = World()

        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1  # Radius

        self.boxes = [Landmark() for _ in range(1)]
        for i, box in enumerate(self.boxes):
            box.name = 'box %d' % i
            box.collide = True
            box.movable = True
            box.size = 0.25
            box.initial_mass = 7.
            box.index = i
            world.landmarks.append(box)

        self.targets = [Landmark() for _ in range(2)]
        for i, target in enumerate(self.targets):
            target.name = 'target %d' % i
            target.collide = False
            target.movable = False
            target.size = 0.05
            target.index = i
            world.landmarks.append(target)

        world.goals = [Goal() for i in range(len(world.agents))]
        for i, goal in enumerate(world.goals):
            goal.name = 'goal %d' % i
            goal.collide = False
            goal.movable = False

        self.reset_world(world)
        
        return world

    def reset_world(self, world):
        """Define random properties for agents, box, and targets.
        Two agents are randomly initialized.
        The box and the left target are initialized at the same location on the left side
        """
        for i, agent in enumerate(world.agents):
            if i == 0:
                agent.color = np.array([1.0, 0.0, 0.0])  # Red
            elif i == 1:
                agent.color = np.array([0.0, 1.0, 0.0])  # Blue
            else:
                raise NotImplementedError()

            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_vel = np.zeros(world.dim_p)

            if "box" in landmark.name and landmark.index == 0:
                landmark.state.p_pos = np.array([-0.25, 0.0])  # Box
            elif "target" in landmark.name and landmark.index == 0:
                landmark.state.p_pos = np.array([-0.85, 0.0])  # Left target
            elif "target" in landmark.name and landmark.index == 1:
                landmark.state.p_pos = np.array([+0.85, 0.0])  # Right target
            else:
                raise ValueError()

        for i, goal in enumerate(world.goals):
            goal.color = world.agents[i].color
            goal.state.p_pos = np.zeros(world.dim_p) - 2  # Initialize outside of the box
            goal.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        """Reward is defined to be large if distance between box and target0 is minimized"""
        for i, landmark in enumerate(world.landmarks):
            if "box" in landmark.name and landmark.index == 0:
                box0 = landmark
            elif "target" in landmark.name and landmark.index == 0:
                target0 = landmark
            elif "target" in landmark.name and landmark.index == 1:
                target1 = landmark
            else:
                raise ValueError()

        dist = np.sum(np.square(box0.state.p_pos - target0.state.p_pos))
        return -dist

    def observation(self, agent, world):
        """For each agent, observation consists of:
        [agent velocity, agent pos, box pos, targets poses, other agent pos]
        """
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos)
        assert len(entity_pos) == len(self.boxes) + len(self.targets)

        other_pos = []
        for other in world.agents:
            if other is agent: 
                continue
            other_pos.append(other.state.p_pos)

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos)
