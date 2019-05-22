import numpy as np
from multiagent.core import World, Agent, Landmark, Goal
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, mode=2):
        """Define two agents, two boxes, and two targets.
        Note that world.goals are used only for hierarchical RL visualization only
        Mode0: Pre-train agents to move left box
        Mode1: Pre-train agents to move right box
        Mode2: Train agents to move both left and right box
        """
        world = World()
        self.mode = mode

        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.1

        self.boxes = [Landmark() for _ in range(2)]
        for i, box in enumerate(self.boxes):
            box.name = 'box %d' % i
            box.size = 0.25 
            box.collide = True
            box.index = i

            # Box modes for pre-training only
            if self.mode == 0 and box.index == 1:
                box.movable = False
            elif self.mode == 1 and box.index == 0:
                box.movable = False
            else:
                box.movable = True

            # Different box mass (Box1 is 3x heavier than Box0)
            if box.index == 0:
                box.initial_mass = 2.
            elif box.index == 1:
                box.initial_mass = 6.
            else:
                raise ValueError()
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
        """Define random properties for agents, boxes, and targets.
        Two agents are randomly initialized.
        One box and target are initialized on the left side.
        The other box and target are initialized on the right side.
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
                landmark.state.p_pos = np.array([-0.30, 0.0])  # Left box
            elif "box" in landmark.name and landmark.index == 1:
                landmark.state.p_pos = np.array([+0.30, 0.0])  # Right box
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
        for i, landmark in enumerate(world.landmarks):
            if "box" in landmark.name and landmark.index == 0:
                box0 = landmark
            elif "box" in landmark.name and landmark.index == 1:
                box1 = landmark
            elif "target" in landmark.name and landmark.index == 0:
                target0 = landmark
            elif "target" in landmark.name and landmark.index == 1:
                target1 = landmark
            else:
                raise ValueError()

        dist1 = np.sum(np.square(box0.state.p_pos - target0.state.p_pos))
        dist2 = np.sum(np.square(box1.state.p_pos - target1.state.p_pos))
        dist = dist1 + dist2

        return -dist

    def observation(self, agent, world):
        """For each agent, observation consists of:
        [agent velocity, agent pos, box poses, targets poses, other agent pos]
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
