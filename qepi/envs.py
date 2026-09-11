import torch
import numpy as np
from gymnasium.envs.classic_control import MountainCarEnv as BaseMountainCarEnv

# device = torch.device("cuda:0")
# device = torch.device("mps")

class DiscretePolicy:
    def __init__(self, pi,dx,dv,x_min = -0.99,v_min = -0.07):
        self.pi = pi
        self.dx = dx
        self.dv = dv
        self.x_min = x_min
        self.v_min = v_min

    def predict(self,state, deterministic = True):
        x_index = torch.div(state[0]+self.dx/2 - self.x_min, self.dx, rounding_mode='floor').item()
        v_index = torch.div(state[1]+self.dv/2 - self.v_min, self.dv, rounding_mode='floor').item()
        pi = self.pi[int(x_index),int(v_index)].item()

        if deterministic:
            return round(pi)
        r = np.random.rand()
        if r<pi:
            a = 1
        else:
            a = 0
        return a



class MountainCarEnv(BaseMountainCarEnv):
    def __init__(self, goal_velocity=0):
        super().__init__(goal_velocity=goal_velocity)

    def done(self,state):
        position, velocity = np.moveaxis(state, -1, 0)
        return (position >= self.goal_position) & (velocity >= self.goal_velocity)

    def reward(self, state, action, state_next):
        position, velocity = np.moveaxis(state, -1, 0)
        dt = np.ones_like(velocity)
        reward = np.where(self.done(state),0,-dt)
        return reward.astype(np.float32)

    def steps(self, action, state):
        state = state.copy()
        position, velocity = np.moveaxis(state, -1, 0)


        velocity = velocity + (action - 1) * self.force + np.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position = position + velocity
        position = np.clip(position, self.min_position, self.max_position)
        velocity[(position == self.min_position) & (velocity < 0)] = 0

        state_next = np.moveaxis((position, velocity), 0, -1)

        done = self.done(state_next)
        dt = np.ones_like(velocity)
        #dt[velocity>0] = np.minimum(1,(self.goal_position-position[velocity>0])/velocity[velocity>0])
        reward = np.where(done,0,-dt)

        return state_next, reward, done, {}

