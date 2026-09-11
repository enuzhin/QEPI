import numpy as np
from .envs import MountainCarEnv
from .discretization import discretize

env = Nx = Nv = dx = dv = gamma = None
X = V = S = None
A_all = S_all = S_new_all = r_all = done_all = not_done = discrete_state_new_all = None


def configure(n_x=4, n_v=4, discount=0.99):
    global env, Nx, Nv, dx, dv, gamma, X, V, S
    global A_all, S_all, S_new_all, r_all, done_all, not_done, discrete_state_new_all
    env = MountainCarEnv()
    Nx, Nv, gamma = n_x, n_v, discount
    dx = (env.max_position - env.min_position) / (Nx - 1)
    dv = 2 * env.max_speed / (Nv - 1)
    X, V = np.meshgrid(np.linspace(env.min_position, env.max_position, Nx),
                       np.linspace(-env.max_speed, env.max_speed, Nv),
                       indexing="ij")
    S = np.concatenate((X[:, :, None], V[:, :, None]), axis=2)
    A_all = np.empty([3, Nx, Nv]);
    A_all[0], A_all[1], A_all[2] = 0, 1, 2
    S_all = np.concatenate([[S], [S], [S]], axis=0)
    S_new_all, r_all, done_all, _ = env.steps(A_all, S_all)
    discrete_state_new_all = discretize(S_new_all)
    not_done = ~env.done(S)
