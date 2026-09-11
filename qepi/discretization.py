"""Stochastic discretisation onto grid nodes and assembly of the
policy-evaluation SLE."""

import numpy as np
import torch


def discretize(state):
    from .grid import env, Nx, Nv, dx, dv

    x, v = torch.moveaxis(torch.tensor(state), -1, 0)

    x_l = torch.div(x - env.min_position, dx, rounding_mode='floor').type(torch.LongTensor)
    p_x_l = 1 - ((x - env.min_position) / dx - x_l)

    v_l = torch.div(v + env.max_speed, dv, rounding_mode='floor').type(torch.LongTensor)
    p_v_l = 1 - ((v + env.max_speed) / dv - v_l)

    x_u, v_u = torch.clip(x_l + 1, 0, Nx - 1), torch.clip(v_l + 1, 0, Nv - 1)

    p_x_u, p_v_u = 1 - p_x_l, 1 - p_v_l

    return ((x_l.numpy(), v_l.numpy()), (x_u.numpy(), v_u.numpy())), ((p_x_l.numpy(), p_v_l.numpy()),
                                                                      (p_x_u.numpy(), p_v_u.numpy()))


def transition_matrix(x_from, v_from, x_to, v_to, p_x, p_v):
    Tx = p_x[:, :, np.newaxis] * (x_to[:, :, np.newaxis] == x_from[np.newaxis, np.newaxis, :])
    Tv = p_v[:, :, np.newaxis] * (v_to[:, :, np.newaxis] == v_from[np.newaxis, np.newaxis, :])
    T = Tx[:, :, :, np.newaxis] * Tv[:, :, np.newaxis, :]
    return T


def value_new(value, discrete_state_new):
    ((x_l, v_l), (x_u, v_u)), ((p_x_l, p_v_l), (p_x_u, p_v_u)) = discrete_state_new
    value_new = p_x_l * p_v_l * value[x_l, v_l] + \
                p_x_u * p_v_l * value[x_u, v_l] + \
                p_x_l * p_v_u * value[x_l, v_u] + \
                p_x_u * p_v_u * value[x_u, v_u]
    return value_new


def to_sle(discrete_state_new, not_done, r):
    from .grid import Nx, Nv, gamma
    x, v = np.arange(0, Nx), np.arange(0, Nv)
    ((x_l, v_l), (x_u, v_u)), ((p_x_l, p_v_l), (p_x_u, p_v_u)) = discrete_state_new
    T_ll = transition_matrix(x_from=x, v_from=v, x_to=x_l, v_to=v_l, p_x=p_x_l, p_v=p_v_l)
    T_lu = transition_matrix(x_from=x, v_from=v, x_to=x_l, v_to=v_u, p_x=p_x_l, p_v=p_v_u)
    T_ul = transition_matrix(x_from=x, v_from=v, x_to=x_u, v_to=v_l, p_x=p_x_u, p_v=p_v_l)
    T_uu = transition_matrix(x_from=x, v_from=v, x_to=x_u, v_to=v_u, p_x=p_x_u, p_v=p_v_u)

    P = (gamma * (T_ll + T_lu + T_ul + T_uu) * not_done[np.newaxis, np.newaxis, :, :] - np.eye(Nx)[
        :, np.newaxis, :, np.newaxis] * np.eye(Nv)[np.newaxis, :, np.newaxis, :]).reshape(Nx * Nv, Nx * Nv)
    r = r.reshape(-1)
    L, b = torch.tensor(P).float(), -torch.tensor(r).float()
    return L, b
