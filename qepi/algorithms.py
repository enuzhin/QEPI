import numpy as np
import torch
from tqdm import tqdm
from torchvision.transforms import GaussianBlur

from .lsmr import lsmr
from .discretization import discretize, to_sle, value_new
from .qubo import (value_to_binary_vector, binary_vector_to_value,
                   linear_equation_to_qubo)
from .solvers import solve_qubo, solve_qubo_dimod

def policy_iteration(n_steps=100):
    from .grid import env, S, Nx, Nv, gamma, A_all, S_all, r_all, done_all, not_done,discrete_state_new_all

    A = np.full((Nx, Nv), 1 / 2)
    value = np.zeros((Nx, Nv))

    for i in tqdm(range(n_steps)):
        S_new, r, _, _ = env.steps(A, S)
        discrete_state_new = discretize(S_new)
        L, b = to_sle(discrete_state_new, not_done, r)

        value, *_ = lsmr(lambda x: L @ x, lambda x: (L.T) @ x, b.clone(), x0=torch.FloatTensor(value).reshape(-1))

        value = value.reshape(Nx, Nv).numpy()

        Q = r_all + gamma * value_new(value, discrete_state_new_all) * (~ done_all)

        A = np.argmax(Q, axis=0)
        A[Q[0] == Q[2]] = 1
        if i % 10 == 0:
            print("Mean value: ",value.mean())

    return A, value

def qepi(n_steps=10, num_bits=10, v_max=100.0,
         num_anneals=100, anneal_duration=1280, backend="qubovert"):
    from .grid import (env, S, Nx, Nv, gamma, r_all, done_all,
                       not_done, discrete_state_new_all)
    history = []
    A = np.empty([Nx, Nv])
    A[:, :] = 1 / 2
    value = np.zeros([Nx, Nv])

    y = value_to_binary_vector(-value, num_bits, v_max=v_max)
    for i in tqdm(range(n_steps)):
        S_new, r, _, _ = env.steps(A, S)
        discrete_state_new = discretize(S_new)
        L, b = to_sle(discrete_state_new, not_done, r)

        P = linear_equation_to_qubo(L.numpy(), b.numpy(), num_bits, v_max)

        if backend == "qubovert":
            y, loss = solve_qubo(P, x0=y, num_anneals=num_anneals,
                                 anneal_duration=anneal_duration)
        else:
            y, loss = solve_qubo_dimod(P, num_anneals=num_anneals,
                                       anneal_duration=anneal_duration, backend=backend)
        loss_sle = loss + (b ** 2).sum()
        value = -binary_vector_to_value(y, num_bits, v_max=v_max)
        value = value.reshape(Nx, Nv)
        Q = r_all + gamma * value_new(value, discrete_state_new_all) * (~ done_all)

        A = np.argmax(Q, axis=0)
        A[Q[0] == Q[2]] = 1

        history.append(float(loss_sle))
    return A, value, history

def soft_vi(n_iters, sigma, kernel=11, device=None):
    from .grid import env, S, Nx, Nv, dx, dv, gamma

    device = _resolve_device(device)
    blur = (lambda q: q) if sigma <= 0 else GaussianBlur(kernel, sigma=sigma)

    A_ = torch.empty(3, Nx, Nv)
    A_[0], A_[1], A_[2] = 0., 1., 2.
    S_ = torch.Tensor(S).unsqueeze(0).expand(3, -1, -1, -1)

    # The dynamics do not depend on the value function, so the successor
    # states, rewards and terminal flags are the same at every iteration.
    state_new, r, done, _ = env.steps(A_.numpy(), S_.numpy())
    state_new = torch.tensor(state_new, dtype=torch.float32, device=device)
    r = torch.tensor(r, dtype=torch.float32, device=device)
    terminal = torch.tensor(env.done(S), device=device)

    (x_new_l, v_new_l), (p_x_new_l, p_v_new_l) = to_index_p(
        state_new, env.min_position, -env.max_speed, dx, dv)
    x_new_u = torch.clip(x_new_l + 1, 0, Nx - 1)
    v_new_u = torch.clip(v_new_l + 1, 0, Nv - 1)

    value = torch.zeros(Nx, Nv, device=device)
    pi = torch.ones(Nx, Nv, device=device)

    for i in tqdm(range(n_iters)):
        value_next = p_x_new_l * p_v_new_l * value[x_new_l, v_new_l] + \
                     (1 - p_x_new_l) * p_v_new_l * value[x_new_u, v_new_l] + \
                     p_x_new_l * (1 - p_v_new_l) * value[x_new_l, v_new_u] + \
                     (1 - p_x_new_l) * (1 - p_v_new_l) * value[x_new_u, v_new_u]

        Q = r + gamma * value_next
        blured_Q = blur(Q)
        value, pi = blured_Q.max(dim=0)
        pi = pi.float()
        pi[blured_Q[0] == blured_Q[2]] = 1.
        pi = blur(pi.unsqueeze(0)).squeeze()
        value[terminal] = 0.
        if i % 100 == 0:
            print("Mean value:", value.mean().item())

    return value, pi

def evaluate_annealing(A_sol, n_steps=10, num_anneals=1000, anneal_duration=10_000,
                       num_bits=10, v_max=100.0, backend="qubovert"):

    from .grid import (env, S, Nx, Nv, gamma, r_all, done_all, not_done,
                       discrete_state_new_all)

    A = np.empty([Nx, Nv])
    A[:, :] = 1 / 2
    value = np.zeros([Nx, Nv])

    y = value_to_binary_vector(-value, num_bits, v_max=v_max)

    history = []
    for _ in range(n_steps):
        S_new, r, _, _ = env.steps(A, S)
        discrete_state_new = discretize(S_new)
        L, b = to_sle(discrete_state_new, not_done, r)

        P = linear_equation_to_qubo(L.numpy(), b.numpy(), num_bits, v_max)
        if backend == "qubovert":
            y, loss = solve_qubo(P, x0=y, num_anneals=num_anneals, anneal_duration=anneal_duration)
        else:
            y, loss = solve_qubo_dimod(P, anneal_duration=anneal_duration, num_anneals=num_anneals,
                                       backend=backend)
        loss_sle = loss + (b ** 2).sum()
        value = -binary_vector_to_value(y, num_bits, v_max=v_max)
        value = value.reshape(Nx, Nv)
        Q = r_all + gamma * value_new(value, discrete_state_new_all) * (~ done_all)

        A = np.argmax(Q, axis=0)
        A[Q[0] == Q[2]] = 1

        history.append((A == A_sol).all().item())
    return history

def estimate_loss_wrt_num_anneals(anneal_duration=20.0, num_anneals=(1,),
                                  num_bits=10, v_max=100.0, backend="qubovert"):
    from .grid import env, S, Nx, Nv, gamma, not_done
    A = np.empty([Nx, Nv])
    A[:, :] = 1 / 2
    value = np.zeros([Nx, Nv])

    y = value_to_binary_vector(-value, num_bits, v_max=v_max)

    history = []
    for num_anneal in num_anneals:
        S_new, r, _, _ = env.steps(A, S)
        discrete_state_new = discretize(S_new)
        L, b = to_sle(discrete_state_new, not_done, r)

        P = linear_equation_to_qubo(L.numpy(), b.numpy(), num_bits, v_max)
        if backend == "qubovert":
            y, loss = solve_qubo(P, x0=y, num_anneals=num_anneal, anneal_duration=anneal_duration)
        else:
            y, loss = solve_qubo_dimod(P, num_anneals=num_anneal, anneal_duration=anneal_duration, backend=backend)
        loss_sle = loss + (b ** 2).sum()
        history.append(loss_sle.item())

    return history

def estimate_loss_wrt_durations(A=None, anneal_durations=(20.0,), num_anneals=1,
                                num_bits=10, v_max=100.0, backend="qubovert"):
    from .grid import env, S, Nx, Nv, gamma, not_done


    if A is None:
        A = np.empty([Nx, Nv])
        A[:, :] = 1 / 2

    value = np.zeros([Nx, Nv])

    y = value_to_binary_vector(-value, num_bits, v_max=v_max)

    history = []
    for anneal_duration in anneal_durations:
        S_new, r, _, _ = env.steps(A, S)
        discrete_state_new = discretize(S_new)
        L, b = to_sle(discrete_state_new, not_done, r)

        P = linear_equation_to_qubo(L.numpy(), b.numpy(), num_bits, v_max)

        if backend == "qubovert":
            y, loss = solve_qubo(P, x0=y, num_anneals=num_anneals, anneal_duration=anneal_duration)
        else:
            y, loss = solve_qubo_dimod(P, num_anneals=num_anneals, anneal_duration=anneal_duration,
                                       backend=backend)
        loss_sle = loss + (b ** 2).sum()
        history.append(loss_sle.item())

    return history

def to_index_p(state, x_min, v_min, dx, dv):
    x_l = torch.div(state[:, :, :, 0] - x_min, dx, rounding_mode='floor').long()
    p_x_l = 1 - ((state[:, :, :, 0] - x_min) / dx - x_l)
    v_l = torch.div(state[:, :, :, 1] - v_min, dv, rounding_mode='floor').long()
    p_v_l = 1 - ((state[:, :, :, 1] - v_min) / dv - v_l)
    return (x_l, v_l), (p_x_l, p_v_l)

def _resolve_device(device=None):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

