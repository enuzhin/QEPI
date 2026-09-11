"""Figures 5 and 9 — QEPI against value iteration on the coarse mesh.

Paper configuration (captions of Figs. 5 and 9): 10 policy update steps on a
4 x 4 grid, gamma = 0.99, n_b = 10 bits, x_min = -100, 100 anneals per policy
update, 1280 annealing duration steps.

With mu = 16 states and n_b = 10 bits the QUBO has mu * n_b = 160 binary
variables, matching Eq. (23).
"""

import argparse

import numpy as np

import qepi
from qepi.discretization import discretize, to_sle
from qepi.plotting import save_policy, save_value, use_paper_style
from qepi.qubo import check_encoding

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=4, help="grid nodes per axis")
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--n-steps", type=int, default=10, help="policy update steps")
p.add_argument("--num-bits", type=int, default=10, help="n_b")
p.add_argument("--v-max", type=float, default=100.0, help="|x_min|, binarisation range")
p.add_argument("--num-anneals", type=int, default=100, help="anneals per policy update")
p.add_argument("--anneal-duration", type=int, default=1280, help="annealing duration steps")
p.add_argument("--backend", default="qubovert",
               choices=["qubovert", "dimod", "dwave", "hybrid"])
p.add_argument("--verify", action="store_true",
               help="check the QUBO encoding identity before running")
args = p.parse_args()

qepi.configure(args.n, args.n, args.gamma)
use_paper_style()

X, V = qepi.grid.X, qepi.grid.V

# Classical reference: the value iteration solution QEPI is validated against.
A_ref, value_ref = qepi.policy_iteration(n_steps=100)

if args.verify:
    # Rebuild the SLE for the reference policy and check that the QUBO energy
    # reproduces the squared residual of the linear system.
    S_new, r, _, _ = qepi.grid.env.steps(A_ref, qepi.grid.S)
    L, b = to_sle(discretize(S_new), qepi.grid.not_done, r)
    energy, residual = check_encoding(L, b, value_ref, args.num_bits, args.v_max)
    print(f"encoding check: {energy:.6e} vs {residual:.6e}")

A, value, history = qepi.qepi(
    n_steps=args.n_steps,
    num_bits=args.num_bits,
    v_max=args.v_max,
    num_anneals=args.num_anneals,
    anneal_duration=args.anneal_duration,
    backend=args.backend,
)

np.save("data/qepi_simulator_history", history)
np.save("data/qepi_simulator_value", value)
np.save("data/qepi_simulator_policy", A)
np.save("data/vi_reference_value", value_ref)
np.save("data/vi_reference_policy", A_ref)

# Figure 5: the two policies.  Figure 9: the two value functions.
save_policy(X, V, A / 2, name="pi_qepi", cmap="PRGn",
            ticks=[0, 0.5, 1], ticklabels=["Left", "None", "Right"],
            vmin=-0.12, vmax=1.12, dpi=600)
save_policy(X, V, A_ref / 2, name="pi_vi", cmap="PRGn",
            ticks=[0, 0.5, 1], ticklabels=["Left", "None", "Right"],
            vmin=-0.12, vmax=1.12, dpi=600)
save_value(X, V, value, name="value_qepi", cmap="magma", norm_gamma=1,
           ticks=None, dpi=600)
save_value(X, V, value_ref, name="value_vi", cmap="magma", norm_gamma=1,
           ticks=None, dpi=600)

print(f"binarisation step: {args.v_max / (2 ** args.num_bits - 1):.4f}")
print(f"QUBO variables:    {args.n * args.n * args.num_bits}")
print(f"policy matches VI: {np.array_equal(A, A_ref)}")
print(f"max |V_QEPI - V_VI|: {np.abs(value - value_ref).max():.3f}")
