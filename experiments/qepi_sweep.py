"""Figure 8 — QEPI accuracy over annealing duration and number of anneals.

Paper configuration (caption of Fig. 8): accuracy estimated over 1000 algorithm
runs at the 10-th policy update step, for annealing durations 10 to 1280 and
1 to 128 anneals, on the 4 x 4 grid with n_b = 10 and x_min = -100.

That is 64 x 1000 = 64000 QEPI runs, hours on one core.  Start with
--n-runs 20 to check the pipeline.  Results are written after every completed
row, so an interrupted sweep still loads.
"""

import argparse

import numpy as np
from tqdm import tqdm

import qepi
from qepi.plotting import save_training_data, use_paper_style

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--num-anneals", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
p.add_argument("--anneal-durations", type=int, nargs="+",
               default=[10, 20, 40, 80, 160, 320, 640, 1280])
p.add_argument("--n-runs", type=int, default=1000)
p.add_argument("--n-steps", type=int, default=10, help="policy update steps")
p.add_argument("--num-bits", type=int, default=10)
p.add_argument("--v-max", type=float, default=100.0)
p.add_argument("--out", default="data/training_qepi_1000")
p.add_argument("--backend", default="qubovert",
               choices=["qubovert", "dimod", "dwave", "hybrid"])
args = p.parse_args()

qepi.configure(args.n, args.n, args.gamma)
use_paper_style()

# The reference policy is computed classically, once.
A_sol, _ = qepi.policy_iteration(n_steps=100)

history_all_full = []
for num_anneals in tqdm(args.num_anneals, desc="anneals"):
    history_all = []
    for anneal_duration in args.anneal_durations:
        history = []
        for _ in range(args.n_runs):
            res = qepi.evaluate_annealing(
                A_sol, n_steps=args.n_steps, num_anneals=num_anneals,
                anneal_duration=anneal_duration, num_bits=args.num_bits,
                v_max=args.v_max, backend=args.backend,
            )
            history.append(res)
        history_all.append(history)
    history_all_full.append(history_all)
    np.save(args.out, history_all_full)

# (num_anneals, durations, runs, steps) -> accuracy at the last update step
data = np.array(history_all_full).mean(axis=2)
accuracy = data[:, :, -1]

n_a, n_d = len(args.num_anneals), len(args.anneal_durations)
save_training_data((-0.5, n_a - 0.5, -0.5, n_d - 0.5), accuracy,
                   name="training_qepi", x_label="Number of anneals",
                   y_label="Annealing duration", cmap="viridis", norm_gamma=1,
                   ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], xticklabels=args.num_anneals,
                   yticklabels=args.anneal_durations)

print("accuracy at the final update step:")
print(np.round(accuracy, 3))
