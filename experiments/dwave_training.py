"""Figure 7 — convergence of QEPI on D-Wave hardware.

Paper configuration (caption of Fig. 7): QEPI implemented with D-Wave's Leap
hybrid algorithm solving the SLE, the fraction of optimal policies estimated
over 40 consecutive algorithm runs, on the 4 x 4 grid with n_b = 10 and
x_min = -100.

The hybrid solver combines the QPU with proprietary classical heuristics, so
this demonstrates that QEPI converges at the algorithmic level on real
hardware — it is not a measurement of quantum speed-up.

Each policy update is one submission, so the defaults are 40 x 10 = 400
submissions.  Use --backend dimod for a free classical dry run.
"""

import argparse

import numpy as np
from tqdm import tqdm

import qepi
from qepi.plotting import save_optimality_bars, use_paper_style

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--n-runs", type=int, default=40, help="consecutive algorithm runs")
p.add_argument("--n-steps", type=int, default=10, help="policy update steps")
p.add_argument("--num-anneals", type=int, default=1, help="ignored by the hybrid solver")
p.add_argument("--annealing-time", type=float, default=20.0,
               help="microseconds; ignored by the hybrid solver")
p.add_argument("--num-bits", type=int, default=10)
p.add_argument("--v-max", type=float, default=100.0)
p.add_argument("--backend", default="hybrid", choices=["hybrid", "dwave", "dimod"])
p.add_argument("--out", default="data/training_qepi_hybrid")
args = p.parse_args()

qepi.configure(args.n, args.n, args.gamma)
use_paper_style()

# The reference policy is computed classically, once.
A_sol, _ = qepi.policy_iteration(n_steps=100)

history = []
for _ in tqdm(range(args.n_runs), desc=args.backend):
    res = qepi.evaluate_annealing(
        A_sol,
        n_steps=args.n_steps,
        num_anneals=args.num_anneals,
        anneal_duration=args.annealing_time,
        num_bits=args.num_bits,
        v_max=args.v_max,
        backend=args.backend,
    )
    history.append(res)
    np.save(args.out, history)

# Step 0 is the policy before any update and is never optimal; the bar chart
# starts there so that convergence is visible from step 1 on.
freq = np.r_[0.0, np.array(history).mean(axis=0)]
save_optimality_bars(freq, name="training_dwave")

print("fraction of optimal policies per update step:")
print(np.round(freq, 3))
