"""Figure 6 — accuracy of the SLE solution against the number of samples.

Paper configuration (caption of Fig. 6): the first iteration in QEPI, i.e. the
SLE built from the initial policy a = 1/2, solved on the standard D-Wave
sampler; accuracy estimated over 100 evaluations, with the Leap hybrid solver
shown for reference and shaded areas giving the standard deviation.

The annealing time is not stated in the caption; 20 us is the value used in the
original runs.  Note that --annealing-time is in microseconds here, unlike the
integer --anneal-duration of the emulator scripts.

Needs a configured Leap token (dwave config create).  The defaults are
4 x 100 = 400 QPU submissions plus --hybrid-repeats more.  Use --backend dimod
for a free classical dry run.
"""

import argparse

import numpy as np
from tqdm import tqdm

import qepi
from qepi.plotting import save_accuracy_curve, use_paper_style

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=4)
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--num-anneals", type=int, nargs="+", default=[1, 10, 100, 1000],
               help="number of samples per solve")
p.add_argument("--annealing-time", type=float, default=20.0, help="microseconds")
p.add_argument("--n-repeats", type=int, default=100, help="evaluations per sweep point")
p.add_argument("--hybrid-repeats", type=int, default=100,
               help="evaluations of the LeapHybrid reference; 0 skips it")
p.add_argument("--num-bits", type=int, default=10)
p.add_argument("--v-max", type=float, default=100.0)
p.add_argument("--backend", default="dwave", choices=["dwave", "dimod", "qubovert"])
args = p.parse_args()

qepi.configure(args.n, args.n, args.gamma)
use_paper_style()

losses = []
for _ in tqdm(range(args.n_repeats), desc=args.backend):
    res = qepi.estimate_loss_wrt_num_anneals(
        num_anneals=args.num_anneals,
        anneal_duration=args.annealing_time,
        num_bits=args.num_bits,
        v_max=args.v_max,
        backend=args.backend,
    )
    losses.append(res)
    np.save("data/loss_qepi_dwave_num_anneals", losses)

hybrid = []
for _ in tqdm(range(args.hybrid_repeats), desc="hybrid"):
    res = qepi.estimate_loss_wrt_num_anneals(
        num_anneals=[1],
        anneal_duration=args.annealing_time,
        num_bits=args.num_bits,
        v_max=args.v_max,
        backend="hybrid",
    )
    hybrid.append(res[0])
    np.save("data/loss_qepi_hybrid", hybrid)

# The stored losses are squared residuals; the figure shows the L2 norm.
losses = np.sqrt(np.array(losses))
mean, std = losses.mean(axis=0), losses.std(axis=0)

reference = None
if args.hybrid_repeats:
    h = np.sqrt(np.array(hybrid))
    reference = (h.mean(), h.std())

save_accuracy_curve(args.num_anneals, mean, std, reference=reference,
                    x_label="Number of samples", name="accuracy_dwave_samples")

for n, m, s in zip(args.num_anneals, mean, std):
    print(f"{n:>5} samples: L2 {m:.3f} +/- {s:.3f}")
if reference:
    print(f"LeapHybrid:    L2 {reference[0]:.3f} +/- {reference[1]:.3f}")
