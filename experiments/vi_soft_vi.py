"""Figures 3 and 4 — value iteration vs soft value iteration.

Paper configuration (captions of Figs. 3 and 4): 400 timesteps on a grid of
6000 x 6000 units, i.e. 6001 nodes per axis, gamma = 0.99, smoothing sigma = 10
grid cells.  sigma = 0 gives plain VI, sigma = 10 gives the soft version; the
two values produce the two panels of each figure.

The kernel size is not stated in the paper.  51 is used here because it gives a
radius of 2.5 sigma at sigma = 10, so the Gaussian is not truncated in a way
that would change the effective smoothing width.

The fine grid needs a GPU and a lot of memory.  For a quick look:
    --n 1024 --n-iters 400 --sigmas 0 2
"""

import argparse

import torch

import qepi
from qepi.plotting import save_policy, save_value, use_paper_style

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--n", type=int, default=6001, help="grid nodes per axis")
p.add_argument("--gamma", type=float, default=0.99)
p.add_argument("--n-iters", type=int, default=400, help="value iteration steps")
p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 10.0],
               help="blur widths in grid cells")
p.add_argument("--kernel", type=int, default=51, help="GaussianBlur kernel size")
p.add_argument("--downsample", type=int, default=1024,
               help="area-average down to this size before plotting")
p.add_argument("--device", default=None, help="cuda, mps or cpu; autodetected by default")

args = p.parse_args()

qepi.configure(args.n, args.n, args.gamma)
use_paper_style()

# save_policy and save_value take the extent from X.min()/X.max() only, so the
# mesh does not have to be resampled along with the data.
X, V = qepi.grid.X, qepi.grid.V


def area_resize(t, size):
    """Area interpolation, as used to bring the fine grid down for plotting."""
    if t.shape[-1] <= size:
        return t
    return torch.nn.functional.interpolate(
        t.unsqueeze(0).unsqueeze(0).float(), size=size, mode="area"
    )[0, 0]


for sigma in args.sigmas:
    tag = f"s{sigma:g}"
    value, pi = qepi.soft_vi(n_iters=args.n_iters, sigma=sigma,
                             kernel=args.kernel, device=args.device)

    torch.save(value, f"data/QValue_{tag}.pth")
    torch.save(pi, f"data/QPolicy_{tag}.pth")

    values = area_resize(value.cpu(), args.downsample).numpy()
    pis = area_resize(pi.cpu(), args.downsample).numpy()

    save_policy(X, V, pis / 2, name=f"pi_soft_vi_{tag}", cmap="PRGn",
                ticks=[0, 0.5, 1], ticklabels=["Left", "None", "Right"],
                vmin=-0.12, vmax=1.12, dpi=600)
    save_value(X, V, values, name=f"value_soft_vi_{tag}",
               cmap="magma", norm_gamma=1, dpi=600)

    print(f"sigma={sigma:g}: value in [{values.min():.1f}, {values.max():.1f}]")
