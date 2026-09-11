import dimod
import numpy as np


def solve_qubo(P, x0=None, num_anneals=1, anneal_duration=1000, temperature_range=None):
    try:
        import qubovert as qv
        from qubovert.sim import anneal_qubo
    except ImportError as e:
        raise ImportError(
            "The qubovert backend is unavailable (no arm64 macOS wheels are "
            "published). Use backend='dimod' instead."
        ) from e

    scale = ((P ** 2).mean()) ** (1 / 2) + 1e-10
    model = qv.utils.matrix_to_qubo(P / scale)
    if x0 is not None:
        x0 = dict(zip(np.arange(len(P)), x0))
    result = anneal_qubo(model, num_anneals=num_anneals, initial_state=x0, anneal_duration=anneal_duration,
                         temperature_range=temperature_range)
    model_solution = result.best.state
    x = np.array([model_solution[i] for i in range(len(P))])
    loss = model.value(model_solution) * scale
    return x, loss


def solve_qubo_dimod(P, num_anneals=1, anneal_duration=20.0, backend="dimod"):
    scale = ((P ** 2).mean()) ** (1 / 2) + 1e-10
    Q = P / scale
    bqm = dimod.BinaryQuadraticModel(Q, dimod.BINARY)

    if backend == "dimod":
        sampler = dimod.SimulatedAnnealingSampler()
    elif backend in ("dwave", "clique", "hybrid"):
        try:
            from dwave.system import (DWaveSampler, DWaveCliqueSampler,
                                      EmbeddingComposite, LeapHybridSampler)
        except ImportError as e:
            raise ImportError(
                f"The '{backend}' backend needs the D-Wave Ocean SDK and a Leap "
                "token: pip install 'qepi[hardware]' && dwave config create. "
                "Use backend='dimod' for a free classical run."
            ) from e

        if backend == "dwave":
            sampler = EmbeddingComposite(DWaveSampler())
        elif backend == "clique":
            sampler = DWaveCliqueSampler()
        else:
            sampler = LeapHybridSampler()
    else:
        raise ValueError(f"unknown backend: {backend}")

    loss = np.inf
    if backend == "hybrid":
        sample_set = sampler.sample(bqm)
    elif backend == "dimod":
        sample_set = sampler.sample(bqm, num_reads=num_anneals,
                                    num_sweeps=int(anneal_duration))
    else:
        sample_set = sampler.sample(bqm, num_reads=num_anneals,
                                    annealing_time=anneal_duration)
    result = sample_set.lowest().first

    anneal_loss = result[1] * scale
    if anneal_loss <= loss:
        # x = np.array(list(result[0].values()))
        x = np.array([result[0][i] for i in range(len(P))])
        loss = anneal_loss

    return x, loss
