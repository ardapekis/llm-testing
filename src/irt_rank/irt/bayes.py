"""Optional NumPyro/NUTS calibration for item-parameter uncertainty."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from irt_rank.irt.model import IRTModel

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BayesResult:
    """Posterior samples returned by the optional full-Bayes path."""

    samples: dict[str, FloatArray]
    model: IRTModel


def fit_full_bayes(
    responses: npt.ArrayLike,
    model: IRTModel | str = IRTModel.TWO_PL,
    *,
    seed: int = 0,
    warmup: int = 500,
    samples: int = 500,
    chains: int = 1,
    progress_bar: bool = False,
) -> BayesResult:
    """Fit a unidimensional IRT model with NumPyro NUTS.

    NumPyro and JAX are optional dependencies. Importing the core package does
    not import either runtime.
    """

    try:
        jnp: Any = importlib.import_module("jax.numpy")
        random: Any = importlib.import_module("jax.random")
        numpyro: Any = importlib.import_module("numpyro")
        dist: Any = importlib.import_module("numpyro.distributions")
        infer: Any = importlib.import_module("numpyro.infer")
    except ImportError as error:
        raise RuntimeError("install irt-rank[bayes] to use full Bayes") from error

    selected_model = IRTModel(model)
    matrix = np.asarray(responses, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("responses must be a model-by-item matrix")
    observed = np.isfinite(matrix)
    if not bool(np.isin(matrix[observed], (0.0, 1.0)).all()):
        raise ValueError("observed responses must be binary")
    filled = np.where(observed, matrix, 0.0)
    n_models, n_items = matrix.shape

    def irt_model() -> None:
        theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand((n_models,)))
        difficulty = numpyro.sample("difficulty", dist.Normal(0.0, 1.5).expand((n_items,)))
        if selected_model is IRTModel.ONE_PL:
            discrimination = jnp.ones((n_items,))
        else:
            log_discrimination = numpyro.sample(
                "log_discrimination", dist.Normal(0.0, 0.5).expand((n_items,))
            )
            discrimination = jnp.exp(log_discrimination)
        if selected_model is IRTModel.THREE_PL:
            guessing = numpyro.sample("guessing", dist.Beta(2.0, 8.0).expand((n_items,)))
        else:
            guessing = jnp.zeros((n_items,))
        logistic = jnp.asarray(1.0) / (
            1.0 + jnp.exp(-discrimination[None, :] * (theta[:, None] - difficulty[None, :]))
        )
        probs = guessing[None, :] + (1.0 - guessing[None, :]) * logistic
        with numpyro.handlers.mask(mask=jnp.asarray(observed)):
            numpyro.sample("responses", dist.Bernoulli(probs=probs), obs=jnp.asarray(filled))

    sampler: Any = infer.MCMC(
        infer.NUTS(irt_model),
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        progress_bar=progress_bar,
    )
    sampler.run(random.PRNGKey(seed))
    posterior = {
        name: np.asarray(value, dtype=np.float64)
        for name, value in sampler.get_samples(group_by_chain=False).items()
    }
    return BayesResult(posterior, selected_model)
