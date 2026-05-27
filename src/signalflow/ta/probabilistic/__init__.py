"""Probabilistic features — emit calibrated probabilities in ``[0, 1]``.

Unlike conventional TA features (raw z-scores, returns, volatilities) these
features output values in the probability simplex, suitable for direct use
as inputs to soft-labeling pipelines, calibration-aware models, and
probability-space MI estimators.

All features are causal and warmup-invariant: bar T's output depends only
on bars ≤ T, and after the warmup window completes the output is
identical regardless of where the input series begins.

Research provenance: iter-34 (gmm_volreg) and iter-35 (extended_gmm,
bayesian_shrinkage_z, posterior_reversal) showed these emit ~3× more
predictive structure for soft labels than raw measurements.
"""
from signalflow.ta.probabilistic.bayesian_shrinkage import BayesianShrinkageZscore
from signalflow.ta.probabilistic.gmm_regime import GMMVolRegime3State, GMMVolRegime5State
from signalflow.ta.probabilistic.posterior_reversal import PosteriorReversalProb

__all__ = [
    "BayesianShrinkageZscore",
    "GMMVolRegime3State",
    "GMMVolRegime5State",
    "PosteriorReversalProb",
]
