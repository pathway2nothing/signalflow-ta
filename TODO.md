# Novel Indicator Research: Cross-Disciplinary Areas for Financial Time Series

## Candidate Areas

### 1. Nonlinear Dynamics & Complexity Science
**Techniques:** Sample entropy, permutation entropy, Lyapunov exponent, recurrence quantification analysis, detrended fluctuation analysis, multifractal spectrum.

**Relevance:** Financial markets are complex adaptive systems. These measures detect regime changes, quantify predictability, and distinguish trending from mean-reverting phases at a fundamental level. They capture nonlinear structure invisible to all linear indicators (MA, RSI, correlation).

**Status:** **SELECTED** — 5 indicators implemented in `stat/complexity.py`.

---

### 2. Information Theory & Information Geometry
**Techniques:** Fisher information, KL divergence, mutual information, transfer entropy, Rényi entropy, relative information gain.

**Relevance:** Measures distribution sharpness, directional information flow between series, and regime transitions. Fisher information uniquely spikes at phase transitions. Transfer entropy captures lead-lag relationships between assets without assuming linearity.

**Status:** **SELECTED** — Fisher Information in `stat/complexity.py`, 5 additional indicators in `stat/information.py`.

## Implemented Indicators — Information Theory

| Indicator | Source Field | What It Measures | Key Insight |
|-----------|-------------|-----------------|-------------|
| **Fisher Information** | Frieden (2004) | Distribution sharpness | Spikes at regime transitions |
| **KL Divergence** | Kullback & Leibler (1951) | Asymmetric distributional divergence (recent vs baseline) | Detects regime deviations from historical norm |
| **Jensen-Shannon Divergence** | Lin (1991) | Symmetric bounded distributional divergence | Stable regime change metric, bounded [0,1] |
| **Rényi Entropy** | Rényi (1961) | Generalized entropy with tunable tail sensitivity | α<1 emphasizes rare events; α>1 emphasizes dominant regime |
| **Auto Mutual Information** | Fraser & Swinney (1986) | Nonlinear temporal dependency | Captures nonlinear structure invisible to autocorrelation |
| **Relative Information Gain** | Schreiber (2000) | Rate of distributional change | High = rapid regime shift; low = stable regime |

---

### 3. Digital Signal Processing / Acoustics
**Techniques:** Spectral flux, zero-crossing rate, spectral rolloff, MFCCs (Mel-frequency cepstral coefficients), spectral flatness (Wiener entropy), chromagram.

**Relevance:** Audio analysis techniques for detecting "timbre" changes in time series. Spectral flux measures how quickly the frequency content changes (regime shift detector). Spectral flatness distinguishes tonal (trending) from noisy (choppy) signals. Partially covered already (spectral centroid, spectral entropy).

**Status:** Candidate for future batch.

---

### 4. Topological Data Analysis (TDA)
**Techniques:** Persistent homology, Betti numbers, persistence landscapes, Wasserstein distance between persistence diagrams.

**Relevance:** Captures shape-based patterns in price trajectories that are invariant to stretching and deformation. Can detect "loops" (oscillations) and "holes" in the attractor space. Mathematically rigorous but computationally expensive. Emerging field in quantitative finance.

**Status:** Candidate for future batch. Requires `ripser` or `giotto-tda` dependency.

---

### 5. Survival Analysis / Extreme Value Theory
**Techniques:** Hazard rate estimation, extremal index, generalized Pareto distribution fitting, return level estimation, peaks-over-threshold analysis.

**Relevance:** Models tail risk and clustering of extreme events. The extremal index measures how much extremes cluster together (vs. occur independently). Hazard rate estimates the conditional probability of a large move given recent history.

**Status:** Candidate for future batch.

---

### 6. Control Theory & Systems Engineering
**Techniques:** Kalman filter innovations (prediction residuals), controllability/observability metrics, Lyapunov stability analysis, PID error signals, system identification (ARX/ARMAX).

**Relevance:** Treats price as output of a dynamical system. Kalman innovations measure how "surprised" an optimal filter is — spikes indicate model breakdown. System identification parameters track changing market dynamics.

**Status:** **SELECTED** — 5 indicators implemented in `stat/control.py`.

## Implemented Indicators — Control Theory

| Indicator | Source Field | What It Measures | Key Insight |
|-----------|-------------|-----------------|-------------|
| **Kalman Innovation** | Harvey (1989) | Normalized innovation statistic from 1-D Kalman filter | NIS >> 1 signals regime shift / model breakdown |
| **AR Coefficient** | Ljung (1999) | First autoregressive coefficient via rolling OLS | Positive = momentum; negative = mean-reversion; tracking = system identification |
| **Lyapunov Exponent** | Rosenstein et al. (1993) | Maximum Lyapunov exponent via phase-space embedding | MLE > 0 = chaos; MLE < 0 = convergent dynamics |
| **PID Error** | Astrom & Murray (2008) | RMS of PID composite tracking error | High = breakout/regime shift; low = stable equilibrium |
| **Prediction Error Decomposition** | Geman et al. (1992) | Bias ratio of linear prediction errors | High = systematic model failure; low = noise-dominated |

---

### 7. Ecology / Biodiversity Metrics
**Techniques:** Simpson's diversity index, Shannon-Wiener index (on discretized returns), species richness analogues, Pielou's evenness, rank-abundance distributions.

**Relevance:** Measures "diversity" of market behavior states. When applied to discretized return magnitudes, captures whether the market exhibits a few dominant behaviors or many equally-weighted ones. Evenness measures are simple, interpretable, and fast.

**Status:** Candidate for future batch.

---

## Selection Rationale

**Nonlinear Dynamics & Complexity Science** was selected because:

1. **Strongest theoretical foundation** for financial markets (complex adaptive systems literature)
2. **Detects regime transitions before they manifest** in price or volatility
3. **Captures nonlinear structure** invisible to all linear indicators
4. **Complements existing indicators** — the codebase has basic entropy but lacks robust variants (permutation entropy, sample entropy) and proper complexity measures
5. **Well-validated** in econophysics literature (hundreds of papers)
6. **No additional dependencies** — pure NumPy implementations
7. **Practical for real-time use** — all O(n) or O(n²) within small windows

## Implemented Indicators

| Indicator | Source Field | What It Measures | Key Insight |
|-----------|-------------|-----------------|-------------|
| **Permutation Entropy** | Bandt & Pompe (2002) | Ordinal pattern diversity | Robust complexity; dropping PE = emerging trend |
| **Sample Entropy** | Richman & Moorman (2000) | Regularity / self-similarity | Low = predictable pattern; high = noise |
| **Lempel-Ziv Complexity** | Lempel & Ziv (1976) | Algorithmic compressibility | LZC << 1 = exploitable structure |
| **Fisher Information** | Frieden (2004) | Distribution sharpness | Spikes at regime transitions |
| **DFA Exponent** | Peng et al. (1994) | Long-range correlations (generalized Hurst) | Handles non-stationarity unlike R/S Hurst |
