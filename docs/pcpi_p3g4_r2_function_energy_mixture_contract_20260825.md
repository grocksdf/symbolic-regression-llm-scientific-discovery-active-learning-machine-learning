# PCPI P3G.4 R-squared function-energy mixture contract

P3G.3 is immutable negative development evidence. It completed 24 audits,
rejected 12, opened no candidate or held-out response, and ran no acquisition.
Its stronger coefficient shrinkage did not narrow the predictive law: signal
variation was transferred into the shared noise scale. Fixed function energy
is therefore rejected as the repair direction.

P3G.4 represents total standardized function signal-to-noise ratio as a proper
latent state. Before response access it applies three-point Gauss--Legendre
quadrature to `R^2 ~ Uniform(0,1)`, then maps every node through
`g=R^2/(1-R^2)`. For a structure with `p` standardized non-intercept terms,
each coefficient has conditional precision `p/g`. The quadrature weights are
the state prior probabilities. Structure, discrepancy, noise law and function
energy are normalized in one ordinary posterior and all enter the acquisition
nuisance target.

This is a frozen finite mixture, not an empirical-Bayes selection of a penalty.
It follows the principle that uncertainty about regression signal scale should
be integrated rather than fixed:

- Liang et al., *Mixtures of g Priors for Bayesian Variable Selection*, JASA
  2008: <https://www2.stat.duke.edu/~clyde/BAS/gpriors.pdf>
- Zhang et al., *Bayesian Regression Using a Prior on the Model Fit: The R2-D2
  Shrinkage Prior*, JASA 2022: <https://arxiv.org/abs/1609.00046>

The real-data boundary is unchanged: identical datasets, splits, seeds,
budgets, row order, PIT betting family, familywise allocation and fail-closed
rule. Candidate responses remain sealed until a 24/24 nonrejection result;
held-out remains closed throughout.
