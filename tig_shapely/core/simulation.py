import numpy as np
import pymc as pm
import arviz as az
import scipy as sp
from itertools import combinations, chain
from typing import Dict, List, Set, Tuple, Union, Any

# Define actors
ACTORS = ["B", "CI", "AI", "CM"]  # Benchmarkers, Code Innovators, Algorithm Innovators, Challenge Maintainers

def powerset(iterable: List[str]) -> List[Tuple[str, ...]]:
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def compute_coalition_value(
    coalition: Tuple[str, ...], 
    alpha_labor: Dict[str, float],
    alpha_capital: Dict[str, float],
    gammas: Dict[str, float],
    delta: float,  # Labor elasticity
    kappa: float   # Capital elasticity
) -> float:
    if not coalition:
        return 0

    # Calculate labor contribution for each actor in the coalition
    labor = 0
    for actor in coalition:
        if actor in ["AI", "CM"]:
            # For actors with gamma factors
            labor += alpha_labor[actor] * (1 + gammas[actor])
        else:
            # For actors without gamma factors
            labor += alpha_labor[actor]
    
    # Calculate capital contribution for each actor in the coalition
    capital = 0
    for actor in coalition:
        capital += alpha_capital[actor]
    
    # Apply Cobb-Douglas function: L^delta * K^kappa
    return (labor ** delta) * (capital ** kappa)

def compute_shapley_values(
    alpha_labor: Dict[str, float],
    alpha_capital: Dict[str, float],
    gammas: Dict[str, float],
    delta: float,
    kappa: float
) -> Dict[str, float]:
    n = len(ACTORS)
    shapley_values = {actor: 0.0 for actor in ACTORS}
    all_coalitions = [set(coalition) for coalition in powerset(ACTORS)]
    
    for actor in ACTORS:
        for coalition in all_coalitions:
            if actor not in coalition:
                # Calculate the marginal contribution
                s = len(coalition)
                weight = (sp.special.factorial(s) * sp.special.factorial(n - s - 1)) / sp.special.factorial(n)
                
                # Convert sets to tuples for the compute_coalition_value function
                coalition_tuple = tuple(coalition)
                coalition_with_actor = tuple(coalition.union({actor}))
                
                marginal = compute_coalition_value(coalition_with_actor, alpha_labor, alpha_capital, gammas, delta, kappa) - \
                           compute_coalition_value(coalition_tuple, alpha_labor, alpha_capital, gammas, delta, kappa)
                
                shapley_values[actor] += weight * marginal
    
    return shapley_values

def sample_hmc_parameters(
    alpha_labor_priors: Dict[str, Tuple[float, float]],
    alpha_capital_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    delta_prior: Tuple[float, float],
    kappa_prior: Tuple[float, float],
    n_samples: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with pm.Model() as model:
        # Sample alpha labor (using lognormal to ensure positivity)
        alpha_labor = {}
        for actor, (mean, std) in alpha_labor_priors.items():
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            alpha_labor[actor] = pm.Lognormal(f"alpha_labor_{actor}", mu=mu, sigma=sigma)
        
        # Sample alpha capital (using lognormal to ensure positivity)
        alpha_capital = {}
        for actor, (mean, std) in alpha_capital_priors.items():
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            alpha_capital[actor] = pm.Lognormal(f"alpha_capital_{actor}", mu=mu, sigma=sigma)
        
        # Sample gammas (using lognormal to ensure positivity)
        gammas = {}
        for actor, (mean, std) in gamma_priors.items():
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            gammas[actor] = pm.Lognormal(f"gamma_{actor}", mu=mu, sigma=sigma)
        
        # Sample delta and kappa (using beta distribution bounded 0-1)
        delta_mean, delta_std = delta_prior
        var = delta_std**2
        if var >= delta_mean * (1 - delta_mean):
            var = 0.9 * delta_mean * (1 - delta_mean)
        
        alpha_param = delta_mean * (delta_mean * (1 - delta_mean) / var - 1)
        beta_param = (1 - delta_mean) * (delta_mean * (1 - delta_mean) / var - 1)
        
        # Ensure alpha and beta parameters are valid
        alpha_param = max(0.01, alpha_param)
        beta_param = max(0.01, beta_param)
        
        delta = pm.Beta("delta", alpha=alpha_param, beta=beta_param)
        
        kappa_mean, kappa_std = kappa_prior
        var = kappa_std**2
        if var >= kappa_mean * (1 - kappa_mean):
            var = 0.9 * kappa_mean * (1 - kappa_mean)
        
        alpha_param = kappa_mean * (kappa_mean * (1 - kappa_mean) / var - 1)
        beta_param = (1 - kappa_mean) * (kappa_mean * (1 - kappa_mean) / var - 1)
        
        # Ensure alpha and beta parameters are valid
        alpha_param = max(0.01, alpha_param)
        beta_param = max(0.01, beta_param)
        
        kappa = pm.Beta("kappa", alpha=alpha_param, beta=beta_param)
        
        # Use No-U-Turn Sampler (NUTS) for efficient HMC sampling
        trace = pm.sample(n_samples, tune=1000, chains=2, cores=1, return_inferencedata=True)
    
    # Extract samples from the trace
    alpha_labor_samples = {actor: trace.posterior[f"alpha_labor_{actor}"].values.flatten() for actor in ACTORS}
    alpha_capital_samples = {actor: trace.posterior[f"alpha_capital_{actor}"].values.flatten() for actor in ACTORS}
    gamma_samples = {actor: trace.posterior[f"gamma_{actor}"].values.flatten() for actor in ["AI", "CM"]}
    delta_samples = trace.posterior["delta"].values.flatten()
    kappa_samples = trace.posterior["kappa"].values.flatten()
    
    return alpha_labor_samples, alpha_capital_samples, gamma_samples, delta_samples, kappa_samples

def compute_hmc_shapley_values(
    alpha_labor_priors: Dict[str, Tuple[float, float]],
    alpha_capital_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    delta_prior: Tuple[float, float],
    kappa_prior: Tuple[float, float],
    n_samples: int
) -> Dict[str, np.ndarray]:
    # Sample parameters using HMC
    alpha_labor, alpha_capital, gammas, delta, kappa = sample_hmc_parameters(
        alpha_labor_priors, alpha_capital_priors, gamma_priors, 
        delta_prior, kappa_prior, n_samples
    )
    
    # Truncate to the requested number of samples
    n_samples = min(n_samples, len(delta))
    
    # Initialize arrays for Shapley values
    shapley_values = {actor: np.zeros(n_samples) for actor in ACTORS}
    
    # Compute Shapley values for each sample
    for i in range(n_samples):
        # Extract parameters for this sample
        alpha_labor_sample = {actor: alpha_labor[actor][i] for actor in ACTORS}
        alpha_capital_sample = {actor: alpha_capital[actor][i] for actor in ACTORS}
        gamma_sample = {actor: gammas[actor][i] for actor in ["AI", "CM"]}
        delta_sample = delta[i]
        kappa_sample = kappa[i]
        
        # Compute Shapley values for this sample
        shapley_values_sample = compute_shapley_values(
            alpha_labor_sample, alpha_capital_sample, gamma_sample, 
            delta_sample, kappa_sample
        )
        
        # Store results
        for actor in ACTORS:
            shapley_values[actor][i] = shapley_values_sample[actor]
    
    return shapley_values

def maximize_and_quantify_uncertainty(
    alpha_labor_priors: Dict[str, Tuple[float, float]],
    alpha_capital_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    delta_prior: Tuple[float, float],
    kappa_prior: Tuple[float, float],
    n_samples: int
) -> Tuple[Dict, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Find optimal parameter values that maximize total value creation, 
    then sample around the optimum to quantify uncertainty.
    
    Returns:
        - map_estimate: The optimal parameter values
        - shapley_values_optimal: Shapley values at the optimum
        - shapley_values_uncertainty: Distribution of Shapley values around the optimum
    """
    with pm.Model() as model:
        # Sample alpha labor (using lognormal to ensure positivity)
        alpha_labor = {}
        for actor, (mean, std) in alpha_labor_priors.items():
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            alpha_labor[actor] = pm.Lognormal(f"alpha_labor_{actor}", mu=mu, sigma=sigma)
        
        # Sample alpha capital (using lognormal to ensure positivity)
        alpha_capital = {}
        for actor, (mean, std) in alpha_capital_priors.items():
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            alpha_capital[actor] = pm.Lognormal(f"alpha_capital_{actor}", mu=mu, sigma=sigma)
        
        # Sample gammas (using lognormal to ensure positivity)
        gammas = {}
        for actor, (mean, std) in gamma_priors.items():
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            gammas[actor] = pm.Lognormal(f"gamma_{actor}", mu=mu, sigma=sigma)
        
        # Sample delta and kappa (using beta distribution bounded 0-1)
        delta_mean, delta_std = delta_prior
        var = delta_std**2
        if var >= delta_mean * (1 - delta_mean):
            var = 0.9 * delta_mean * (1 - delta_mean)
        
        alpha_param = delta_mean * (delta_mean * (1 - delta_mean) / var - 1)
        beta_param = (1 - delta_mean) * (delta_mean * (1 - delta_mean) / var - 1)
        
        # Ensure alpha and beta parameters are valid
        alpha_param = max(0.01, alpha_param)
        beta_param = max(0.01, beta_param)
        
        delta = pm.Beta("delta", alpha=alpha_param, beta=beta_param)
        
        kappa_mean, kappa_std = kappa_prior
        var = kappa_std**2
        if var >= kappa_mean * (1 - kappa_mean):
            var = 0.9 * kappa_mean * (1 - kappa_mean)
        
        alpha_param = kappa_mean * (kappa_mean * (1 - kappa_mean) / var - 1)
        beta_param = (1 - kappa_mean) * (kappa_mean * (1 - kappa_mean) / var - 1)
        
        # Ensure alpha and beta parameters are valid
        alpha_param = max(0.01, alpha_param)
        beta_param = max(0.01, beta_param)
        
        kappa = pm.Beta("kappa", alpha=alpha_param, beta=beta_param)
        
        # Define a potential function that increases with total value (for maximization)
        def total_value_potential(_alpha_labor=alpha_labor, _alpha_capital=alpha_capital, 
                             _gammas=gammas, _delta=delta, _kappa=kappa):
            # Convert gammas dict to include all actors (with 0 for B and CI)
            full_gammas = {actor: _gammas.get(actor, 0.0) for actor in ACTORS}
            
            # Calculate total value for all actors combined (grand coalition)
            all_actors = tuple(ACTORS)
            val = compute_coalition_value(all_actors, _alpha_labor, _alpha_capital, 
                                        full_gammas, _delta, _kappa)
            return val  # PyMC will try to maximize this value
        
        # Create the potential
        pm.Potential("total_value", total_value_potential())
        
        # First find MAP estimate (maximization)
        map_estimate = pm.find_MAP()
        
        # Then sample around MAP to quantify uncertainty
        trace = pm.sample(n_samples, tune=1000, chains=2, cores=1, 
                         return_inferencedata=True, start=map_estimate)
    
    # Extract optimal parameter values
    optimal_params = {
        'alpha_labor': {actor: map_estimate[f'alpha_labor_{actor}'] for actor in ACTORS},
        'alpha_capital': {actor: map_estimate[f'alpha_capital_{actor}'] for actor in ACTORS},
        'gammas': {actor: map_estimate[f'gamma_{actor}'] for actor in ['AI', 'CM']},
        'delta': map_estimate['delta'],
        'kappa': map_estimate['kappa']
    }
    
    # Calculate Shapley values at the optimum
    optimal_gammas = {actor: optimal_params['gammas'].get(actor, 0.0) for actor in ACTORS}
    shapley_values_optimal = compute_shapley_values(
        optimal_params['alpha_labor'], 
        optimal_params['alpha_capital'], 
        optimal_gammas,
        optimal_params['delta'], 
        optimal_params['kappa']
    )
    
    # Calculate Shapley values distribution around the optimum
    # Extract samples from the trace (similar to original code)
    n_samples = min(n_samples, len(trace.posterior['delta'].values.flatten()))
    shapley_values_uncertainty = {actor: np.zeros(n_samples) for actor in ACTORS}
    
    # Sample Shapley values around the optimum
    for i in range(n_samples):
        alpha_labor_sample = {actor: trace.posterior[f'alpha_labor_{actor}'].values.flatten()[i] for actor in ACTORS}
        alpha_capital_sample = {actor: trace.posterior[f'alpha_capital_{actor}'].values.flatten()[i] for actor in ACTORS}
        gamma_sample = {actor: trace.posterior[f'gamma_{actor}'].values.flatten()[i] for actor in ['AI', 'CM']}
        full_gamma_sample = {actor: gamma_sample.get(actor, 0.0) for actor in ACTORS}
        delta_sample = trace.posterior['delta'].values.flatten()[i]
        kappa_sample = trace.posterior['kappa'].values.flatten()[i]
        
        # Compute Shapley values for this sample
        shapley_values_sample = compute_shapley_values(
            alpha_labor_sample, alpha_capital_sample, full_gamma_sample, 
            delta_sample, kappa_sample
        )
        
        # Store results
        for actor in ACTORS:
            shapley_values_uncertainty[actor][i] = shapley_values_sample[actor]
    
    return optimal_params, shapley_values_optimal, shapley_values_uncertainty 