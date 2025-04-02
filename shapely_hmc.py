import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
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
    alphas: Dict[str, float], 
    gammas: Dict[str, float],
    betas: Dict[str, float],
    k: float
) -> float:
    if not coalition:
        return 0

    # Initialize the product for Cobb-Douglas
    product = k
    
    # Calculate the contribution of each actor in the coalition
    for actor in coalition:
        if actor in ["AI", "CM"]:
            # For actors with gamma factors
            product *= (alphas[actor] * (1 + gammas[actor])) ** betas[actor]
        else:
            # For actors without gamma factors
            product *= alphas[actor] ** betas[actor]
    
    return product

def compute_shapley_values(
    alphas: Dict[str, float], 
    gammas: Dict[str, float],
    betas: Dict[str, float],
    k: float
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
                
                marginal = compute_coalition_value(coalition_with_actor, alphas, gammas, betas, k) - \
                           compute_coalition_value(coalition_tuple, alphas, gammas, betas, k)
                
                shapley_values[actor] += weight * marginal
    
    return shapley_values

def sample_hmc_parameters(
    alpha_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    beta_priors: Dict[str, Tuple[float, float]],
    k_prior: Tuple[float, float],
    n_samples: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    with pm.Model() as model:
        # Sample alphas (using lognormal to ensure positivity)
        alphas = {}
        for actor, (mean, std) in alpha_priors.items():
            # Convert normal mean, std to lognormal parameters
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            alphas[actor] = pm.Lognormal(f"alpha_{actor}", mu=mu, sigma=sigma)
        
        # Sample gammas (using lognormal to ensure positivity)
        gammas = {}
        for actor, (mean, std) in gamma_priors.items():
            # Convert normal mean, std to lognormal parameters
            mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            gammas[actor] = pm.Lognormal(f"gamma_{actor}", mu=mu, sigma=sigma)
        
        # Sample betas (using beta distribution bounded 0-1)
        betas = {}
        for actor, (mean, std) in beta_priors.items():
            # Convert mean and std to alpha and beta parameters
            var = std**2
            if var >= mean * (1 - mean):
                var = 0.9 * mean * (1 - mean)
            
            alpha_param = mean * (mean * (1 - mean) / var - 1)
            beta_param = (1 - mean) * (mean * (1 - mean) / var - 1)
            
            # Ensure alpha and beta parameters are valid
            alpha_param = max(0.01, alpha_param)
            beta_param = max(0.01, beta_param)
            
            betas[actor] = pm.Beta(f"beta_{actor}", alpha=alpha_param, beta=beta_param)
        
        # Sample k (using lognormal to ensure positivity)
        k_mean, k_std = k_prior
        mu = np.log(k_mean**2 / np.sqrt(k_mean**2 + k_std**2))
        sigma = np.sqrt(np.log(1 + (k_std**2 / k_mean**2)))
        k = pm.Lognormal("k", mu=mu, sigma=sigma)
        
        # Use No-U-Turn Sampler (NUTS) for efficient HMC sampling
        trace = pm.sample(n_samples, tune=1000, chains=2, cores=1, return_inferencedata=True)
    
    # Extract samples from the trace
    alpha_samples = {actor: trace.posterior[f"alpha_{actor}"].values.flatten() for actor in ACTORS}
    gamma_samples = {actor: trace.posterior[f"gamma_{actor}"].values.flatten() for actor in ["AI", "CM"]}
    beta_samples = {actor: trace.posterior[f"beta_{actor}"].values.flatten() for actor in ACTORS}
    k_samples = trace.posterior["k"].values.flatten()
    
    return alpha_samples, gamma_samples, beta_samples, k_samples

def compute_hmc_shapley_values(
    alpha_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    beta_priors: Dict[str, Tuple[float, float]],
    k_prior: Tuple[float, float],
    n_samples: int
) -> Dict[str, np.ndarray]:
    # Sample parameters using HMC
    alphas, gammas, betas, k = sample_hmc_parameters(
        alpha_priors, gamma_priors, beta_priors, k_prior, n_samples
    )
    
    # Truncate to the requested number of samples
    n_samples = min(n_samples, len(k))
    
    # Initialize arrays for Shapley values
    shapley_values = {actor: np.zeros(n_samples) for actor in ACTORS}
    
    # Compute Shapley values for each sample
    for i in range(n_samples):
        # Extract parameters for this sample
        alpha_sample = {actor: alphas[actor][i] for actor in ACTORS}
        gamma_sample = {actor: gammas[actor][i] for actor in ["AI", "CM"]}
        beta_sample = {actor: betas[actor][i] for actor in ACTORS}
        k_sample = k[i]
        
        # Compute Shapley values for this sample
        shapley_values_sample = compute_shapley_values(
            alpha_sample, gamma_sample, beta_sample, k_sample
        )
        
        # Store results
        for actor in ACTORS:
            shapley_values[actor][i] = shapley_values_sample[actor]
    
    return shapley_values

def create_shapley_distribution_chart(shapley_values: Dict[str, np.ndarray]) -> alt.Chart:
    # Prepare data
    data = []
    for actor, values in shapley_values.items():
        for val in values:
            data.append({
                "Actor": actor,
                "Shapley Value": val
            })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Create the chart
    chart = alt.Chart(df).transform_density(
        'Shapley Value', 
        groupby=['Actor'],
        as_=['Shapley Value', 'Density']
    ).mark_area(opacity=0.5).encode(
        x=alt.X('Shapley Value:Q'),
        y=alt.Y('Density:Q'),
        color='Actor:N'
    ).properties(
        width=600,
        height=400,
        title="Shapley Value Distributions"
    )
    
    return chart

def create_shapley_boxplot(shapley_values: Dict[str, np.ndarray]) -> alt.Chart:
    # Prepare data
    data = []
    for actor, values in shapley_values.items():
        for val in values:
            data.append({
                "Actor": actor,
                "Shapley Value": val
            })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    # Create the chart
    chart = alt.Chart(df).mark_boxplot().encode(
        x='Actor:N',
        y='Shapley Value:Q',
        color='Actor:N'
    ).properties(
        width=400,
        height=300,
        title="Shapley Value Distributions"
    )
    
    return chart

def create_shapley_summary_table(shapley_values: Dict[str, np.ndarray]) -> pd.DataFrame:
    # Prepare data
    data = []
    for actor in ACTORS:
        values = shapley_values[actor]
        data.append({
            "Actor": actor,
            "Mean": values.mean(),
            "Std": values.std(),
            "25th Percentile": np.percentile(values, 25),
            "Median": np.median(values),
            "75th Percentile": np.percentile(values, 75)
        })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    return df

def main():
    st.set_page_config(layout="wide", page_title="TIG Shapley Value Simulation")
    
    st.title("TIG Shapley Value Simulation with Cobb-Douglas Model")
    st.markdown("""
    This application simulates reward allocation using Shapley values for four key actors in the TIG ecosystem using a Cobb-Douglas production model:
    """)
    
    st.latex(r"Y = k \cdot \alpha_B^{\beta_B} \cdot \alpha_{CI}^{\beta_{CI}} \cdot (\alpha_{AI} \cdot (1+\gamma_{AI}))^{\beta_{AI}} \cdot (\alpha_{CM} \cdot (1+\gamma_{CM}))^{\beta_{CM}}")
    
    st.markdown("""
    Where:
    - **Benchmarkers (B)**
    - **Code Innovators (CI)**
    - **Algorithm Innovators (AI)**
    - **Challenge Maintainers (CM)**
    
    The simulation uses Hamiltonian Monte Carlo to sample from prior distributions.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Simulation Settings")
        n_samples = st.slider("Number of HMC samples", 50, 1000, 200, 50)
        
        st.subheader("Base Contributions (α) Priors")
        alpha_B_mean = st.number_input("Benchmarkers (αB) Mean", 0.1, 5.0, 1.0, 0.1)
        alpha_B_std = st.number_input("Benchmarkers (αB) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CI_mean = st.number_input("Code Innovators (αCI) Mean", 0.1, 5.0, 1.0, 0.1)
        alpha_CI_std = st.number_input("Code Innovators (αCI) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_AI_mean = st.number_input("Algorithm Innovators (αAI) Mean", 0.1, 5.0, 2.0, 0.1)
        alpha_AI_std = st.number_input("Algorithm Innovators (αAI) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CM_mean = st.number_input("Challenge Maintainers (αCM) Mean", 0.1, 5.0, 2.0, 0.1)
        alpha_CM_std = st.number_input("Challenge Maintainers (αCM) Std", 0.01, 2.0, 0.2, 0.05)

        st.subheader("Bonus Factors (γ) Priors")
        gamma_AI_mean = st.number_input("Algorithm Innovators (γAI) Mean", 0.0, 5.0, 2.0, 0.1)
        gamma_AI_std = st.number_input("Algorithm Innovators (γAI) Std", 0.01, 2.0, 0.2, 0.05)
        
        gamma_CM_mean = st.number_input("Challenge Maintainers (γCM) Mean", 0.0, 5.0, 1.0, 0.1)
        gamma_CM_std = st.number_input("Challenge Maintainers (γCM) Std", 0.01, 2.0, 0.2, 0.05)
        
        st.subheader("Elasticity Parameters (β) Priors")
        st.write("Beta distribution parameters (must be between 0 and 1)")
        beta_B_mean = st.slider("Benchmarkers (βB) Mean", 0.01, 0.99, 0.25, 0.01)
        beta_B_std = st.slider("Benchmarkers (βB) Std", 0.01, 0.5, 0.1, 0.01)
        
        beta_CI_mean = st.slider("Code Innovators (βCI) Mean", 0.01, 0.99, 0.25, 0.01)
        beta_CI_std = st.slider("Code Innovators (βCI) Std", 0.01, 0.5, 0.1, 0.01)
        
        beta_AI_mean = st.slider("Algorithm Innovators (βAI) Mean", 0.01, 0.99, 0.25, 0.01)
        beta_AI_std = st.slider("Algorithm Innovators (βAI) Std", 0.01, 0.5, 0.1, 0.01)
        
        beta_CM_mean = st.slider("Challenge Maintainers (βCM) Mean", 0.01, 0.99, 0.25, 0.01)
        beta_CM_std = st.slider("Challenge Maintainers (βCM) Std", 0.01, 0.5, 0.1, 0.01)
        
        st.subheader("Scaling Factor (k) Prior")
        k_mean = st.number_input("Scaling Factor (k) Mean", 0.1, 10.0, 1.0, 0.1)
        k_std = st.number_input("Scaling Factor (k) Std", 0.01, 5.0, 0.2, 0.05)
    
    # Prepare priors
    alpha_priors = {
        "B": (alpha_B_mean, alpha_B_std),
        "CI": (alpha_CI_mean, alpha_CI_std),
        "AI": (alpha_AI_mean, alpha_AI_std),
        "CM": (alpha_CM_mean, alpha_CM_std)
    }
    
    gamma_priors = {
        "AI": (gamma_AI_mean, gamma_AI_std),
        "CM": (gamma_CM_mean, gamma_CM_std)
    }
    
    beta_priors = {
        "B": (beta_B_mean, beta_B_std),
        "CI": (beta_CI_mean, beta_CI_std),
        "AI": (beta_AI_mean, beta_AI_std),
        "CM": (beta_CM_mean, beta_CM_std)
    }
    
    k_prior = (k_mean, k_std)
    
    # Start HMC sampling with progress indicator
    with st.spinner("Running Hamiltonian Monte Carlo sampling... This may take a few minutes."):
        shapley_values = compute_hmc_shapley_values(
            alpha_priors, gamma_priors, beta_priors, k_prior, n_samples
        )
    
    # Create visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Shapley Value Distributions")
        density_chart = create_shapley_distribution_chart(shapley_values)
        st.altair_chart(density_chart, use_container_width=True)
    
    with col2:
        st.subheader("Shapley Value Boxplot")
        boxplot_chart = create_shapley_boxplot(shapley_values)
        st.altair_chart(boxplot_chart, use_container_width=True)
    
    st.subheader("Shapley Value Summary Statistics")
    summary_df = create_shapley_summary_table(shapley_values)
    st.dataframe(summary_df, use_container_width=True)
    
    # Add reward distribution visualization
    st.subheader("Expected Reward Distribution")
    mean_shapley_values = {actor: shapley_values[actor].mean() for actor in ACTORS}
    total = sum(mean_shapley_values.values())
    percentages = {actor: value/total*100 for actor, value in mean_shapley_values.items()}
    
    # Create donut chart
    pie_data = pd.DataFrame({
        "Actor": list(percentages.keys()),
        "Percentage": list(percentages.values())
    })
    
    pie_chart = alt.Chart(pie_data).mark_arc(innerRadius=50).encode(
        theta=alt.Theta(field="Percentage", type="quantitative"),
        color=alt.Color(field="Actor", type="nominal", scale=alt.Scale(scheme="category10")),
        tooltip=[
            alt.Tooltip("Actor", title="Actor"),
            alt.Tooltip("Percentage", title="Percentage", format=".1f")
        ]
    ).properties(
        width=400,
        height=400,
        title="Expected Reward Distribution (%)"
    )
    
    st.altair_chart(pie_chart, use_container_width=True)

if __name__ == "__main__":
    main()