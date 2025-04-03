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
    alpha_labor: Dict[str, float],
    alpha_capital: Dict[str, float],
    gammas: Dict[str, float],
    delta: float,  # Labor elasticity
    kappa: float,  # Capital elasticity
    A: float       # Total factor productivity
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
    
    # Apply Cobb-Douglas function: A * L^delta * K^kappa
    return A * (labor ** delta) * (capital ** kappa)

def compute_shapley_values(
    alpha_labor: Dict[str, float],
    alpha_capital: Dict[str, float],
    gammas: Dict[str, float],
    delta: float,
    kappa: float,
    A: float
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
                
                marginal = compute_coalition_value(coalition_with_actor, alpha_labor, alpha_capital, gammas, delta, kappa, A) - \
                           compute_coalition_value(coalition_tuple, alpha_labor, alpha_capital, gammas, delta, kappa, A)
                
                shapley_values[actor] += weight * marginal
    
    return shapley_values

def sample_hmc_parameters(
    alpha_labor_priors: Dict[str, Tuple[float, float]],
    alpha_capital_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    delta_prior: Tuple[float, float],
    kappa_prior: Tuple[float, float],
    A_prior: Tuple[float, float],
    n_samples: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
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
        
        # Sample A (using lognormal to ensure positivity)
        A_mean, A_std = A_prior
        mu = np.log(A_mean**2 / np.sqrt(A_mean**2 + A_std**2))
        sigma = np.sqrt(np.log(1 + (A_std**2 / A_mean**2)))
        A = pm.Lognormal("A", mu=mu, sigma=sigma)
        
        # Use No-U-Turn Sampler (NUTS) for efficient HMC sampling
        trace = pm.sample(n_samples, tune=1000, chains=2, cores=1, return_inferencedata=True)
    
    # Extract samples from the trace
    alpha_labor_samples = {actor: trace.posterior[f"alpha_labor_{actor}"].values.flatten() for actor in ACTORS}
    alpha_capital_samples = {actor: trace.posterior[f"alpha_capital_{actor}"].values.flatten() for actor in ACTORS}
    gamma_samples = {actor: trace.posterior[f"gamma_{actor}"].values.flatten() for actor in ["AI", "CM"]}
    delta_samples = trace.posterior["delta"].values.flatten()
    kappa_samples = trace.posterior["kappa"].values.flatten()
    A_samples = trace.posterior["A"].values.flatten()
    
    return alpha_labor_samples, alpha_capital_samples, gamma_samples, delta_samples, kappa_samples, A_samples

def compute_hmc_shapley_values(
    alpha_labor_priors: Dict[str, Tuple[float, float]],
    alpha_capital_priors: Dict[str, Tuple[float, float]],
    gamma_priors: Dict[str, Tuple[float, float]],
    delta_prior: Tuple[float, float],
    kappa_prior: Tuple[float, float],
    A_prior: Tuple[float, float],
    n_samples: int
) -> Dict[str, np.ndarray]:
    # Sample parameters using HMC
    alpha_labor, alpha_capital, gammas, delta, kappa, A = sample_hmc_parameters(
        alpha_labor_priors, alpha_capital_priors, gamma_priors, 
        delta_prior, kappa_prior, A_prior, n_samples
    )
    
    # Truncate to the requested number of samples
    n_samples = min(n_samples, len(A))
    
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
        A_sample = A[i]
        
        # Compute Shapley values for this sample
        shapley_values_sample = compute_shapley_values(
            alpha_labor_sample, alpha_capital_sample, gamma_sample, 
            delta_sample, kappa_sample, A_sample
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
    
    st.write("This application simulates reward allocation using Shapley values for four key actors in the TIG ecosystem. Visit the 'Explanation' page to learn more about the model.")
    
    st.latex(r"Y = A \cdot L^{\delta} \cdot K^{\kappa}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Simulation Settings")
        n_samples = st.slider("Number of HMC samples", 50, 1000, 200, 50)
        
        st.subheader("Labor Contribution (α^L) Priors")
        alpha_B_L_mean = st.number_input("Benchmarkers (αB^L) Mean", 0.1, 5.0, 1.0, 0.1)
        alpha_B_L_std = st.number_input("Benchmarkers (αB^L) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CI_L_mean = st.number_input("Code Innovators (αCI^L) Mean", 0.1, 5.0, 1.0, 0.1)
        alpha_CI_L_std = st.number_input("Code Innovators (αCI^L) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_AI_L_mean = st.number_input("Algorithm Innovators (αAI^L) Mean", 0.1, 5.0, 2.0, 0.1)
        alpha_AI_L_std = st.number_input("Algorithm Innovators (αAI^L) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CM_L_mean = st.number_input("Challenge Maintainers (αCM^L) Mean", 0.1, 5.0, 2.0, 0.1)
        alpha_CM_L_std = st.number_input("Challenge Maintainers (αCM^L) Std", 0.01, 2.0, 0.2, 0.05)

        st.subheader("Capital Contribution (α^K) Priors")
        alpha_B_K_mean = st.number_input("Benchmarkers (αB^K) Mean", 0.1, 5.0, 2.0, 0.1)
        alpha_B_K_std = st.number_input("Benchmarkers (αB^K) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CI_K_mean = st.number_input("Code Innovators (αCI^K) Mean", 0.1, 5.0, 0.5, 0.1)
        alpha_CI_K_std = st.number_input("Code Innovators (αCI^K) Std", 0.01, 2.0, 0.1, 0.05)
        
        alpha_AI_K_mean = st.number_input("Algorithm Innovators (αAI^K) Mean", 0.1, 5.0, 0.5, 0.1)
        alpha_AI_K_std = st.number_input("Algorithm Innovators (αAI^K) Std", 0.01, 2.0, 0.1, 0.05)
        
        alpha_CM_K_mean = st.number_input("Challenge Maintainers (αCM^K) Mean", 0.1, 5.0, 0.5, 0.1)
        alpha_CM_K_std = st.number_input("Challenge Maintainers (αCM^K) Std", 0.01, 2.0, 0.1, 0.05)
        
        st.subheader("Bonus Factors (γ) Priors")
        gamma_AI_mean = st.number_input("Algorithm Innovators (γAI) Mean", 0.0, 5.0, 2.0, 0.1)
        gamma_AI_std = st.number_input("Algorithm Innovators (γAI) Std", 0.01, 2.0, 0.2, 0.05)
        
        gamma_CM_mean = st.number_input("Challenge Maintainers (γCM) Mean", 0.0, 5.0, 1.0, 0.1)
        gamma_CM_std = st.number_input("Challenge Maintainers (γCM) Std", 0.01, 2.0, 0.2, 0.05)
        
        st.subheader("Production Function Parameters")
        delta_mean = st.slider("Labor Elasticity (δ) Mean", 0.01, 0.99, 0.6, 0.01)
        delta_std = st.slider("Labor Elasticity (δ) Std", 0.01, 0.3, 0.1, 0.01)
        
        kappa_mean = st.slider("Capital Elasticity (κ) Mean", 0.01, 0.99, 0.4, 0.01)
        kappa_std = st.slider("Capital Elasticity (κ) Std", 0.01, 0.3, 0.1, 0.01)
        
        st.subheader("Total Factor Productivity (A) Prior")
        A_mean = st.number_input("Total Factor Productivity (A) Mean", 0.1, 10.0, 1.0, 0.1)
        A_std = st.number_input("Total Factor Productivity (A) Std", 0.01, 5.0, 0.2, 0.05)
    
    # Prepare priors
    alpha_labor_priors = {
        "B": (alpha_B_L_mean, alpha_B_L_std),
        "CI": (alpha_CI_L_mean, alpha_CI_L_std),
        "AI": (alpha_AI_L_mean, alpha_AI_L_std),
        "CM": (alpha_CM_L_mean, alpha_CM_L_std)
    }
    
    alpha_capital_priors = {
        "B": (alpha_B_K_mean, alpha_B_K_std),
        "CI": (alpha_CI_K_mean, alpha_CI_K_std),
        "AI": (alpha_AI_K_mean, alpha_AI_K_std),
        "CM": (alpha_CM_K_mean, alpha_CM_K_std)
    }
    
    gamma_priors = {
        "AI": (gamma_AI_mean, gamma_AI_std),
        "CM": (gamma_CM_mean, gamma_CM_std)
    }
    
    delta_prior = (delta_mean, delta_std)
    kappa_prior = (kappa_mean, kappa_std)
    A_prior = (A_mean, A_std)
    
    # Start HMC sampling with progress indicator
    with st.spinner("Running Hamiltonian Monte Carlo sampling... This may take a few minutes."):
        shapley_values = compute_hmc_shapley_values(
            alpha_labor_priors, alpha_capital_priors, gamma_priors, 
            delta_prior, kappa_prior, A_prior, n_samples
        )
    
    # Create tabs to organize the visualizations
    tab1, tab2 = st.tabs(["Shapley Value Results", "Prior Distributions"])
    
    # Tab 1: Shapley Value Results
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Shapley Value Distributions")
            density_chart = create_shapley_distribution_chart(shapley_values)
            st.altair_chart(density_chart, use_container_width=True)
        
        with col2:
            st.subheader("Shapley Value Boxplot")
            boxplot_chart = create_shapley_boxplot(shapley_values)
            st.altair_chart(boxplot_chart, use_container_width=True)
        
        with col3:
            # Replace pie chart with a horizontal bar chart for the reward distribution
            st.subheader("Expected Reward Distribution")
            mean_shapley_values = {actor: shapley_values[actor].mean() for actor in ACTORS}
            total = sum(mean_shapley_values.values())
            percentages = {actor: value/total*100 for actor, value in mean_shapley_values.items()}
            
            # Create horizontal bar chart
            bar_data = pd.DataFrame({
                "Actor": list(percentages.keys()),
                "Percentage": list(percentages.values())
            })
            
            # Sort bars in descending order
            bar_data = bar_data.sort_values("Percentage", ascending=False)
            
            bar_chart = alt.Chart(bar_data).mark_bar().encode(
                y=alt.Y('Actor:N', sort='-x', title=None),
                x=alt.X('Percentage:Q', title='Percentage (%)'),
                color=alt.Color('Actor:N', scale=alt.Scale(scheme="category10")),
                tooltip=[
                    alt.Tooltip("Actor", title="Actor"),
                    alt.Tooltip("Percentage", title="Percentage", format=".1f")
                ]
            ).properties(
                title="Expected Reward (%)"
            )
            
            # Add text labels to the bars
            text = bar_chart.mark_text(
                align='left',
                baseline='middle',
                dx=3  # Offset the text from the bar
            ).encode(
                text=alt.Text('Percentage:Q', format='.1f')
            )
            
            # Combine the bar chart and text
            final_chart = (bar_chart + text)
            
            st.altair_chart(final_chart, use_container_width=True)
    
    # Tab 2: Prior Distributions
    with tab2:
        # Create function to visualize priors
        def plot_prior_distribution(name, mean, std, is_lognormal=True):
            if is_lognormal:
                # For lognormal priors
                x = np.linspace(0.001, mean + 4*std, 1000)
                mu = np.log(mean**2 / np.sqrt(mean**2 + std**2))
                sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
                pdf = sp.stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))
            else:
                # For beta priors
                x = np.linspace(0.001, 0.999, 1000)
                var = std**2
                if var >= mean * (1 - mean):
                    var = 0.9 * mean * (1 - mean)
                alpha_param = max(0.01, mean * (mean * (1 - mean) / var - 1))
                beta_param = max(0.01, (1 - mean) * (mean * (1 - mean) / var - 1))
                pdf = sp.stats.beta.pdf(x, alpha_param, beta_param)
            
            data = pd.DataFrame({
                'x': x,
                'density': pdf,
                'parameter': [name] * len(x)
            })
            
            chart = alt.Chart(data).mark_line().encode(
                x=alt.X('x', title='Value'),
                y=alt.Y('density', title='Density'),
                color=alt.Color('parameter', legend=None)
            ).properties(
                height=200,
                width=200,
                title=name
            )
            return chart
        
        st.subheader("Model Parameter Priors")
        
        # Use streamlit columns for layout instead of Altair concatenation
        # Group all parameters
        all_params = []
        
        # Labor contribution priors
        for actor in ACTORS:
            name = f"{actor} Labor (α{actor}^L)"
            mean, std = alpha_labor_priors[actor]
            all_params.append((name, mean, std, True))
        
        # Capital contribution priors
        for actor in ACTORS:
            name = f"{actor} Capital (α{actor}^K)"
            mean, std = alpha_capital_priors[actor]
            all_params.append((name, mean, std, True))
        
        # Gamma priors
        for actor in ["AI", "CM"]:
            name = f"{actor} Bonus (γ{actor})"
            mean, std = gamma_priors[actor]
            all_params.append((name, mean, std, True))
        
        # Delta and Kappa (beta distribution)
        all_params.append(("Labor Elasticity (δ)", delta_mean, delta_std, False))
        all_params.append(("Capital Elasticity (κ)", kappa_mean, kappa_std, False))
        
        # Total Factor Productivity (lognormal)
        all_params.append(("TFP (A)", A_mean, A_std, True))
        
        # Create a 3-column layout
        # Divide all parameters into rows with 3 params each
        charts_per_row = 3
        for i in range(0, len(all_params), charts_per_row):
            cols = st.columns(charts_per_row)
            for j in range(charts_per_row):
                if i + j < len(all_params):
                    name, mean, std, is_lognormal = all_params[i + j]
                    with cols[j]:
                        chart = plot_prior_distribution(name, mean, std, is_lognormal)
                        st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main() 