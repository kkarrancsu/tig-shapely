import altair as alt
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Union, Any
import scipy as sp
from tig_shapely import (
    ACTORS
)

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

def create_optimal_vs_uq_chart(optimal_shapley: Dict[str, float], uq_shapley: Dict[str, np.ndarray]) -> alt.Chart:
    """Create a chart comparing optimal Shapley values to UQ distribution medians"""
    # Prepare data
    data = []
    for actor in ACTORS:
        data.append({
            "Actor": actor,
            "Optimal": optimal_shapley[actor],
            "UQ Median": np.median(uq_shapley[actor])
        })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    df_melted = pd.melt(df, id_vars=['Actor'], value_vars=['Optimal', 'UQ Median'],
                        var_name='Method', value_name='Value')
    
    # Create the chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('Actor:N'),
        y=alt.Y('Value:Q'),
        color=alt.Color('Method:N'),
        column=alt.Column('Method:N')
    ).properties(
        width=250,
        title="Optimal vs. UQ Median Shapley Values"
    )
    
    return chart

def create_optimal_params_table(optimal_params: Dict) -> pd.DataFrame:
    """Create a table displaying the optimal parameter values"""
    # Prepare data
    data = []
    
    # Alpha labor parameters
    for actor in ACTORS:
        data.append({
            "Parameter": f"α{actor}^L (Labor)",
            "Optimal Value": optimal_params['alpha_labor'][actor]
        })
    
    # Alpha capital parameters
    for actor in ACTORS:
        data.append({
            "Parameter": f"α{actor}^K (Capital)",
            "Optimal Value": optimal_params['alpha_capital'][actor]
        })
    
    # Gamma parameters
    for actor in ['AI', 'CM']:
        data.append({
            "Parameter": f"γ{actor} (Bonus)",
            "Optimal Value": optimal_params['gammas'][actor]
        })
    
    # Production function parameters
    data.append({
        "Parameter": "δ (Labor Elasticity)",
        "Optimal Value": optimal_params['delta']
    })
    data.append({
        "Parameter": "κ (Capital Elasticity)",
        "Optimal Value": optimal_params['kappa']
    })
    
    # Convert to dataframe
    df = pd.DataFrame(data)
    
    return df

def plot_prior_distribution(name: str, mean: float, std: float, is_lognormal: bool = True) -> alt.Chart:
    """Create a chart showing the prior distribution for a parameter"""
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