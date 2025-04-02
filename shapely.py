import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
from itertools import combinations, chain
from typing import Dict, List, Set, Tuple, Union

# Define actors
ACTORS = ["B", "CI", "AI", "CM"]  # Benchmarkers, Code Innovators, Algorithm Innovators, Challenge Maintainers

def powerset(iterable: List[str]) -> List[Tuple[str, ...]]:
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

def compute_coalition_value(
    coalition: Tuple[str, ...], 
    alphas: Dict[str, float], 
    gammas: Dict[str, float]
) -> float:
    if not coalition:
        return 0

    # Calculate the sum of base contributions
    base_sum = sum(alphas[actor] for actor in coalition)
    
    # Calculate the multiplier based on bonus actors present
    multiplier = 1.0
    for actor in coalition:
        if actor in ["AI", "CM"] and actor in gammas:
            multiplier *= (1 + gammas[actor])
    
    return base_sum * multiplier

def compute_shapley_values(
    alphas: Dict[str, float], 
    gammas: Dict[str, float]
) -> Dict[str, float]:
    n = len(ACTORS)
    shapley_values = {actor: 0.0 for actor in ACTORS}
    all_coalitions = [set(coalition) for coalition in powerset(ACTORS)]
    
    for actor in ACTORS:
        for coalition in all_coalitions:
            if actor not in coalition:
                # Calculate the marginal contribution
                s = len(coalition)
                weight = (np.math.factorial(s) * np.math.factorial(n - s - 1)) / np.math.factorial(n)
                
                # Convert sets to tuples for the compute_coalition_value function
                coalition_tuple = tuple(coalition)
                coalition_with_actor = tuple(coalition.union({actor}))
                
                marginal = compute_coalition_value(coalition_with_actor, alphas, gammas) - \
                           compute_coalition_value(coalition_tuple, alphas, gammas)
                
                shapley_values[actor] += weight * marginal
    
    return shapley_values

def create_sensitivity_data(
    param_name: str,
    param_values: List[float],
    fixed_params: Dict[str, float]
) -> pd.DataFrame:
    results = []
    
    for value in param_values:
        current_params = fixed_params.copy()
        
        # Determine if the parameter is an alpha or gamma
        if param_name.startswith("alpha_"):
            actor = param_name[6:]  # Extract actor name after "alpha_"
            alphas = {
                "B": current_params.get("alpha_B", 1.0),
                "CI": current_params.get("alpha_CI", 1.0),
                "AI": current_params.get("alpha_AI", 1.0),
                "CM": current_params.get("alpha_CM", 0.5)
            }
            alphas[actor] = value
            
            gammas = {
                "AI": current_params.get("gamma_AI", 0.5),
                "CM": current_params.get("gamma_CM", 1.0)
            }
        else:  # It's a gamma parameter
            actor = param_name[6:]  # Extract actor name after "gamma_"
            gammas = {
                "AI": current_params.get("gamma_AI", 0.5),
                "CM": current_params.get("gamma_CM", 1.0)
            }
            gammas[actor] = value
            
            alphas = {
                "B": current_params.get("alpha_B", 1.0),
                "CI": current_params.get("alpha_CI", 1.0),
                "AI": current_params.get("alpha_AI", 1.0),
                "CM": current_params.get("alpha_CM", 0.5)
            }
        
        # Compute Shapley values with current parameters
        shapley_values = compute_shapley_values(alphas, gammas)
        
        for actor, shapley_value in shapley_values.items():
            results.append({
                "Parameter": param_name,
                "Parameter Value": value,
                "Actor": actor,
                "Shapley Value": shapley_value
            })
    
    return pd.DataFrame(results)

def create_all_sensitivity_data(
    param_ranges: Dict[str, Tuple[float, float, int]],
    fixed_params: Dict[str, float]
) -> pd.DataFrame:
    all_results = []
    
    for param_name, (start, end, steps) in param_ranges.items():
        param_values = np.linspace(start, end, steps)
        param_df = create_sensitivity_data(param_name, param_values, fixed_params)
        all_results.append(param_df)
    
    return pd.concat(all_results, ignore_index=True)

def create_shapley_chart(shapley_values: Dict[str, float]) -> alt.Chart:
    # Convert to dataframe
    df = pd.DataFrame({
        "Actor": list(shapley_values.keys()),
        "Shapley Value": list(shapley_values.values())
    })
    
    # Calculate percentages
    total = df["Shapley Value"].sum()
    df["Percentage"] = (df["Shapley Value"] / total * 100).round(1)
    df["Label"] = df.apply(lambda x: f"{x['Actor']}: {x['Percentage']}%", axis=1)
    
    # Create the chart
    chart = alt.Chart(df).mark_arc().encode(
        theta=alt.Theta(field="Shapley Value", type="quantitative"),
        color=alt.Color(field="Actor", type="nominal", scale=alt.Scale(scheme="category10")),
        tooltip=[
            alt.Tooltip("Actor", title="Actor"),
            alt.Tooltip("Shapley Value", title="Shapley Value", format=".3f"),
            alt.Tooltip("Percentage", title="Percentage", format=".1f")
        ]
    ).properties(
        width=400,
        height=300,
        title="Reward Distribution"
    )
    
    # Add text labels
    text = alt.Chart(df).mark_text(radius=140, size=14).encode(
        theta=alt.Theta(field="Shapley Value", type="quantitative"),
        text="Label"
    )
    
    return chart + text

def create_sensitivity_chart(df: pd.DataFrame) -> alt.Chart:
    selector = alt.selection_point(fields=['Actor'], bind='legend')
    
    base = alt.Chart(df).encode(
        x=alt.X('Parameter Value:Q', title='Parameter Value'),
        y=alt.Y('Shapley Value:Q'),
        color=alt.Color('Actor:N', scale=alt.Scale(scheme="category10")),
        tooltip=['Actor', 'Parameter Value', 'Shapley Value']
    ).properties(
        width=250,
        height=200
    ).add_params(selector)
    
    lines = base.mark_line().encode(
        opacity=alt.condition(selector, alt.value(1), alt.value(0.2))
    )
    
    points = base.mark_point(size=60).encode(
        opacity=alt.condition(selector, alt.value(1), alt.value(0))
    )
    
    facet = (lines + points).facet(
        facet=alt.Facet('Parameter:N', title=None),
        columns=3
    ).resolve_scale(
        x='independent'
    ).properties(
        title='Sensitivity Analysis for All Parameters'
    )
    
    return facet

def main():
    st.set_page_config(layout="wide", page_title="TIG Shapley Value Simulation")
    
    st.title("TIG Shapley Value Simulation")
    st.write("""
    This application simulates reward allocation using Shapley values for four key actors in the TIG ecosystem:
    - **Benchmarkers (B)**
    - **Code Innovators (CI)**
    - **Algorithm Innovators (AI)**
    - **Challenge Maintainers (CM)**
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Base Contributions (α)")
        alpha_B = st.slider("Benchmarkers (αB)", 0.1, 2.0, 1.0, 0.1)
        alpha_CI = st.slider("Code Innovators (αCI)", 0.1, 2.0, 1.0, 0.1)
        alpha_AI = st.slider("Algorithm Innovators (αAI)", 0.1, 2.0, 1.0, 0.1)
        alpha_CM = st.slider("Challenge Maintainers (αCM)", 0.1, 2.0, 0.5, 0.1)

        st.subheader("Bonus Factors (γ)")
        gamma_AI = st.slider("Algorithm Innovators (γAI)", 0.0, 2.0, 0.5, 0.1)
        gamma_CM = st.slider("Challenge Maintainers (γCM)", 0.0, 2.0, 1.0, 0.1)
    
        st.subheader("Sensitivity Analysis")
        sensitivity_range = st.slider(
            "Parameter range",
            0.0, 2.0, (0.1, 2.0), 0.1
        )
        sensitivity_steps = st.slider("Number of steps per parameter", 5, 30, 10)
    
    # Compute Shapley values with current parameters
    alphas = {
        "B": alpha_B,
        "CI": alpha_CI,
        "AI": alpha_AI,
        "CM": alpha_CM
    }
    
    gammas = {
        "AI": gamma_AI,
        "CM": gamma_CM
    }
    
    shapley_values = compute_shapley_values(alphas, gammas)
    
    # Fixed parameters for sensitivity analysis
    fixed_params = {
        "alpha_B": alpha_B,
        "alpha_CI": alpha_CI,
        "alpha_AI": alpha_AI,
        "alpha_CM": alpha_CM,
        "gamma_AI": gamma_AI,
        "gamma_CM": gamma_CM
    }
    
    # Parameter ranges for sensitivity analysis
    param_ranges = {
        "alpha_B": (sensitivity_range[0], sensitivity_range[1], sensitivity_steps),
        "alpha_CI": (sensitivity_range[0], sensitivity_range[1], sensitivity_steps),
        "alpha_AI": (sensitivity_range[0], sensitivity_range[1], sensitivity_steps),
        "alpha_CM": (sensitivity_range[0], sensitivity_range[1], sensitivity_steps),
        "gamma_AI": (sensitivity_range[0], sensitivity_range[1], sensitivity_steps),
        "gamma_CM": (sensitivity_range[0], sensitivity_range[1], sensitivity_steps)
    }
    
    # Create sensitivity data for all parameters
    all_sensitivity_df = create_all_sensitivity_data(param_ranges, fixed_params)
    
    # Create two columns for the main content
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Create and display the Shapley value pie chart
        pie_chart = create_shapley_chart(shapley_values)
        st.altair_chart(pie_chart, use_container_width=True)
        
        # Create a table with detailed Shapley values
        st.subheader("Shapley Values")
        shapley_df = pd.DataFrame({
            "Actor": list(shapley_values.keys()),
            "Shapley Value": list(shapley_values.values()),
            "Percentage": [val/sum(shapley_values.values())*100 for val in shapley_values.values()]
        })
        shapley_df = shapley_df.sort_values("Shapley Value", ascending=False)
        shapley_df["Percentage"] = shapley_df["Percentage"].round(2).astype(str) + "%"
        st.dataframe(shapley_df, use_container_width=True)
    
    with col2:
        # Create and display the sensitivity charts
        sensitivity_chart = create_sensitivity_chart(all_sensitivity_df)
        st.altair_chart(sensitivity_chart, use_container_width=True)

if __name__ == "__main__":
    main()