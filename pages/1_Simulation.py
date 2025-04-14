import streamlit as st
import numpy as np
import altair as alt
import pandas as pd
import scipy as sp
from tig_shapely import (
    ACTORS,
    compute_hmc_shapley_values,
    maximize_and_quantify_uncertainty,
    create_shapley_distribution_chart,
    create_shapley_boxplot,
    create_shapley_summary_table,
    create_optimal_vs_uq_chart,
    create_optimal_params_table,
    plot_prior_distribution
)

def main():
    st.set_page_config(layout="wide", page_title="TIG Shapley Value Simulation")
    
    st.title("TIG Shapley Value Simulation with Cobb-Douglas Model")
    
    st.write("This application simulates reward allocation using Shapley values for four key actors in the TIG ecosystem. Visit the 'Explanation' page to learn more about the model.")
    
    st.latex(r"Y = L^{\delta} \cdot K^{\kappa}")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Simulation Settings")
        n_samples = st.slider("Number of HMC samples", 50, 1000, 200, 50)
        
        st.subheader("Labor Contribution (α^L) Priors")
        alpha_B_L_mean = st.number_input("Benchmarkers (αB^L) Mean", 0.1, 5.0, 1.0, 0.1)
        alpha_B_L_std = st.number_input("Benchmarkers (αB^L) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CI_L_mean = st.number_input("Code Innovators (αCI^L) Mean", 0.1, 5.0, 0.2, 0.1)
        alpha_CI_L_std = st.number_input("Code Innovators (αCI^L) Std", 0.01, 2.0, 0.1, 0.05)
        
        alpha_AI_L_mean = st.number_input("Algorithm Innovators (αAI^L) Mean", 0.1, 5.0, 2.4, 0.1)
        alpha_AI_L_std = st.number_input("Algorithm Innovators (αAI^L) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CM_L_mean = st.number_input("Challenge Maintainers (αCM^L) Mean", 0.1, 5.0, 0.4, 0.1)
        alpha_CM_L_std = st.number_input("Challenge Maintainers (αCM^L) Std", 0.01, 2.0, 0.1, 0.05)

        st.subheader("Capital Contribution (α^K) Priors")
        alpha_B_K_mean = st.number_input("Benchmarkers (αB^K) Mean", 0.1, 5.0, 1.0, 0.1)
        alpha_B_K_std = st.number_input("Benchmarkers (αB^K) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CI_K_mean = st.number_input("Code Innovators (αCI^K) Mean", 0.1, 5.0, 0.2, 0.1)
        alpha_CI_K_std = st.number_input("Code Innovators (αCI^K) Std", 0.01, 2.0, 0.1, 0.05)
        
        alpha_AI_K_mean = st.number_input("Algorithm Innovators (αAI^K) Mean", 0.1, 5.0, 2.4, 0.1)
        alpha_AI_K_std = st.number_input("Algorithm Innovators (αAI^K) Std", 0.01, 2.0, 0.2, 0.05)
        
        alpha_CM_K_mean = st.number_input("Challenge Maintainers (αCM^K) Mean", 0.1, 5.0, 0.4, 0.1)
        alpha_CM_K_std = st.number_input("Challenge Maintainers (αCM^K) Std", 0.01, 2.0, 0.1, 0.05)
        
        st.subheader("Bonus Factors (γ) Priors")
        gamma_AI_mean = st.number_input("Algorithm Innovators (γAI) Mean", 0.0, 5.0, 1.0, 0.1)
        gamma_AI_std = st.number_input("Algorithm Innovators (γAI) Std", 0.01, 2.0, 0.2, 0.05)
        
        gamma_CM_mean = st.number_input("Challenge Maintainers (γCM) Mean", 0.0, 5.0, 0.5, 0.1)
        gamma_CM_std = st.number_input("Challenge Maintainers (γCM) Std", 0.01, 2.0, 0.1, 0.05)
        
        st.subheader("Production Function Parameters")
        delta_mean = st.slider("Labor Elasticity (δ) Mean", 0.01, 0.99, 0.6, 0.01)
        delta_std = st.slider("Labor Elasticity (δ) Std", 0.01, 0.3, 0.1, 0.01)
        
        kappa_mean = st.slider("Capital Elasticity (κ) Mean", 0.01, 0.99, 0.4, 0.01)
        kappa_std = st.slider("Capital Elasticity (κ) Std", 0.01, 0.3, 0.1, 0.01)
    
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
    
    progress_container = st.empty()
    with progress_container.container():
        st.info("Running Hamiltonian Monte Carlo sampling for Forward UQ...")
        uq_shapley_values = compute_hmc_shapley_values(
            alpha_labor_priors, alpha_capital_priors, gamma_priors, 
            delta_prior, kappa_prior, n_samples
        )
    with progress_container.container():
        st.info("Optimizing parameters to maximize total value...")
        optimal_params, optimal_shapley, shapley_uncertainty = maximize_and_quantify_uncertainty(
            alpha_labor_priors, alpha_capital_priors, gamma_priors, 
            delta_prior, kappa_prior, n_samples
        )
    progress_container.empty()

    
    # Create tabs to organize the visualizations
    tab1, tab2, tab3 = st.tabs([
        "Forward UQ Results", 
        "Optimization Results", 
        "Prior Distributions"
    ])
    
    # Tab 1: Shapley Value Results
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Shapley Value Distributions")
            density_chart = create_shapley_distribution_chart(uq_shapley_values)
            st.altair_chart(density_chart, use_container_width=True)
        
        with col2:
            st.subheader("Shapley Value Boxplot")
            boxplot_chart = create_shapley_boxplot(uq_shapley_values)
            st.altair_chart(boxplot_chart, use_container_width=True)
        
        with col3:
            # Replace pie chart with a horizontal bar chart for the reward distribution
            st.subheader("Expected Reward Distribution")
            mean_shapley_values = {actor: uq_shapley_values[actor].mean() for actor in ACTORS}
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
    
    # Tab 2: Optimization Results
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Optimal Parameter Values")
            params_table = create_optimal_params_table(optimal_params)
            st.dataframe(params_table, use_container_width=True)
        
        with col2:
            st.subheader("Optimal Reward Distribution")
            # Calculate percentages
            total = sum(optimal_shapley.values())
            percentages = {actor: value/total*100 for actor, value in optimal_shapley.items()}
            
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
                title="Optimal Reward Distribution (%)"
            )
            
            # Add text labels to the bars
            text = bar_chart.mark_text(
                align='left',
                baseline='middle',
                dx=3
            ).encode(
                text=alt.Text('Percentage:Q', format='.1f')
            )
            
            final_chart = (bar_chart + text)
            st.altair_chart(final_chart, use_container_width=True)
        
        # Uncertainty around optimal distribution
        st.subheader("Uncertainty Around Optimal Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            uncertainty_chart = create_shapley_distribution_chart(shapley_uncertainty)
            st.altair_chart(uncertainty_chart, use_container_width=True)
        
        with col2:
            uncertainty_boxplot = create_shapley_boxplot(shapley_uncertainty)
            st.altair_chart(uncertainty_boxplot, use_container_width=True)
    
    # Tab 3: Prior Distributions
    with tab3:
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