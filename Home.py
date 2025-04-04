import streamlit as st

def main():
    st.set_page_config(layout="wide", page_title="TIG Shapley Value Simulation - Explanation")
    
    st.title("TIG Shapley Value Simulation")
    
    # Introduction and problem context
    st.markdown("""
    ## Problem Context
    
    In TIG, multiple actors with different roles collaborate to drive progress. In this framework, we explore: how should rewards be distributed 
    fairly among contributors when their inputs to the network are different, but interdependent?
    
    The four actors in TIG are:
    
    - **Benchmarkers (B)**: Evaluate and run algorithms created by other actors for various challenges
    - **Code Innovators (CI)**: Develop efficient implementations of algorithms
    - **Algorithm Innovators (AI)**: Invent new mathematical approaches to challenges
    - **Challenge Maintainers (CM)**: Manage the benchmarks and foster the competitive ecosystem
    
    Each group contributes differently to the overall value creation, but all are part of the ecosystem. For example, 
    algorithm innovations are only valuable when they can be implemented in code and tested against benchmarks.
    """)

    st.markdown("### Approach - Shapley Values")
    st.markdown("""
    Shapley values are one way to assess the individual contributions of each actor group, and have some desirable properties:
    See [here](https://en.wikipedia.org/wiki/Shapley_value#Properties) for more details.
    
    #### Calculation Method
    
    For each actor $i$, the Shapley value is calculated as:
    """)
    
    st.latex(r"\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \cdot [v(S \cup \{i\}) - v(S)]")
    
    st.markdown("""
    Where:
    - $N$ is the set of all actors
    - $S$ is a subset of actors not containing $i$
    - $v$ is the characteristic function that assigns a value to each coalition
    - $v(S)$ is the value produced by coalition $S$
    
    In our model, the characteristic function $v$ is the Cobb-Douglas production function (described below).
    
    Shapley values can then be translated to: 
    - how much each actor group should be rewarded relative to others, 
    - which actor groups provide the most critical marginal value, and 
    - how rewards should change as parameters in the model change.
    """)
    
    # Why Cobb-Douglas is appropriate
    st.markdown("""
    ## Cobb-Douglas Production Function
    
    The Cobb-Douglas production function is a model that can represent the relationship between inputs and how much output can be produced by those inputs.
    See [here](https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function) for more details.
    
    It takes the form:
    """)
    st.latex(r"Y = A \cdot L^{\delta} \cdot K^{\kappa}")
    
    st.markdown("""
    Where:
    - $Y$ is the total value created
    - $A$ is the total factor productivity (scaling factor)
    - $L$ is the aggregated labor input
    - $K$ is the aggregated capital input
    - $\\delta$ is the labor elasticity (indicating how output changes with labor input)
    - $\\kappa$ is the capital elasticity (indicating how output changes with capital input)
    
    If either $L$ or $K$ is zero, then $Y$ is zero.

    We extend the this model to incorporate both labor and capital inputs from each actor group, with bonus factors for certain actors to reflect their unique contributions.
    """)
    st.markdown("### Production Function Components")
    
    # Labor component with proper LaTeX rendering
    st.markdown("#### Labor Input (L)")
    st.latex(r"L = L_B + L_{CI} + L_{AI} + L_{CM}")
    st.markdown("""
    Labor represents the direct effort, time, and skills contributed by each actor group. For Algorithm Innovators and 
    Challenge Maintainers, we include a bonus factor ($\\gamma$) to account for their potentially outsized impact:
    """)
    st.latex(r"L_{AI} = \alpha_{AI}^L \cdot (1+\gamma_{AI})")
    st.latex(r"L_{CM} = \alpha_{CM}^L \cdot (1+\gamma_{CM})")
    
    # Capital component with proper LaTeX rendering
    st.markdown("#### Capital Input (K)")
    st.latex(r"K = K_B + K_{CI} + K_{AI} + K_{CM}")
    st.markdown("""
    Capital represents the accumulated knowledge, infrastructure, tools, and resources contributed by each actor group. 
    It reflects long-term investments rather than immediate effort.
    """)
    
    st.markdown("### Actor Contributions")
    st.markdown("""
    Each actor contributes both labor and capital to the production function, in different proportions:
    """)
    
    # Actor equations with proper LaTeX rendering
    st.markdown("#### Benchmarkers (B)")
    st.latex(r"L_B = \alpha_B^L")
    st.latex(r"K_B = \alpha_B^K")
    st.markdown("""
    Benchmarkers provide capital because they require hardware to run the benchmarks. They also contribute labor through the ongoing work of updating 
    benchmarks, analyzing results, and setting standards.
    """)
    
    st.markdown("#### Code Innovators (CI)")
    st.latex(r"L_{CI} = \alpha_{CI}^L")
    st.latex(r"K_{CI} = \alpha_{CI}^K")
    st.markdown("""
    Code Innovators primarily contribute labor through implementation work, but also provide capital in the form 
    of reusable code repositories, development frameworks, and technical expertise.
    """)
    
    st.markdown("#### Algorithm Innovators (AI)")
    st.latex(r"L_{AI} = \alpha_{AI}^L \cdot (1+\gamma_{AI})")
    st.latex(r"K_{AI} = \alpha_{AI}^K")
    st.markdown("""
    Algorithm Innovators contribute labor through research effort, but their contribution is amplified by a 
    bonus factor $\\gamma_{AI}$ to reflect the potentially breakthrough nature of algorithmic innovations. 
    Their capital contributions include accumulated knowledge, research paradigms, and intellectual property.
    """)
    
    st.markdown("#### Challenge Maintainers (CM)")
    st.latex(r"L_{CM} = \alpha_{CM}^L \cdot (1+\gamma_{CM})")
    st.latex(r"K_{CM} = \alpha_{CM}^K")
    st.markdown("""
    Challenge Maintainers contribute labor in community management, challenge organization, and ecosystem 
    development, with a bonus factor $\\gamma_{CM}$ to reflect their impact in coordinating the broader 
    ecosystem. Their capital includes established community networks, reputation, and organizational structures.
    """)
    
    st.markdown("### Modeling Uncertainty with Hamiltonian Monte Carlo")
    st.markdown("""
    Rather than using fixed point estimates for the model parameters, which would lead to a single set of 
    Shapley values, we use Hamiltonian Monte Carlo (HMC) sampling to account for uncertainty in the parameter values.
    
    #### Parameter Distributions
    
    The simulation samples from prior distributions for each parameter:
    
    - Labor contribution parameters ($\\alpha^L$): Lognormal distributions to ensure positivity
    - Capital contribution parameters ($\\alpha^K$): Lognormal distributions to ensure positivity
    - Bonus factors ($\\gamma$): Lognormal distributions to ensure positivity
    - Labor elasticity ($\\delta$): Beta distribution to constrain between 0 and 1
    - Capital elasticity ($\\kappa$): Beta distribution to constrain between 0 and 1
    - Total factor productivity ($A$): Lognormal distribution to ensure positivity
    
    The resulting distributions of Shapley values provide insight into:
    
    1. The expected reward for each actor
    2. The uncertainty in those rewards given parameter uncertainty
    3. The relationships between parameters and rewards
    4. The robustness of the reward allocation to parameter changes
    
    The primary difficulty in this approach is how to set the prior distributions for the parameters.
    """)

if __name__ == "__main__":
    main()