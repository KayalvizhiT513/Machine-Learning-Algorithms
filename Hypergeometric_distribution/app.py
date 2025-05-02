import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import hypergeom
import google.generativeai as genai
import os

# Set your Gemini API key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]  # Add this to .streamlit/secrets.toml
genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="Hypergeometric PMF")

st.title("🎲 Hypergeometric PMF Explorer")

# Sidebar controls
N = st.sidebar.slider("Population size (N)", 10, 500, 100)
k = st.sidebar.slider("Sample size (k)", 1, N, 90)
m_min, m_max = st.sidebar.slider("Successes in population range (m)", 0, N, (k, N))
m_step = st.sidebar.number_input("Step size for m", min_value=1, max_value=max(1, m_max - m_min), value=1)
x_min, x_max = st.sidebar.slider("Observed successes in sample range (x)", 0, k, (0, k))

# Plotting
fig, ax = plt.subplots()
for m in range(m_min, m_max + 1, m_step):
    x = np.arange(x_min, x_max + 1)
    rv = hypergeom(N, m, k)
    probs = rv.pmf(x)
    ax.plot(x, probs, label=f"m = {m}")
ax.set_title(f"Hypergeometric PMF\nN={N}, k={k}, m∈[{m_min},{m_max}], x∈[{x_min},{x_max}]")
ax.set_xlabel("x (successes in sample)")
ax.set_ylabel("Probability")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Prompt for Gemini
prompt = f"""
You are a data analyst assistant.

The user plotted the hypergeometric PMF with the following settings:
- Population size (N): {N}
- Sample size (k): {k}
- Successes in population (m): from {m_min} to {m_max}
- Range of successes in sample (x): from {x_min} to {x_max}

Please explain:
- How the PMF curve shape changes as m increases
- Where the peaks in probability occur
- How likely it is to get higher x as m increases
- Any interesting or non-obvious observations
"""

try:
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    st.success("Summary from GenAI:")
    st.write(response.text)
except Exception as e:
    st.error(f"Error calling Gemini API: {e}")
