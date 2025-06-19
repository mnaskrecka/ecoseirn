import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Model równań różniczkowych
def model(t, y, params):
    S, E, I, R, NE, M = y
    βI, βN, σ1, σ2, γr, γi, α, r, r0, m, N, Mo = params
    dS_dt = (-βI * (E + I + R) / N * M / (m + M) * S
             - βN * NE / N * S 
             + σ2 * NE / N * E
             + α * M / (m + M) * (E + I + R) / N * NE)

    dE_dt = (βI * (E + I + R) / N * M / (m + M) * S
             - σ1 * M / (m + M) * (I + R) / N * E
             - σ2 * NE / N * E)

    dI_dt = (σ1 * M / (m + M) * (I + R) / N * E
             - γr * I 
             + γi * R)

    dR_dt = (-γi * R 
             + γr * I)

    dNE_dt = (βN * NE / N * S
              - α * M / (m + M) * (E + I + R) / N * NE)

    dM_dt = (r * (NE + S) / N  
             - r0 * (M - Mo))

    return [dS_dt, dE_dt, dI_dt, dR_dt, dNE_dt, dM_dt]

# Tytuł aplikacji
st.title("Simulation of the Greenish Consumer Dynamics Model (2024–2064)")

# Parametry modelu
with st.sidebar:
    st.header("Model Parameters")
    βI = st.slider('βI', 0.001, 0.1, 0.01175, step=0.00001)
    βN = st.slider('βN', 0.01, 0.1, 0.07083, step=0.00001)
    σ1 = st.slider('σ1', 0.0001, 0.1, 0.00068, step=0.00001)
    σ2 = st.slider('σ2', 0.01, 0.1, 0.07201, step=0.00001)
    γr = st.slider('γr', 0.01, 0.1, 0.05635, step=0.00001)
    γi = st.slider('γi', 0.01, 0.1, 0.04575, step=0.00001)
    α = st.slider('α', 0.02, 0.045, 0.04150, step=0.00001)
    r = st.slider('r', 0.001, 0.3, 0.02393, step=0.0001)
    r0 = st.slider('r0', 0.01, 0.1, 0.06188, step=0.00001)
    m = st.slider('m', 0.001, 0.1, 10/365, step=0.00001)

    st.header("Initial Conditions")
    S0 = st.slider('S₀', 0.0, 100.0, 26.0, step=0.5)
    E0 = st.slider('E₀', 0.0, 100.0, 28.0, step=0.5)
    I0 = st.slider('I₀', 0.0, 100.0, 15.5, step=0.5)
    R0 = st.slider('R₀', 0.0, 100.0, 15.5, step=0.5)
    NE0 = st.slider('NE₀', 0.0, 100.0, 15.0, step=0.5)
    M0 = st.slider('M₀', 0.0, 100.0, 10.0, step=1.0)

# Obliczenia
N = S0 + E0 + I0 + R0 + NE0
Mo = 10
params = [βI, βN, σ1, σ2, γr, γi, α, r, r0, m, N, Mo]
y0 = [S0, E0, I0, R0, NE0, M0]
t_span = (0, 3652*4)
t_eval = np.arange(0, 3652*4 + 1, 365)

solution = solve_ivp(model, t_span, y0, t_eval=t_eval, args=(params,), method='RK45')
t = solution.t
S, E, I, R, NE, M = solution.y

# Wykresy
fig, axs = plt.subplots(2, 1, figsize=(12, 8))

axs[0].plot(t, S, 'b-o', label='S(t)')
axs[0].plot(t, E, 'g-o', label='E(t)')
axs[0].plot(t, I, 'r-o', label='I(t)')
axs[0].plot(t, R, 'm-o', label='R(t)')
axs[0].plot(t, NE, 'c-o', label='NE(t)')
axs[0].set_title('Population Groups Over Time')
axs[0].set_ylabel('Share in Population (%)')
axs[0].set_xticks(t)
axs[0].set_xticklabels([str(2024 + i) for i in range(len(t))])
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, M, 'k-o', label='M(t)')
axs[1].set_title('Media Intensity')
axs[1].set_ylabel('Intensity')
axs[1].set_xticks(t)
axs[1].set_xticklabels([str(2024 + i) for i in range(len(t))])
axs[1].legend()
axs[1].grid(True)

st.pyplot(fig)
