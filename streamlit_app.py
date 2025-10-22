import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Helper functions
# -----------------------------
def solve_speed(P_front, CdA, m, rho, Crr, g=9.81):
    """Solve steady-state speed for given front power and CdA."""
    a = 0.5 * rho * CdA
    b = Crr * m * g
    coeffs = [a, 0, b, -P_front]
    roots = np.roots(coeffs)
    v_candidates = [r.real for r in roots if abs(r.imag) < 1e-9 and r.real > 0]
    if v_candidates:
        return v_candidates[0]
    # Newton fallback
    v = 12.0
    for _ in range(200):
        f = a*v**3 + b*v - P_front
        df = 3*a*v**2 + b
        v_new = v - f/df
        if abs(v_new - v) < 1e-12:
            break
        v = v_new
    return v

def rolling_mean_last_N(x, N):
    out = np.full_like(x, np.nan, dtype=float)
    csum = np.cumsum(np.insert(x, 0, 0.0))
    for i in range(N-1, len(x)):
        out[i] = (csum[i+1] - csum[i+1-N]) / N
    return out

def calc_np(p):
    return np.power(np.mean(np.power(p, 4)), 0.25)

def np_over_window(p, N):
    out = np.full_like(p, np.nan, dtype=float)
    p30 = rolling_mean_last_N(p, 30)
    for i in range(N-1, len(p)):
        seg = p30[i-N+1:i+1]
        out[i] = calc_np(seg)
    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("2-up Time Trial Simulator")

st.header("Environment settings")
rho = st.number_input("Air density (kg/m^3)", 1.225)
Crr = st.number_input("Rolling resistance coefficient (Crr)", 0.004)
wind_speed = st.number_input("Wind speed (m/s)", 0.0)
wind_dir = st.selectbox("Wind direction", ["None", "Head/Tail (out and back)", "Crosswind"])
st.caption("Future options: Add gradient/hills, accelerations, transitions (inactive).")

st.header("Rider A settings")
mass_A = st.number_input("Rider A + bike mass (kg)", 80.0)
CdA_A = st.number_input("Rider A CdA", 0.25)
FTP_A = st.number_input("Rider A FTP (W)", 300.0)
Pfront_A = st.number_input("Rider A target front power (W)", 330.0)
pull_A = st.number_input("Rider A pull duration (s)", 120)

st.header("Rider B settings")
mass_B = st.number_input("Rider B + bike mass (kg)", 80.0)
CdA_B = st.number_input("Rider B CdA", 0.25)
FTP_B = st.number_input("Rider B FTP (W)", 300.0)
Pfront_B = st.number_input("Rider B target front power (W)", 330.0)
pull_B = st.number_input("Rider B pull duration (s)", 120)

st.header("Course settings")
course_km = st.number_input("Course length (km)", 40.0)

st.header("Plots to generate")
plot_1minNP = st.checkbox("Rolling 1-min Normalised Power")
plot_NP_to_date = st.checkbox("Cumulative NP since start")
plot_AvgP = st.checkbox("Cumulative average power since start")

if st.button("Run Simulation"):
    # Effective CdA for draft (25% reduction)
    CdA_draft_A = CdA_A * 0.75
    CdA_draft_B = CdA_B * 0.75

    # Solve for steady speeds for each rider when pulling
    v_A = solve_speed(Pfront_A, CdA_A, mass_A, rho, Crr)
    v_B = solve_speed(Pfront_B, CdA_B, mass_B, rho, Crr)
    v = (v_A + v_B) / 2  # simplified common speed

    # Build time series with asymmetric pulls
    course_m = course_km * 1000
    T_total = int(np.ceil(course_m / v))
    t = 0
    powers_A, powers_B = [], []
    while t < T_total:
        # A pulls
        dur = min(pull_A, T_total - t)
        powers_A.extend([Pfront_A] * dur)
        draft_power_B = 0.5*rho*CdA_draft_B*v**3 + Crr*mass_B*9.81*v
        powers_B.extend([draft_power_B] * dur)
        t += dur
        if t >= T_total: break
        # B pulls
        dur = min(pull_B, T_total - t)
        powers_B.extend([Pfront_B] * dur)
        draft_power_A = 0.5*rho*CdA_draft_A*v**3 + Crr*mass_A*9.81*v
        powers_A.extend([draft_power_A] * dur)
        t += dur

    time = np.arange(len(powers_A))
    power_A = np.array(powers_A)
    power_B = np.array(powers_B)

    if plot_1minNP:
        NP_A_1min = np_over_window(power_A, 60)
        NP_B_1min = np_over_window(power_B, 60)
        fig, ax = plt.subplots()
        ax.plot(time/60, NP_A_1min, label="Rider A 1-min NP")
        ax.plot(time/60, NP_B_1min, label="Rider B 1-min NP")
        ax.axhline(FTP_A, color="gray", linestyle="--", label="FTP A")
        ax.axhline(FTP_B, color="red", linestyle="--", label="FTP B")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Normalised Power (W)")
        ax.set_title("Rolling 1-min NP")
        ax.legend()
        st.pyplot(fig)

    if plot_NP_to_date:
        def np_to_date_series(p):
            p30 = rolling_mean_last_N(p, 30)
            out = np.full(len(p), np.nan, dtype=float)
            for i in range(len(p)):
                s = max(0, i-29)
                out[i] = calc_np(p30[s:i+1])
            return out
        NP_A_to_date = np_to_date_series(power_A)
        NP_B_to_date = np_to_date_series(power_B)
        fig, ax = plt.subplots()
        ax.plot(time/60, NP_A_to_date, label="Rider A NP to date")
        ax.plot(time/60, NP_B_to_date, label="Rider B NP to date")
        ax.axhline(FTP_A, color="gray", linestyle="--", label="FTP A")
        ax.axhline(FTP_B, color="red", linestyle="--", label="FTP B")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Normalised Power (W)")
        ax.set_title("Cumulative NP since start")
        ax.legend()
        st.pyplot(fig)

    if plot_AvgP:
        avgP_A = np.cumsum(power_A)/(np.arange(len(power_A))+1)
        avgP_B = np.cumsum(power_B)/(np.arange(len(power_B))+1)
        fig, ax = plt.subplots()
        ax.plot(time/60, avgP_A, label="Rider A Avg Power")
        ax.plot(time/60, avgP_B, label="Rider B Avg Power")
        ax.axhline(FTP_A, color="gray", linestyle="--", label="FTP A")
        ax.axhline(FTP_B, color="red", linestyle="--", label="FTP B")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Average Power (W)")
        ax.set_title("Cumulative Average Power since start")
        ax.legend()
        st.pyplot(fig)