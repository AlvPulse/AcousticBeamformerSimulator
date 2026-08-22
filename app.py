import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import io

from acoustic_sim.dsp_array import ArrayGeometry
from acoustic_sim.dsp_signal import tone_burst, synthesize_array_recording, compute_spatial_coherence
from acoustic_sim.beamformer import array_factor, delay_and_sum, compute_map_papr, compute_isl

st.set_page_config(page_title="Acoustic Array Diagnostics", layout="wide")

st.markdown("""
<style>
.metric-card {
    background-color: #f7f7f5;
    border: 1px solid #e2e1dc;
    border-radius: 8px;
    padding: 10px 15px;
    margin-bottom: 10px;
}
.metric-label { font-size: 12px; color: #8a8a86; }
.metric-val { font-size: 20px; font-weight: 600; color: #16161a; }
.metric-sub { font-size: 11px; color: #5b5b58; }
</style>
""", unsafe_allow_html=True)

st.title("Acoustic Array Beamforming Diagnostic Explorer")

# --- Default Array Configurations ---
PRESETS = {
    "grid4x4": "\n".join([f"{x*0.15:.3f}, {y*0.15:.3f}" for x in range(4) for y in range(4)]),
    "ula8": "\n".join([f"{i*0.15:.3f}, 0.0" for i in range(8)]),
    "uca6": "\n".join([f"{0.3*np.cos(i*2*np.pi/6):.3f}, {0.3*np.sin(i*2*np.pi/6):.3f}" for i in range(6)])
}

def parse_sensors(text):
    x_vals, y_vals = [], []
    for line in text.split('\n'):
        if line.strip():
            try:
                parts = line.split(',')
                x_vals.append(float(parts[0].strip()))
                y_vals.append(float(parts[1].strip()))
            except:
                pass
    return x_vals, y_vals

# --- UI Controls ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Array Geometry")
    preset = st.selectbox("Preset", ["grid4x4", "ula8", "uca6", "custom"])

    if "sensor_text" not in st.session_state:
        st.session_state.sensor_text = PRESETS["grid4x4"]

    if preset != "custom":
        st.session_state.sensor_text = PRESETS[preset]

    sensor_text = st.text_area("Sensors (x, y in meters)", st.session_state.sensor_text, height=150)

    spatial_window = st.selectbox("Spatial Tapering (Window)", ["None", "Blackman", "Hanning", "Hamming"])

    ideal_mode = st.checkbox("Ideal array factor only (no noise, no path loss)", value=True)

with col2:
    st.subheader("Environment & Source")
    c1, c2 = st.columns(2)
    with c1:
        freq = st.slider("Frequency (Hz)", 100, 8000, 2000, 10)
        az = st.slider("Source azimuth (deg)", -180, 180, 45, 1)
        el = st.slider("Source elevation (deg)", 0, 90, 25, 1)
        dist = st.slider("Source distance (m)", 1, 100, 10, 1, disabled=ideal_mode)
        level = st.slider("Source level (dB SPL @ 1m)", 40, 100, 80, 1, disabled=ideal_mode)
    with c2:
        noise = st.slider("Noise floor (dB SPL)", 0, 80, 30, 1, disabled=ideal_mode)
        temp = st.slider("Temperature (C)", -10, 40, 20, 1, disabled=ideal_mode)
        hum = st.slider("Relative humidity (%)", 10, 100, 50, 1, disabled=ideal_mode)
        ground = st.slider("Ground effect loss (dB)", 0.0, 10.0, 0.0, 0.5, disabled=ideal_mode)
        shadowing = st.slider("Shadowing std dev (dB)", 0.0, 5.0, 0.0, 0.5, disabled=ideal_mode)

# --- Compute ---
x_vals, y_vals = parse_sensors(sensor_text)
if len(x_vals) < 2:
    st.warning("Please provide at least 2 sensors.")
    st.stop()

arr = ArrayGeometry(x_vals, y_vals)
fs = max(16000, int(freq * 2.5))
c = 331.4 + 0.6 * temp

AZ_STEP = 2.0
EL_STEP = 2.0
grid_az = np.arange(-180, 180 + AZ_STEP, AZ_STEP)
grid_el = np.arange(0, 90 + EL_STEP, EL_STEP)

spatial_weights = arr.get_spatial_weights(spatial_window.lower())

if ideal_mode:
    # Run Stage 0
    power_map = array_factor(arr, freq, az, el, grid_az, grid_el, c=c, spatial_weights=spatial_weights)
    # Convert to dB manually since array_factor returns linear power normalized
    power_db = 10 * np.log10(power_map + 1e-15)
else:
    # Run Stage 1 full pipeline
    env = {'c': c, 'temp_c': temp, 'rel_humidity': hum, 'pressure_kpa': 101.325,
           'ground_effect_loss_db': ground, 'shadowing_std_db': shadowing}
    sig = tone_burst(freq, 0.05, fs)
    recording = synthesize_array_recording(arr, sig, az, el, dist, level, fs, noise, freq, env)
    power_map = delay_and_sum(recording, arr, fs, grid_az, grid_el, c=c, spatial_weights=spatial_weights)
    power_db = 10 * np.log10(power_map + 1e-15)

# Compute sanity check features
papr = compute_map_papr(power_map)
isl = compute_isl(power_map, grid_az, grid_el, az, el)
spatial_coh = 1.0 if ideal_mode else compute_spatial_coherence(recording, fs)

# Theoretical vs Empirical Gain
# Theoretical AG for random noise = 10 * log10(N)
theoretical_ag = 10 * np.log10(arr.n_sensors)

input_snr = np.nan
empirical_ag = np.nan
if not ideal_mode:
    # Estimate input SNR from config (Source SPL - Path Loss - Noise Floor)
    # This is a rough theoretical estimation of what hit the array.
    from acoustic_sim.propagation import path_loss_db
    pl = path_loss_db(freq, dist, temp, hum, 101.325, ground, shadowing)
    input_snr = level - pl - noise

    # Estimate empirical output SNR by measuring the peak vs the average background
    # We use PAPR as a proxy for empirical output SNR because peak = signal+noise, avg = noise
    # It's an approximation for the dashboard sanity check.
    output_snr = papr
    empirical_ag = output_snr - input_snr


# Normalize power to 0 dB max
max_val = np.max(power_db)
power_db -= max_val

# --- Metrics ---
def find_secondary_peak(power_db):
    # simple 2D peak finding suppressing the mainlobe
    flat = power_db.flatten()
    max_idx = np.argmax(flat)

    # Mask out the mainlobe area
    masked_db = power_db.copy()
    az_idx = max_idx % len(grid_az)
    el_idx = max_idx // len(grid_az)

    # zero out a box around mainlobe
    box = int(10 / AZ_STEP) # ~10 degree mask
    for r in range(max(0, el_idx-box), min(power_db.shape[0], el_idx+box)):
        for col_idx in range(max(0, az_idx-box), min(power_db.shape[1], az_idx+box)):
            masked_db[r, col_idx] = -np.inf

    sec_idx = np.argmax(masked_db.flatten())
    sec_az = grid_az[sec_idx % len(grid_az)]
    sec_el = grid_el[sec_idx // len(grid_az)]
    sec_val = masked_db.flatten()[sec_idx]

    return sec_val, sec_az, sec_el

def beamwidth_3db(power_db):
    flat = power_db.flatten()
    max_idx = np.argmax(flat)
    el_idx = max_idx // len(grid_az)

    row = power_db[el_idx, :]
    peak_az_idx = np.argmax(row)

    l_idx = peak_az_idx
    while l_idx > 0 and row[l_idx] > -3:
        l_idx -= 1

    r_idx = peak_az_idx
    while r_idx < len(row)-1 and row[r_idx] > -3:
        r_idx += 1

    return (r_idx - l_idx) * AZ_STEP

sec_val, sec_az, sec_el = find_secondary_peak(power_db)
bw = beamwidth_3db(power_db)

st.markdown("### Automated Sanity Check Features")
sm1, sm2, sm3, sm4 = st.columns(4)

def status_color(val, thresh_good, thresh_bad, invert=False):
    if invert:
        if val <= thresh_good: return "#1d7a4c" # green
        if val >= thresh_bad: return "#c23b3b" # red
        return "#b8790b" # orange
    else:
        if val >= thresh_good: return "#1d7a4c"
        if val <= thresh_bad: return "#c23b3b"
        return "#b8790b"

papr_color = status_color(papr, 15, 5)
with sm1:
    st.markdown(f'<div class="metric-card" style="border-left: 4px solid {papr_color}"><div class="metric-label">Map PAPR</div><div class="metric-val">{papr:.1f} dB</div><div class="metric-sub">>15dB is sharp, <5dB is noisy</div></div>', unsafe_allow_html=True)

coh_color = status_color(spatial_coh, 0.7, 0.3)
with sm2:
    st.markdown(f'<div class="metric-card" style="border-left: 4px solid {coh_color}"><div class="metric-label">Spatial Coherence</div><div class="metric-val">{spatial_coh:.2f}</div><div class="metric-sub">1.0 is perfect wavefield</div></div>', unsafe_allow_html=True)

ag_diff = empirical_ag - theoretical_ag if not ideal_mode else 0
ag_color = status_color(ag_diff, -3, -10)
with sm3:
    st.markdown(f'<div class="metric-card" style="border-left: 4px solid {ag_color}"><div class="metric-label">Empirical Array Gain</div><div class="metric-val">{"N/A" if ideal_mode else f"{empirical_ag:.1f} dB"}</div><div class="metric-sub">Theoretical limit: {theoretical_ag:.1f} dB</div></div>', unsafe_allow_html=True)

isl_color = status_color(isl, -5, 5, invert=True)
with sm4:
    st.markdown(f'<div class="metric-card" style="border-left: 4px solid {isl_color}"><div class="metric-label">Integrated Sidelobes</div><div class="metric-val">{isl:.1f} dB</div><div class="metric-sub">< 0dB means mainlobe dominates</div></div>', unsafe_allow_html=True)


st.markdown("### Classic Beamformer Metrics")
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Mainlobe SNR</div><div class="metric-val">{"N/A (ideal)" if ideal_mode else f"{np.round(level - noise)} dB"}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Highest secondary peak</div><div class="metric-val">{sec_val:.1f} dB</div><div class="metric-sub">at az {sec_az:.0f}, el {sec_el:.0f}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">-3dB azimuth beamwidth</div><div class="metric-val">{bw:.1f}°</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Array sensors</div><div class="metric-val">{arr.n_sensors}</div></div>', unsafe_allow_html=True)

# --- Plots ---

# Interpolate for smoother polar heatmap if desired, but Plotly polar contour handles it decently.
# For Polar, we need R and Theta. R is (90 - el), Theta is az.
# Actually, plotly doesn't have a direct polar heatmap. We can use a Scatterpolar with markers,
# or transform the grid to Cartesian and use an Image or Heatmap with aspect ratio.
# Let's map el to radius (0 at center=90 deg el, R at edge=0 deg el).

import plotly.express as px

# Cartesian Heatmap
fig_rect = go.Figure(data=go.Heatmap(
    z=power_db,
    x=grid_az,
    y=grid_el,
    colorscale='Jet',
    zmin=-40, zmax=0
))
fig_rect.update_layout(
    title="Rectangular Heatmap (Azimuth vs Elevation)",
    xaxis_title="Azimuth (deg)",
    yaxis_title="Elevation (deg)",
    height=400,
    margin=dict(l=40, r=40, t=40, b=40)
)
# Add true DOA marker
fig_rect.add_trace(go.Scatter(x=[az], y=[el], mode='markers', marker=dict(color='white', symbol='cross', size=10), showlegend=False))

# Cuts
az_idx = np.argmin(np.abs(grid_az - az))
el_idx = np.argmin(np.abs(grid_el - el))

fig_az_cut = go.Figure()
fig_az_cut.add_trace(go.Scatter(x=grid_az, y=power_db[el_idx, :], mode='lines', name='Power'))
fig_az_cut.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3dB")
fig_az_cut.update_layout(title=f"Power vs Azimuth (at True El={el}°)", yaxis_range=[-40, 2], height=300)

fig_el_cut = go.Figure()
fig_el_cut.add_trace(go.Scatter(x=grid_el, y=power_db[:, az_idx], mode='lines', name='Power'))
fig_el_cut.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3dB")
fig_el_cut.update_layout(title=f"Power vs Elevation (at True Az={az}°)", yaxis_range=[-40, 2], height=300)


pc1, pc2 = st.columns(2)
with pc1:
    st.plotly_chart(fig_rect, width="stretch")
with pc2:
    st.plotly_chart(fig_az_cut, width="stretch")

pc3, pc4 = st.columns(2)
with pc3:
    st.plotly_chart(fig_el_cut, width="stretch")
with pc4:
    # A simple polar visualization by mapping to x,y Cartesian
    # Create a dense grid in x,y
    n_pts = 200
    x_c = np.linspace(-1, 1, n_pts)
    y_c = np.linspace(-1, 1, n_pts)
    X, Y = np.meshgrid(x_c, y_c)
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(X, -Y) * 180 / np.pi # -180 to 180, 0 is at top

    # Map R to elevation: R=0 is el=90, R=1 is el=0
    EL_map = 90 * (1 - R)
    AZ_map = Theta

    # Interpolate
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((grid_el, grid_az), power_db, bounds_error=False, fill_value=-40)

    pts = np.column_stack((EL_map.flatten(), AZ_map.flatten()))
    Z = interp(pts).reshape((n_pts, n_pts))
    Z[R > 1] = np.nan # mask outside circle

    fig_polar = go.Figure(data=go.Heatmap(
        z=Z, x=x_c, y=y_c, colorscale='Jet', zmin=-40, zmax=0, showscale=False
    ))
    # True source
    r_src = 1 - el/90.0
    th_src = az * np.pi / 180.0
    x_src = r_src * np.sin(th_src)
    y_src = -r_src * np.cos(th_src)
    fig_polar.add_trace(go.Scatter(x=[x_src], y=[y_src], mode='markers', marker=dict(color='white', symbol='cross', size=10), showlegend=False))

    fig_polar.update_layout(
        title="Polar Heatmap (Center=Boresight)",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor='black'
    )
    st.plotly_chart(fig_polar, width="stretch")

st.caption("A flat (z=0) array cannot distinguish +elevation from -elevation. The ambiguity is folded in.")
