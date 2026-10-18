import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os

# To import beamformers from simulator
import sys
sys.path.append('.')
from simulator.beamformers import LCMV_SMI, RobustSOCP, RobustLCMV

def create_ula(n_mics=10, d=0.5):
    """ Creates a Uniform Linear Array along x-axis, d is spacing in wavelengths """
    c = 343.0
    f = 1000.0
    lam = c / f
    spacing = d * lam
    pos = np.zeros((n_mics, 3))
    pos[:, 0] = np.arange(n_mics) * spacing
    pos -= np.mean(pos, axis=0)
    return pos

def simulate_narrowband_covariance(array_pos, target_az, int_azs, snr=10, inrs=[30]):
    """ Generates theoretical covariance matrix R for narrowband signals """
    c = 343.0
    f = 1000.0
    M = array_pos.shape[0]

    def sv(az):
        rad = np.deg2rad(az)
        k = np.array([np.cos(rad), np.sin(rad), 0])
        tau = -np.dot(array_pos, k) / c
        return np.exp(-1j * 2 * np.pi * f * tau)

    a_t = sv(target_az)

    # Noise covariance (white noise)
    R_nn = np.eye(M)

    # Interferer covariance
    R_i = np.zeros((M, M), dtype=complex)
    for i, az in enumerate(int_azs):
        a_i = sv(az)
        p_i = 10**(inrs[i]/10.0)
        R_i += p_i * np.outer(a_i, np.conj(a_i))

    # Target covariance
    p_s = 10**(snr/10.0)
    R_s = p_s * np.outer(a_t, np.conj(a_t))

    R = R_s + R_i + R_nn
    return R, R_i + R_nn, a_t

def calc_sinr(w, R_s, R_in):
    num = np.abs(np.conj(w).T @ R_s @ w)
    den = np.abs(np.conj(w).T @ R_in @ w)
    if den == 0: return 0
    return 10 * np.log10(num / den)

def reproduce_vorobyov_socp():
    """
    Reproduces Fig 1 from Vorobyov et al. 2003.
    Output SINR vs Steering Direction Error.
    """
    os.makedirs("docs/figures", exist_ok=True)
    print("Reproducing Vorobyov (2003) SOCP vs LCMV...")
    pos = create_ula(n_mics=10)

    true_target_az = 0.0
    int_az = 30.0
    snr = 10.0
    inr = 30.0

    errors = np.linspace(0, 5, 20)
    sinr_lcmv = []
    sinr_socp = []

    # Theoretical matrices
    R, R_in, true_a = simulate_narrowband_covariance(pos, true_target_az, [int_az], snr=snr, inrs=[inr])
    R_s = 10**(snr/10.0) * np.outer(true_a, np.conj(true_a))

    # We must mock the beamformers since they expect STFT data. We will just use their math directly.
    import cvxpy as cp
    M = pos.shape[0]
    c = 343.0
    f = 1000.0

    for err in errors:
        steer_az = true_target_az + err

        # Calculate nominal steering vector with error
        rad = np.deg2rad(steer_az)
        k = np.array([np.cos(rad), np.sin(rad), 0])
        tau = -np.dot(pos, k) / c
        a_0 = np.exp(-1j * 2 * np.pi * f * tau)

        # 1. Standard LCMV (MVDR)
        R_inv = np.linalg.inv(R + 1e-3*np.eye(M))
        w_lcmv = (R_inv @ a_0) / (np.conj(a_0) @ R_inv @ a_0)
        sinr_lcmv.append(calc_sinr(w_lcmv, R_s, R_in))

        # 2. SOCP
        eps = 1.0 # Mismatch bound
        w = cp.Variable(M, complex=True)
        R_psd = 0.5 * (R + np.conj(R.T))
        obj = cp.Minimize(cp.quad_form(w, R_psd))
        constraints = [
            cp.real(a_0.conj().T @ w) >= eps * cp.norm(w, 2) + 1,
            cp.imag(a_0.conj().T @ w) == 0
        ]
        prob = cp.Problem(obj, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            w_socp = w.value if w.value is not None else w_lcmv
        except:
            w_socp = w_lcmv

        sinr_socp.append(calc_sinr(w_socp, R_s, R_in))

    plt.figure(figsize=(8,6))
    plt.plot(errors, sinr_socp, 'b-o', label='Robust SOCP (Vorobyov)')
    plt.plot(errors, sinr_lcmv, 'r-s', label='Standard LCMV/SMI')
    plt.title("Reproduction of Vorobyov 2003: Output SINR vs Mismatch")
    plt.xlabel("Look Direction Error (Degrees)")
    plt.ylabel("Output SINR (dB)")
    plt.grid(True)
    plt.legend()
    plt.savefig("docs/figures/vorobyov_socp_mismatch.png")
    plt.close()

def reproduce_mailloux_cmt():
    """
    Reproduces Beampattern null broadening from Mailloux (1995) / CMT
    """
    print("Reproducing Mailloux (1995) CMT Null Broadening...")
    pos = create_ula(n_mics=10)
    target_az = 0.0
    int_az = 30.0

    # We want to show the beampattern, so we simulate a very loud interferer
    R, _, _ = simulate_narrowband_covariance(pos, target_az, [int_az], snr=0, inrs=[40])

    c = 343.0
    f = 1000.0
    M = pos.shape[0]

    def sv(az):
        rad = np.deg2rad(az)
        k = np.array([np.cos(rad), np.sin(rad), 0])
        tau = -np.dot(pos, k) / c
        return np.exp(-1j * 2 * np.pi * f * tau)

    a_0 = sv(target_az)

    # 1. Standard LCMV
    R_inv = np.linalg.inv(R + 1e-3*np.eye(M))
    w_lcmv = (R_inv @ a_0) / (np.conj(a_0) @ R_inv @ a_0)

    # 2. CMT LCMV
    delta_theta = 5.0 # 5 degree broadening
    delta_rad = np.deg2rad(delta_theta)
    T_mat = np.ones((M, M), dtype=complex)
    for i in range(M):
        for j in range(M):
            dist = np.linalg.norm(pos[i] - pos[j])
            arg = 2 * np.pi * f * dist * np.sin(delta_rad) / c
            if arg != 0:
                T_mat[i, j] = np.sin(arg) / arg

    R_cmt = R * T_mat
    R_cmt_inv = np.linalg.inv(R_cmt + 1e-3*np.eye(M))
    w_cmt = (R_cmt_inv @ a_0) / (np.conj(a_0) @ R_cmt_inv @ a_0)

    # Plot Beampattern
    angles = np.linspace(-90, 90, 360)
    bp_lcmv = []
    bp_cmt = []

    for az in angles:
        a_scan = sv(az)
        bp_lcmv.append(np.abs(np.conj(w_lcmv).T @ a_scan)**2)
        bp_cmt.append(np.abs(np.conj(w_cmt).T @ a_scan)**2)

    # Convert to dB and normalize
    bp_lcmv = 10 * np.log10(np.array(bp_lcmv) + 1e-10)
    bp_cmt = 10 * np.log10(np.array(bp_cmt) + 1e-10)

    bp_lcmv -= np.max(bp_lcmv)
    bp_cmt -= np.max(bp_cmt)

    plt.figure(figsize=(8,6))
    plt.plot(angles, bp_lcmv, 'r-', label='Standard LCMV (Sharp Null)')
    plt.plot(angles, bp_cmt, 'b-', linewidth=2, label='CMT LCMV (Broadened Null)')
    plt.axvline(x=int_az, color='k', linestyle='--', label='Interferer Location')
    plt.title("Reproduction of Mailloux 1995: CMT Null Broadening")
    plt.xlabel("Angle (Degrees)")
    plt.ylabel("Beampattern Power (dB)")
    plt.ylim(-60, 5)
    plt.grid(True)
    plt.legend()
    plt.savefig("docs/figures/mailloux_cmt_beampattern.png")
    plt.close()

if __name__ == "__main__":
    reproduce_vorobyov_socp()
    reproduce_mailloux_cmt()
    print("Reproductions complete. Plots saved to docs/figures/")
