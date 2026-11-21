"""
Physics-based numerical simulation of a 4-DOF shear-building structure,
used as a stand-in for the LANL/IASC-ASCE class of benchmark structures
(direct network access to the original recorded datasets, e.g. the LANL
three-story structure archive and the KW51 Zenodo record, was not reachable
from this execution environment -- see paper Limitations section).

Model: lumped-mass shear frame, 4 stories.
  M x'' + C x' + K x = F(t)
  M = diag(m1..m4)
  K = standard shear-building stiffness assembly from story stiffnesses k1..k4
  C = Rayleigh damping (alpha*M + beta*K)
  F(t) = band-limited random excitation force applied at each floor (ambient-
         vibration-type loading, consistent with common SHM operational practice)

"Damage" = stiffness reduction in a single story (story 1, the most common
damage-scenario convention in the shear-building SHM literature), applied at
one of several severities. Each run is a different random excitation /
measurement-noise realization, sampled at 200 Hz for 20 s, giving 4-channel
(one accelerometer per floor) time series that are then windowed for the
classification task.
"""
import numpy as np
from scipy.integrate import solve_ivp

RNG_GLOBAL = np.random.default_rng(42)

FS = 200.0          # sampling rate, Hz
DT = 1.0 / FS
DURATION = 20.0      # seconds per run
N_SAMPLES = int(DURATION * FS)

# Nominal structure parameters (SI-consistent arbitrary units, chosen to give
# first-mode frequencies in the few-Hz range typical of small test structures)
M_VALUES = np.array([1.0, 1.0, 1.0, 1.0])          # story masses
K_NOMINAL = np.array([800.0, 800.0, 800.0, 800.0])  # story stiffnesses
ALPHA_RAYLEIGH = 0.6
BETA_RAYLEIGH = 0.0015


def assemble_matrices(k):
    n = len(k)
    M = np.diag(M_VALUES)
    K = np.zeros((n, n))
    for i in range(n):
        k_below = k[i]
        k_above = k[i + 1] if i + 1 < n else 0.0
        K[i, i] = k_below + k_above
        if i > 0:
            K[i, i - 1] = -k_below
            K[i - 1, i] = -k_below
    C = ALPHA_RAYLEIGH * M + BETA_RAYLEIGH * K
    return M, C, K


def simulate_run(damage_story=None, severity=0.0, seed=0, noise_snr_db=15.0):
    """Simulate one 20s run. damage_story: 0-indexed story with reduced
    stiffness, or None for healthy. severity: fractional stiffness loss."""
    rng = np.random.default_rng(seed)
    k = K_NOMINAL.copy()
    if damage_story is not None:
        k[damage_story] *= (1.0 - severity)
    M, C, K = assemble_matrices(k)
    Minv = np.linalg.inv(M)

    # band-limited random excitation force on each floor: white noise low-pass
    # filtered to ~0-25 Hz, generated once per run and interpolated during integration
    t_grid = np.arange(0, DURATION + DT, DT)
    raw_noise = rng.normal(0, 1.0, size=(len(t_grid), 4))
    # simple moving-average low-pass filter (zero-phase) to band-limit the force
    win = 5
    kernel = np.ones(win) / win
    force_signal = np.apply_along_axis(
        lambda col: np.convolve(col, kernel, mode="same"), 0, raw_noise
    ) * 15.0

    def force_at(t):
        idx = min(int(t / DT), len(t_grid) - 1)
        return force_signal[idx]

    def rhs(t, y):
        x = y[:4]
        v = y[4:]
        f = force_at(t)
        a = Minv @ (f - C @ v - K @ x)
        return np.concatenate([v, a])

    y0 = np.zeros(8)
    sol = solve_ivp(rhs, [0, DURATION], y0, t_eval=t_grid, method="RK45", max_step=DT)

    # acceleration output (what accelerometers measure) = a = Minv(F - Cv - Kx)
    X = sol.y[:4, :].T
    V = sol.y[4:, :].T
    A = np.zeros_like(X)
    for i, ti in enumerate(t_grid):
        f = force_at(ti)
        A[i] = Minv @ (f - C @ V[i] - K @ X[i])

    # measurement noise (additive Gaussian, fixed SNR)
    sig_power = np.mean(A ** 2, axis=0, keepdims=True)
    snr_lin = 10 ** (noise_snr_db / 10.0)
    noise_power = sig_power / snr_lin
    A_noisy = A + rng.normal(0, 1, A.shape) * np.sqrt(noise_power)

    return A_noisy[:N_SAMPLES].astype(np.float32)  # shape (N_SAMPLES, 4)


def build_dataset(n_healthy_runs=40, n_damaged_runs=40, severities=(0.20, 0.35, 0.50)):
    runs = []
    labels = []
    meta = []
    seed = 1000
    for i in range(n_healthy_runs):
        A = simulate_run(damage_story=None, severity=0.0, seed=seed)
        runs.append(A)
        labels.append(0)
        meta.append({"class": "healthy", "severity": 0.0, "seed": seed})
        seed += 1
    for i in range(n_damaged_runs):
        sev = severities[i % len(severities)]
        A = simulate_run(damage_story=0, severity=sev, seed=seed)
        runs.append(A)
        labels.append(1)
        meta.append({"class": "damaged", "severity": sev, "seed": seed})
        seed += 1
    return runs, np.array(labels), meta


if __name__ == "__main__":
    import time
    t0 = time.time()
    runs, labels, meta = build_dataset()
    print(f"Simulated {len(runs)} runs in {time.time()-t0:.1f}s; "
          f"healthy={np.sum(labels==0)}, damaged={np.sum(labels==1)}")
    np.savez_compressed(
        "outputs/raw_runs.npz",
        runs=np.stack(runs), labels=labels,
        meta=np.array([str(m) for m in meta]),
    )
    print("Saved outputs/raw_runs.npz, shape:", np.stack(runs).shape)
