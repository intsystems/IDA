

import numpy as np
import matplotlib.pyplot as plt

def run_demo(seed=7, save_plots=False):
    rng = np.random.default_rng(seed)

    # -------------------------------
    # Parameters (more stable)
    # -------------------------------
    T = 75
    S = 40
    dt = 0.05
    r = 0.6
    K = 1.5
    D = 0.12
    sigma_eta = 0.06
    b = -0.7  # lower baseline so exp doesn't blow up

    def laplacian_1d(x):
        left = np.concatenate([x[..., :1], x[..., :-1]], axis=-1)
        right = np.concatenate([x[..., 1:], x[..., -1:]], axis=-1)
        return left - 2.0 * x + right

    # Simulate latent field
    x_true = np.zeros((T, S))
    grid = np.linspace(-2, 2, S)
    x0 = 0.8 * np.exp(-grid**2) + 0.15 * np.sin(2 * np.pi * grid)
    x_true[0] = x0 + rng.normal(scale=0.03, size=S)

    clip_min, clip_max = -2.5, 2.5

    for t in range(T - 1):
        x = x_true[t]
        growth = r * x * (1.0 - x / K)
        diff = D * laplacian_1d(x)
        mean_next = x + dt * (growth + diff)
        mean_next = np.clip(mean_next, clip_min, clip_max)
        x_true[t + 1] = mean_next + rng.normal(scale=sigma_eta, size=S)

    lam = np.exp(x_true + b)
    y = rng.poisson(lam)

    # Particle Filter
    Np = 400
    ess_threshold = Np / 2

    particles = np.tile(x_true[0], (Np, 1)) + rng.normal(scale=0.2, size=(Np, S))
    weights = np.ones(Np) / Np

    x_filt_mean = np.zeros((T, S))
    loglik = 0.0

    def systematic_resample(weights, rng):
        N = weights.size
        positions = (rng.random() + np.arange(N)) / N
        cumulative_sum = np.cumsum(weights)
        indexes = np.zeros(N, dtype=int)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    for t in range(T):
        lam_particles = np.exp(particles + b)
        log_w = (y[t] * particles).sum(axis=1) - lam_particles.sum(axis=1)
        log_w -= log_w.max()
        w = np.exp(log_w)
        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum == 0.0:
            w = np.ones_like(w)
            w_sum = w.sum()
        w /= w_sum
        weights = w

        loglik += log_w.max() + np.log(w_sum + 1e-300)

        x_filt_mean[t] = (weights[:, None] * particles).sum(axis=0)

        ess = 1.0 / np.sum(weights**2)
        if ess < ess_threshold:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.ones(Np) / Np

        if t < T - 1:
            growth = r * particles * (1.0 - particles / K)
            diff = D * laplacian_1d(particles)
            mean_next = particles + dt * (growth + diff)
            mean_next = np.clip(mean_next, clip_min, clip_max)
            particles = mean_next + rng.normal(scale=sigma_eta, size=particles.shape)

    # Visualizations
    plt.figure(figsize=(8, 4))
    plt.imshow(x_true, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel("Space index s")
    plt.ylabel("Time t")
    plt.title("True latent field x_t(s)")
    plt.tight_layout()
    if save_plots:
        plt.savefig('true_latent_field.png', dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.imshow(x_filt_mean, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel("Space index s")
    plt.ylabel("Time t")
    plt.title("Filtered mean of x_t(s) (Particle Filter)")
    plt.tight_layout()
    if save_plots:
        plt.savefig('filtered_mean_field.png', dpi=200)
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.imshow(y, aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel("Space index s")
    plt.ylabel("Time t")
    plt.title("Observed counts y_t(s)")
    plt.tight_layout()
    if save_plots:
        plt.savefig('observed_counts.png', dpi=200)
    plt.show()

    sel_sites = [5, S//2, S-6]
    for s_idx in sel_sites:
        plt.figure(figsize=(8, 3))
        plt.plot(x_true[:, s_idx], label="true x")
        plt.plot(x_filt_mean[:, s_idx], label="filtered mean", linestyle='--')
        plt.xlabel("Time t")
        plt.ylabel(f"x_t(s={s_idx})")
        plt.title(f"State at site s={s_idx}")
        plt.legend()
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'site_{s_idx}_timeseries.png', dpi=200)
        plt.show()

    print(f"Approx. marginal log-likelihood (PF estimate): {loglik:.2f}")

if __name__ == '__main__':
    run_demo(save_plots=True)
