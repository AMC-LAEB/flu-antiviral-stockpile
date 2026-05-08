import sys, os
import time
import numpy as np
import scipy.stats as st
from scipy.integrate import odeint
import pickle
import emcee

# Load treatment data as global (required by within_host_model.py functions)
with open('data/treatment_to_data.pkl', 'rb') as f:
    treatment_to_data = pickle.load(f)

# Load model functions — they rely on np, st, odeint, treatment_to_data in scope
exec(open('src/within_host_model.py').read())

# ── Parameter setup ──────────────────────────────────────────────────────────
# Full 16-parameter order (matches fobj signature):
# bw, bm, l, r, d, k, q, a, pw, cw, pmb, cmb, Vw0, mub, eb, eo
ALL_PARAM_NAMES = ['bw','bm','l','r','d','k','q','a','pw','cw','pmb','cmb','Vw0','mub','eb','eo']

# 13 sampled parameters and their indices in the full vector
SAMPLED_NAMES   = ['bw','bm','l','r','d','k','q','a','pw','cw','Vw0','eb','eo']
SAMPLED_INDICES = [0,   1,   2,  3,  4,  5,  6,  7,  8,   9,   12,   14,  15 ]

# 3 fixed parameters (only constrain baloxavir_resistance arm, excluded from MCMC)
FIXED_NAMES   = ['pmb','cmb','mub']
FIXED_INDICES = [10,   11,   13  ]

# Load DE best-fit vector
_pars_file = np.load('data/within_host_pars.npz')
_all_pars  = _pars_file['within_host_model_pars']  # shape (16,)

best_fit  = _all_pars[SAMPLED_INDICES]
fixed_pars = {name: _all_pars[idx] for name, idx in zip(FIXED_NAMES, FIXED_INDICES)}

# ── Prior bounds (13 sampled parameters) ─────────────────────────────────────
# Order: bw, bm, l, r, d, k, q, a, pw, cw, Vw0, eb, eo
MIN_BOUND = np.asarray([1e-8, 1e-8, 1e-5, 1e-3, 1e-10, 1.,   1e-10, 1e-1, 1e-10, 0.,  0., 0., 0.])
MAX_BOUND = np.asarray([1e-5, 1e-2, 3.,   3.,   5.,    5.,   1e-8,  3.,   1e-1,  20., 1., 1., 1.])

# Parameters with min=0 get uniform priors; those with min>0 get log-uniform priors
_LOG_UNIFORM_MASK = MIN_BOUND > 0  # True for indices 0-8 (bw..pw)


def _build_full_state(state13):
    """Reconstruct the 16-parameter vector from the 13 sampled values + fixed_pars."""
    full = np.empty(16)
    full[SAMPLED_INDICES] = state13
    for name, idx in zip(FIXED_NAMES, FIXED_INDICES):
        full[idx] = fixed_pars[name]
    return full


def fobj_mcmc(state13, fp=None):
    """Modified objective: positive negative log-likelihood over placebo/baloxavir/oseltamivir only."""
    if fp is None:
        fp = fixed_pars
    bw, bm, l, r, d, k, q, a, pw, cw, Vw0, eb, eo = state13
    rate_pars = np.asarray([bw, bm, l, r, d, k, q, a, pw, cw])
    T0 = 4e8

    ll = 0.0
    for treatment in ['placebo', 'baloxavir', 'oseltamivir']:
        tspan = np.linspace(0., 11., (11 * 24) + 1)
        t_eval = treatment_to_data[treatment]['data_t'].mean(axis=0)[1:]
        tspan = np.unique(np.concatenate((tspan, t_eval)))
        y_true_pars = treatment_to_data[treatment]['vl_norm_pars']

        if treatment == 'placebo':
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0., tspan=tspan, y_init=y0_init)
            Vw = s0[:, -2].copy()
            Vw[Vw < 5.] = 5.
            ll += compute_norm_ll(np.log10(Vw)[np.isin(tspan, t_eval)], y_true_pars)

        else:  # baloxavir or oseltamivir
            e_val = eb if treatment == 'baloxavir' else eo
            t0 = tspan[tspan < t_eval[0]]
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0., tspan=t0, y_init=y0_init)
            t1 = tspan[tspan >= t_eval[0]]
            s1 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=e_val, tspan=t1, y_init=s0[-1, :])
            Vw = s1[:, -2].copy()
            Vw[Vw < 5.] = 5.
            ll += compute_norm_ll(np.log10(Vw)[np.isin(t1, t_eval)], y_true_pars)

    return ll  # positive = negative log-likelihood


# Informative Beta prior on eb (index 11): baloxavir has near-complete efficacy
# against susceptible virus. Beta(50, 2) → mean≈0.96, 2.5th pct≈0.91.
_EB_IDX    = SAMPLED_NAMES.index('eb')
_EB_BETA_A = 50.
_EB_BETA_B = 2.


def log_prior(state13):
    """Log-uniform priors for +ve-bounded params; uniform for zero-bounded;
    Beta(50,2) for eb."""
    if np.any(state13 <= 0):
        return -np.inf
    if np.any(state13 < MIN_BOUND) or np.any(state13 > MAX_BOUND):
        return -np.inf
    lp = 0.0
    for i, (lo, hi, log_uni) in enumerate(zip(MIN_BOUND, MAX_BOUND, _LOG_UNIFORM_MASK)):
        if i == _EB_IDX:
            lp += st.beta.logpdf(state13[i], _EB_BETA_A, _EB_BETA_B)
        elif log_uni:
            lp -= np.log(state13[i]) + np.log(np.log(hi / lo))
        else:
            lp -= np.log(hi - lo)
    return lp


def log_likelihood(state13):
    try:
        nll = fobj_mcmc(state13)
        if not np.isfinite(nll):
            return -np.inf
        return -nll
    except Exception:
        return -np.inf


def log_posterior(state13):
    lp = log_prior(state13)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(state13)


# ── emcee settings ───────────────────────────────────────────────────────────
NDIM     = len(SAMPLED_NAMES)   # 13
NWALKERS = 32
NSTEPS   = 5000
CHAIN_FILE = 'outputs/mcmc_chain.h5'


def init_walkers(seed=42):
    """Tight Gaussian ball (1% perturbation) around best_fit, clipped to bounds."""
    rng = np.random.default_rng(seed)
    p0 = best_fit * (1.0 + 0.01 * rng.standard_normal((NWALKERS, NDIM)))
    p0 = np.clip(p0, MIN_BOUND + 1e-30, MAX_BOUND - 1e-30)
    return p0


def run_mcmc(nsteps=NSTEPS, trial_steps=10, chain_file=CHAIN_FILE):
    backend = emcee.backends.HDFBackend(chain_file)

    # Resume if chain already exists, otherwise reset
    try:
        completed = backend.iteration
    except Exception:
        completed = 0

    if completed > 0:
        print(f"Resuming from step {completed}/{nsteps}")
        p0 = None
    else:
        backend.reset(NWALKERS, NDIM)
        p0 = init_walkers()

        # ── Trial run to estimate total runtime ──
        print(f"Running {trial_steps}-step trial to estimate runtime...")
        t0 = time.time()
        sampler_trial = emcee.EnsembleSampler(NWALKERS, NDIM, log_posterior)
        sampler_trial.run_mcmc(p0, trial_steps, progress=False)
        trial_secs = time.time() - t0
        est_total_min = trial_secs / trial_steps * nsteps / 60.0
        print(f"  Trial ({trial_steps} steps): {trial_secs:.1f}s")
        print(f"  Estimated total ({nsteps} steps): {est_total_min:.1f} min")
        completed = 0  # backend still empty; we discard the trial sampler

    sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_posterior, backend=backend)
    remaining = nsteps - completed
    print(f"\nRunning {remaining} steps with {NWALKERS} walkers ({NDIM} dims)...")
    sampler.run_mcmc(p0, remaining, progress=True)
    print("Done.")
    return sampler


# ── Steps 5 & 6: VL posterior draws and figure ───────────────────────────────

PLOT_TSPAN = np.linspace(0., 11., 300)   # smooth time axis for plotting
ARMS       = ['placebo', 'baloxavir', 'oseltamivir']
ARM_COLORS = {'placebo': '#4393c3', 'baloxavir': '#d6604d', 'oseltamivir': '#74c476'}
ARM_LABELS = {'placebo': 'Placebo', 'baloxavir': 'Baloxavir', 'oseltamivir': 'Oseltamivir'}
N_DRAWS    = 500


def _simulate_vl(state13, arm, T0=4e8):
    """Return log10 VL on PLOT_TSPAN for a given arm and parameter draw."""
    bw, bm, l, r, d, k, q, a, pw, cw, Vw0, eb, eo = state13
    rate_pars = np.asarray([bw, bm, l, r, d, k, q, a, pw, cw])

    t_eval = treatment_to_data[arm]['data_t'].mean(axis=0)[1:]
    tspan_full = np.unique(np.concatenate((PLOT_TSPAN, t_eval)))

    y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])

    if arm == 'placebo':
        sol = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0.,
                          tspan=tspan_full, y_init=y0_init)
        Vw = sol[:, -2]
    else:
        e_val = eb if arm == 'baloxavir' else eo
        t0 = tspan_full[tspan_full < t_eval[0]]
        s0 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0.,
                         tspan=t0, y_init=y0_init)
        t1 = tspan_full[tspan_full >= t_eval[0]]
        s1 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=e_val,
                         tspan=t1, y_init=s0[-1, :])
        Vw = np.concatenate([s0[:, -2], s1[:, -2]])

    Vw = np.where(Vw < 5., 5., Vw)
    log_vl = np.log10(Vw)
    # Interpolate onto PLOT_TSPAN
    return np.interp(PLOT_TSPAN, tspan_full, log_vl)


def run_vl_posterior(chain_file=CHAIN_FILE, n_draws=N_DRAWS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    # ── Draw posterior samples ──
    backend  = emcee.backends.HDFBackend(chain_file, read_only=True)
    tau      = backend.get_autocorr_time(quiet=True)
    burn_in  = int(2 * np.max(tau))
    thin     = max(1, int(0.5 * np.min(tau)))
    flat     = backend.get_chain(discard=burn_in, thin=thin, flat=True)

    rng      = np.random.default_rng(0)
    idx      = rng.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    samples  = flat[idx]
    print(f"Using {len(samples)} posterior draws (burn-in={burn_in}, thin={thin})")

    # ── Simulate VL for all draws and arms ──
    vl_draws = {}   # arm -> (n_draws, len(PLOT_TSPAN))
    for arm in ARMS:
        print(f"  Simulating {arm}...", flush=True)
        trajs = []
        for s in samples:
            try:
                trajs.append(_simulate_vl(s, arm))
            except Exception:
                trajs.append(np.full(len(PLOT_TSPAN), np.nan))
        vl_draws[arm] = np.array(trajs)

    # Save arrays
    save_dict = {f'vl_{arm}': vl_draws[arm] for arm in ARMS}
    save_dict['tspan'] = PLOT_TSPAN
    np.save('outputs/posterior_vl_draws.npy', save_dict, allow_pickle=True)
    print("Saved outputs/posterior_vl_draws.npy")

    # MLE trajectories
    mle_vl = {arm: _simulate_vl(best_fit, arm) for arm in ARMS}

    # ── Figure ──
    try:
        font = FontProperties(family='DejaVu Serif')
    except Exception:
        font = None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    panel_labels = ['A', 'B', 'C']

    for ax, arm, label in zip(axes, ARMS, panel_labels):
        draws = vl_draws[arm]
        median = np.nanpercentile(draws, 50, axis=0)
        lo     = np.nanpercentile(draws, 2.5, axis=0)
        hi     = np.nanpercentile(draws, 97.5, axis=0)
        color  = ARM_COLORS[arm]

        # 95% CI band
        ax.fill_between(PLOT_TSPAN, lo, hi, color=color, alpha=0.25, linewidth=0)
        # Posterior median
        ax.plot(PLOT_TSPAN, median, color=color, linewidth=1.8, label='Posterior median')
        # MLE
        ax.plot(PLOT_TSPAN, mle_vl[arm], color='black', linewidth=1.2,
                linestyle='--', label='MLE')

        # Data mean ± SD
        d_arm  = treatment_to_data[arm]
        t_data = d_arm['data_t'].mean(axis=0)[1:]
        mu_d   = d_arm['vl_norm_pars'][0]
        sd_d   = d_arm['vl_norm_pars'][1]
        ax.errorbar(t_data, mu_d, yerr=sd_d, fmt='o', color='black',
                    markersize=4, capsize=3, linewidth=1., label='Data mean±SD', zorder=5)

        ax.set_title(ARM_LABELS[arm], fontsize=11)
        ax.set_xlabel('Days post infection', fontsize=9)
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=8)

        # Panel label
        ax.text(-0.12, 1.02, label, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='top')

    axes[0].set_ylabel('log₁₀ viral load (TCID₅₀/mL)', fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)

    fig.tight_layout()
    fig.savefig('outputs/fig_vl_posterior.pdf', bbox_inches='tight')
    fig.savefig('outputs/fig_vl_posterior.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("Saved outputs/fig_vl_posterior.pdf and .png")


# ── Step 4: convergence diagnostics ─────────────────────────────────────────

def run_diagnostics(chain_file=CHAIN_FILE):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import corner

    backend = emcee.backends.HDFBackend(chain_file, read_only=True)
    chain = backend.get_chain()          # (nsteps, nwalkers, ndim)
    nsteps_done = chain.shape[0]
    print(f"Chain shape: {chain.shape}")

    # Autocorrelation time
    tau = backend.get_autocorr_time(quiet=True)
    print("\nIntegrated autocorrelation time (tau):")
    for name, t in zip(SAMPLED_NAMES, tau):
        print(f"  {name:6s}: {t:.1f}")

    burn_in = int(2 * np.max(tau))
    thin    = max(1, int(0.5 * np.min(tau)))
    print(f"\nBurn-in: {burn_in} steps  |  Thinning: {thin}")

    flat = backend.get_chain(discard=burn_in, thin=thin, flat=True)
    ess  = flat.shape[0]
    print(f"Flat samples after burn-in + thinning: {ess}")
    print(f"ESS per parameter (approx = flat samples / 1 since already thinned): {ess}")
    ess_per_param = nsteps_done * NWALKERS / tau
    low_ess = []
    print("\nESS per parameter:")
    for name, e in zip(SAMPLED_NAMES, ess_per_param):
        flag = "  *** LOW ***" if e < 200 else ""
        print(f"  {name:6s}: {e:.0f}{flag}")
        if e < 200:
            low_ess.append(name)
    if low_ess:
        print(f"\nWARNING: ESS < 200 for {low_ess}. Consider running more steps.")
    else:
        print("\n✓ All parameters have ESS >= 200")

    # ── Trace plots ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(NDIM, 1, figsize=(10, 2 * NDIM), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, SAMPLED_NAMES)):
        ax.plot(chain[:, :, i], color='k', alpha=0.1, linewidth=0.4, rasterized=True)
        ax.axvline(burn_in, color='red', linestyle='--', linewidth=0.8, label='burn-in' if i == 0 else '')
        ax.set_ylabel(name, fontsize=8)
    axes[-1].set_xlabel('Step')
    axes[0].legend(fontsize=7)
    fig.suptitle('Walker traces', y=1.001)
    fig.tight_layout()
    fig.savefig('outputs/mcmc_traces.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\nSaved outputs/mcmc_traces.png")

    # ── Corner plot ──────────────────────────────────────────────────────────
    fig = corner.corner(
        flat,
        labels=SAMPLED_NAMES,
        quantiles=[0.025, 0.5, 0.975],
        show_titles=True,
        title_fmt='.3g',
        title_kwargs={'fontsize': 8},
        label_kwargs={'fontsize': 8},
    )
    fig.savefig('outputs/mcmc_corner.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("Saved outputs/mcmc_corner.png")

    return burn_in, thin, flat


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'trial'

    if mode == 'trial':
        print("── Step 3: trial run ──")
        backend = emcee.backends.HDFBackend(CHAIN_FILE)
        try:
            backend.reset(NWALKERS, NDIM)
        except Exception:
            pass
        p0 = init_walkers()
        print(f"Running 10-step trial with {NWALKERS} walkers ({NDIM} dims)...")
        t0 = time.time()
        sampler = emcee.EnsembleSampler(NWALKERS, NDIM, log_posterior)
        sampler.run_mcmc(p0, 10, progress=False)
        trial_secs = time.time() - t0
        est_total_min = trial_secs / 10 * NSTEPS / 60.0
        print(f"  Trial (10 steps): {trial_secs:.1f}s")
        print(f"  Estimated total ({NSTEPS} steps): {est_total_min:.1f} min")
        print("\nTo run the full chain: python mcmc_within_host.py run")

    elif mode == 'run':
        run_mcmc()

    elif mode == 'diagnostics':
        print("── Step 4: convergence diagnostics ──")
        run_diagnostics()

    elif mode == 'figures':
        print("── Steps 5 & 6: VL posterior draws + figure ──")
        run_vl_posterior()

    elif mode == 'sanity':
        print("Sampled parameters (13):")
        for name, val in zip(SAMPLED_NAMES, best_fit):
            print(f"  {name:6s}: {val:.6g}")
        print("\nFixed parameters (3):")
        for name, val in fixed_pars.items():
            print(f"  {name:6s}: {val:.6g}")
        print("\n── Step 2 sanity check ──")
        lpost = log_posterior(best_fit)
        llike = log_likelihood(best_fit)
        lp    = log_prior(best_fit)
        print(f"  log_prior    : {lp:.4f}")
        print(f"  log_likelihood: {llike:.4f}")
        print(f"  log_posterior : {lpost:.4f}")
        assert np.isfinite(lpost), "log_posterior is not finite at DE best-fit!"
        print("  ✓ log_posterior is finite")
