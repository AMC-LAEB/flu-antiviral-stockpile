"""
mcmc_renewal_posterior.py

For each of 500 MCMC posterior draws:
  1. Re-anchors Hill k1/k2 to the CENTERSTONE constraint
     (29% BXM transmission reduction at treatment_t ~ 2.47 days post-infection,
      integrated over 6 days post-treatment).
  2. Computes full infectiousness profiles (infpro, R_theta) using the
     draw's within-host VL trajectories.
  3. Runs the renewal (between-host) model for the specified pandemic/treatment.

Pandemics : covid (R=3.0), pandemic_1918 (R=2.0), pandemic_1968 (R=1.8),
            pandemic_2009 (R=1.5)
Treatments: BXM (baloxavir), OSL (oseltamivir)

Outputs per scenario (all in rebuttal/outputs/):
  posterior_deaths_{pandemic}_{treatment}.npy   — deaths per draw
  posterior_courses_{pandemic}_{treatment}.npy  — treatment courses per draw
  posterior_k1k2_draws.npy                      — Hill k1/k2 per draw (shared)
  baseline_deaths_{pandemic}.npy                — no-treatment deaths (shared per pandemic)
  pop_n.npy                                     — USA population size (shared)
  renewal_checkpoint_{pandemic}_{treatment}.npz — checkpoint every 10 draws
  summary_{pandemic}_{treatment}.csv            — summary table
  fig_deaths_averted_{pandemic}_{treatment}.pdf — distribution figure

Modes (run from rebuttal/ directory):
    python mcmc_renewal_posterior.py --pandemic covid --treatment BXM
    python mcmc_renewal_posterior.py run_all          # all 8 scenarios sequentially
    python mcmc_renewal_posterior.py summarise        # all completed scenarios
    python mcmc_renewal_posterior.py summarise --pandemic covid --treatment BXM
"""

import sys
import os
import time
import numpy as np
import emcee
from scipy.optimize import minimize
from scipy.integrate import trapezoid

# ── Paths ─────────────────────────────────────────────────────────────────────
REBUTTAL_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR     = os.path.dirname(REBUTTAL_DIR)
OUTPUTS_DIR    = os.path.join(REBUTTAL_DIR, 'outputs')
CHAIN_FILE     = os.path.join(OUTPUTS_DIR, 'mcmc_chain.h5')
K1K2_FILE      = os.path.join(OUTPUTS_DIR, 'posterior_k1k2_draws.npy')
POP_FILE       = os.path.join(OUTPUTS_DIR, 'pop_n.npy')
PARENT_DATA_DIR = os.path.join(PARENT_DIR, 'data')

# renewal module lives in parent
sys.path.insert(0, PARENT_DIR)
import renewal
import sciris as sc
from renewal.utils import (viral_load_trajectory, compute_infectiousness,
                            discretize_pmf, auc)

# ── Scenario configuration ────────────────────────────────────────────────────
PANDEMIC_CONFIG = {
    'covid':         {'R': 3.0, 'profile': 'covid'},
    'pandemic_1918': {'R': 2.0, 'profile': 'pandemic_1918'},
    'pandemic_1968': {'R': 1.8, 'profile': 'pandemic_1968'},
    'pandemic_2009': {'R': 1.5, 'profile': 'pandemic_2009'},
}
TREATMENT_CONFIG = {
    'BXM': {'plan': 'tat_BXM.xlsx', 'treated_n_idx': 0},  # axis-2 index in treated_n
    'OSL': {'plan': 'tat_OSL.xlsx', 'treated_n_idx': 1},
}
ALL_SCENARIOS = [(p, t) for p in PANDEMIC_CONFIG for t in TREATMENT_CONFIG]


def _out(pandemic, treatment, name):
    """Return path for a scenario-specific output file."""
    return os.path.join(OUTPUTS_DIR, f'{name}_{pandemic}_{treatment}.npy')

def _ckpt(pandemic, treatment):
    return os.path.join(OUTPUTS_DIR, f'renewal_checkpoint_{pandemic}_{treatment}.npz')

def _baseline_file(pandemic):
    return os.path.join(OUTPUTS_DIR, f'baseline_deaths_{pandemic}.npy')

def _summary_csv(pandemic, treatment):
    return os.path.join(OUTPUTS_DIR, f'summary_{pandemic}_{treatment}.csv')

def _fig_pdf(pandemic, treatment):
    return os.path.join(OUTPUTS_DIR, f'fig_deaths_averted_{pandemic}_{treatment}.pdf')


# ── Within-host parameter setup (mirrors mcmc_within_host.py) ─────────────────
SAMPLED_NAMES   = ['bw','bm','l','r','d','k','q','a','pw','cw','Vw0','eb','eo']
SAMPLED_INDICES = [0,   1,   2,  3,  4,  5,  6,  7,  8,   9,   12,   14,  15]
FIXED_NAMES     = ['pmb','cmb','mub']
FIXED_INDICES   = [10,   11,   13]

_npz         = np.load(os.path.join(REBUTTAL_DIR, 'data', 'within_host_pars.npz'))
_all_pars    = _npz['within_host_model_pars']
best_fit     = _all_pars[SAMPLED_INDICES]
fixed_pars   = {n: _all_pars[i] for n, i in zip(FIXED_NAMES, FIXED_INDICES)}

TREATMENT_T  = float(_npz['treatment_t'][0])   # 2.468 days
ENDPOINT_T   = 6
TSPAN_INFPRO = np.arange(0, (14 * 24) + 1) / 24

_K1_INIT = 5.6884
_K2_INIT = 1.0000


def _build_full_state(state13):
    full = np.empty(16)
    full[SAMPLED_INDICES] = state13
    for name, idx in zip(FIXED_NAMES, FIXED_INDICES):
        full[idx] = fixed_pars[name]
    return full


# ── Parallel worker (module-level so multiprocessing can pickle) ───────────────
# These globals are populated by _worker_init in each subprocess; they are
# None in the main process and never used there.
_w_pars  = None
_w_pop   = None
_w_trt   = None
_w_dummy = None

def _worker_init(pandemic, treatment):
    global _w_pars, _w_pop, _w_trt, _w_dummy
    _w_pars  = make_pars(pandemic, treatment, treatment_bool=True)
    _w_pop   = renewal.Pop(_w_pars)
    _w_trt   = treatment
    _w_dummy = os.path.join(OUTPUTS_DIR, '_tmp_renewal')
    os.makedirs(_w_dummy, exist_ok=True)

def _run_draw_worker(args):
    i, state13 = args
    try:
        all_16          = _build_full_state(state13)
        k1, k2          = fit_hill_k1k2(all_16)
        deaths, courses = run_sim_with_infpro(
            all_16, k1, k2, _w_pars, _w_pop, _w_trt, _w_dummy)
        return i, deaths, courses, k1, k2
    except Exception as exc:
        print(f"  Draw {i}: ERROR — {exc}", flush=True)
        return i, np.nan, np.nan, np.nan, np.nan


# ── Hill k1/k2 fitting ────────────────────────────────────────────────────────

def fit_hill_k1k2(all_16_pars, target_eff=0.29, k1_init=_K1_INIT, k2_init=_K2_INIT):
    """
    Find Hill k1, k2 so BXM at TREATMENT_T gives target_eff transmission
    reduction (CENTERSTONE constraint), integrated over ENDPOINT_T days.
    Uses L-BFGS-B with bounds k1∈[0.01,10], k2∈[0.5,10] in log-space.
    k1/k2 depend only on within-host parameters, not on pandemic/treatment.
    """
    tspan = TSPAN_INFPRO
    vlt   = viral_load_trajectory(all_16_pars, tspan)

    pl_V, _, _ = vlt.get_vl_traj(treatment='placebo')
    pl_V       = np.where(pl_V < 5., 5., pl_V)
    log_pl     = np.log10(pl_V)

    bxm_V, _, bxm_ts = vlt.get_vl_traj(treatment_t=TREATMENT_T, treatment='baloxavir')
    bxm_V   = np.where(bxm_V < 5., 5., bxm_V)
    log_bxm = np.log10(bxm_V)[np.isin(bxm_ts, tspan)]

    mask  = tspan <= TREATMENT_T + ENDPOINT_T
    t_m   = tspan[mask]
    pl_m  = log_pl[mask]
    bxm_m = log_bxm[mask]

    def _obj(log_pars):
        k1, k2  = np.exp(log_pars)
        hp      = np.array([k1, k2])
        pl_inf  = compute_infectiousness(pl_m,  hp, 'hill')
        bxm_inf = compute_infectiousness(bxm_m, hp, 'hill')
        pl_auc  = trapezoid(pl_inf,  t_m)
        bxm_auc = trapezoid(bxm_inf, t_m)
        eff     = 1. - bxm_auc / pl_auc if pl_auc > 0 else 0.
        return (target_eff - eff) ** 2

    log_bounds = [(np.log(0.01), np.log(10.)), (np.log(0.5), np.log(10.))]
    x0 = np.clip(np.log([k1_init, k2_init]),
                 [b[0] for b in log_bounds],
                 [b[1] for b in log_bounds])
    res = minimize(_obj, x0=x0, method='L-BFGS-B', bounds=log_bounds,
                   options={'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000})
    return float(np.exp(res.x[0])), float(np.exp(res.x[1]))


# ── Infectiousness profile computation ────────────────────────────────────────

def compute_infpro_for_draw(all_16_pars, k1, k2):
    """
    Replicates renewal.utils.load_infectiousness_data with per-draw
    within_host_pars and Hill k1/k2.  Returns the same 8-tuple.
    Infpro is independent of pandemic and treatment scenario.
    """
    tspan     = TSPAN_INFPRO
    hill_pars = np.array([k1, k2])
    vlt       = viral_load_trajectory(all_16_pars, tspan)

    pl_V, _, _ = vlt.get_vl_traj(treatment='placebo')
    pl_V       = np.where(pl_V < 5., 5., pl_V)
    log_pl     = np.log10(pl_V)

    unt_inf     = compute_infectiousness(log_pl, hill_pars, 'hill')
    unt_auc     = auc(tspan, unt_inf)
    unt_profile = discretize_pmf(14, tspan, unt_inf)
    if unt_profile.sum() > 0:
        unt_profile /= unt_profile.sum()
    timepoints_n = unt_profile.size   # 15

    infpro_list  = []
    profile_id   = 0
    res_id_start = np.zeros(2, dtype=np.int32)

    for t_idx, treatment in enumerate(['baloxavir', 'oseltamivir', 'baloxavir_resistance']):
        is_resistance = (treatment == 'baloxavir_resistance')
        if is_resistance:
            t_idx = 0
            res_id_start[0] = profile_id

        for treatment_d in range(15):
            pts = tspan[(tspan >= treatment_d - 0.5) & (tspan <= treatment_d + 0.5)]
            pts = pts[pts > 0.]
            if len(pts) == 0:
                pts = np.array([max(0.042, float(treatment_d))])

            vl_acc = np.zeros((len(pts), tspan.size))
            vm_acc = np.zeros((len(pts), tspan.size)) if is_resistance else None

            for tt_i, tt in enumerate(pts):
                V, Vm, ts = vlt.get_vl_traj(treatment_t=tt, treatment=treatment)
                V = np.where(V < 5., 5., V)
                vl_acc[tt_i] = V[np.isin(ts, tspan)]
                if is_resistance and Vm is not None:
                    Vm = np.where(Vm < 5., 5., Vm)
                    vm_acc[tt_i] = Vm[np.isin(ts, tspan)]

            mean_V  = np.where(vl_acc.mean(axis=0) < 5., 5., vl_acc.mean(axis=0))
            log_V   = np.log10(mean_V)
            tr_inf  = compute_infectiousness(log_V, hill_pars, 'hill')
            tr_auc  = auc(tspan, tr_inf)
            tr_prof = discretize_pmf(14, tspan, tr_inf)
            tr_theta = (tr_auc / unt_auc) - 1. if unt_auc > 0 else -1.

            if is_resistance:
                mean_Vm = np.where(vm_acc.mean(axis=0) < 5., 5., vm_acc.mean(axis=0))
                log_Vm  = np.log10(mean_Vm)
                mt_inf  = compute_infectiousness(log_Vm, hill_pars, 'hill')
                mt_auc  = auc(tspan, mt_inf)
                mt_prof = discretize_pmf(14, tspan, mt_inf)
                mt_theta = (mt_auc / unt_auc) - 1. if unt_auc > 0 else -1.
                tr_tp, mt_tp = 1., 1.
                if mt_prof.sum() > 0:
                    mt_prof /= mt_prof.sum()
            else:
                mt_theta = -1.
                mt_prof  = np.zeros(timepoints_n, dtype=np.float32)
                tr_tp, mt_tp = 1., 0.

            if tr_prof.sum() > 0:
                tr_prof /= tr_prof.sum()
            else:
                tr_prof[:] = 0.

            infpro_list.append({
                'treatment_k':    np.int32(t_idx),
                'treatment_d':    np.int32(treatment_d),
                'resistance':     np.int32(1) if is_resistance else np.int32(0),
                'theta':          np.float32(tr_theta),
                'profile':        tr_prof,
                'treated_trans_p': tr_tp,
                'mutant_theta':   np.float32(mt_theta),
                'mutant_profile': mt_prof,
                'mutant_trans_p': mt_tp,
            })
            profile_id += 1

    infpro_n    = len(infpro_list) + 1
    infpro      = np.zeros((infpro_n, timepoints_n), dtype=np.float32)
    mutpro      = np.zeros((infpro_n, timepoints_n), dtype=np.float32)
    R_theta     = np.zeros(infpro_n, dtype=np.float32)
    Rmut_theta  = np.zeros(infpro_n, dtype=np.float32)
    trans_p     = np.ones(infpro_n,  dtype=np.float32)
    mut_trans_p = np.ones(infpro_n,  dtype=np.float32)
    infpro_map  = np.zeros((infpro_n - 1, 2, 2), dtype=np.int32) - 1

    infpro[0] = unt_profile
    mutpro[0] = unt_profile

    for i, elem in enumerate(infpro_list):
        infpro[i+1]      = elem['profile']
        R_theta[i+1]     = elem['theta']
        trans_p[i+1]     = elem['treated_trans_p']
        mutpro[i+1]      = elem['mutant_profile']
        Rmut_theta[i+1]  = elem['mutant_theta']
        mut_trans_p[i+1] = elem['mutant_trans_p']
        infpro_map[elem['treatment_d'], elem['treatment_k'], elem['resistance']] = i + 1

    return (infpro_map, infpro, R_theta, trans_p,
            mutpro, Rmut_theta, mut_trans_p, res_id_start + 1)


# ── Renewal simulation setup ──────────────────────────────────────────────────

def make_pars(pandemic, treatment, treatment_bool=True, seed=42):
    cfg   = PANDEMIC_CONFIG[pandemic]
    tplan = os.path.join(PARENT_DIR, 'treatment_plans', TREATMENT_CONFIG[treatment]['plan'])
    return sc.objdict(
        country='USA',
        hr_non_elderly_p=0.2,
        ndays=365,
        init=10,
        R=cfg['R'],
        symptom_onset_lognorm_pars=np.asarray([0.35, 0.41, 14]),
        seek_test_lognorm_pars=np.asarray([1., 0.6, 8]),
        time_to_sev_gamma_pars=np.asarray([4.471, 1.097, 21]),
        time_to_dea_lognorm_pars=np.asarray([2.157, 0.408, 28]),
        asymp_prob=0.16,
        profile=cfg['profile'],
        treatment_start_t=7,
        predistribute_antiviral=False,
        treatment_predistributed='baloxavir',
        starting_background_activity_t=0,
        treatment_bool=treatment_bool,
        baloxavir_transmission_effect=0.29,
        vl_to_inf_model='hill',
        treatment_plan=tplan,
        treatment_window=2,
        test_willingness=1.0,
        test_sensitivity=0.70,
        treat_compliance=np.asarray([0.95, 0.65]),
        treat_success=np.asarray([1., 1.]),
        treat_sev_odds_ratio=np.asarray([0.75, 0.75]),
        treat_dea_odds_ratio=np.asarray([0.23, 0.23]),
        treat_resistance_prob=np.asarray([0., 0., 0., 0.]),
        resistance_fitness_cost=np.asarray([0., 0.]),
        iso_likelihood=0.,
        iso_contact_reduction_factor=np.asarray([0.1, 0.85, 0.85, 0.60]),
        contact_tracing=False,
        prophylaxis_effect=np.asarray([0.43, 0.40]),
        prophylaxis_effect_t=np.asarray([10, 10]),
        verbose=0,
        datadir=PARENT_DATA_DIR,
        seed=seed,
    )


def run_sim_with_infpro(all_16_pars, k1, k2, pars, pop, treatment, dummy_outdir):
    """
    Run one renewal simulation with per-draw infpro.
    Returns (total_deaths, total_treatment_courses).
    treatment_courses uses the correct treated_n axis for BXM (0) or OSL (1).
    """
    infpro_data = compute_infpro_for_draw(all_16_pars, k1, k2)

    sim = renewal.Sim(pars, pop)
    (sim.infpro_map, sim.infpro, sim.R_theta, sim.trans_p,
     sim.mutpro, sim.mut_R_theta, sim.mut_trans_p,
     sim.resistance_profile_id_start) = infpro_data

    sim.save_results = lambda _outdir: None
    sim.run(dummy_outdir)

    deaths  = int(sim.deaths_n.sum())
    t_idx   = TREATMENT_CONFIG[treatment]['treated_n_idx']
    courses = int(sim.treated_n[:, :, t_idx, :, :].sum())
    return deaths, courses


# ── Load MCMC posterior samples ───────────────────────────────────────────────

def load_posterior_samples(chain_file=CHAIN_FILE, n_draws=500, seed=0):
    backend = emcee.backends.HDFBackend(chain_file, read_only=True)
    tau     = backend.get_autocorr_time(quiet=True)
    burn_in = int(2 * np.max(tau))
    thin    = max(1, int(0.5 * np.min(tau)))
    flat    = backend.get_chain(discard=burn_in, thin=thin, flat=True)
    rng     = np.random.default_rng(seed)
    idx     = rng.choice(len(flat), size=min(n_draws, len(flat)), replace=False)
    print(f"MCMC chain: burn-in={burn_in}, thin={thin}, "
          f"flat samples={len(flat)}, using {len(idx)}")
    return flat[idx]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(pandemic, treatment, n_workers=1):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    deaths_file  = _out(pandemic, treatment, 'posterior_deaths')
    courses_file = _out(pandemic, treatment, 'posterior_courses')
    baseline_file = _baseline_file(pandemic)
    ckpt_file    = _ckpt(pandemic, treatment)
    dummy_outdir = os.path.join(OUTPUTS_DIR, '_tmp_renewal')
    os.makedirs(dummy_outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Scenario: {pandemic} / {treatment}")
    print(f"{'='*60}")

    samples   = load_posterior_samples()
    n_draws   = len(samples)
    pars_treat = make_pars(pandemic, treatment, treatment_bool=True)
    pars_notrt = make_pars(pandemic, treatment, treatment_bool=False)
    pop        = renewal.Pop(pars_treat)
    pop_n      = int(pop.agecat_n.sum())
    np.save(POP_FILE, np.array(pop_n))

    # ── Baseline (no treatment, MLE pars, per pandemic) ──
    if os.path.exists(baseline_file):
        baseline_deaths = int(np.load(baseline_file))
        print(f"Loaded baseline deaths: {baseline_deaths:,}")
    else:
        print("Running baseline (no treatment, MLE pars)...")
        mle_16 = _build_full_state(best_fit)
        k1_mle, k2_mle = fit_hill_k1k2(mle_16)
        infpro_mle = compute_infpro_for_draw(mle_16, k1_mle, k2_mle)
        sim_base = renewal.Sim(pars_notrt, pop)
        (sim_base.infpro_map, sim_base.infpro, sim_base.R_theta, sim_base.trans_p,
         sim_base.mutpro, sim_base.mut_R_theta, sim_base.mut_trans_p,
         sim_base.resistance_profile_id_start) = infpro_mle
        sim_base.save_results = lambda _: None
        sim_base.run(dummy_outdir)
        baseline_deaths = int(sim_base.deaths_n.sum())
        np.save(baseline_file, np.array(baseline_deaths))
        print(f"Baseline deaths: {baseline_deaths:,}")

    # ── Load checkpoint ──
    if os.path.exists(ckpt_file):
        ckpt         = np.load(ckpt_file)
        deaths_arr   = ckpt['deaths'].tolist()
        courses_arr  = ckpt['courses'].tolist()
        k1k2_arr     = ckpt['k1k2'].tolist()
        start_idx    = int(ckpt['next_idx'])
        print(f"Resuming from draw {start_idx}/{n_draws}")
    else:
        deaths_arr  = []
        courses_arr = []
        k1k2_arr    = []
        start_idx   = 0

    # ── Per-draw runs ──
    t_total = time.time()
    if n_workers > 1:
        from multiprocessing import Pool
        batch_n = max(n_workers * 4, 40)
        print(f"Using {n_workers} parallel workers (batch size {batch_n})")
        remaining = list(range(start_idx, n_draws))
        with Pool(n_workers, initializer=_worker_init,
                  initargs=(pandemic, treatment)) as pool:
            for b_start in range(0, len(remaining), batch_n):
                batch_idx  = remaining[b_start : b_start + batch_n]
                batch_args = [(i, samples[i]) for i in batch_idx]
                t_b = time.time()
                results = sorted(pool.map(_run_draw_worker, batch_args),
                                 key=lambda r: r[0])
                for i_r, deaths, courses, k1, k2 in results:
                    deaths_arr.append(deaths)
                    courses_arr.append(courses)
                    k1k2_arr.append([k1, k2])
                n_done    = start_idx + len(deaths_arr)
                t_elapsed = time.time() - t_b
                eta_min   = t_elapsed / len(batch_idx) * (n_draws - n_done) / 60.
                print(f"  Batch {b_start // batch_n + 1}: "
                      f"{n_done}/{n_draws} draws done  "
                      f"{t_elapsed:.0f}s  ETA {eta_min:.0f} min")
                np.savez(ckpt_file,
                         deaths=np.array(deaths_arr,  dtype=float),
                         courses=np.array(courses_arr, dtype=float),
                         k1k2=np.array(k1k2_arr,      dtype=float),
                         next_idx=n_done)
    else:
        for i in range(start_idx, n_draws):
            t0      = time.time()
            all_16  = _build_full_state(samples[i])
            try:
                k1, k2          = fit_hill_k1k2(all_16)
                deaths, courses = run_sim_with_infpro(
                    all_16, k1, k2, pars_treat, pop, treatment, dummy_outdir)
            except Exception as exc:
                print(f"  Draw {i}: ERROR — {exc}")
                deaths, courses = np.nan, np.nan
                k1, k2          = np.nan, np.nan
            deaths_arr.append(deaths)
            courses_arr.append(courses)
            k1k2_arr.append([k1, k2])
            elapsed = time.time() - t0
            eta_min = elapsed * (n_draws - start_idx - (i - start_idx + 1)) / 60.
            print(f"Draw {i+1:3d}/{n_draws}  deaths={deaths:>10,}  "
                  f"courses={courses:>10,}  "
                  f"k1={k1:.3f} k2={k2:.3f}  "
                  f"{elapsed:.0f}s  ETA {eta_min:.0f} min")
            if (i + 1) % 10 == 0 or (i + 1) == n_draws:
                np.savez(ckpt_file,
                         deaths=np.array(deaths_arr,  dtype=float),
                         courses=np.array(courses_arr, dtype=float),
                         k1k2=np.array(k1k2_arr,      dtype=float),
                         next_idx=i + 1)

    # ── Save k1/k2 (shared across scenarios — overwrite is fine, same draws) ──
    deaths_arr  = np.array(deaths_arr,  dtype=float)
    courses_arr = np.array(courses_arr, dtype=float)
    k1k2_arr    = np.array(k1k2_arr,    dtype=float)
    np.save(deaths_file,  deaths_arr)
    np.save(courses_file, courses_arr)
    np.save(K1K2_FILE,    k1k2_arr)
    print(f"\nSaved {deaths_file}")
    print(f"Saved {courses_file}")
    print(f"Total wall time: {(time.time()-t_total)/3600:.2f} h")

    make_figure(pandemic, treatment, deaths_arr, courses_arr, baseline_deaths, pop_n)


# ── Figure and summary ────────────────────────────────────────────────────────

def make_figure(pandemic, treatment, deaths_arr, courses_arr,
                baseline_deaths, pop_n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid   = np.isfinite(deaths_arr) & np.isfinite(courses_arr)
    d_valid = deaths_arr[valid]
    c_valid = courses_arr[valid]
    averted = baseline_deaths - d_valid

    mle_16         = _build_full_state(best_fit)
    k1_mle, k2_mle = fit_hill_k1k2(mle_16)
    pars_mle       = make_pars(pandemic, treatment, treatment_bool=True)
    pop_mle        = renewal.Pop(pars_mle)
    dummy          = os.path.join(OUTPUTS_DIR, '_tmp_renewal')
    mle_deaths, mle_courses = run_sim_with_infpro(
        mle_16, k1_mle, k2_mle, pars_mle, pop_mle, treatment, dummy)
    mle_averted = baseline_deaths - mle_deaths

    lo, hi = np.percentile(averted, [2.5, 97.5])
    label  = f"{pandemic.replace('_',' ')} / {treatment}"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(averted / 1e3, bins=30, color='#4393c3', edgecolor='white',
            linewidth=0.4, alpha=0.85, label='Posterior draws')
    ax.axvline(mle_averted / 1e3, color='black', linewidth=1.5,
               linestyle='--', label=f'MLE ({mle_averted/1e3:.0f}k)')
    ax.axvline(np.median(averted) / 1e3, color='#d6604d', linewidth=1.5,
               label=f'Posterior median ({np.median(averted)/1e3:.0f}k)')
    ax.axvspan(lo / 1e3, hi / 1e3, alpha=0.15, color='#d6604d', label='95% CI')
    ax.set_xlabel('Deaths averted (thousands)', fontsize=10)
    ax.set_ylabel('Number of draws', fontsize=10)
    ax.set_title(f'USA / {label}\n(1-day TAT, window=2, start=7)\n'
                 f'n={valid.sum()} posterior draws', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.text(-0.10, 1.02, 'A', transform=ax.transAxes,
            fontsize=13, fontweight='bold', va='top')
    fig.tight_layout()

    out_pdf = _fig_pdf(pandemic, treatment)
    out_png = out_pdf.replace('.pdf', '.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_pdf}")

    _print_summary(pandemic, treatment, d_valid, c_valid,
                   baseline_deaths, pop_n, mle_deaths, mle_courses, mle_averted)


SUMMARY_CSV_HEADER = ['metric', 'mle', 'p2.5', 'median', 'p97.5']


def _print_summary(pandemic, treatment, d_valid, c_valid,
                   baseline_deaths, pop_n, mle_deaths, mle_courses, mle_averted):
    import csv

    averted        = baseline_deaths - d_valid
    pct_averted    = 100. * averted / baseline_deaths
    averted_pm     = 1e6 * averted / pop_n
    courses_pc     = c_valid / pop_n

    mle_pct        = 100. * mle_averted / baseline_deaths
    mle_averted_pm = 1e6 * mle_averted / pop_n
    mle_courses_pc = mle_courses / pop_n

    def _ci(arr):
        return np.percentile(arr, [2.5, 50, 97.5])

    a_lo,  a_med,  a_hi  = _ci(averted)
    p_lo,  p_med,  p_hi  = _ci(pct_averted)
    pm_lo, pm_med, pm_hi = _ci(averted_pm)
    c_lo,  c_med,  c_hi  = _ci(courses_pc)

    label = f"{pandemic.replace('_',' ')} / {treatment}"
    W = 62
    print()
    print("=" * W)
    print(f"  SUMMARY: USA / {label}  (1-day TAT, window=2, start=7)")
    print("=" * W)
    print(f"  Population (USA):                  {pop_n:>15,}")
    print(f"  Baseline deaths (no treatment):    {baseline_deaths:>15,}")
    print(f"  Valid posterior draws:             {len(d_valid):>15,}")
    print("-" * W)
    print(f"  {'Metric':<38} {'MLE':>8}  {'Median (95% CI)'}")
    print("-" * W)
    print(f"  {'Deaths averted':<38} "
          f"{mle_averted:>8,.0f}  {a_med:,.0f}  [{a_lo:,.0f}, {a_hi:,.0f}]")
    print(f"  {'Deaths averted per million pop.':<38} "
          f"{mle_averted_pm:>8,.1f}  {pm_med:,.1f}  [{pm_lo:,.1f}, {pm_hi:,.1f}]")
    print(f"  {'% deaths averted':<38} "
          f"{mle_pct:>8.1f}  {p_med:.1f}  [{p_lo:.1f}, {p_hi:.1f}]")
    print(f"  {f'{treatment} courses per capita':<38} "
          f"{mle_courses_pc:>8.4f}  {c_med:.4f}  [{c_lo:.4f}, {c_hi:.4f}]")
    print("=" * W)

    csv_file = _summary_csv(pandemic, treatment)
    rows = [
        ['country', 'USA'], ['pandemic', pandemic], ['treatment', treatment],
        ['tat_delay', '1D'], ['treatment_window', 2], ['treatment_start_t', 7],
        ['population', pop_n], ['baseline_deaths', baseline_deaths],
        ['n_draws', len(d_valid)],
        [],
        SUMMARY_CSV_HEADER,
        ['deaths_averted',
         round(mle_averted), round(a_lo), round(a_med), round(a_hi)],
        ['deaths_averted_per_million',
         round(mle_averted_pm, 2), round(pm_lo, 2), round(pm_med, 2), round(pm_hi, 2)],
        ['pct_deaths_averted',
         round(mle_pct, 3), round(p_lo, 3), round(p_med, 3), round(p_hi, 3)],
        [f'{treatment.lower()}_courses_per_capita',
         round(mle_courses_pc, 6), round(c_lo, 6), round(c_med, 6), round(c_hi, 6)],
    ]
    with open(csv_file, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"Saved {csv_file}")


# ── Summarise completed scenarios ─────────────────────────────────────────────

def summarise(pandemic=None, treatment=None):
    """Print summary for one or all completed scenarios."""
    if pandemic and treatment:
        targets = [(pandemic, treatment)]
    else:
        targets = ALL_SCENARIOS

    dummy = os.path.join(OUTPUTS_DIR, '_tmp_renewal')
    os.makedirs(dummy, exist_ok=True)

    for pan, trt in targets:
        d_file  = _out(pan, trt, 'posterior_deaths')
        c_file  = _out(pan, trt, 'posterior_courses')
        b_file  = _baseline_file(pan)

        missing = [f for f in [d_file, c_file, b_file, POP_FILE]
                   if not os.path.exists(f)]
        if missing:
            print(f"\n[{pan}/{trt}] Not yet complete — missing:")
            for f in missing:
                print(f"  {f}")
            continue

        deaths_arr  = np.load(d_file)
        courses_arr = np.load(c_file)
        baseline_deaths = int(np.load(b_file))
        pop_n           = int(np.load(POP_FILE))

        valid   = np.isfinite(deaths_arr) & np.isfinite(courses_arr)
        d_valid = deaths_arr[valid]
        c_valid = courses_arr[valid]

        print(f"\nComputing MLE reference for {pan}/{trt}...")
        mle_16         = _build_full_state(best_fit)
        k1_mle, k2_mle = fit_hill_k1k2(mle_16)
        pars_mle       = make_pars(pan, trt, treatment_bool=True)
        pop_mle        = renewal.Pop(pars_mle)
        mle_deaths, mle_courses = run_sim_with_infpro(
            mle_16, k1_mle, k2_mle, pars_mle, pop_mle, trt, dummy)
        mle_averted = baseline_deaths - mle_deaths

        _print_summary(pan, trt, d_valid, c_valid,
                       baseline_deaths, pop_n, mle_deaths, mle_courses, mle_averted)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MCMC renewal posterior propagation')
    parser.add_argument('mode', nargs='?', default='run',
                        choices=['run', 'run_all', 'summarise'],
                        help='run, run_all, or summarise')
    parser.add_argument('--pandemic', choices=list(PANDEMIC_CONFIG), default='covid')
    parser.add_argument('--treatment', choices=list(TREATMENT_CONFIG), default='BXM')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='parallel workers per scenario (default 1 = serial)')
    args = parser.parse_args()

    if args.mode == 'run_all':
        for pan, trt in ALL_SCENARIOS:
            d_file = _out(pan, trt, 'posterior_deaths')
            if os.path.exists(d_file):
                print(f"\nSkipping {pan}/{trt} — results already exist.")
                continue
            main(pan, trt, n_workers=args.n_workers)
    elif args.mode == 'summarise':
        pan = args.pandemic if '--pandemic' in sys.argv else None
        trt = args.treatment if '--treatment' in sys.argv else None
        summarise(pan, trt)
    else:
        main(args.pandemic, args.treatment, n_workers=args.n_workers)
