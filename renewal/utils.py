import os
import re
import sciris as sc
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
from scipy.integrate import odeint
from scipy.special import expit
from sklearn.metrics import auc, root_mean_squared_error
import numba as nb
import pandas as pd
import matplotlib.pyplot as plt

def parse_treatment_plan(fpath, ndays, agecat_n_size):
    treatment_plan = np.zeros((ndays+1, 3, agecat_n_size), dtype=np.int8)

    strategy_idx_dict = {"tat":0, "pep":1, "prep":2}
    treatment_idx_dict = {"baloxavir":1, "oseltamivir":2}

    # read dataframe
    df = pd.read_excel(fpath)
    for r, row in df.iterrows():
        strategy_idx = strategy_idx_dict[row.strategy]
        treatment_idx = treatment_idx_dict[row.treatment]
        min_age_bin = np.int32(row.min_age/5)
        max_age_bin = np.int32((row.max_age+1)/5)
        min_t = np.int32((ndays+1)*row.min_t)
        max_t = np.int32((ndays+1)*row.max_t)
        # store treatment plan
        treatment_plan[min_t:max_t, strategy_idx, min_age_bin:max_age_bin] = treatment_idx

    return treatment_plan

def get_discrete_prob_dist(par1, par2, trucate_day, dist_type='lognorm', mode='pdf', ll=1e-6, actual_mu_sd=0):
    time_bins = np.asarray([0.]+list(np.linspace(1.5, 28.5, 28)))
    # set up distribution type
    if dist_type == 'lognorm':
        if actual_mu_sd > 0:
            raise Exception("LADIDA")
        else:
            p1 = par1
            p2 = par2
        dist = st.lognorm(s=p2, scale=np.exp(p1))
    elif dist_type == 'gamma':
        if actual_mu_sd > 0:
            p1 = (par1/par2)**2
            p2 = (par2**2)/par1
        else:
            p1 = par1
            p2 = par2
        dist = st.gamma(a=p1, scale=p2)

    # compute probability
    discrete_probability = np.zeros(time_bins.size, dtype=np.float32)
    if mode == 'cdf':
        discrete_probability[0] = 1.

    for t in range(time_bins.size):
        if t > 0:
            if mode == 'pdf':
                p = np.float32(dist.cdf(time_bins[t]) - dist.cdf(time_bins[t-1]))
            elif mode == 'cdf':
                p = np.float32(1. - dist.cdf(time_bins[t]))
            discrete_probability[t] = p

    # any probability below ll is zero-ed
    discrete_probability[discrete_probability<ll] = np.float32(0.)
    # truncate from day onwards
    discrete_probability = discrete_probability[:int(trucate_day)+1]
    # cumsum
    discrete_probability /= discrete_probability.sum()
    return discrete_probability

def discretize_time_dist(symptom_onset_lognorm_pars, seek_test_lognorm_pars, time_to_sev_gamma_pars, time_to_dea_lognorm_pars):
    # generate discrete symptom onset probability distribution
    symp_onset_p = get_discrete_prob_dist(par1=symptom_onset_lognorm_pars[0], par2=symptom_onset_lognorm_pars[1], trucate_day=symptom_onset_lognorm_pars[2], dist_type='lognorm', mode='pdf')
    # generate discrete test seeking probability distribution
    test_p = get_discrete_prob_dist(par1=seek_test_lognorm_pars[0], par2=seek_test_lognorm_pars[1], trucate_day=seek_test_lognorm_pars[2], dist_type='gamma', mode='cdf', actual_mu_sd=1)
    # load time to hospitalization since symptom onset
    time_to_sev_p = get_discrete_prob_dist(par1=time_to_sev_gamma_pars[0], par2=time_to_sev_gamma_pars[1], trucate_day=time_to_sev_gamma_pars[2], dist_type='gamma', mode='pdf')
    # load time fo death since hospitalization
    hosp_time_to_dea_p = get_discrete_prob_dist(par1=time_to_dea_lognorm_pars[0], par2=time_to_dea_lognorm_pars[1], trucate_day=time_to_dea_lognorm_pars[2], dist_type='lognorm', mode='pdf')
    # load onset time to death
    onset_time_to_dea_p = get_discrete_prob_dist(par1=time_to_dea_lognorm_pars[0], par2=time_to_dea_lognorm_pars[1], trucate_day=time_to_dea_lognorm_pars[2], dist_type='lognorm', mode='pdf')
    return symp_onset_p, test_p, time_to_sev_p, hosp_time_to_dea_p, onset_time_to_dea_p

class viral_load_trajectory():
    """
    Object to compute viral load trajectory
    """
    def __init__(self, within_host_model_pars, tspan):
        self.within_host_model_pars = within_host_model_pars
        self.tspan = tspan
        return

    def within_host_model(self, y, t, pars):
        # T = target cells, R = refractory cells, I = infected cells,
        # F = interferon response, V = concentration of free infectious virions measured via TCID50 infectivity assay
        T, R, Iw, Im, F, Vw, Vm = y
        # b = beta infection rate, l = IFN-induced antiviral efficacy
        # r = reversion rate from refractory, d = death rate of infected cells
        # k = killing rate of infected cells by NK cells, q = production rate of IFN
        # a = decay rate of IFN, pw = wild-type virus replication rate, pm = mutant virus replication rate
        # c = virus clearance rate, mu = mutation rate, e = treatment effectiveness rate
        bw, bm, l, r, d, k, q, a, pw, pm, cw, cm, mu, e = pars

        dT = -(bw * T * Vw) - (bm * T * Vm) - (l * T * F) + (r * R)
        dR = (l * T * F) - (r * R)
        dIw = (bw * T * Vw) - (d * Iw) - (k * Iw * F)
        dIm = (bm * T * Vm) - (d * Im) - (k * Im * F)
        dF = (q * (Iw + Im)) - (a * F)
        dVw = ((1 - mu) * (1 - e) * pw * Iw) - (cw * Vw)
        dVm = (pm * Im) + (mu * pw * Iw) - (cm * Vm)

        return [dT, dR, dIw, dIm, dF, dVw, dVm]

    def solve_model(self, rate_pars, pm, cm, mu, e, tspan, y_init):
        bw, bm, l, r, d, k, q, a, pw, cw = rate_pars
        pars = np.asarray([bw, bm, l, r, d, k, q, a, pw, pm, cw, cm, mu, e])
        sol = odeint(func=self.within_host_model, y0=y_init, t=tspan, args=(pars,))
        return sol

    def get_vl_traj(self, treatment_t=None, treatment="placebo", T0=4e8):
        # rate parameters
        bw, bm, l, r, d, k, q, a, pw, cw, pmb, cmb, Vw0, mub, eb, eo = self.within_host_model_pars
        rate_pars = np.asarray([bw, bm, l, r, d, k, q, a, pw, cw])

        # treatment
        if treatment == 'placebo':
            y_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            # no treatment
            s0 = self.solve_model(rate_pars, pm=0., cm=0.,  mu=0., e=0., tspan=self.tspan, y_init=y_init)
            V = s0[:,-2]
            return V, None, self.tspan

        elif treatment == 'baloxavir' or treatment == 'oseltamivir':
            tspan = np.sort(np.unique(np.concatenate((self.tspan, [treatment_t]))))
            # pre-treatment
            t0 = tspan[tspan < treatment_t]
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = self.solve_model(rate_pars, pm=0., cm=0.,  mu=0., e=0., tspan=t0, y_init=y0_init)
            # post-treatment
            t1 = tspan[tspan >= treatment_t]
            y1_init = s0[-1,:]
            s1 = self.solve_model(rate_pars, pm=0., cm=0., mu=0., e=eb if treatment == 'baloxavir' else eo, tspan=t1, y_init=y1_init)
            # fit viral load to baloxavir treated data
            V = np.concatenate((s0[:,-2], s1[:,-2]))
            return V, None, tspan

        elif treatment == 'baloxavir_resistance':
            tspan = np.sort(np.unique(np.concatenate((self.tspan, [treatment_t]))))
            # pre-treatment + no mutation
            t0 = tspan[tspan < treatment_t]
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = self.solve_model(rate_pars, pm=0., cm=0.,  mu=0., e=0., tspan=t0, y_init=y0_init)
            # post-treatment
            t1 = tspan[tspan >= treatment_t]
            y1_init = s0[-1,:]
            s1 = self.solve_model(rate_pars, pm=pmb, cm=cmb, mu=mub, e=eb, tspan=t1, y_init=y1_init)
            # fit viral load to baloxavir treated data
            Vw = np.concatenate((s0[:,-2], s1[:,-2]))
            Vm = np.concatenate((s0[:,-1], s1[:,-1]))
            return Vw, Vm, tspan

def get_disc_lognorm_pdf(y, N):
    # get pdf of lognorm dist
    mu, sigma = y
    dist = st.lognorm(s=sigma, scale=np.exp(mu))
    pdf = np.asarray([dist.cdf(0.), dist.cdf(1.5)] + [dist.cdf(i+0.5) - dist.cdf(i-0.5) for i in range(2, N-1)])
    pdf = np.concatenate((pdf, [1 - dist.cdf(N-0.5)]))
    return pdf

def fit_contact_data(y, centerstone_raw_contacts):
    # household contact from CENTERSTONE study
    centerstone_raw_pdf = centerstone_raw_contacts/centerstone_raw_contacts.sum()
    return root_mean_squared_error(get_disc_lognorm_pdf(y, centerstone_raw_contacts.size), centerstone_raw_pdf)

def compute_infectiousness(V, pars, model="negexp"):
    pt = np.zeros(V.size, dtype=np.float32)
    if model == "negexp":
        p0 = pars[0]
        pt[V > 0.] = 1 - np.exp(-V[V > 0.]/p0)
    elif model == "hill":
        p0, p1 = pars
        pt[V > 0.] = 1 / (1 + (p0/V[V > 0.])**p1)
    elif model == "logit":
        p0, p1, p2 = pars
        pt[V <= p1] = p2
        pt[V > p1] = expit(p2 + p0 * (V[V > p1] - p1))
    return pt

class fit_infectiousness():
    def __init__(self, datadir, centerstone_contact_data=np.asarray([[0, 63, 122, 110, 69, 28, 11, 2], [0, 70, 127, 101, 56, 29, 11, 5]]),
                 endpoint_t=6):
        # previously fitted within host model parameters
        self.within_host_model_pars = np.load(datadir + "/within_host_pars.npz")['within_host_model_pars']
        # time of baloxavir treatment
        self.treatment_t = np.load(datadir + "/within_host_pars.npz")['treatment_t'][0]
        # endpoint of trial (after administration of treatment)
        self.endpoint_t = endpoint_t
        # compute average numeber of contacts for placebo and treatment arm in CENTERSTONE
        minfn_placebo = minimize(fit_contact_data, x0=[.5, .5], args=(centerstone_contact_data[0,:]))
        minfn_treatment = minimize(fit_contact_data, x0=[.5, .5], args=(centerstone_contact_data[1,:]))
        self.contact_dist_placebo = st.lognorm(s=minfn_placebo.x[1], scale=np.exp(minfn_placebo.x[0]))
        self.contacts_dist_treatment = st.lognorm(s=minfn_treatment.x[1], scale=np.exp(minfn_treatment.x[0]))
        return

    def fit(self, target_effect, model="negexp", seed=42):
        if model not in ["negexp", "hill", "logit", "linear"]:
            raise Exception("Invalid viral load to infectiousness model.")

        # get infectiousness factor
        print ("Fitting infectiousnesss factor using %s model..."%(model))
        best_fit_pars, best_fitness_score = self.transmission_risk_de(args=(target_effect, model, seed))
        return best_fit_pars, best_fitness_score

    def transmission_risk_de(self, args, popsize=50, its=500, mut=0.3, crossp=0.8):

        target_effect, model, seed = args
        rng = np.random.RandomState(seed)

        if model == "negexp":
            bounds = ((1., 1000.),)
        elif model == "hill":
            bounds = ((0., 10.), (1., 10.),)
        elif model == "logit":
            bounds = ((1e-3, 1000), (0., 10.), (0., 0.2),)
        else:
            bounds = ((-1000, 1000), (-1000, 1000),)

        # bounds of variables
        min_bound = np.asarray([b[0] for b in bounds])
        max_bound = np.asarray([b[1] for b in bounds])
        diff = max_bound - min_bound # difference between bounds
        idx = np.arange(popsize)
        # generate population (normalized)
        dim = len(bounds)
        population = rng.rand(popsize, dim)
        # compute fitness
        population_denorm = min_bound + population * diff

        # compute fitness
        fitness = np.asarray([self.fit_vl_to_trans_factor(population_denorm[i,:], target_effect, model, seed) for i in range(popsize)])
        best_fitness = fitness.min()
        prev_best_fitness = best_fitness
        mean_fitness_memory = np.zeros(its, dtype=np.float32)

        for i in range(its):
            for j in range(popsize):
                # randomly choose three indices without replacement excluding j
                a, b, c = population[rng.choice(idx[idx != j], 3, replace=False)]
                # generate mutant
                mutant = np.clip(a + mut * (b - c), 0, 1)

                if model == "negexp":
                    trial = mutant.copy()
                else:
                    # recombination
                    cross_points = rng.rand(dim) < crossp
                    if not np.any(cross_points):
                        cross_points[rng.randint(0, dim)] = True
                    # get trial candidate
                    trial = np.where(cross_points, mutant, population[j])

                trial_denorm = min_bound + trial * diff
                trial_f = self.fit_vl_to_trans_factor(trial_denorm, target_effect, model, seed)
                # selection
                if trial_f < fitness[j]:
                    fitness[j] = trial_f
                    population[j] = trial

            best_idx = fitness.argmin()
            mean_fitness_memory[i] = fitness.mean()

            #print (i, min_bound + population[best_idx] * diff, fitness.min())
            if fitness.min() < best_fitness:
                best_fitness = fitness.min()
                best_pars = min_bound + population[best_idx] * diff
                if model == "negexp":
                    print ("improved fit at step %i (fitness score = %.4e; f = %.4f)"%(i, best_fitness, best_pars[0]))
                elif model == "hill" or model == "linear":
                    print ("improved fit at step %i (fitness score = %.4e; l = %.4f, a = %.4f)"%(i, best_fitness, best_pars[0], best_pars[1]))
                elif model == "logit":
                    print ("improved fit at step %i (fitness score = %.4e; b = %.4f, V0 = %.4f, p0 = %.4f)"%(i, best_fitness, best_pars[0], best_pars[1], best_pars[2]))
                #else:
                #    print ("improved fit at step %i (fitness score = %.4e; l = %.4f, a = %.4f)"%(i, best_fitness, best_pars[0], best_pars[1]))

            if np.isclose(fitness.min(), 0.):
                break

            # run minimally for 50 iterations
            # no improvement in mean fitness in the past 20 iterations
            if i >= 50 and all((mean_fitness_memory[i]/mean_fitness_memory[i-20:i]) == 1.):
                break

        print ("best fitness = %.4e"%(fitness.min()))
        return min_bound + population[best_idx] * diff, fitness.min()

    def fit_vl_to_trans_factor(self, x, target_effect, model, seed):

        # get placebo viral load trajectory
        tspan = np.linspace(0., 14, (24*14)+1)

        # compute trial endpoint
        trial_endpoint_t = np.around(tspan[tspan <= self.treatment_t+self.endpoint_t][-1]).astype(np.int32)

        # create vlt object
        vlt_obj = viral_load_trajectory(self.within_host_model_pars, tspan)

        # get placebo viral load trajectory adnd infectiousness
        placebo_V, placebo_Vm, placebo_tspan = vlt_obj.get_vl_traj(treatment="placebo")
        placebo_V[placebo_V < 5.] = 5.
        placebo_V = np.log10(placebo_V)
        placebo_infectiousness = compute_infectiousness(placebo_V, x, model)
        placebo_infectiousness_auc = auc(tspan[tspan<=self.treatment_t+self.endpoint_t], placebo_infectiousness[tspan<=self.treatment_t+self.endpoint_t])

        # get baloxavir viral load trajectory
        baloxavir_V, baloxavir_Vm, baloxavir_tspan = vlt_obj.get_vl_traj(treatment_t=self.treatment_t, treatment="baloxavir")
        baloxavir_V[baloxavir_V < 5.] = 5.
        baloxavir_V = np.log10(baloxavir_V)

        # compute infectiousness
        baloxavir_infectiousness = compute_infectiousness(baloxavir_V, x, model)
        baloxavir_infectiousness = baloxavir_infectiousness[np.isin(baloxavir_tspan, tspan)]
        baloxavir_infectiousness_auc = auc(tspan[tspan<=self.treatment_t+self.endpoint_t], baloxavir_infectiousness[tspan<=self.treatment_t+self.endpoint_t])

        if placebo_infectiousness_auc > 0.:
            eff = 1. - baloxavir_infectiousness_auc/placebo_infectiousness_auc
        else:
            eff = 1. - np.inf

        return abs(target_effect - eff)

    def compute_transmissions(self, discrete_p, contact_dist, seed):
        rng = np.random.RandomState(seed)
        contact_n = np.around(contact_dist.rvs(1000, random_state=rng)).astype(np.int32)
        contact_n = contact_n[contact_n>0]
        # household ids
        contact_hid = np.arange(contact_n.size).astype(np.int32)
        # get current contacts labeled based on hid
        curr_contacts = np.repeat(contact_hid, contact_n)

        infected_n = np.zeros(contact_n.size, dtype=np.int32)
        for t in range(discrete_p.size):
            # bernoulli trial to identify infected contacts
            infected_mask = rng.random(curr_contacts.size) < discrete_p[t]
            infected_hids, infected_counts = np.unique(curr_contacts[infected_mask], return_counts=True)
            infected_n[infected_hids] += infected_counts
            # update current contacts
            curr_contacts = np.repeat(contact_hid, contact_n-infected_n)
        # compute average fraction of contacts infected
        return (infected_n/contact_n).mean()

def compute_conditional_infectiousness(treated_infectiousness, mutant_infectiousness):
    sens_p_arr = np.zeros(treated_infectiousness.size, dtype=np.float32)
    resi_p_arr = np.zeros(treated_infectiousness.size, dtype=np.float32)
    for tt in range(treated_infectiousness.size):
        p_sens, p_resi = treated_infectiousness[tt], mutant_infectiousness[tt]
        sens_p_arr[tt] = p_sens * (1. - p_resi)
        resi_p_arr[tt] = (1. - p_sens) * p_resi
    return sens_p_arr, resi_p_arr

def compute_variant_inf_prob(treatment_d, treated_profile, mutant_profile, rng, n_sample = 10000):

    adj_treated_profile = np.zeros(treated_profile.size, dtype=np.float32)
    adj_mutant_profile = np.zeros(mutant_profile.size, dtype=np.float32)
    for tt in range(treated_profile.size):
        p_sens, p_resi = treated_profile[tt], mutant_profile[tt]
        if p_sens == 0. and p_resi == 0.:
            continue
        else:
            adj_treated_profile[tt] = p_sens*(1-p_resi)
            adj_mutant_profile[tt] = p_resi*(1-p_sens)

    fig = plt.figure()
    plt.plot(range(adj_treated_profile.size), adj_treated_profile, c='k')
    plt.plot(range(adj_mutant_profile.size), adj_mutant_profile, c='r')
    #plt.savefig("./TD%i_infectiousness.pdf"%(treatment_d), bbox_inches='tight')

    bernoulli_p = rng.random((n_sample, adj_treated_profile.size))
    for tt in range(adj_treated_profile.size):
        bernoulli_p[:,tt] *= adj_treated_profile[tt] + adj_mutant_profile[tt]

    inf_results = np.zeros(2, dtype=np.int32)
    for i in range(n_sample):
        sens_bool = bernoulli_p[i,1:] < adj_treated_profile[1:]
        resi_bool = bernoulli_p[i,1:] >= adj_treated_profile[1:]
        if np.argmax(resi_bool) < np.argmax(sens_bool):
            inf_results[1] += 1
        else:
            inf_results[0] += 1
    sens_trans_chance = inf_results[0]/n_sample
    resi_trans_chance =  inf_results[1]/n_sample

    return adj_treated_profile, adj_mutant_profile, sens_trans_chance, resi_trans_chance

def load_infectiousness_data(datadir, baloxavir_transmission_effect, vl_to_inf_model, seed):

    # check if infectiousness parameters have already been fitted before
    try:
        previous_fit_df = pd.read_csv(datadir + "/vl_to_inf_fit.csv").set_index(["vl_to_inf_model", "baloxavir_transmission_effect"])
        prev_records_boolean = 1
    except:
        prev_records_boolean = 0

    if prev_records_boolean > 0:
        try:
            fitted_infectiousness_pars = previous_fit_df.loc[(vl_to_inf_model, baloxavir_transmission_effect)].to_numpy()
            fitted_infectiousness_pars = fitted_infectiousness_pars[~np.isnan(fitted_infectiousness_pars)]
        except:
            previous_fit_df = previous_fit_df.reset_index()
            fitted_infectiousness_pars = fit_infectiousness(datadir).fit(baloxavir_transmission_effect, vl_to_inf_model)[0]
            entry = {"vl_to_inf_model":vl_to_inf_model, "baloxavir_transmission_effect":baloxavir_transmission_effect}
            for i in range(3):
                try:
                    entry["p%i"%(i)] = fitted_infectiousness_pars[i]
                except:
                    entry["p%i"%(i)] = None
            previous_fit_df = pd.concat([previous_fit_df, pd.DataFrame.from_dict([entry])])
            previous_fit_df.to_csv(datadir + "/vl_to_inf_fit.csv", index=False)

    else:
        # fit infectiousness parameter based on viral load data for desired baloxavir transmission effect
        fitted_infectiousness_pars = fit_infectiousness(datadir).fit(baloxavir_transmission_effect, vl_to_inf_model)[0]
        entry = {"vl_to_inf_model":vl_to_inf_model, "baloxavir_transmission_effect":baloxavir_transmission_effect}
        for i in range(3):
            try:
                entry["p%i"%(i)] = fitted_infectiousness_pars[i]
            except:
                entry["p%i"%(i)] = None
        previous_fit_df = pd.DataFrame.from_dict([entry])
        previous_fit_df.to_csv(datadir + "/vl_to_inf_fit.csv", index=False)

    # compute infectiousness profiles
    tspan = np.arange(0, (14*24)+1)/24
    within_host_model_pars = np.load(datadir + "/within_host_pars.npz")['within_host_model_pars']
    vlt_obj = viral_load_trajectory(within_host_model_pars, tspan)

    # compute placebo viral load
    placebo_V, placebo_Vm, placebo_tspan = vlt_obj.get_vl_traj(treatment='placebo')
    placebo_V[placebo_V < 5.] = 5.
    placebo_V = np.log10(placebo_V)

    # get untreated generation interval
    untreated_infectiousness = compute_infectiousness(placebo_V, fitted_infectiousness_pars, vl_to_inf_model)
    untreated_infectiousness_auc = auc(tspan, untreated_infectiousness)
    untreated_profile = discretize_pmf(14, tspan, untreated_infectiousness)
    untreated_profile /= untreated_profile.sum() # normalize
    timepoints_n = untreated_profile.size

    # generate list of infection profile
    infpro_list = []
    profile_id = 0
    resistance_profile_id_start = np.zeros(2, dtype=np.int32)
    for t, treatment in enumerate(['baloxavir', 'oseltamivir', 'baloxavir_resistance']):
        if treatment == "baloxavir_resistance":
            t = 0
            resistance_profile_id_start[0] = profile_id

        for td, treatment_d in enumerate(range(15)):
            #print (treatment_d)
            # get potential time of treatment for day d
            potential_time_of_treatment = tspan[(tspan>=treatment_d-0.5)&(tspan<=treatment_d+0.5)]
            potential_time_of_treatment = potential_time_of_treatment[potential_time_of_treatment>0.]

            if treatment == "baloxavir_resistance":
                potential_viral_load = np.zeros((potential_time_of_treatment.size, tspan.size))
                potential_mutant_load = np.zeros((potential_time_of_treatment.size, tspan.size))
            else:
                potential_viral_load = np.zeros((potential_time_of_treatment.size, tspan.size))

            for tt, treatment_t in enumerate(potential_time_of_treatment):
                treat_V, treat_Vm, treat_tspan = vlt_obj.get_vl_traj(treatment_t=treatment_t, treatment=treatment)
                if treatment == "baloxavir_resistance":
                    potential_viral_load[tt,:] = treat_V
                    potential_mutant_load[tt,:] = treat_Vm
                else:
                    potential_viral_load[tt,:] = treat_V

            # get average antiviral-sensitive viral load across all potential treatment times
            treatment_vl = potential_viral_load.mean(axis=0)
            treatment_vl[treatment_vl < 5.] = 5.
            treatment_vl = np.log10(treatment_vl)
            # generation interval of antiviral-sensitive virus
            treated_infectiousness = compute_infectiousness(treatment_vl, fitted_infectiousness_pars, vl_to_inf_model)

            if treatment == "baloxavir_resistance":
                # get average antiviral resistant viral load across all potential treatment times
                mutant_vl = potential_mutant_load.mean(axis=0)
                mutant_vl[mutant_vl < 5.] = 5.
                mutant_vl = np.log10(mutant_vl)

                nd_abv_inf_mutant = np.argwhere(mutant_vl > np.log10(5.)).T[0].size
                nd_abv_inf_treated = np.argwhere(treatment_vl > np.log10(5.)).T[0].size
                if nd_abv_inf_mutant > 0 and nd_abv_inf_treated > 0:
                    if (tspan[np.argwhere(treatment_vl > np.log10(5.)).T[0].max()] + 0.5 < tspan[np.argwhere(mutant_vl > np.log10(5.)).T[0].min()]):
                        nd_abv_inf_mutant = 0

                # mutant should not emerge when there are no more infectious wild-type virus
                if nd_abv_inf_mutant == 0:
                    mutant_theta = -1.
                    mutant_profile = np.zeros(15, dtype=np.float32)

                    # compute treated drug-sensitive virus auc
                    treated_infectiousness_auc = auc(tspan, treated_infectiousness)
                    treated_profile = discretize_pmf(14, tspan, treated_infectiousness)
                    # compute reduction in infectiousness
                    treated_theta = (treated_infectiousness_auc/untreated_infectiousness_auc) - 1.

                    treated_trans_p, mutant_trans_p = 1., 0.

                    # normalize treated_profile
                    if treated_profile.sum() > 0:
                        treated_profile /= treated_profile.sum() # normalize
                    else:
                        treated_profile[:] = 0.

                elif nd_abv_inf_treated == 0:
                    treated_theta = -1.
                    treated_profile = np.zeros(15, dtype=np.float32)

                    # generation interval of resistant virus
                    mutant_infectiousness = compute_infectiousness(mutant_vl, fitted_infectiousness_pars, vl_to_inf_model)

                    # compute treated drug-sensitive virus auc
                    mutant_infectiousness_auc = auc(tspan, mutant_infectiousness)
                    mutant_profile = discretize_pmf(14, tspan, mutant_infectiousness)
                    # compute reduction in infectiousness
                    mutant_theta = (mutant_infectiousness_auc/untreated_infectiousness_auc) - 1.

                    treated_trans_p, mutant_trans_p = 0., 1.

                    # normalize mutant_profile
                    if mutant_profile.sum() > 0:
                        mutant_profile /= mutant_profile.sum() # normalize
                    else:
                        mutant_profile[:] = 0.

                else:
                    # generation interval of resistant virus
                    mutant_infectiousness = compute_infectiousness(mutant_vl, fitted_infectiousness_pars, vl_to_inf_model)

                    # compute cond infectiousness
                    #treated_infectiousness, mutant_infectiousness = compute_conditional_infectiousness(treated_infectiousness, mutant_infectiousness)
                    mutant_infectiousness_auc = auc(tspan, mutant_infectiousness)
                    mutant_profile = discretize_pmf(14, tspan, mutant_infectiousness)

                    # compute treated drug-sensitive virus auc
                    treated_infectiousness_auc = auc(tspan, treated_infectiousness)
                    treated_profile = discretize_pmf(14, tspan, treated_infectiousness)

                    """
                    fig = plt.figure()
                    plt.plot(range(treated_profile.size), treated_profile, c='k')
                    plt.plot(range(mutant_profile.size), mutant_profile, c='r')
                    plt.savefig("./TD%i_infectiousness.pdf"%(treatment_d), bbox_inches='tight')
                    """

                    # bootstrap to compute overall probability of drug-sensitive and resistant variants
                    treated_profile, mutant_profile, treated_trans_p, mutant_trans_p = compute_variant_inf_prob(treatment_d, treated_profile, mutant_profile, np.random.RandomState(seed))
                    #print (treated_trans_p, mutant_trans_p)
                    # compute reduction in infectiousness
                    treated_theta = (treated_infectiousness_auc/untreated_infectiousness_auc) - 1.
                    mutant_theta = (mutant_infectiousness_auc/untreated_infectiousness_auc) - 1.

                    # normalize treated_profile
                    treated_profile /= treated_profile.sum()
                    # normalize mutant_profile
                    mutant_profile /= mutant_profile.sum()
            else:
                mutant_theta = -1.
                mutant_profile = np.zeros(15, dtype=np.float32)

                # compute treated drug-sensitive virus auc
                treated_infectiousness_auc = auc(tspan, treated_infectiousness)
                treated_profile = discretize_pmf(14, tspan, treated_infectiousness)
                # compute reduction in infectiousness
                treated_theta = (treated_infectiousness_auc/untreated_infectiousness_auc) - 1.

                treated_trans_p, mutant_trans_p = 1., 0.

                # normalize treated_profile
                if treated_profile.sum() > 0:
                    treated_profile /= treated_profile.sum() # normalize
                else:
                    treated_profile[:] = 0.

            infpro_list.append({"treatment_k":np.int32(t), "treatment_d":np.int32(treatment_d), "resistance":np.int32(1) if treatment == "baloxavir_resistance" else np.int32(0), "theta":np.float32(treated_theta), "profile":treated_profile, "treated_trans_p":treated_trans_p, "mutant_theta":np.float32(mutant_theta), "mutant_profile":mutant_profile, "mutant_trans_p":mutant_trans_p})

            profile_id += 1

    # generate simulation array
    infpro_n = len(infpro_list) + 1
    # infection profiles (number of profiles x timepoints)
    infpro = np.zeros((infpro_n, timepoints_n), dtype=np.float32)
    infpro[0,:] = untreated_profile

    mutpro = np.zeros((infpro_n, timepoints_n), dtype=np.float32)
    # assume mutant has the same untreated profile as wild-type
    mutpro[0,:] = untreated_profile

    # R_theta (number of profiles)
    R_theta = np.zeros(infpro_n, dtype=np.float32)
    Rmut_theta = np.zeros(infpro_n, dtype=np.float32)

    trans_p = np.ones(infpro_n, dtype=np.float32)
    mut_trans_p = np.ones(infpro_n, dtype=np.float32)

    # infpro_map (number of profiles minus untreated x treatment type x resistance profile)
    infpro_map = np.zeros((infpro_n-1, 2, 2), dtype=np.int32) - 1

    for i, element in enumerate(infpro_list):
        infpro[i+1,:] = element["profile"]
        R_theta[i+1] = element["theta"]
        trans_p[i+1] = element['treated_trans_p']

        mutpro[i+1,:] = element["mutant_profile"]
        Rmut_theta[i+1] = element["mutant_theta"]
        mut_trans_p[i+1] = element['mutant_trans_p']

        infpro_map[element["treatment_d"], element["treatment_k"], element["resistance"]] = i+1

    return infpro_map, infpro, R_theta, trans_p, mutpro, Rmut_theta, mut_trans_p, resistance_profile_id_start+1

def discretize_pmf(tlim, T, w):
    discrete_time_bins = np.asarray([0.5] + list(np.linspace(1.5, 0.5 + tlim, tlim)))
    discrete_probability = [0.]
    for t in range(discrete_time_bins.size):
        if t > 0:
            Tmask = (T>=discrete_time_bins[t-1])&(T<discrete_time_bins[t])
            discrete_probability.append(auc(T[Tmask], w[Tmask]))
    discrete_probability = np.asarray(discrete_probability)
    return discrete_probability.astype(np.float32)

def compute_case_r(case_data, si_data, t):
    # compute effective reproduction number based on case data
    curr_t_new_cases = case_data[t,:].sum()
    if curr_t_new_cases == 0:
        return 0.
    prev_t_cases = case_data[:t,:].sum(axis=1)
    if prev_t_cases.sum() == 0:
        return 0.
    T = np.minimum(si_data.size, prev_t_cases.size)
    return curr_t_new_cases/np.sum(si_data[:T][::-1]/si_data.sum() * prev_t_cases[-T:])

def compute_relative_risk_matrix(contact_matrix, relative_susceptibility, iso_likelihood, iso_contact_reduction_factor):
    # compute relative risk of undiagnosed individuals
    A = relative_susceptibility.size
    rr_matrix = np.zeros((2, A, A), dtype=np.float32) # (isolation status x from-age x to-age)
    # and fraction of contacts if isolated
    contact_frac = np.ones((2, A, A), dtype=np.float32)
    # all contacts
    non_d_rr = contact_matrix[0] * relative_susceptibility
    rr_matrix[0,:,:] = rr_matrix[1,:,:] = non_d_rr/np.linalg.eig(non_d_rr)[0].real.max() # normalize

    if iso_likelihood > 0.:
        # compute relative risk for positively diagnosed (and isolated) individuals
        # contact matrix for those who isolate
        iso_contact_matrix = np.zeros((A, A), dtype=np.float32)
        for i in range(iso_contact_reduction_factor.size):
            iso_contact_matrix += (iso_contact_reduction_factor[i] * contact_matrix[i+1,:,:])
        # relative risk would change since distribution of contacts has changed
        d_rr = iso_contact_matrix * relative_susceptibility
        rr_matrix[1,:,:] = d_rr/np.linalg.eig(d_rr)[0].real.max() # normalize
        # change fraction of contact made if there is isolation
        contact_frac[1,:,:] = iso_contact_matrix / contact_matrix[0]

    # normalized rel contacts in households
    hh_rel_contact_matrix = np.zeros((A, A), dtype=np.float32)
    hh_rel_contact_matrix = contact_matrix[1]/np.linalg.eig(contact_matrix[1])[0].real.max()

    return rr_matrix, contact_frac, hh_rel_contact_matrix

@nb.jit(nopython=True)
def compute_group_rt(agecat_n, R0, R_theta, trans_p, rr_matrix, infected_n, primary_mut_infected_n, sus_on_pep, prophylaxis_effect, prophylaxis_effect_t, curr_t, variant):

    # compute Rt for different groups (isolation status, pep status, dist-gen profile, a_infector, a_susceptible)
    K = R_theta.size
    A = agecat_n.size
    rt = np.zeros((2, K, A, A), dtype=np.float32)

    # compute number of individuals that cannot be infected
    curr_infected_n = infected_n.reshape((-1, A)).sum(axis=0) + primary_mut_infected_n.reshape((-1, A)).sum(axis=0) # all infected
    # compute number susceptibles that can be infected
    infectible_sus_n = agecat_n - curr_infected_n

    if variant == 0 and sus_on_pep.sum() > 0: # pep only works for drug sensitive variant
        for treatment in range(2):
            # get all susceptibles on pep during effective period
            sus_w_eff_pep = sus_on_pep[treatment,-prophylaxis_effect_t[treatment]:,:].sum(axis=0) # shape = A
            if sus_w_eff_pep.sum() == 0:
                continue
            # compute the expected number of individuals that would not be infected
            n_uninfected_by_pep = np.int32(sus_w_eff_pep.sum() * prophylaxis_effect[treatment])
            infectible_sus_n -= sainte_lague_distribution(n_uninfected_by_pep, sus_w_eff_pep)

    # compute fraction of susceptibles that can be infected in each age group
    S = infectible_sus_n/agecat_n
    for d in range(2): # non-isolated, isolated
        for k in range(K): # for each infectious profile
            rt[d,k,:,:] = rr_matrix[d] * S * R0 * (1 + R_theta[k]) * trans_p[k]

    # return rt and number of susceptible individuals that are NOT on pep (n_sus_wo_pep)
    return rt, agecat_n - curr_infected_n - sus_on_pep.reshape((-1, A)).sum(axis=0)

@nb.jit(nopython=True)
def compute_transmission(rt, infpro, mutpro, contact_frac, infected_n, curr_t, seed, resistance_profile_id_start, resist_mutant_fitness_cost, sus_on_pep, n_sus_wo_pep, prophylaxis_effect, variant):
    T, TS, IS, K, A = infected_n.shape # time of infection, diagnosis status, isolation status, infpro groups, age group
    age_groups = np.arange(A).astype(np.int32)
    transmissions = np.zeros(A, dtype=np.float32)
    # for different isolation status of infector
    for d in range(IS): # for isolation status
        # for each infectiousness profile
        for k in range(K):
            # for each previous timepoint
            for tau in range(T):
                if curr_t - tau >= infpro.shape[-1]:
                    continue

                dkt_infected_n = infected_n[tau,:,d,k,:].sum(axis=0).astype(np.float32) # sum across testing status
                if dkt_infected_n.sum() == 0:
                    continue
                # effective reproduction number * infectiousness * infected
                # for now those tested do not change their behaviour
                if k >= resistance_profile_id_start[variant-1]:
                    if variant > 0: # resistant mutant
                        profile = mutpro[k] * np.float32(1. - resist_mutant_fitness_cost[variant-1])
                    else: # sensitive wild-type
                        profile = infpro[k]
                else:
                    if variant > 0: # resistant mutant
                        profile = mutpro[k] * np.float32(1. - resist_mutant_fitness_cost[variant-1])
                    else: # sensitive wild-type
                        profile = infpro[k]
                transmissions += np.dot(rt[d,k,:,:].T * contact_frac[d], dkt_infected_n.astype(np.float32)) * profile[curr_t-tau] #infpro[k,curr_t-tau]

    # compute expected number of transmissions
    if transmissions.sum() > 10:
        n_transmissions = np.int32(np.around(transmissions.sum()))
    else:
        np.random.seed(seed)
        n_transmissions = np.int32(np.floor(transmissions.sum()))
        residual = transmissions.sum() - n_transmissions
        if np.random.random() < residual:
            n_transmissions += 1
        else:
            n_transmissions -= 1
            if n_transmissions < 0:
                n_transmissions = 0

    # distribute across age group and pep status
    transmissions = sainte_lague_distribution(n_transmissions, transmissions, check_max=0)

    # compute number of effective pep failures (based on fraction of individuals with effective PEP in the population)
    if variant > 0:
        age_pep_failures = np.around(transmissions * sus_on_pep.reshape((-1, A)).sum(axis=0)/(sus_on_pep.reshape((-1, A)).sum(axis=0) + n_sus_wo_pep)).astype(np.int32)
    else:
        age_pep_failures = np.zeros(A, dtype=np.int32)
        for treatment in range(2):
            age_pep_failures += np.around(transmissions * (1 - prophylaxis_effect[treatment]) * sus_on_pep[treatment].sum(axis=0)/(sus_on_pep.reshape((-1, A)).sum(axis=0) + n_sus_wo_pep)).astype(np.int32)

    age_pep_failures[age_pep_failures > transmissions] = transmissions[age_pep_failures > transmissions]

    pep_failures = np.zeros(sus_on_pep.shape, dtype=np.int32)
    for a in range(A):
        if age_pep_failures[a] > 0:
            pep_failures[:,:,a] = sainte_lague_distribution(age_pep_failures[a], sus_on_pep[:,:,a].flatten()).reshape(sus_on_pep[:,:,a].shape)

    return transmissions, pep_failures


@nb.jit(nopython=True)
def sainte_lague_distribution(target, votes, check_max=0):
    seats = np.zeros(votes.size, dtype=np.int32)
    for i in range(target):
        quotient = votes/((2*seats)+1)
        max_idx = np.argmax(quotient)
        if check_max > 0:
            if votes[max_idx] - seats[max_idx] > 0:
                seats[max_idx] += 1
            else:
                print (votes[max_idx], seats[max_idx])
                raise Exception("Incorrect Sanite Lague distribution.")
        else:
            seats[max_idx] += 1
    return seats

@nb.jit(nopython=True)
def compute_burden(symp_onset_p, burden_p, infected_n, treat_odds_ratio, infpro_map):
    T, K, A = infected_n.shape # time of infection, testing status, isolation status, infpro groups, age group
    burden = np.zeros((T, 3, A), dtype=np.float32)
    discrete_burden = np.zeros((T, 3, A), dtype=np.int32)

    for curr_t in range(T):
        for tau in range(curr_t+1): # tau = time of infection
            xi = curr_t - tau # xi = age of infection

            # must be within timeframe of symptom onset and burden
            eta_range = np.arange(xi+1).astype(np.int32)
            eta_range = eta_range[(eta_range<symp_onset_p.size)&(xi-eta_range<burden_p.shape[1])]

            for eta in eta_range: # eta = day of symptom onset (includes today)
                # calculate number of individuals who are burdened for different age groups
                prop_burden = symp_onset_p[eta] * burden_p[:,xi-eta]
                burden[curr_t, 0, :] += prop_burden * infected_n[tau,0,:]
                for treatment in [1, 2]:
                    # get profile of those under specified treatment
                    treatment_k = np.unique(infpro_map[:,treatment-1,0])
                    treatment_k = treatment_k[treatment_k>0]
                    # compute proportion of individuals who would be hospitalized after treatment
                    C = prop_burden/(1 - prop_burden)
                    C *= treat_odds_ratio[treatment-1]
                    prop_burden_treated = C/(1+C)
                    burden[curr_t, treatment, :] += prop_burden_treated * infected_n[tau,treatment_k,:].sum(axis=0) # treated individuals

        for k in range(3):
            # compute integer expected burden
            n_burden = np.int32(np.around(burden[curr_t, k, :].sum()))
            # distribute across age groups
            discrete_burden[curr_t, k, :] = sainte_lague_distribution(n_burden, burden[curr_t, k, :], check_max=0)
        #print (curr_t, discrete_burden[curr_t].sum())
    return discrete_burden

def compute_predistributed_antiviral_usage(symp_onset_p, seek_av_p, infected_n, treated_n, pri_mut_infected_n, curr_t, start_treatment_bool, treatment_window, treat_compliance, treat_success, treat_resistance_prob, infpro_map, iso_likelihood, predistribute_antiviral, treatment_plan, background_respiratory_prob):

    T, TS, IS, K, A = infected_n.shape[1:] # time of infection x testing status x isolation status x infpro groups x age group
    MS, T, TS, IS, K, A = pri_mut_infected_n.shape[1:] # mutant acquisition status (0 = within-host, 1 = transmission), time of infection, testing status, isolation status, infpro groups, age group

    for tau in range(T): # tau = time of infection
        xi = curr_t - tau # xi = age of infection

        eta_range = np.arange(xi+1).astype(np.int32)
        # must be within timeframe of symptom onset (we assume that test_p = seek_av_p)
        eta_range = eta_range[(eta_range<symp_onset_p.size)&(xi-eta_range<seek_av_p.size)]

        # calculate probability of who did not experience a non-relevant respiratory complaint before
        for eta in eta_range: # eta = day of symptom onset (includes today)

            # get numebr of people infected at time tau who have not been treated for the relevant pathogen
            # we take the historical snapshot of transmission force yesterday (and update for today)
            tau_untreated_n = infected_n[curr_t-1,tau,0,:,0,:]
            # only include those infected with mutant but are untested
            mut_tau_untreated_n = pri_mut_infected_n[curr_t-1,1,tau,0,:,0,:]

            # array to update infected_n (isolation status x infpro group x age group)
            update_infected_arr = np.zeros((IS,K,A), dtype=np.int32)
            update_infected_arr[:,0,:] = tau_untreated_n.copy()

            # array to update pri_mut_infected_n (isolation status x infpro group x age group)
            mut_update_infected_arr = np.zeros((IS,K,A), dtype=np.int32)
            mut_update_infected_arr[:,0,:] = mut_tau_untreated_n.copy()

            ## -- isolation -- ##
            # symptomatic individuals may choose to isolate, treated or untreated
            if iso_likelihood > 0.:
                ## -- wild type infected -- ##
                # those who are not isolated might isolate
                n_isolated = np.int32(np.around(iso_likelihood * update_infected_arr[0,0,:].sum()))
                # update infected arr those not isolating to isolate
                iso_symp_inds = sainte_lague_distribution(n_isolated, update_infected_arr[0,0,:])
                update_infected_arr[0,0,:] -= iso_symp_inds
                update_infected_arr[1,0,:] += iso_symp_inds

                ## -- primary resistant mutant infected -- ##
                # those who are not isolated might isolate
                mut_n_isolated = np.int32(np.around(iso_likelihood * mut_update_infected_arr[0,0,:].sum()))
                # update infected arr those not isolating to isolate
                mut_iso_symp_inds = sainte_lague_distribution(mut_n_isolated, mut_update_infected_arr[0,0,:])
                mut_update_infected_arr[0,0,:] -= mut_iso_symp_inds
                mut_update_infected_arr[1,0,:] += mut_iso_symp_inds

            if start_treatment_bool > 0:
                # calculate probability of taking the drug today
                prop_wanting_test = symp_onset_p[eta] * seek_av_p[xi-eta]

                for variant in range(2):
                    if variant == 0: # wild-type
                        # compute number of individuals infected at tau who would want to get treated at curr_t
                        n_seek_treat = np.int32(np.around(prop_wanting_test * update_infected_arr[:,0,:].sum()))
                    elif variant == 1: # primary resistant mutant
                        # compute number of individuals infected at tau who would want to get treated at curr_t
                        n_seek_treat = np.int32(np.around(prop_wanting_test * mut_update_infected_arr[:,0,:].sum()))

                    if n_seek_treat == 0:
                        continue

                    if variant == 0:
                        # symp_inds_to_treat.shape = (isolation status, age group)
                        symp_inds_to_treat = sainte_lague_distribution(n_seek_treat, update_infected_arr[:,0,:].flatten()).reshape(update_infected_arr[:,0,:].shape)
                    elif variant == 1:
                        symp_inds_to_treat = sainte_lague_distribution(n_seek_treat, mut_update_infected_arr[:,0,:].flatten()).reshape(mut_update_infected_arr[:,0,:].shape)

                    # compute the age-structured probability that individuals have not had a respiratory episode in the past up to when they were infected by the relevant pathogen (if they have, we assumed that they have used the predistributed antiviral)
                    prob_of_no_resp_so_far = np.prod(1 - background_respiratory_prob[:,:tau], axis=1)
                    treated_symp_inds = np.zeros(symp_inds_to_treat.shape, dtype=np.int32)
                    for i in range(symp_inds_to_treat.shape[-1]):
                        n = np.int32(np.around(prob_of_no_resp_so_far[i] * symp_inds_to_treat[:,i].sum()))
                        if n > 0:
                            treated_symp_inds[:,i] = sainte_lague_distribution(n, symp_inds_to_treat[:,i])

                    for treatment_predistributed in np.unique(treatment_plan):
                        if treatment_predistributed == 0:
                            continue
                        # treatment groups are differentiated by age groups
                        treated_groups = np.arange(A)[treatment_plan == treatment_predistributed].astype(np.int32)

                        # save all individuals tho are treated
                        # treated_n shape = (time x testing or contact tracing x treatment type x age)
                        if variant == 0:
                            treated_n[curr_t,0,treatment_predistributed-1,1,treated_groups] += treated_symp_inds[:,treated_groups].sum(axis=0)
                        elif variant == 1:
                            treated_n[curr_t,0,treatment_predistributed-1,2,treated_groups] += treated_symp_inds[:,treated_groups].sum(axis=0)

                        # those who are treated sucessfully must have their time since symptom onset within treatment window
                        if variant == 0 and xi-eta <= treatment_window:
                            # compute number of individuals who comply and treated successfully (i.e. efficacy)
                            n_success = np.int32(np.around(treat_compliance[treatment_predistributed-1] * treat_success[treatment_predistributed-1] * treated_symp_inds[:,treated_groups].sum()))
                            # distribute across treated inds
                            successfully_treated = sainte_lague_distribution(n_success, treated_symp_inds[:,treated_groups].flatten()).reshape(treated_symp_inds[:,treated_groups].shape)
                            # remove from untreated group
                            update_infected_arr[:,0,treated_groups] -= successfully_treated

                            # distinguish between children and adsolescent+adults
                            successfully_treated_with_resistance = np.zeros(successfully_treated.shape, dtype=np.int32)
                            for age_status in range(2):
                                if age_status == 0:
                                    age_status_n_success = successfully_treated.sum(axis=0)[:2].sum()
                                else:
                                    age_status_n_success = successfully_treated.sum(axis=0)[2:].sum()
                                # compute number of individuals who develop resistance within-host
                                n_developing_resistance = np.int32(np.around(age_status_n_success * treat_resistance_prob[(treatment_predistributed - 1)*2 + age_status]))
                                if age_status == 0:
                                    successfully_treated_with_resistance[:,:2] += sainte_lague_distribution(n_developing_resistance, successfully_treated[:,:2].flatten()).reshape(successfully_treated[:,:2].shape)
                                else:
                                    successfully_treated_with_resistance[:,2:] += sainte_lague_distribution(n_developing_resistance, successfully_treated[:,2:].flatten()).reshape(successfully_treated[:,2:].shape)

                            # distribute resistance across treated inds
                            successfully_treated = successfully_treated - successfully_treated_with_resistance

                            ## -- no resistance -- ##
                            new_k = infpro_map[xi,treatment_predistributed-1,0] # new infection profile without resistant mutant
                            # treated without developing resistance
                            update_infected_arr[:,new_k,treated_groups] += successfully_treated

                            ## -- resistance -- ##
                            # resistance for oseltamivir is simply regarded as treatment failure but assumed to not spread (given that there is no reliable data to fit against viral load)
                            if treatment_predistributed == 1 and successfully_treated_with_resistance.sum() > 0:
                                # baloxavir has resistant mutation viral load profile
                                new_mut_k = infpro_map[xi, treatment_predistributed-1, 1] # new infection profile with resistant mutant
                                update_infected_arr[:,new_mut_k,treated_groups] += successfully_treated_with_resistance
                                mut_update_infected_arr[:,new_mut_k,treated_groups] += successfully_treated_with_resistance

            # update infected arr
            # history x time of infection x testing status x isolation status x infpro groups x age group
            infected_n[curr_t:,tau,0,:,0,:] -= tau_untreated_n # remove those isolated and/or treated
            infected_n[curr_t:,tau,0,:,:,:] += update_infected_arr

            # updat pri_mut_infected_n (second index = 0 - resistance developed within-host)
            pri_mut_infected_n[curr_t:,0,tau,0,:,1:,:] += mut_update_infected_arr[:,1:,:]

            # update pri_mut_infected_n (second index = 1 - transmitted)
            pri_mut_infected_n[curr_t:,1,tau,0,:,0,:] -= mut_tau_untreated_n
            pri_mut_infected_n[curr_t:,1,tau,0,:,0,:] += mut_update_infected_arr[:,0,:]

    return

def tat_isolation_contact_tracing(pos_cases, symp_onset_p, test_p, infected_n, treated_n, pri_mut_infected_n, curr_t, start_treatment_bool, test_sensitivity, treatment_bool, treatment_plan, treatment_window, treat_compliance, treat_success, treat_resistance_prob, infpro_map, iso_likelihood, contact_tracing, hh_rel_contact_matrix, mean_average_hh_contacts, sus_on_pep, agecat_n):

    HT, T, TS, IS, K, A = infected_n.shape # historical time, time of infection, diagnosis status, isolation status, infpro groups, age group

    MS = pri_mut_infected_n.shape[1] # mutant acquisition status (0 = within-host, 1 = transmission), time of infection, testing status, household pep status, infpro groups, age group

    if contact_tracing == True:
        ct_n = np.zeros(A, dtype=np.float32) # number of contacts traced by age

    for tau in range(T): # tau = time of infection
        xi = curr_t - tau # xi = age of infection

        eta_range = np.arange(xi+1).astype(np.int32)
        # must be within timeframe of symptom onset and testing
        eta_range = eta_range[(eta_range<symp_onset_p.size)&(xi-eta_range<test_p.size)]

        # calculate total probability of individuals wanting a test
        for eta in eta_range: # eta = day of symptom onset (includes today)

            # calculate probability of seeking a test
            prop_wanting_test = symp_onset_p[eta] * test_p[xi-eta]

            # get numebr of people infected at time tau who are undiagnosed and untreated (for infection but may have taken a post-exposure prophylaxis which failed to block infection)
            # we take the historical snapshot of transmission force yesterday (and update for today)
            tau_untested_n = infected_n[curr_t-1,tau,0,:,0,:] # shape = (IS x A)
            # only include those infected with mutant but are untested
            mut_tau_untested_n = pri_mut_infected_n[curr_t-1,1,tau,0,:,0,:] # shape = (IS x A)

            # compute number of individuals infected at tau who want a test at curr_t
            # number of individuals wanting test is dependent on total number of individuals infected
            # because probability of symptom onset and probability of testing are based simply the course of an infection
            n_wanting_test = np.int32(np.around(prop_wanting_test * tau_untested_n.sum()))
            mut_n_wanting_test = np.int32(np.around(prop_wanting_test * mut_tau_untested_n.sum()))

            if n_wanting_test + mut_n_wanting_test == 0:
                continue

            ## ------- mutant ------- ##
            if mut_n_wanting_test > 0:
                # currently assumes that all individuals who wants a test gets a test
                mut_n_tested = np.int32(np.around(1. * mut_n_wanting_test))
                if mut_n_tested > 0:
                    mut_tested_symp_inds = sainte_lague_distribution(mut_n_tested, mut_tau_untested_n.flatten()).reshape(mut_tau_untested_n.shape) # shape = (IS x A)

                    # identify tests that were positive
                    mut_n_positive = np.int32(np.around(test_sensitivity * mut_n_tested))
                    pos_cases[1] += mut_n_positive
                    # distribute positive tests across tested individuals
                    mut_pos_symp_inds = sainte_lague_distribution(mut_n_positive, mut_tested_symp_inds.flatten()).reshape(mut_tested_symp_inds.shape) # shape = (IS x A)

                    if contact_tracing == True:
                        # compute probability of contact tracing by age for positively-tested infectors infected during tau
                        #ct_n += np.dot(mut_pos_symp_inds.reshape((-1, A)).sum(axis=0)/pri_mut_infected_n[curr_t-1,1,tau].reshape((-1, A)).sum(axis=0), hh_rel_contact_matrix)
                        ct_n += np.dot(mut_pos_symp_inds.reshape((-1, A)).sum(axis=0), hh_rel_contact_matrix * mean_average_hh_contacts)

                    # array to update pri_mut_infected_n
                    mut_update_infected_arr = np.zeros((IS,K,A), dtype=np.int32)
                    mut_update_infected_arr[:,0,:] = mut_pos_symp_inds

                    ## -- isolation -- ##
                    if iso_likelihood > 0.:
                        # those who are not isolated might isolate
                        mut_n_isolated = np.int32(np.around(iso_likelihood * mut_pos_symp_inds[0].sum()))
                        # update infected arr those not isolating to isolate
                        mut_iso_symp_inds = sainte_lague_distribution(mut_n_isolated, mut_pos_symp_inds[0])
                        mut_update_infected_arr[0,0,:] -= mut_iso_symp_inds
                        mut_update_infected_arr[1,0,:] += mut_iso_symp_inds

                    ## -- treatment -- ## - assumed that test does not distinguish wt and mt
                    if treatment_bool == True and start_treatment_bool > 0:
                        # time since symptom onset must be within treatment window
                        if xi-eta <= treatment_window:
                            # for each type of treatment in treatment plan (diagnosis)
                            for treatment in np.unique(treatment_plan[0]):
                                if treatment > 0:
                                    # treatment groups are differentiated by age groups
                                    treated_groups = np.arange(A)[treatment_plan[0] == treatment].astype(np.int32)
                                    # identify treated inds
                                    mut_treated_symp_inds = mut_update_infected_arr[:,0,treated_groups]
                                    # save test-and-treat (assume all that are tested positive within treatment window are treated)
                                    # shape = (time, test-and-treat (0) or post-exposure prophylaxis (1), treatment (baloxavir or oseltamivir), (susceptible (0), drug-sensitive (1), resistant mutant-infected (2)), age)
                                    treated_n[curr_t,0,treatment-1,2,treated_groups] += mut_treated_symp_inds.sum(axis=0)
                                    # treatment however has no effect

                # updat pri_mut_infected_n (second index = 1 - transmitted)
                # change testing status
                pri_mut_infected_n[curr_t:,1,tau,0,:,0,:] -= mut_tested_symp_inds
                pri_mut_infected_n[curr_t:,1,tau,1,:,0,:] += mut_tested_symp_inds
                # update those who isolate and/or treated
                pri_mut_infected_n[curr_t:,1,tau,1,:,0,:] -= mut_pos_symp_inds
                pri_mut_infected_n[curr_t:,1,tau,1] += mut_update_infected_arr

            ## ------- wild-type ------- ##
            if n_wanting_test > 0:

                # currently assumes that all individuals who wants a test gets a test
                n_tested = np.int32(np.around(1. * n_wanting_test))
                if n_tested > 0:
                    tested_symp_inds = sainte_lague_distribution(n_tested, tau_untested_n.flatten()).reshape(tau_untested_n.shape) # shape = (IS x A)

                    # identify tests that were positive
                    n_positive = np.int32(np.around(test_sensitivity * n_tested))
                    pos_cases[0] += n_positive
                    # distribute positive tests across tested individuals
                    pos_symp_inds = sainte_lague_distribution(n_positive, tested_symp_inds.flatten()).reshape(tested_symp_inds.shape) # shape = (IS x A)

                    if contact_tracing == True:
                        # compute probability of contact tracing by age for positively-tested infectors infected during tau
                        #ct_n += np.dot(pos_symp_inds.reshape((-1, A)).sum(axis=0)/infected_n[curr_t-1,tau].reshape((-1, A)).sum(axis=0), hh_rel_contact_matrix)
                        ct_n += np.dot(pos_symp_inds.reshape((-1, A)).sum(axis=0), hh_rel_contact_matrix * mean_average_hh_contacts)

                    # array to update infected_n
                    update_infected_arr = np.zeros((IS,K,A), dtype=np.int32)
                    update_infected_arr[:,0,:] = pos_symp_inds

                    mut_update_infected_arr = np.zeros((IS,K,A), dtype=np.int32)

                    ## -- isolation -- ##
                    if iso_likelihood > 0.:
                        # those who are not isolated might isolate
                        n_isolated = np.int32(np.around(iso_likelihood * pos_symp_inds[0].sum()))
                        # update infected arr those not isolating to isolate
                        iso_symp_inds = sainte_lague_distribution(n_isolated, pos_symp_inds[0].flatten()).reshape(pos_symp_inds[0].shape)

                        update_infected_arr[0,0,:] -= iso_symp_inds
                        update_infected_arr[1,0,:] += iso_symp_inds

                    ## -- treatment -- ##
                    if treatment_bool == True and start_treatment_bool > 0:

                        # time since symptom onset must be within treatment window
                        if xi-eta <= treatment_window:

                            # for each type of treatment in treatment plan (diagnosis)
                            for treatment in np.unique(treatment_plan[0]):
                                if treatment > 0:
                                    # treatment groups are differentiated by age groups
                                    treated_groups = np.arange(A)[treatment_plan[0] == treatment].astype(np.int32)
                                    # identify treated inds
                                    treated_symp_inds = update_infected_arr[:,0,treated_groups]
                                    # save treated by test and treat to global treated_n (assume all that are tested positive within treatment window are treated)
                                    # treated_n shape = (time, test-and-treat (0) or post-exposure prophylaxis (1), treatment (baloxavir or oseltamivir), (susceptible (0), drug-sensitive (1), resistant mutant-infected (2)), age)
                                    treated_n[curr_t,0,treatment-1,1,treated_groups] += treated_symp_inds.sum(axis=0)

                                    # compute number of individuals who comply and are treated successfully (i.e. efficacy)
                                    n_success = np.int32(np.around(treat_compliance[treatment-1] * treat_success[treatment-1] * treated_symp_inds.sum()))
                                    # distribute across treated inds
                                    successfully_treated = sainte_lague_distribution(n_success, treated_symp_inds.flatten()).reshape(treated_symp_inds.shape)

                                    # remove from untreated group
                                    update_infected_arr[:,0,treated_groups] -= successfully_treated

                                    treated_groups_range = np.arange(treated_groups.size).astype(np.int32)
                                    # distinguish between children and adsolescent+adults
                                    successfully_treated_with_resistance = np.zeros((IS, treated_groups.size), dtype=np.int32)

                                    for age_status in range(2):
                                        if age_status == 0:
                                            age_status_n_success = successfully_treated[:,treated_groups_range[treated_groups<2]].sum()
                                        else:
                                            age_status_n_success = successfully_treated[:,treated_groups_range[treated_groups>=2]].sum()
                                        if age_status_n_success == 0:
                                            continue

                                        # compute number of individuals who develop resistance within-host
                                        n_developing_resistance = np.int32(np.around(age_status_n_success * treat_resistance_prob[(treatment-1)*2 + age_status]))
                                        if n_developing_resistance == 0:
                                            continue

                                        if age_status == 0:
                                            successfully_treated_with_resistance[:,treated_groups_range[treated_groups<2]] += sainte_lague_distribution(n_developing_resistance, successfully_treated[:,treated_groups_range[treated_groups<2]].flatten()).reshape(successfully_treated[:,treated_groups_range[treated_groups<2]].shape)
                                        else:
                                            successfully_treated_with_resistance[:,treated_groups_range[treated_groups>=2]] += sainte_lague_distribution(n_developing_resistance, successfully_treated[:,treated_groups_range[treated_groups>=2]].flatten()).reshape(successfully_treated[:,treated_groups_range[treated_groups>=2]].shape)

                                    # distribute resistance across treated inds
                                    successfully_treated_wo_resistance = successfully_treated - successfully_treated_with_resistance

                                    ## -- no resistance -- ##
                                    new_k = infpro_map[xi,treatment-1,0] # new infection profile without resistant mutant
                                    # treated without developing resistance
                                    update_infected_arr[:,new_k,treated_groups] += successfully_treated_wo_resistance

                                    ## -- resistance -- ##
                                    if treatment == 1: # baloxavir
                                        if successfully_treated_with_resistance.sum() > 0:
                                            # baloxavir has resistant mutation viral load profile
                                            new_mut_k = infpro_map[xi, treatment-1, 1] # new infection profile with resistant mutant
                                            update_infected_arr[:,new_mut_k,treated_groups] += successfully_treated_with_resistance

                                            mut_update_infected_arr[:,new_mut_k,treated_groups] += successfully_treated_with_resistance
                                    else:
                                        # resistance for oseltamivir is simply regarded as treatment failure but assumed to not spread (given that there is no reliable data to fit against viral load)
                                        update_infected_arr[:,0,treated_groups] += successfully_treated_with_resistance

                    # update infected arr
                    # change testing status
                    infected_n[curr_t:,tau,0,:,0,:] -= tested_symp_inds # IS x A
                    infected_n[curr_t:,tau,1,:,0,:] += tested_symp_inds # IS x A
                    # update those who isolate and/or treated
                    infected_n[curr_t:,tau,1,:,0,:] -= pos_symp_inds # IS x A
                    infected_n[curr_t:,tau,1] += update_infected_arr # (IS x K x A)

                    # updat pri_mut_infected_n (second index = 0 - resistance developed within-host)
                    pri_mut_infected_n[curr_t:,0,tau,1] += mut_update_infected_arr # (IS x K x A)

    # contact tracing
    if contact_tracing == True and start_treatment_bool > 0 and ct_n.sum() > 0.:
        ## -- compute number of susceptibles who will receive post-exposure prophylaxis during curr_t -- ##
        # compute total number of susceptible persons left in each age group
        total_infected_n = infected_n[curr_t,:curr_t].reshape((-1, agecat_n.size)).sum(axis=0)
        total_pri_mut_infected_n = pri_mut_infected_n[curr_t,1,:curr_t].reshape((-1, agecat_n.size)).sum(axis=0)
        sus_left = agecat_n - total_infected_n - total_pri_mut_infected_n
        # compute total number of susceptibles who did not receive pep in the period of its effectiveness
        sus_not_on_pep = sus_left - sus_on_pep.reshape((-1, agecat_n.size)).sum(axis=0)
        sus_not_on_pep[sus_not_on_pep<0] = 0
        # compute the number of susceptible individuals that would be given PEP
        sus_n_given_pep = np.around(ct_n * (sus_not_on_pep/agecat_n)).astype(np.int32)
        
        for treatment in np.unique(treatment_plan[1]):
            if treatment > 0:
                # treatment groups are differentiated by age groups
                treated_groups = np.arange(A)[treatment_plan[1] == treatment].astype(np.int32)
                sus_on_pep[treatment-1,-1,treated_groups] += sus_n_given_pep[treated_groups]
                # save PEP distributed for susceptibles
                # treated_n shape = (time, test-and-treat (0) or post-exposure prophylaxis (1), treatment (baloxavir or oseltamivir), (susceptible (0), drug-sensitive (1), resistant mutant-infected (2)), age)
                treated_n[curr_t,1,treatment-1,0,treated_groups] += sus_n_given_pep[treated_groups]

        ## -- compute the number of previously untested, untreated, infected individuals who will receive post-exposure prophylaxis during curr_t -- ##
        #inf_n_given_pep = np.around(ct_n * (total_infected_n/agecat_n)).astype(np.int32)
        #pri_mut_inf_n_given_pep = np.around(ct_n * (total_pri_mut_infected_n/agecat_n)).astype(np.int32)
        untreated_inf_n = infected_n[curr_t,:curr_t,0,0,0,:] # T x A
        untreated_pri_mut_inf_n = pri_mut_infected_n[curr_t,1,:curr_t,0,0,0,:] # T x A

        # only limit to period when antiviral works
        xi_range = (curr_t - np.arange(T)).astype(np.int32)
        xi_range = xi_range[xi_range <= np.argwhere(infpro_map>0)[:,0].max()]

        for xi in xi_range: # xi = age of infection
            tau = curr_t - xi # tau = time of infection

            # for each type of treatment in treatment plan (pep)
            for treatment in np.unique(treatment_plan[1]):
                if treatment == 0:
                    continue
                # treatment groups are differentiated by age groups
                treated_groups = np.arange(A)[treatment_plan[1] == treatment].astype(np.int32)

                ## ------- mutant ------- ##
                new_mut_k = infpro_map[xi,treatment-1,1]
                if new_mut_k > 0:
                    # get all untested, untreated individual infected during tau
                    tau_untested_n = untreated_pri_mut_inf_n[tau,treated_groups]
                    if tau_untested_n.sum() > 0:
                        # individuals that would be given pep for treatment
                        n_pep_given = np.int32((ct_n[treated_groups] * (tau_untested_n/agecat_n[treated_groups])).sum())
                        if n_pep_given > 0:
                            tau_treated_inds = sainte_lague_distribution(n_pep_given, tau_untested_n)
                            # treated_n shape = (time, test-and-treat (0) or post-exposure prophylaxis (1), treatment (baloxavir or oseltamivir), (susceptible (0), drug-sensitive (1), resistant mutant-infected (2)), age)
                            treated_n[curr_t,1,treatment-1,2,treated_groups] += tau_treated_inds
                            # update pri_mut_infected_n for pep-diagnosed individuals
                            pri_mut_infected_n[curr_t:,1,tau,0,0,0,treated_groups] -= tau_treated_inds
                            pri_mut_infected_n[curr_t:,1,tau,2,0,0,treated_groups] += tau_treated_inds

                ## ------- wild-type ------- ##
                new_k = infpro_map[xi,treatment-1,0]
                if new_k > 0:
                    # get all untested, untreated individual infected during tau
                    tau_untested_n = untreated_inf_n[tau,treated_groups]
                    if tau_untested_n.sum() > 0:

                        # individuals that would be given pep for treatment
                        n_pep_given = np.int32((ct_n[treated_groups] * (tau_untested_n/agecat_n[treated_groups])).sum())
                        if n_pep_given > 0:
                            tau_treated_inds = sainte_lague_distribution(n_pep_given, tau_untested_n)
                            # treated_n shape = (time, test-and-treat (0) or post-exposure prophylaxis (1), treatment (baloxavir or oseltamivir), (susceptible (0), drug-sensitive (1), resistant mutant-infected (2)), age)
                            treated_n[curr_t,1,treatment-1,1,treated_groups] += tau_treated_inds

                            # update infected_n for pep-diagnosed individuals
                            infected_n[curr_t:,tau,0,0,0,treated_groups] -= tau_treated_inds
                            infected_n[curr_t:,tau,2,0,0,treated_groups] += tau_treated_inds

                            # compute number of individuals who comply and are treated successfully (i.e. efficacy)
                            n_success = np.int32(np.around(treat_compliance[treatment-1] * treat_success[treatment-1] * n_pep_given))
                            # distribute across treated inds
                            successfully_treated = sainte_lague_distribution(n_success, tau_treated_inds)

                            # update infected_n of sucessfully treated individuals
                            infected_n[curr_t:,tau,2,0,0,treated_groups] -= successfully_treated

                            # distinguish between children and adsolescent+adults
                            successfully_treated_with_resistance = np.zeros(treated_groups.size, dtype=np.int32)
                            for age_status in range(2):
                                if age_status == 0:
                                    age_status_n_success = successfully_treated[treated_groups<2].sum()
                                else:
                                    age_status_n_success = successfully_treated[treated_groups>=2].sum()
                                if age_status_n_success == 0:
                                    continue

                                # compute number of individuals who develop resistance within-host
                                n_developing_resistance = np.int32(np.around(age_status_n_success * treat_resistance_prob[(treatment-1)*2 + age_status]))
                                if n_developing_resistance == 0:
                                    continue

                                if age_status == 0:
                                    successfully_treated_with_resistance[treated_groups<2] += sainte_lague_distribution(n_developing_resistance, successfully_treated[treated_groups<2])
                                else:
                                    successfully_treated_with_resistance[treated_groups>=2] += sainte_lague_distribution(n_developing_resistance, successfully_treated[treated_groups>=2])

                            # distribute resistance across treated inds
                            successfully_treated_wo_resistance = successfully_treated - successfully_treated_with_resistance

                            ## -- no resistance -- ##
                            new_k = infpro_map[xi,treatment-1,0] # new infection profile without resistant mutant
                            # update infected_n treated without developing resistance
                            infected_n[curr_t:,tau,2,0,new_k,treated_groups] += successfully_treated_wo_resistance

                            ## -- resistance -- ##
                            if treatment == 1: # baloxavir
                                if successfully_treated_with_resistance.sum() > 0:
                                    # baloxavir has resistant mutation viral load profile
                                    new_mut_k = infpro_map[xi, treatment-1, 1] # new infection profile with resistant mutant
                                    # update infected_n and pri_mut_infected_n treated developing resistance
                                    infected_n[curr_t:,tau,2,0,new_mut_k,treated_groups] += successfully_treated_with_resistance
                                    pri_mut_infected_n[curr_t:,0,tau,2,0,new_mut_k,treated_groups] += successfully_treated_with_resistance
                            else:
                                # resistance for oseltamivir is simply regarded as treatment failure but assumed to not spread (given that there is no reliable data to fit against viral load)
                                # update infected_n
                                infected_n[curr_t:,tau,2,0,0,treated_groups] += successfully_treated_with_resistance

    return

@nb.jit(nopython=True)
def safe_divide(num, den):
    y = np.zeros(num.size, dtype=np.float32)
    irange = np.arange(num.size)[den > 0.]
    for i in irange:
        y[i] = num[i]/den[i]
    return y

def get_contact_data(datadir, country, country_iso):
    contact_data = pd.read_csv(datadir + "/prem-et-al_synthetic_contacts_2020.csv").set_index(["iso3c", 'setting', 'location_contact']).sort_index()
    try:
        contact_data.loc[(country, 'overall', 'all')]
    except:
        contact_data = pd.read_csv(datadir + "/prem-et-al_synthetic_contacts_2017.csv").set_index(["iso3c", 'setting', 'location_contact']).sort_index()

    try:
        contact_data.loc[(country, 'overall', 'all')]
    except:
        raise Exception()

    contact_matrix = np.zeros((5, 20, 20), dtype=np.float32)
    for i, setting in enumerate(['all', 'home', 'work', 'school', 'others']):
        country_contact_data = contact_data.loc[(country, 'overall', setting)].copy()
        country_contact_data = country_contact_data.reset_index()[['age_contactor', 'age_contactee', 'mean_number_of_contacts']]
        country_contact_data = country_contact_data.pivot(index='age_contactor', columns='age_contactee', values='mean_number_of_contacts').to_numpy().astype(np.float32)
        contact_matrix[i,:,:] += country_contact_data # all data is average contacts per day per person = summation of all settings

    hh_data = pd.read_excel(datadir + "/undesa_pd_2022_hh-size-composition.xlsx")
    hh_data = hh_data[hh_data["Average household size (number of members)"] != ".."]
    mean_hh_contacts = np.float32(hh_data[hh_data["ISO Code"]==country_iso].sort_values(by="Reference date (dd/mm/yyyy)", ascending=False)['Average household size (number of members)'].iloc[0] - 1.)

    return contact_matrix, mean_hh_contacts
