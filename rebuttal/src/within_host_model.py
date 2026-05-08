def within_host_model(y, t, pars):
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

def solve_model(rate_pars, pm, cm, mu, e, tspan, y_init):
    bw, bm, l, r, d, k, q, a, pw, cw = rate_pars
    pars = np.asarray([bw, bm, l, r, d, k, q, a, pw, pm, cw, cm, mu, e])
    sol = odeint(func=within_host_model, y0=y_init, t=tspan, args=(pars,))
    return sol

def compute_norm_ll(y_pred, y_true_pars):
    mu_ll = -np.sum([st.norm(loc=y_true_pars[0,i], scale=y_true_pars[1,i]).logpdf(yp).sum() for i, yp in enumerate(y_pred)])
    if np.isnan(mu_ll):
        return np.inf
    else:
        return mu_ll

def de(min_bound, max_bound, popsize, its):
    mut = 0.3
    crossp = 0.8
    dim = min_bound.size
    diff = max_bound - min_bound # difference between bounds
    idx = np.arange(popsize) # index
    # generate population (normalized)
    population = np.random.rand(popsize, dim)
    # de-normalize population
    population_denorm = min_bound + population * diff
    # compute fitness
    fitness = np.asarray([fobj(state=population_denorm[i,:]) for i in range(popsize)])
    best_fitness = fitness.min()
    best_idx = fitness.argmin()

    #print (fitness.mean(), best_fitness)
    for i in range(its):
        if i == 0 or ((i+1)%(its/100) == 0 and fitness.min() < best_fitness):
            best_fitness = fitness.min()
            print (i+1, list(min_bound + population[best_idx] * diff), np.mean(fitness), fitness.min())

            plot_wh_traj(min_bound + population[best_idx] * diff)

        for j in range(popsize):
            # randomly choose three indices without replacement excluding j
            a, b, c = population[np.random.choice(idx[idx != j], 3, replace=False)]
            # generate mutant
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dim) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            # get trial candidate
            trial = np.where(cross_points, mutant, population[j])
            trial_denorm = min_bound + trial * diff
            trial_f = fobj(state=trial_denorm)
            # selection
            if trial_f < fitness[j]:
                fitness[j] = trial_f
                population[j] = trial

        best_idx = fitness.argmin()

    print ("best:", min_bound + population[best_idx] * diff, fitness.min(), fitness.mean())
    return min_bound + population[best_idx] * diff

def fobj(state, T0=4e8): # state

    #bw, bm, l, r, d, k, q, a, pw, cw, pmb, cmb, pmo, cmo, Vw0, mub, eb, muo, eo = state
    bw, bm, l, r, d, k, q, a, pw, cw, pmb, cmb, Vw0, mub, eb, eo = state
    rate_pars = np.asarray([bw, bm, l, r, d, k, q, a, pw, cw])

    ll = 0 # total log likelihood
    for treatment in ['placebo', 'baloxavir', 'oseltamivir', 'baloxavir_resistance']: # oseltamivir_resistance
        # get tspan
        tspan = np.linspace(0., 11., (11*24)+1)
        t_eval = treatment_to_data[treatment]['data_t'].mean(axis=0)[1:] # first time point = treatment
        tspan = np.unique(np.concatenate((tspan, t_eval)))

        # get fitting data
        y_true_pars = treatment_to_data[treatment]['vl_norm_pars']
        if treatment == 'baloxavir_resistance':
            y_Vw_true_pars = treatment_to_data['baloxavir']['vl_norm_pars']
        elif treatment == 'oseltamivir_resistance':
            y_Vw_true_pars = treatment_to_data['oseltamivir']['vl_norm_pars']

        if treatment == 'placebo':
            # fit placebo data - mu = e = 0
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = solve_model(rate_pars, pm=0., cm=0.,  mu=0., e=0., tspan=tspan, y_init=y0_init)
            # fit wild-type vl to placebo viral load data
            Vw = s0[:,-2]
            Vw[Vw < 5.] = 5.
            Vw = np.log10(Vw)
            # compute log-likelihood
            ll += compute_norm_ll(Vw[np.isin(tspan, t_eval)], y_true_pars)

        elif treatment == 'baloxavir' or treatment == 'oseltamivir':
            # two phase (pre-treatment and post-treatment)
            # pre-treatment
            t0 = tspan[tspan < t_eval[0]]
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0., tspan=t0, y_init=y0_init)
            # post-treatment
            t1 = tspan[tspan >= t_eval[0]]
            y1_init = s0[-1,:]
            s1 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=eb if treatment == 'baloxavir' else eo, tspan=t1, y_init=y1_init)
            # fit viral load to baloxavir treated data
            s1_Vw = s1[:,-2]
            s1_Vw[s1_Vw < 5.] = 5.
            s1_Vw = np.log10(s1_Vw)
            ll += compute_norm_ll(s1_Vw[np.isin(t1, t_eval)], y_true_pars)

        elif treatment == 'baloxavir_resistance':
            # three phases
            # pre-treatment + no mutation
            t0 = tspan[tspan < t_eval[0]]
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0., tspan=t0, y_init=y0_init)
            # post-treatment + mutation
            t1 = tspan[tspan >= t_eval[0]]
            y1_init = s0[-1,:]
            s1 = solve_model(rate_pars, pm=pmb, cm=pmb, mu=mub, e=eb, tspan=t1, y_init=y1_init)
            # fit viral load to baloxavir treated data
            Vw = np.concatenate((s0[:,-2], s1[:,-2]))
            Vm = np.concatenate((s0[:,-1], s1[:,-1]))
            Vw[Vw < 5.] = 5.
            Vm[Vm < 5.] = 5.
            Vw = np.log10(Vw)
            Vm = np.log10(Vm)
            ll += compute_norm_ll(Vw[np.isin(tspan, t_eval)], y_Vw_true_pars)
            ll += compute_norm_ll(Vm[np.isin(tspan, t_eval[1:])], y_true_pars[:,1:])

        elif treatment == 'oseltamivir_resistance':
            # pre-treatment + no mutation
            t0 = tspan[tspan < t_eval[0]]
            y0_init = np.asarray([T0, 0., 0., 0., 0., Vw0, 0.])
            s0 = solve_model(rate_pars, pm=0., cm=0., mu=0., e=0., tspan=t0, y_init=y0_init)
            # post-treatment + mutation
            t1 = tspan[tspan >= t_eval[0]]
            y1_init = s0[-1,:]
            s1 = solve_model(rate_pars, pm=pmo, cm=cmo, mu=muo, e=eo, tspan=t1, y_init=y1_init)
            # fit viral load to oseltamivir treated data
            Vw = np.concatenate((s0[:,-2], s1[:,-2]))
            mt_t_eval = treatment_to_data['oseltamivir']['data_t'].mean(axis=0)[1:]
            ll += compute_norm_ll(Vw[np.isin(tspan, mt_t_eval)], y_Vw_true_pars)

            Vm = np.concatenate((s0[:,-1], s1[:,-1]))
            ll += compute_norm_ll(Vm[np.isin(tspan, t_eval)], y_true_pars)

    return ll
