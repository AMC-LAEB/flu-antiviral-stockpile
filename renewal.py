# import libraries
import renewal
import argparse
import sciris as sc
import numpy as np
import sys
import os

def simulate(params):

    pars = sc.objdict(
        ## -- population inputs -- ##
        country = params.country,
        hr_non_elderly_p = params.hr_non_elderly_p,

        ## -- epidemic inputs -- ##
        ndays = params.ndays,
        init = params.init,
        R = params.R,
        symptom_onset_lognorm_pars = np.asarray(params.symptom_onset_lognorm_pars),
        seek_test_lognorm_pars = np.asarray(params.seek_test_lognorm_pars),
        time_to_sev_gamma_pars = np.asarray(params.time_to_sev_gamma_pars),
        time_to_dea_lognorm_pars = np.asarray(params.time_to_dea_lognorm_pars),
        asymp_prob = params.asymp_prob,
        profile = params.profile,

        treatment_start_t = params.treatment_start_t,

        ## -- predistribution inputs -- ##
        predistribute_antiviral = params.predistribute_antiviral,
        treatment_predistributed = params.treatment_predistributed,
        starting_background_activity_t = params.starting_background_activity_t,

        ## -- test and treat inputs -- ##
        treatment_bool = params.treatment,
        baloxavir_transmission_effect = params.baloxavir_transmission_effect,
        vl_to_inf_model = params.vl_to_inf_model,
        treatment_plan = params.treatment_plan,
        treatment_window = params.window,
        test_willingness = params.test_willingness,
        test_sensitivity = params.test_sensitivity,
        treat_compliance = np.asarray(params.treat_compliance),
        treat_success = np.asarray(np.asarray(params.treat_efficacy)),
        treat_sev_odds_ratio = np.asarray(params.treat_sev_odds_ratio),
        treat_dea_odds_ratio = np.asarray(params.treat_dea_odds_ratio),
        treat_resistance_prob = np.asarray(params.treat_resistance),
        resistance_fitness_cost = np.asarray(params.resistance_trans_fitness_cost),

        ## -- isolation -- ##
        iso_likelihood = params.iso_likelihood,
        iso_contact_reduction_factor = 1 - np.asarray(params.iso_contact_reduction_factor),

        ## -- contact tracting -- ##
        contact_tracing = params.contact_tracing,
        prophylaxis_effect = params.prophylaxis_effect,
        prophylaxis_effect_t = params.prophylaxis_effect_t,

        ## -- simulator options -- ##
        verbose = params.verbose,
        datadir = params.datadir,
        seed = params.seed,
        )

    # make output directory
    if os.path.isdir(params.outdir) == False:
        os.makedirs(params.outdir)

    # setup population
    pop = renewal.Pop(pars)

    # run simulation
    sim = renewal.Sim(pars, pop)
    sim.run(params.outdir)

    return #sim.deaths_n.sum(axis=0), sim.agecat_n

def make_parser():
    '''
    Make argument parser
    '''
    parser = argparse.ArgumentParser(description='antiviral renewal')
    subparsers = parser.add_subparsers()

    ### -- simulation --- ###
    sim_parser = subparsers.add_parser('simulate', description='create and run simulation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## -- population inputs -- ##
    sim_parser.add_argument('--country', type = str, default='USA', help='country to simulate')
    sim_parser.add_argument('--hr_non_elderly_p', type = float, default=0.2, help='proportion of non-elderly (5-64y) at risk of severe disease')

    ## -- epidemic inputs -- ##
    sim_parser.add_argument('--ndays', type = int, default = 365, help='duration of simulation')
    sim_parser.add_argument('--init', type = float, default = 10, help='number of initial infections (distributed according to demography); if less than one, taken as proportion of population')
    sim_parser.add_argument('--R', '-R', type = float, default = 1.20, help='basic reproduction number of drug sensitive pandemic virus')
    sim_parser.add_argument('--symptom_onset_lognorm_pars', type = float, nargs = 3, default = [0.35, 0.41, 14], help='lognormal distribution parameters + max time horizon for distribution of time to symptom onset since infection')
    sim_parser.add_argument('--seek_test_lognorm_pars', type = float, nargs = 3, default = [5., 3., 8], help='(mean, sd) + max time horizon for distribution of time to seeking a test since symptom onset')
    sim_parser.add_argument('--asymp_prob', type = float, default = 0.16, help='asymptomatic probability (source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4586318/)')
    sim_parser.add_argument('--time_to_sev_gamma_pars', type = float, nargs = 3, default = [4.471, 1.097, 21], help='gamma distribution parameters + max time horizon for distribution of time to severe disease (hospitalization) since symptom onset')
    sim_parser.add_argument('--time_to_dea_lognorm_pars', type = float, nargs = 3, default = [2.157, 0.408, 28], help='lognormal distribution parameters + max time horizon for distribution of time to death since severe disease (hospitalization)')
    sim_parser.add_argument('--profile', type = str, default = "seasonal_us1718", help="relative susceptibility, hospitalization and death probabilities for different age groups - choices: 'seasonal_us1718', 'pandemic_2009' or 'pandemic_1918'")

    sim_parser.add_argument('--treatment_start_t', type = int, default = 28, help='number of days before any treatment interventions begins since first case detection')

    ## -- background activity -- ##
    sim_parser.add_argument("--predistribute_antiviral", action = 'store_true', help= "predistribute one antiviral course per person")
    sim_parser.add_argument("--treatment_predistributed", type = str, default = "baloxavir", help= "treatment to predistribute")
    sim_parser.add_argument("--starting_background_activity_t", type = int, default = 0, help = 'start of background activity week')

    ## -- test and treat inputs -- ##
    sim_parser.add_argument('--treatment', '-t', action = 'store_true', help = 'treatment boolean')
    sim_parser.add_argument('--baloxavir_transmission_effect', type=float, default=0.29, help='average effect of baloxavir on onward transmission based on CENTERSTONE trial')
    sim_parser.add_argument("--vl_to_inf_model", type = str, default = "negexp", help="viral load to infectiousness model ('negexp' (default), 'hill', 'logit')")
    sim_parser.add_argument('--treatment_plan', type=str, default="./treatment_plans/treatment_template.xlsx", help='treatment plan excel file')
    sim_parser.add_argument('--window', type = int, default = 2, help = 'treatment time window since symptom onset')
    sim_parser.add_argument('--test_willingness', '-w', type = float, default = 0., help='willingness to test')
    sim_parser.add_argument('--test_sensitivity', type = float, default = 0.7, help='test sensitivity')
    sim_parser.add_argument('--treat_compliance', type = float, nargs=2, default = [0.95, 0.65], help='compliance to completing (baloxavir, oseltamivir) course')
    sim_parser.add_argument('--treat_efficacy', type = float, nargs=2, default = [1., 1.], help='treatment efficacy (baloxavir, oseltamivir)')
    sim_parser.add_argument('--treat_resistance', type = float, nargs=4, default = [0.217, 0.088, 0.05, 0.05], help='conditional probability of developing resistance to (baloxavir - children (<12y), adolescents and adults (>=12y); oseltamivir - children (<12y), adolescents and adults (>=12y)) when treated')
    sim_parser.add_argument('--resistance_trans_fitness_cost', type = float, nargs=2, default = [0., 0.], help='Fitness cost of resistance mutant (baloxavir, oseltamivir) on its transmissibility; Negative values = fitness benefit of resistant mutant')

    sim_parser.add_argument('--treat_sev_odds_ratio', type = float, nargs=2, default = [0.75, 0.75], help='odds ratio of (baloxavir, oseltamivir) high-risk treated individuals who will be hospitalized (https://pubmed.ncbi.nlm.nih.gov/22371849/)')
    sim_parser.add_argument('--treat_dea_odds_ratio', type = float, nargs=2, default = [0.23, 0.23], help='odds ratio of (baloxavir, oseltamivir) high-risk treated individuals who will die (https://pubmed.ncbi.nlm.nih.gov/22371849/)')

    ## -- isolation inputs -- ##
    sim_parser.add_argument('--iso_likelihood', '-i', type = float, default = 0., help = 'likelihood positively tested or contact traced individuals isolating')
    sim_parser.add_argument('--iso_contact_reduction_factor', type = float, nargs=4, default = [0.1, 0.85, 0.85, 0.60], help='reduction in (home, workplace, school, other community) contacts if individual isolate; the higher the less contacts')

    ## -- household contact tracing and post-exposure prophylaxis -- ##
    sim_parser.add_argument('--contact_tracing', '-c', action = 'store_true', help = 'contact tracing for distributing post-exposure prophylaxis')
    sim_parser.add_argument('--prophylaxis_effect', type = float, nargs = 2, default = [0.43, 0.40], help = 'relative risk reduction to influenza infection due to prophylaxis use (baloxavir, oseltamivir)')
    sim_parser.add_argument('--prophylaxis_effect_t', type = float, nargs = 2, default = [10, 10], help = 'duration of prophylatic effect in days (baloxavir, oseltamivir)')

    ## -- simulator options -- ##
    sim_parser.add_argument('--outdir', type=str, default="./renewal_model_output", help='output directory')
    sim_parser.add_argument('--datadir', type = str, default='./data', help='data folder')
    sim_parser.add_argument('--seed', type=int, default=42, help='random seed')
    sim_parser.add_argument('--verbose', '-v', type = int, default=1, help='verbose')

    sim_parser.set_defaults(func=simulate)

    return parser

def main():
    # parse arguments
    parser = make_parser()
    params = parser.parse_args()
    # run function
    if params == argparse.Namespace():
        parser.print_help()
        return_code = 0
    else:
        return_code = params.func(params)
    sys.exit(return_code)

if __name__ == '__main__':
    main()
