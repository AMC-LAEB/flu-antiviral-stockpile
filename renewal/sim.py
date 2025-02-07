# import libraries
import re
import os
import numpy as np
import scipy as sp
import pandas as pd
import sciris as sc

from . import utils

class Sim():
    def __init__(self, pars=None, pop=None):
        self.verbose = pars.verbose
        # discretize time distribution data
        self.symp_onset_p, self.test_p, onset_time_to_sev_p, hosp_time_to_dea_p, onset_time_to_dea_p = utils.discretize_time_dist(pars.symptom_onset_lognorm_pars, pars.seek_test_lognorm_pars, pars.time_to_sev_gamma_pars, pars.time_to_dea_lognorm_pars)
        # load infectiousness data
        self.infpro_map, self.infpro, self.R_theta, self.trans_p, self.mutpro, self.mut_R_theta, self.mut_trans_p, self.resistance_profile_id_start = utils.load_infectiousness_data(pars.datadir, pars.baloxavir_transmission_effect, pars.vl_to_inf_model, pars.seed)
        # load relative susceptibility, hospitalization (severe disease) and death probabilities
        sussevdea_par = np.zeros((3, 20), dtype=np.float32)

        if pars.profile in ['pandemic_2009', 'covid', 'pandemic_1918', 'pandemic_1968', 'seasonal_us1718']: # 'seasonal_us1718'
            sevsus_pars_df = pd.read_excel(pars.datadir + "/burden_profiles.xlsx")
            sevsus_pars_df = sevsus_pars_df[sevsus_pars_df['profile']==pars.profile]
            for pi, pname in enumerate(['rel_sus', 'chr', 'cfr']):
                # load relative susceptiblity based on the different profile
                par_fdf = sevsus_pars_df[sevsus_pars_df['par']==pname]
                min_age_arr = np.int32(np.floor(par_fdf['min_age'].to_numpy()/5))
                max_age_arr = np.int32(np.floor((par_fdf['max_age'].to_numpy()+1)/5))
                val_arr = par_fdf['val'].to_numpy()
                for min_age, max_age, val in zip(min_age_arr, max_age_arr, val_arr):
                    sussevdea_par[pi,min_age:max_age] = val
        else:
            raise Exception("Incorrect profile entry")

        # adjust symp_onset_p by proportion of infections expected to be symptomatic
        self.symp_onset_p *= (1 - pars.asymp_prob)

        # initialize
        ## -- test willingness = total probability of testing -- ##
        self.test_willingness = pars.test_willingness

        ## -- contact tracing -- ##
        self.contact_tracing = pars.contact_tracing

        ## -- relative risk to infection due to prophylaxis prophylaxis -- ##
        self.prophylaxis_effect = np.float32(pars.prophylaxis_effect)
        self.prophylaxis_effect_t = np.int32(pars.prophylaxis_effect_t)

        ## -- isolation -- ##
        self.iso_likelihood = np.float32(pars.iso_likelihood)
        self.iso_contact_reduction_factor = np.float32(pars.iso_contact_reduction_factor)

        ## -- population -- ##
        self.agecat_n = pop.agecat_n

        # get household contact matrix (assuming that these are the number of contacts that can be contact traced)
        self.mean_average_hh_contacts = pop.mean_average_hh_contacts

        # get relative risk matrix
        self.rr_matrix, self.contact_frac, self.hh_rel_contact_matrix = utils.compute_relative_risk_matrix(pop.contact_matrix, sussevdea_par[0], self.iso_likelihood, self.iso_contact_reduction_factor)
        # compute conditional hospitalization (severe) probabilities stratified by age
        cond_sev_p = sussevdea_par[1]/(1 - pars.asymp_prob)
        self.onset_to_sev_p = np.float32(np.outer(cond_sev_p, onset_time_to_sev_p))
        # compute death probabilities since onset
        cond_dea_p = sussevdea_par[2]/(1 - pars.asymp_prob)
        self.onset_to_dea_p = np.float32(np.outer(cond_dea_p, onset_time_to_dea_p))

        ## -- epidemic -- ##
        self.R0 = pars.R
        self.ndays = pars.ndays
        self.treatment_start_t = pars.treatment_start_t

        ## -- test and treat -- ##
        self.treatment_bool = pars.treatment_bool
        self.treatment_window = np.int32(pars.treatment_window)
        self.test_sensitivity = np.float32(pars.test_sensitivity)
        self.treat_compliance = pars.treat_compliance.astype(np.float32)
        self.treat_success = pars.treat_success.astype(np.float32)
        self.treat_resistance_prob = pars.treat_resistance_prob.astype(np.float32)
        self.resist_mutant_fitness_cost = pars.resistance_fitness_cost.astype(np.float32)
        self.treat_sev_odds_ratio = pars.treat_sev_odds_ratio.astype(np.float32)
        self.treat_dea_odds_ratio = pars.treat_dea_odds_ratio.astype(np.float32)

        ## -- predistribution -- ##
        self.predistribute_antiviral = pars.predistribute_antiviral
        if self.predistribute_antiviral == True:
            # get age-structured daily probability of outpatient visit
            daily_prob_of_medical_visit = pd.read_csv(pars.datadir + "/edited_moses-e-al_lancetPH_2018.csv")
            daily_prob_of_medical_visit = daily_prob_of_medical_visit[daily_prob_of_medical_visit["country"]==pars.country.upper()]
            daily_prob_of_medical_visit = daily_prob_of_medical_visit.sort_values(by="age_category")
            daily_prob_of_medical_visit = daily_prob_of_medical_visit["p"].to_numpy()
            # get background probability of background respiratory complaints
            background_respiratory_prob = np.load(pars.datadir + "/us_average_ili_2016-2019.npz")['p']
            for i in range(background_respiratory_prob.shape[0]):
                background_respiratory_prob[i,:] *= daily_prob_of_medical_visit[i]
            # reorder time
            self.background_respiratory_prob = np.zeros(background_respiratory_prob.shape, dtype=np.float32)
            self.background_respiratory_prob[:,:366-pars.starting_background_activity_t] = background_respiratory_prob[:,pars.starting_background_activity_t:]
            self.background_respiratory_prob[:,366-pars.starting_background_activity_t:] = background_respiratory_prob[:,:pars.starting_background_activity_t]

        # treatment plan: time x (0 = diagnosed, 1 = contact-traced/pep, 2 = prep) x age groups
        self.treatment_plan = utils.parse_treatment_plan(pars.treatment_plan, self.ndays, self.agecat_n.size)

        ## -- get test probability over course of infection -- ##
        self.test_p *= self.test_willingness

        # timer
        self.t = 0

        # result arrays (wild-type)
        self.Rt = np.zeros((self.ndays+1, 2, self.infpro.shape[0], self.agecat_n.size, self.agecat_n.size), dtype=np.float32) # Rt over time (time, isolation status, dist-gen profile, age infector, age infectee)
        self.infected_n = np.zeros((self.ndays+1, self.ndays+1, 3, 2, self.infpro.shape[0], self.agecat_n.size), dtype=np.int32) # infected stratified by (time (to track history of changes between household pep status for contact tracing estimation), time of infection, diagnosis status (0=undiagnosed, 1=tested, 2=pep-treated), isolation status, dist-gen profile, age)
        self.treated_n =  np.zeros((self.ndays+1, 2, 2, 3, self.agecat_n.size), dtype=np.int32) # number of treatments given each day (time, test-and-treat or post-exposure prophylaxis, treatment (baloxavir or oseltamivir), (susceptible, drug-sensitive, resistant mutant-infected), age)

        # result arrays (primary (baloxavir) antiviral resistant mutant)
        self.primary_mut_Rt = np.zeros((self.ndays+1, 2, self.mutpro.shape[0], self.agecat_n.size, self.agecat_n.size), dtype=np.float32) # Rt over time (time, isolation status, dist-gen profile, age infector, age infectee)
        self.primary_mut_infected_n = np.zeros((self.ndays+1, 2, self.ndays+1, 3, 2, self.mutpro.shape[0], self.agecat_n.size), dtype=np.int32) # infected with mutant virus stratified by (time (to track history of changes between household pep status for contact tracing estimation), mutant acquisition status (0 = within-host, 1 = transmission), time of infection, diagnosis status (0=undiagnosed, 1=tested, 2=pep-treated), isolation status, pep status (0 = no pep, 1 = balxoavir, 2 = oseltamivir),, dist-gen profile, age)

        # number of positively tested cases over time (time x variant)
        self.pos_cases = np.zeros((self.ndays+1, 2), dtype=np.int32)
        # number of susceptible individuals that are currently on EFFECTIVE pep (treatment (baloxavir or oseltamivir) x treatment effective time (max) x age)
        self.sus_on_pep = np.zeros((2, self.prophylaxis_effect_t.max(), self.agecat_n.size), dtype=np.int32)

        # initialize infection
        self.seed = pars.seed
        self.init_infections(pars.init, sussevdea_par[0])

    def init_infections(self, init, relative_susceptibility):
        if self.verbose > 0:
            print ("Initializing epidemic: R = %.3f, Test willingness = %.3f, Treatment = %i"%(self.R0, self.test_willingness, self.treatment_bool))
        # initalize infections
        if init < 1.:
            init = np.int32(np.around(self.agecat_n.sum() * init))
        else:
            init = np.int32(init)
        # seed generator
        rng = np.random.default_rng(self.seed)
        # normalize relative susceptiblity
        normalized_relative_susceptibility = relative_susceptibility/relative_susceptibility.sum()
        # initialize epidemic based on relative susceptibility
        sample = np.int32(rng.choice(np.arange(self.agecat_n.size), init, p=normalized_relative_susceptibility, replace=True))
        chosen_agecat, chosen_agecat_counts = np.unique(sample, return_counts=True)
        chosen_agecat = np.int32(chosen_agecat)
        if any(chosen_agecat_counts > self.agecat_n[chosen_agecat]):
            chosen_agecat_counts = np.int32(np.around(chosen_agecat_counts/chosen_agecat_counts.sum() * init))
        # initialize drug-sensitive infections
        self.infected_n[self.t:,self.t,0,0,0,chosen_agecat] += chosen_agecat_counts

    def hospitalization_and_deaths(self):
        curr_infected_n = self.infected_n[self.ndays] + self.primary_mut_infected_n[self.ndays,1]
        curr_infected_n = curr_infected_n.reshape((self.ndays+1, -1, self.infpro.shape[0], self.agecat_n.size)).sum(axis=1)

        self.hospitalized_n = utils.compute_burden(self.symp_onset_p, self.onset_to_sev_p, curr_infected_n, self.treat_sev_odds_ratio, self.infpro_map) # number of hospitalized individuals each day (x treatment status (0 = untreated, 1 = baloxavir, 2 = oseltamivir) x age category)
        self.deaths_n = utils.compute_burden(self.symp_onset_p, self.onset_to_dea_p, curr_infected_n, self.treat_dea_odds_ratio, self.infpro_map).sum(axis=1) # number of deaths each day

        if self.verbose > 0:
            # print hospitalisation
            print ("{:,} hospitalisation".format(self.hospitalized_n.sum()))
            hosp_by_age = self.hospitalized_n.reshape((-1, self.agecat_n.size)).sum(axis=0)
            print ("0-19y, {:.2f} ... 20-65y, {:.2f} ... 65+y, {:.2f} per 100,000 individuals".format(1e5*hosp_by_age[:int(20/5)].sum()/self.agecat_n[:int(20/5)].sum(), 1e5*hosp_by_age[int(20/5):int(65/5)].sum()/self.agecat_n[int(20/5):int(65/5)].sum(), 1e5*hosp_by_age[int(65/5):].sum()/self.agecat_n[int(65/5):].sum()))

            # print mortality rate
            print ("{:,} deaths (CFR = {:.5f}%)".format(self.deaths_n.sum(), 100*self.deaths_n.sum()/curr_infected_n.sum()))
            deaths_by_age = self.deaths_n.sum(axis=0)
            print ("0-19y, {:.2f} ... 20-65y, {:.2f} ... 65+y, {:.2f} per 100,000 individuals".format(1e5*deaths_by_age[:int(20/5)].sum()/self.agecat_n[:int(20/5)].sum(), 1e5*deaths_by_age[int(20/5):int(65/5)].sum()/self.agecat_n[int(20/5):int(65/5)].sum(), 1e5*deaths_by_age[int(65/5):].sum()/self.agecat_n[int(65/5):].sum()))

        return

    def transmission(self):
        ## -- wild type -- ##
        # compute Rt for the day
        rt, n_sus_wo_pep = utils.compute_group_rt(self.agecat_n, self.R0, self.R_theta, self.trans_p, self.rr_matrix, self.infected_n[self.t,:self.t+1], self.primary_mut_infected_n[self.t,1,:self.t+1], self.sus_on_pep, self.prophylaxis_effect, self.prophylaxis_effect_t, self.t, 0)
        self.Rt[self.t] += rt # save historical Rt
        # compute drug-sensitive variant transmissions
        newly_infected, pep_failures = utils.compute_transmission(rt, self.infpro, self.mutpro, self.contact_frac, self.infected_n[self.t,:self.t], self.t, self.seed+self.t, self.resistance_profile_id_start, self.resist_mutant_fitness_cost, self.sus_on_pep, n_sus_wo_pep, self.prophylaxis_effect, 0)
        # update infected_n and sus_on_pep
        self.infected_n[self.t:,self.t,0,0,0,:] += newly_infected
        self.sus_on_pep -= pep_failures

        ## -- primary (baloxavir) mutant -- ##
        if self.primary_mut_infected_n[self.t,:,:self.t].sum() > 0:
            # compute Rt for the day
            primary_mut_rt, n_sus_wo_pep = utils.compute_group_rt(self.agecat_n, self.R0, self.mut_R_theta, self.mut_trans_p, self.rr_matrix, self.infected_n[self.t,:self.t+1], self.primary_mut_infected_n[self.t,1,:self.t+1], self.sus_on_pep, self.prophylaxis_effect, self.prophylaxis_effect_t, self.t, 1)
            self.primary_mut_Rt[self.t] += primary_mut_rt # save historical primary_mut_rt
            # compute transmissions
            pri_mut_newly_infected, pep_failures = utils.compute_transmission(primary_mut_rt, self.infpro, self.mutpro, self.contact_frac, self.primary_mut_infected_n[self.t,:,:self.t].sum(axis=0), self.t, self.seed+self.t, self.resistance_profile_id_start, self.resist_mutant_fitness_cost, self.sus_on_pep, n_sus_wo_pep, self.prophylaxis_effect, 1)
            # update primary_mut_infected_n and sus_on_pep
            self.primary_mut_infected_n[self.t:,1,self.t,0,0,0,:] += pri_mut_newly_infected
            self.sus_on_pep -= pep_failures

        return

    def update_sus_on_pep(self):
        # shift timeframe of sus_on_pep by 1 day
        self.sus_on_pep[:,:-1,:] = self.sus_on_pep[:,1:,:]
        self.sus_on_pep[:,-1,:] = 0.
        for t in range(2):
            # individuals beyond effective period is considered not on PEP anymore
            self.sus_on_pep[:,:-self.prophylaxis_effect_t[t],:] = 0.

    def save_results(self, outdir):
        # save demography
        np.savez(outdir + "/demography.npz", arr=self.agecat_n)
        # save newly infected over time
        np.savez(outdir + "/infected.npz", arr=self.infected_n[self.ndays]) # only save the last day
        np.savez(outdir + "/primary_mut_infected_n.npz", arr=self.primary_mut_infected_n[self.ndays]) # only save the last day
        # save amount of treatment given
        np.savez(outdir + "/treatment.npz", arr=self.treated_n)
        # save positive cases over time
        np.savez(outdir + "/pos_cases.npz", arr=self.pos_cases)
        # save hospitalized data
        np.savez(outdir + '/hospitalized.npz', arr=self.hospitalized_n)
        # save deaths
        np.savez(outdir + "/deaths.npz", arr=self.deaths_n)

    def run(self, outdir):
        # simulate
        while self.t < self.ndays:

            if self.t >= self.treatment_start_t:
                start_treatment_bool = 1
            else:
                start_treatment_bool = 0

            self.t += 1

            if self.predistribute_antiviral == True:
                # compute number of predistributed baloxavir course used
                # we assume that test_p = seek_av_p
                utils.compute_predistributed_antiviral_usage(self.symp_onset_p, self.test_p, self.infected_n[:,:self.t], self.treated_n[:self.t+1], self.primary_mut_infected_n[:,:,:self.t], self.t, start_treatment_bool, self.treatment_window, self.treat_compliance, self.treat_success, self.treat_resistance_prob, self.infpro_map, self.predistribute_antiviral, self.treatment_plan[self.t,0], self.background_respiratory_prob)
            else:
                # compute individuals who were tested, with positive results, isolation, treatment if any
                utils.tat_isolation_contact_tracing(self.pos_cases[self.t], self.symp_onset_p, self.test_p, self.infected_n[:,:self.t], self.treated_n[:self.t+1], self.primary_mut_infected_n[:,:,:self.t], self.t, start_treatment_bool, self.test_sensitivity, self.treatment_bool, self.treatment_plan[self.t], self.treatment_window, self.treat_compliance, self.treat_success, self.treat_resistance_prob, self.infpro_map, self.iso_likelihood, self.contact_tracing, self.hh_rel_contact_matrix, self.mean_average_hh_contacts, self.sus_on_pep, self.agecat_n)

            # compute transmissions
            self.transmission()

            # update sus_on_pep (i.e. those on effective pep)
            if self.sus_on_pep.sum() > 0:
                self.update_sus_on_pep()

            if self.verbose > 0:
                print ("-- D{:03d}/W{:02d} --".format(self.t, np.int32(np.floor(self.t/7))+1))
                today_wt_cases = self.infected_n[self.t,self.t].sum()
                total_wt_cases = self.infected_n[self.t,:self.t+1].sum()
                total_wt_cases_with_resistance_mutant = self.primary_mut_infected_n[self.t,0,:self.t+1].sum()
                today_mt_cases = self.primary_mut_infected_n[self.t,1,self.t].sum()
                total_mt_cases = self.primary_mut_infected_n[self.t,1,:self.t+1].sum()

                print ("{:,} susceptibles on effective PEP".format(self.sus_on_pep.sum()))
                print (self.sus_on_pep)
                if self.sus_on_pep.sum() < 0:
                    print (self.sus_on_pep)
                    raise Exception
                print ("New cases: {:,} (W), {:,} (M)".format(today_wt_cases, today_mt_cases))
                print ("Total cases: {:,} (W) [{:,} ({:.2f}%) developed resistance], {:,} (M)".format(total_wt_cases, total_wt_cases_with_resistance_mutant, 100 * total_wt_cases_with_resistance_mutant/total_wt_cases, total_mt_cases))

                wt_cases_tested = self.infected_n[self.t,:self.t+1,1].sum()
                mt_cases_tested = self.primary_mut_infected_n[self.t,1,:self.t+1,1].sum()
                print ("Tested: {:,} (W), {:,} (M)".format(wt_cases_tested, mt_cases_tested))

                cumm_wt_pos_cases, cumm_mt_pos_cases, cumm_all_pos_cases = self.pos_cases[:self.t,0].sum(), self.pos_cases[:self.t,1].sum(), self.pos_cases[:self.t,:].sum()
                print ("+cases: {:,} (W), {:,} (M)".format(cumm_wt_pos_cases, cumm_mt_pos_cases))

                # number of treatments given each day (time, test-and-treat or post-exposure prophylaxis, treatment (baloxavir or oseltamivir), (susceptible, drug-sensitive, resistant mutant-infected), age)
                wt_cases_tat = self.treated_n[:self.t,0,:,1,:].sum()
                mt_cases_tat = self.treated_n[:self.t,0,:,2,:].sum()
                sus_pep_given = self.treated_n[:self.t,1,:,0,:].sum()
                wt_pep_given = self.treated_n[:self.t,1,:,1,:].sum()
                mt_pep_given = self.treated_n[:self.t,1,:,2,:].sum()

                print ("Treated +cases: {:,} (W), {:,} (M)".format(wt_cases_tat, mt_cases_tat))
                print ("PEP given: {:,} (S), {:,} (W), {:,} (M)".format(sus_pep_given, wt_pep_given, mt_pep_given))
                print ("")
            #if self.t == 23:
            #    raise Exception
        # simulate hospitalization and deaths
        self.hospitalization_and_deaths()

        # save results
        self.save_results(outdir)

class Pop():
    def __init__(self, pars=None):
        self.country = pars.country.upper()

        country_iso_codes_df = pd.read_csv(pars.datadir + "/country_iso_codes.csv")
        country_iso_codes_df = country_iso_codes_df[country_iso_codes_df['alpha-3'] == self.country]
        self.country_iso = np.int64(country_iso_codes_df['country-code'].iloc[0])
        country_name = country_iso_codes_df['name'].iloc[0]

        if pars.verbose > 0:
            print ("Creating population object for %s..."%(country_name))

        self.setup_pop_obj(pars.datadir)

    def setup_pop_obj(self, datadir):

        self.agecat_n = np.zeros(20, dtype=np.int32)

        # get age demography
        wpp = pd.read_excel(datadir + "/WPP2022_POP_F02_1_POPULATION_5-YEAR_AGE_GROUPS_BOTH_SEXES.xlsx")
        country_wpp = wpp[(wpp['ISO3 Alpha-code']==self.country)&(wpp['Year']==2021)]
        age_demography = (country_wpp[['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95-99', '100+']].to_numpy()[0])
        age_demography[-2] += age_demography[-1]
        age_demography = age_demography[:-1] * 1e3
        self.agecat_n = np.int32(age_demography)

        # get contact data (average contacts per day) and mean household size
        self.contact_matrix, self.mean_average_hh_contacts = utils.get_contact_data(datadir, self.country, self.country_iso)
