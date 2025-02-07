import re
import os
from subprocess import run
import numpy as np
import multiprocessing as mp

ncpus = 50

country_list = ['USA', 'AFG', 'AGO', 'ALB', 'ARE', 'ARG', 'ARM', 'ATG', 'AUS', 'AUT', 'AZE', 'BDI', 'BEL', 'BEN', 'BFA', 'BGD', 'BGR', 'BHR', 'BHS', 'BIH', 'BLR', 'BLZ', 'BOL', 'BRA', 'BRB', 'BRN', 'BTN', 'BWA', 'CAF', 'CAN', 'CHE', 'CHL', 'CHN', 'CIV', 'CMR', 'COD', 'COG', 'COL', 'COM', 'CPV', 'CRI', 'CUB', 'CYP', 'CZE', 'DEU', 'DJI', 'DNK', 'DOM', 'DZA', 'ECU', 'EGY', 'ERI', 'ESP', 'EST', 'ETH', 'FIN', 'FJI', 'FRA', 'GAB', 'GBR', 'GEO', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'GRC', 'GTM', 'GUY', 'HKG', 'HND', 'HRV', 'HTI', 'HUN', 'IDN', 'IND', 'IRL', 'IRN', 'IRQ', 'ISL', 'ISR', 'ITA', 'JAM', 'JOR', 'JPN', 'KAZ', 'KEN', 'KGZ', 'KHM', 'KIR', 'KOR', 'KWT', 'LAO', 'LBN', 'LBR', 'LBY', 'LCA', 'LKA', 'LSO', 'LTU', 'LUX', 'LVA', 'MAC', 'MAR', 'MCO', 'MDA', 'MDG', 'MDV', 'MEX', 'MKD', 'MLI', 'MLT', 'MMR', 'MNE', 'MNG', 'MOZ', 'MRT', 'MUS', 'MWI', 'MYS', 'NAM', 'NER', 'NGA', 'NIC', 'NLD', 'NOR', 'NPL', 'NZL', 'OMN', 'PAK', 'PAN', 'PER', 'PHL', 'PNG', 'POL', 'PRI', 'PRK', 'PRT', 'PRY', 'PSE', 'QAT', 'ROU', 'RUS', 'RWA', 'SAU', 'SDN', 'SEN', 'SGP', 'SLB', 'SLE', 'SLV', 'SRB', 'SSD', 'STP', 'SUR', 'SVK', 'SVN', 'SWE', 'SWZ', 'SYC', 'SYR', 'TCD', 'TGO', 'THA', 'TJK', 'TKM', 'TLS', 'TON', 'TTO', 'TUN', 'TUR', 'TWN', 'TZA', 'UGA', 'UKR', 'URY', 'UZB', 'VCT', 'VEN', 'VNM', 'VUT', 'WSM', 'YEM', 'ZAF', 'ZMB', 'ZWE']

pandemic_to_R = {'pandemic_1918':2.0, 'pandemic_1968':1.8, 'pandemic_2009':1.5, 'covid':3.0}

time_to_treat_pars = {"1D":(1., 0.6), "2D":(2., 1.2), "5D":(5., 3.)}

treatment_list = ["BXM", "BXM05-24", "BXM25-64", "BXM65-99"] # OSL #["BXM10-44", "BXM10-64", ] "OSL", #

resistance_p_dict = {"H1RESIST":[0.094, 0.022], "H3RESIST":[0.210, 0.085]} #

treatment_window = [2, 8]

treatment_start_t_arr = [7, 28, 84]

init = 10
model = "hill"
TS = 70
TW = 100

parallel_cmds_to_run = []

for pandemic, R in pandemic_to_R.items():
    for country in country_list:
        for treatment_start_t in treatment_start_t_arr:
            for treatment in treatment_list:
                treatment_plan = "./treatment_plans/tat_%s.xlsx"%(treatment)
                for ttt_label, ttt_pars in time_to_treat_pars.items():
                    for treat_win in treatment_window:
                        """
                        outdir = "./simulations/%s/%s/%s/%s_%s_AVS%03i_%s_%s_TS%03i_TW%03i_WIN%02i_PEP"%(model, pandemic.replace("_", ""), country, country, pandemic.replace("_", ""), treatment_start_t, ttt_label, treatment, TS, TW, treat_win)
                        cmd = ["python", "renewal.py", "simulate", "--country", country, "--ndays", "365", "--profile", pandemic,
                               "--treat_resistance", "0.", "0.",  "0.", "0.", # no resistance mutations
                               "--vl_to_inf_model", model, "--init", str(init), "--R", str(R),
                               "--outdir", outdir,
                               "--test_willingness", str(TW/100), "--test_sensitivity", str(TS/100),
                               "--seek_test_lognorm_pars", str(ttt_pars[0]), str(ttt_pars[1]), "8",
                               "--treatment_start_t", str(treatment_start_t),
                               "--treatment", "--treatment_plan", treatment_plan, "--window", str(treat_win),
                               "--contact_tracing"]
                        if not os.path.isfile(outdir + "/deaths.npz"):
                            parallel_cmds_to_run.append(cmd)
                        """
                        for resist_label, resist_p_arr in resistance_p_dict.items():
                            outdir = "./simulations/%s/%s/%s/%s_%s_AVS%03i_%s_%s_%s_TS%03i_TW%03i_WIN%02i_PEP"%(model, pandemic.replace("_", ""), country, country, pandemic.replace("_", ""), treatment_start_t, ttt_label, treatment, resist_label, TS, TW, treat_win)
                            cmd = ["python", "renewal.py", "simulate", "--country", country, "--ndays", "365", "--profile", pandemic,
                                   "--treat_resistance", str(resist_p_arr[0]), str(resist_p_arr[1]),  "0.", "0.", # no resistance mutations
                                   "--vl_to_inf_model", model, "--init", str(init), "--R", str(R),
                                   "--outdir", outdir,
                                   "--test_willingness", str(TW/100), "--test_sensitivity", str(TS/100),
                                   "--seek_test_lognorm_pars", str(ttt_pars[0]), str(ttt_pars[1]), "8",
                                   "--treatment_start_t", str(treatment_start_t),
                                   "--treatment", "--treatment_plan", treatment_plan, "--window", str(treat_win),
                                   "--contact_tracing"]
                            if not os.path.isfile(outdir + "/deaths.npz"):
                                parallel_cmds_to_run.append(cmd)


def run_worker(cmd):
    run(cmd)

def main():
    pool = mp.Pool(processes=ncpus)
    results = [pool.apply_async(run_worker, args=(parallel_cmds_to_run[x],)) for x in range(len(parallel_cmds_to_run))]
    output = [p.get() for p in results]
    pool.close()

if __name__ == "__main__":
    main()
