import re
from subprocess import run
import numpy as np
import multiprocessing as mp

# viral load to infectiousness models
# assuming baloxavir transmission reduction effect = 29%

models_to_calibrated_initialization = {"logit":{"init":1e-4, "R":1.27}, "hill":{"init":1e-4, "R":1.3}, "negexp":{"init":1e-4, "R":1.3}}
time_to_treat_pars = {"1D":(1., 0.6), "2D":(2., 1.2), "5D":(5., 3.)} #

ncpus = 10
TW = 30
parallel_cmds_to_run = []

treatment_list = ["BXM"] # "OSL"

for model in ['hill', 'logit', 'negexp']: # models_to_calibrated_initialization.keys():
    init = str(models_to_calibrated_initialization[model]["init"])
    R = str(models_to_calibrated_initialization[model]["R"])
    # run baseline
    cmd = ["python", "renewal.py", "simulate", "--country", "USA", "--ndays", "365", "--profile", "seasonal_us1718",
           "--treat_resistance", "0.", "0.", "0.", "0.", # no resistance mutations
           "--vl_to_inf_model", model, "--init", init, "--R", R,
           "--outdir", "./simulations/us_seasonal_1718/%s/us_seasonal_1718"%(model),
           "--test_willingness", "0."]
    parallel_cmds_to_run.append(cmd)

    for TW in [10, 20, 30, 40, 50]:
        for label, pars in time_to_treat_pars.items():
            p0, p1 = str(pars[0]), str(pars[1])

            if label == "1D":
                TS_range = [100, 70]
            else:
                TS_range = [70]

            for treatment in treatment_list:
                treatment_plan = "./treatment_plans/tat_%s.xlsx"%(treatment)

                for TS in TS_range:
                    outdir = "./simulations/us_seasonal_1718/%s/us_seasonal_1718_%s_%s_TS%03i_TW%03i"%(model, label, treatment, TS, TW)
                    cmd = ["python", "renewal.py", "simulate", "--country", "USA", "--ndays", "365", "--profile", "seasonal_us1718",
                           "--treat_resistance", "0.", "0.", "0.", "0.", # no resistance mutations
                           "--vl_to_inf_model", model, "--init", init, "--R", R,
                           "--outdir", outdir,
                           "--test_willingness", str(TW/100), "--test_sensitivity", str(TS/100),
                           "--treat_compliance", "1.", "1.",
                           "--seek_test_lognorm_pars", p0, p1, "8",
                           "--treatment_start_t", "0",
                           "--treatment", "--treatment_plan", treatment_plan]
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
