import sys
from pathlib import Path
project_root = Path.cwd()  # importing functions from other folders
sys.path.insert(0, str(project_root))

import os
from multiprocessing import Pool
from _data.data_utils import read_in
from _fitting.model_utils import model_fit, data_settings_to_name, model_settings_to_name, compare_models, model_fit_Bdropcentred
import arviz as az
import pandas as pd

az.style.use("arviz-darkgrid")

################################################################################
if '___laptop' in os.listdir('.'):
    # laptop folder
    folder = "../_data/p-dengue/"
    outpath = "../_data/p-dengue/model_fits/"
elif '___server' in os.listdir('.'):
    # server folder
    folder = "../../../../data/lucaratzinger_data/p_dengue/"
    outpath = "../../../../data/lucaratzinger_data/p_dengue/model_fits"
else:
    print('something wrong')

################################################################################
data_settings = {'admin':2, 'max_lag':6, 'start_year':2016, 'start_month':1, 'end_year':2019, 'end_month':12}
data_name = data_settings_to_name(data_settings)

fitting_task = 'intercept_prior'
################################################################################

def init_worker():
    """Initialize worker process with data"""
    global worker_data, worker_data_name, worker_outpath
    worker_data = read_in(folder, **data_settings, standardise=True, dropna=True, celsius=True, tp_log=True)
    worker_data_name = data_name
    worker_outpath = outpath
    print(f"Worker initialized with data: {worker_data_name}")

def worker(task):
    """Fit a single model"""
    model_name, model_settings = task
    print(f'Fitting model: {model_name}')
    try:
        model_fit_Bdropcentred(
            worker_data, 
            worker_data_name, 
            model_settings, 
            worker_outpath,
            fitting_task,
            n_chains=4, 
            n_draws=4000, 
            n_tune=1000,
            sampler='nutpie',
            invert_log=True,
            centred_w=True,
            check_report=True,
            check_idata=True,
            clear_idata=False,
            pars_in_name=['intercept_sigma']
        )

        return (model_name, "success")
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")
        return (model_name, f"failed: {e}")

################################################################################

if __name__ == "__main__":
    # Build model dictionary
    model_dict = {}
    
    for intercept_sigma_val in [0.5, 1.0, 2.5, 5.0, 10.0]:
        settings = {'stat_names': [],
                    'disp_sigma': 0.5,
                    'intercept_sigma': intercept_sigma_val,
                    'intercept_mu': -10,
                    'degree': 3, 
                    'num_knots': 5,
                    'beta_u_sigma': 1.0,
                    'sigma_w_sigma': 0.5,
                    'surveillance_name': None,
                    'urbanisation_name': 'urbanisation_pop_weighted_std',
        }
        model_name = model_settings_to_name(settings)
        pars_in_name = ['intercept_sigma']
        for par in pars_in_name:
            model_name += f"_{par}_{settings[par]}"
        model_dict[model_name] = settings
    
    # Create tasks list
    tasks = list(model_dict.items())
    
    print(f"Fitting {len(tasks)} models in total...")
    print(f"Data: {data_name}")
    
    # Number of workers (adjust based on your server)
    # Each model uses n_chains, so N_WORKERS * n_chains = total cores used
    N_WORKERS = 5
    
    with Pool(N_WORKERS, initializer=init_worker) as p:
        results = p.map(worker, tasks)
    
    # Print summary
    print("\n" + "="*50)
    print("Fitting Summary:")
    print("="*50)
    for model_name, status in results:
        print(f"{model_name}: {status}")

    # Now compare models after all fitting is done
    print("\n" + "="*50)
    print("Model Comparison:")
    print("="*50)
    metric="loo"
    comparison_df = compare_models(outpath, data_name, task=fitting_task, metric=metric)
    
    # Save comparison results
    save_path = os.path.join(outpath, f'{data_name}[{fitting_task}]', f"model_comparison({metric}).csv")
    comparison_df.to_csv(save_path)
    print(f"\nComparison saved to: {save_path}")

    elpd_metrics = pd.read_csv(os.path.join(outpath, f'{data_name}[{fitting_task}]', f"_model_elpd_metrics.csv"))
    elpd_metrics = elpd_metrics.sort_values(by='loo', ascending=False).reset_index(drop=True)
    elpd_metrics.to_csv(os.path.join(outpath, f'{data_name}[{fitting_task}]', f"_model_elpd_metrics.csv"), index=False)

    ## RESULTS - intercept_sigma 1.0
    # - all values fit the same, same mean, hdi_%3 and hdi_%97, and same posterior so likelihood is dominating.
    # Mean set at -10 to be more reasonable (so that exp of the mean is around 4.5e-5 which is closer to the data
    # value = common dengue incidence rates in indonesia (data has rate mean 3.7-e5)). Fits are basically indistinguishable,
    # will keep intercept_sigma at 1.0 as this already allows enough variation.