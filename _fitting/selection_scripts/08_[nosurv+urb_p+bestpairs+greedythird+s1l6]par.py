import sys
from pathlib import Path
project_root = Path.cwd()  # importing functions from other folders
sys.path.insert(0, str(project_root))

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import pandas as pd
pd.options.mode.string_storage = "python"
pd.options.future.infer_string = False

from multiprocessing import Pool
from _data.data_utils import read_in
from _fitting.model_utils import model_fit, data_settings_to_name, model_settings_to_name, compare_models
import arviz as az


az.style.use("arviz-darkgrid")

################################################################################
if '___laptop' in os.listdir('.'):
    # laptop folder
    folder = "../_data/p-dengue/"
    outpath = "../_data/p-dengue/model_fits/"
elif '___server' in os.listdir('.'):
    # server folder
    folder = "../../../../data/lucaratzinger_data/p_dengue/"
    outpath = "../../../../data/lucaratzinger_data/p_dengue/model_fits/"
else:
    print('something wrong')

################################################################################
data_settings = {'admin':2, 'max_lag':6, 'start_year':2016, 'start_month':1, 'end_year':2019, 'end_month':12}
data_name = data_settings_to_name(data_settings)

fitting_task = 'nosurv+urb_p+bestpairs+greedythird+s1l6'  # fitting task name
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
        model_fit(
            worker_data, 
            worker_data_name, 
            model_settings, 
            worker_outpath, 
            n_chains=4, 
            n_draws=4000, 
            n_tune=1000, 
            invert_log=True,
            task=fitting_task,
            replace=False,
            clear_idata=True
        )
        return (model_name, "success")
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")
        return (model_name, f"failed: {e}")

################################################################################
_data = read_in(folder, **data_settings, standardise=True, dropna=True, celsius=True, tp_log=True)
statistics = _data.columns.tolist()
# that start with t2m, rh, tp
statistics = [name for name in statistics if name.startswith(('t2m','rh','tp'))]
# that contains 'pop_weighted'
statistics = [name for name in statistics if 'pop_weighted' in name]
# if it contains 'tp' then it should contain 'log('
statistics = [name for name in statistics if not (name.startswith('tp') and 'log(' not in name)]
print(statistics)
# sys.exit(0)

greedy_base = []
# best pairs + greedy third variable
greedy_base.append(['tp_24hmean_pop_weighted_log(2)', 'tp_24hmean_pop_weighted_log(5)', 't2m_max_pop_weighted(0)'])
greedy_base.append(['tp_24hmean_pop_weighted_log(2)', 'tp_24hmean_pop_weighted_log(5)', 't2m_max_pop_weighted(3)'])
greedy_base.append(['tp_24hmean_pop_weighted_log(5)', 'rh_mean_pop_weighted(1)', 't2m_mean_pop_weighted(3)'])
greedy_base.append(['tp_24hmean_pop_weighted_log(2)', 'tp_24hmean_pop_weighted_log(5)', 't2m_max_pop_weighted(2)'])
greedy_base.append(['tp_24hmean_pop_weighted_log(2)', 'tp_24hmean_pop_weighted_log(5)', 't2m_max_pop_weighted(1)'])
greedy_base.append(['tp_24hmean_pop_weighted_log(5)', 'rh_mean_pop_weighted(1)', 't2m_mean_pop_weighted(1)'])

# statistics = [stat for stat in statistics if stat not in greedy_base]

#print(len(statistics))
#print(statistics)

# sys.exit(0)
################################################################################
if __name__ == "__main__":
    # Build model dictionary
    model_dict = {}
    for surv in [None]:
        for urb in ['urbanisation_pop_weighted']:
            for base in greedy_base:
                for s1 in statistics:
                    if s1 in base:
                        continue
                    stat_names = base + [s1]
                    settings = {
                        'surveillance_name': surv,
                        'urbanisation_name': urb,
                        'stat_names': stat_names, 
                        'degree': 3, 
                        'num_knots': 5, 
                        'knot_type': 'quantile',
                        'orthogonal': True
                    }
                    model_dict[model_settings_to_name(settings)] = settings
    
    # Create tasks list
    tasks = list(model_dict.items())[:]
    
    print(f"Fitting {len(tasks)} models in total...")
    print(f"Data: {data_name}")
    # sys.exit(0)
    
    # Number of workers (adjust based on your server)
    # Each model uses n_chains, so N_WORKERS * n_chains = total cores used
    N_WORKERS = 50
    
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