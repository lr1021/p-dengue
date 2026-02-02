import sys
from pathlib import Path
project_root = Path.cwd()  # importing functions from other folders
sys.path.insert(0, str(project_root))


import os
from _data.data_utils import read_in
from _fitting.model_utils import model_fit, data_settings_to_name, model_settings_to_name
import arviz as az

az.style.use("arviz-darkgrid")

################################################################################
if '___laptop' in os.listdir('.'):
    # laptop folder
    folder = "../_data/p-dengue/"
elif '___server' in os.listdir('.'):
    # server folder
    folder = "../../../../data/lucaratzinger_data/p_dengue/"
else:
    print('something wrong')

if '___laptop' in os.listdir('.'):
    # laptop folder
    outpath = "../_data/p-dengue/model_fits/"
elif '___server' in os.listdir('.'):
    # server folder
    outpath = "../../../../data/lucaratzinger_data/p_dengue/model_fits"

################################################################################
data_settings = {'admin':2, 'max_lag':6, 'start_year':2016, 'start_month':1, 'end_year':2019, 'end_month':12}
data_name = data_settings_to_name(data_settings)
data = read_in(folder, **data_settings, standardise=True, dropna=True, celsius=True, tp_log=True)
print('data: ',data_name)

################################################################################

model_dict = {}
for surv in [None, 'surveillance_pop_weighted', 'urban_surveillance_pop_weighted'][::]:
    settings = {
            'surveillance_name':surv,
            'urbanisation_name':None,
            'stat_names':[], 'degree':3, 'num_knots':5, 'knot_type':'quantile','orthogonal':True}
    model_dict[model_settings_to_name(settings)] = settings

################################################################################

for model_name, model_settings in model_dict.items():
    print('model: ', model_name)
    model_fit(data, data_name, model_settings, outpath, n_chains=4, n_draws=40, n_tune=10, invert_log=True)