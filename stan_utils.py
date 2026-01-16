import __main__
import numpy as np
import pandas as pd
from spline_utils import spline_values

def i_split_spline_values_inputs(s):
    ni_inc = __main__.ni_inc
    i_inc = __main__.i_inc

    order_ = s[2]
    ni_input_ = np.array(ni_inc[f'{s[0]}[{s[1]}]'])
    i_input_ = np.array(i_inc[f'{s[0]}[{s[1]}]'])

    # regular interval
    knots_ = np.linspace(min(np.min(ni_input_), np.min(i_input_)),
                         max(np.max(ni_input_), np.max(i_input_)), s[3])
    

    ext_knots_ = np.concatenate((np.repeat(knots_[0], order_-1), knots_, np.repeat(knots_[-1], order_-1)))
    return order_, ni_input_, i_input_, ext_knots_

def input_data(model_name, template=False, ss=False):
    ni_inc = __main__.ni_inc
    i_inc = __main__.i_inc

    if template:
        if ss:
            print("modular")
            ###
            if template == '24':
                surv_value = 'surveillance_pop_pct'
                urb_value = 'urban_pop_pct'

                r_ = {'ni_n': ni_inc.shape[0],
                        'i_n': i_inc.shape[0],

                        'ni_y': ni_inc['cases'].astype(int).tolist(),
                        'i_y': i_inc['cases'].astype(int).tolist(),

                        'ni_log_pop': ni_inc['log_pop'].tolist(),
                        'i_log_pop': i_inc['log_pop'].tolist(),

                        'ni_log_surv': np.log(ni_inc[surv_value]).tolist(),
                        'i_log_surv': np.log(i_inc[surv_value]).tolist(),

                        'ni_urb': (ni_inc[urb_value]/100).tolist(),
                        'i_urb': (i_inc[urb_value]/100).tolist()
                        }
                
                # modular spline inputs
                for j, s in enumerate(ss):
                    j += 1
                    order_, ni_input_, i_input_, ext_knots_ = i_split_spline_values_inputs(s)
                    B_ni_ = spline_values(ni_input_, order_, ext_knots=ext_knots_)
                    B_i_ = spline_values(i_input_, order_, ext_knots=ext_knots_)

                    r_[f'B_n_{j}'] = B_ni_.shape[1]
                    r_[f'B_ni_{j}'] = B_ni_
                    r_[f'B_i_{j}'] = B_i_
                # return
                return r_
            ###
        else:
            if model_name == '23':
                surv_value = 'surveillance_pop_pct'
                urb_value = 'urban_pop_pct'

                ## spline values (variable, lag, order)
                s1 = ['mean_temp', 0, 2]
                order_, ni_input_, i_input_, ext_knots_ = i_split_spline_values_inputs(s1)
                s1_B_ni = spline_values(ni_input_, order_, ext_knots=ext_knots_)
                s1_B_i = spline_values(i_input_, order_, ext_knots=ext_knots_)
                ##

                return {'ni_n': ni_inc.shape[0],
                        'i_n': i_inc.shape[0],

                        'ni_y': ni_inc['cases'].astype(int).tolist(),
                        'i_y': i_inc['cases'].astype(int).tolist(),

                        'ni_log_pop': ni_inc['log_pop'].tolist(),
                        'i_log_pop': i_inc['log_pop'].tolist(),

                        'ni_log_surv': np.log(ni_inc[surv_value]).tolist(),
                        'i_log_surv': np.log(i_inc[surv_value]).tolist(),

                        'ni_urb': (ni_inc[urb_value]/100).tolist(),
                        'i_urb': (i_inc[urb_value]/100).tolist(),

                        'B_n': s1_B_ni.shape[1],
                        'B_ni': s1_B_ni,
                        'B_i': s1_B_i
                        }
            else:
                print('Model data not configured')

def write_model_code(template, ss, variant, reg=False):
    data_target = '// spline data'
    parameter_target = '// spline parameters'
    prob_target = '// spline priors and calculations'
    targets = [data_target, parameter_target, prob_target]

    with open(f"1-Models/m{template}.stan", "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if any([t in line for t in targets]):
            if data_target in line:
                for j, s in enumerate(ss):
                    j += 1
                    lines.insert(i + 1, f'\tmatrix[i_n, B_n_{j}] B_i_{j};\n')
                    lines.insert(i + 1, f'\tmatrix[ni_n, B_n_{j}] B_ni_{j};\n')
                    lines.insert(i + 1, f'\tint<lower=1> B_n_{j};\n')
                    lines.insert(i + 1, '\n')

            if parameter_target in line:
                for j, s in enumerate(ss):
                    j += 1
                    lines.insert(i + 1, f'\tvector[B_n_{j}] alpha_{j};\n')

            if prob_target in line:
                for j, s in enumerate(ss):
                    j += 1
                    lines.insert(i + 1, f'\ti_sadd += B_i_{j}*alpha_{j};\n')
                    lines.insert(i + 1, f'\tni_sadd += B_ni_{j}*alpha_{j};\n')
                    lines.insert(i + 1, f'\talpha_{j} ~ normal(0,1);\n')
                    lines.insert(i + 1, '\n')
                    
    with open(f"1-Models/m{template}{variant}.stan", "w") as f:
        f.writelines(lines)