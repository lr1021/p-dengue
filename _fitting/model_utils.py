import pymc as pm
import numpy as np
from patsy import dmatrix
import re

def abbrev_surveillance(name):
    if name is None:
        return "nosurv"
    base = "surv"
    if "urban" in name:
        base = "urb_surv"
    weight = "p" if "pop_weighted" in name else "u"
    return f"{base}_{weight}"

def abbrev_urbanisation(name):
    if name is None:
        return "nourb"
    base = "urb"
    weight = "p" if "pop_weighted" in name else "u"
    std = "_std" if "std" in name else ""
    return f"{base}_{weight}{std}"

def abbrev_stat(stat):
    # remove spaces
    s = stat.replace(" ", "")
    
    # lag extraction: "(k)"
    lag = re.search(r"\((\d+)\)", s)
    lag_str = f"({lag.group(1)})" if lag else ""
    
    # weighting
    if "pop_weighted" in s:
        w = "p"
    elif "unweighted" in s:
        w = "u"
    else:
        w = ""
    
    # remove weighting + lag from base
    base = re.sub(r"_?(pop_weighted|unweighted).*", "", s)
    
    return f"{base}_{w}{lag_str}"

def model_settings_to_name(settings):
    surv = abbrev_surveillance(settings.get("surveillance_name"))
    urb = abbrev_urbanisation(settings.get("urbanisation_name"))
    
    stats = settings.get("stat_names", [])
    if len(stats) == 0:
        stat_str = "nostat"
    else:
        stat_str = "+".join(abbrev_stat(s) for s in stats)
    
    deg = settings.get("degree")
    k = settings.get("num_knots")
    
    knot_map = {"quantile": "q", "uniform": "u"}
    kt = knot_map.get(settings.get("knot_type"), settings.get("knot_type"))
    
    orth = "o" if settings.get("orthogonal") else "no"
    if len(stats) == 0:
        return f"[{surv} {urb}] [{stat_str}] []"
    else:
        return f"[{surv} {urb}] [{stat_str}] [{deg},{k},{kt},{orth}]"

def build_model(data, stat_names, degree=3, num_knots = 3, knot_type='quantile', orthogonal=True,
                surveillance_name='urban_surveillance_pop_weighted', urbanisation_name='urbanisation_pop_weighted_std'):
    model = pm.Model()
    with model:
        # Priors
        alpha = pm.Exponential("alpha", 0.5)
        intercept = pm.Normal("intercept", mu=0, sigma=2.5)
        if urbanisation_name:
            beta_u = pm.Normal("beta_u", mu=0, sigma=1)

        # splines
        knot_list = {}
        B = {}
        sigma_w = {}
        w = {}
        f = {}
        for stat_name in stat_names:
            d = data[stat_name]
            if knot_type=='equispaced':
                knot_list[stat_name] = np.linspace(np.min(d), np.max(d), num_knots+2)[1:-1]
            elif knot_type=='quantile':
                knot_list[stat_name] = np.percentile(d, np.linspace(0, 100, num_knots + 2))[1:-1]
            else:
                print('knot_list must be quantile or equispaced')

            B[stat_name] = dmatrix(f"bs(s, knots=knots, degree=degree, include_intercept=False)-1",
                        {"s": data[stat_name], "knots": knot_list[stat_name], "degree":degree})
            if orthogonal:
                B[stat_name] = np.asarray(B[stat_name])
                B[stat_name] = (B[stat_name] - B[stat_name].mean(axis=0)) / B[stat_name].std(axis=0)
                B[stat_name], _ = np.linalg.qr(B[stat_name])
        
            # Spline coefficients
            sigma_w[stat_name] = pm.HalfNormal(f"sigma_w({stat_name})", sigma=0.5)
            w[stat_name] = pm.Normal(f"w({stat_name})", mu=0, sigma=1, size=B[stat_name].shape[1], dims="splines")
        
            # Spline contribution (with scaled mean to zero soft constraint)
            f_raw = pm.math.dot(B[stat_name], sigma_w[stat_name]* w[stat_name])
            f_mean = pm.math.mean(f_raw)
            f_var  = pm.math.mean((f_raw - f_mean) ** 2)
            f_std = pm.math.sqrt(f_var + 1e-6)
            pm.Potential(
            f"f_centred_prior({stat_name})",
            pm.logp(pm.Normal.dist(mu=0.0, sigma=0.01), f_mean/(f_std+1e-6)))
            f[stat_name] = f_raw

        # Link
        log_mu = intercept + pm.math.log(data['population'])
        if surveillance_name:
            log_mu += pm.math.log(data[surveillance_name]+1e-3)
        if urbanisation_name:
            log_mu += beta_u*data[urbanisation_name]
        for stat_name in stat_names:
            log_mu += f[stat_name]

        # Likelihood
        y_obs = pm.NegativeBinomial('y_obs', mu=pm.math.exp(log_mu), alpha=alpha, observed=data['cases'])

    return model, B, knot_list

def data_settings_to_name(s):
    admin = s["admin"]
    start = f"{s['start_year']}{s['start_month']:02d}"
    end = f"{s['end_year']}{s['end_month']:02d}"
    return f"a{admin}_{start}_{end}"