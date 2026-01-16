functions {
  real sparse_car_lpdf(vector phi, real alpha,
                       array[,] int W_sparse, vector D_sparse, vector lambda,
                       int n, int W_n) {
    row_vector[n] phit_D; // phi' * D
    row_vector[n] phit_W; // phi' * W
    vector[n] ldet_terms;
    
    phit_D = (phi .* D_sparse)';
    phit_W = rep_row_vector(0, n);
    for (i in 1 : W_n) {
      phit_W[W_sparse[1, i]] = phit_W[W_sparse[1, i]] + phi[W_sparse[2, i]];
      phit_W[W_sparse[2, i]] = phit_W[W_sparse[2, i]] + phi[W_sparse[1, i]];
    }
    
    for (i in 1 : n) {
      ldet_terms[i] = log1m(alpha * lambda[i]);
    }
    return 0.5 * (sum(ldet_terms) - (phit_D * phi - alpha * (phit_W * phi)));
  }
}
data {
    int<lower=1> ni_n;
    int<lower=1> i_n;

    int<lower=1> ni_n_regions;
    int<lower=1> i_n_regions;

    int<lower=1> n_timesteps;

    array[ni_n] int<lower=0> ni_y;
    array[i_n] int<lower=0> i_y;

    array[ni_n] int<lower=0> ni_region_index;
    array[i_n] int<lower=0> i_region_index;

    array[ni_n] int<lower=0> ni_time_index;
    array[i_n] int<lower=0> i_time_index;

    vector[ni_n] ni_log_pop;
    vector[i_n] i_log_pop;

    vector[ni_n] ni_log_surv;
    vector[i_n] i_log_surv;

    vector[ni_n] ni_urb;
    vector[i_n] i_urb;

    int W_n;
    vector[ni_n_regions] lambda;
    array[2, W_n] int W_sparse;
    vector[ni_n_regions] D_diag;
}
transformed data {
    int ni_n_zeros = 0;
    int ni_n_non_zeros = 0;

    for (n in 1:ni_n) {
        if (ni_y[n] == 0)
            ni_n_zeros += 1;
        else
            ni_n_non_zeros += 1;
    }

    array[ni_n_zeros] int ni_zeros;
    array[ni_n_non_zeros] int ni_non_zeros;
    {
        int z = 1;
        int nz = 1;

        for (n in 1:ni_n) {
            if (ni_y[n] == 0) {
                ni_zeros[z] = n;
                z = z + 1;
            } else {
                ni_non_zeros[nz] = n;
                nz = nz + 1;
            }
        }
    }
    int i_n_zeros = 0;
    int i_n_non_zeros = 0;

    for (n in 1:i_n) {
        if (i_y[n] == 0)
            i_n_zeros += 1;
        else
            i_n_non_zeros += 1;
    }

    array[i_n_zeros] int i_zeros;
    array[i_n_non_zeros] int i_non_zeros;
    {
        int z = 1;
        int nz = 1;

        for (n in 1:i_n) {
            if (i_y[n] == 0) {
                i_zeros[z] = n;
                z = z + 1;
            } else {
                i_non_zeros[nz] = n;
                nz = nz + 1;
            }
        }
    }
}
parameters {
    real beta_0;
    real<lower=1e-12, upper=1e6> disp;
    real<lower=0, upper=1> theta;
    real beta_urb;
}
transformed parameters {
}
model {
    beta_0 ~ normal(0, 1);
    disp ~ exponential(1);
    beta_urb ~ normal(0, 1);

    real log_theta = log(theta);
    real log1m_theta = log1m(theta);

    // non isolated
    for (n in 1:ni_n_zeros){
    target += log_sum_exp(log_theta,
                          log1m_theta + neg_binomial_2_log_lpmf(0 | beta_0 + ni_log_pop[ni_zeros[n]]
                          + ni_log_surv[ni_zeros[n]] + beta_urb*ni_urb[ni_zeros[n]], disp));
    }
    target += ni_n_non_zeros * log1m_theta;
    target += neg_binomial_2_log_lpmf(ni_y[ni_non_zeros] | beta_0 + ni_log_pop[ni_non_zeros]
                                      + ni_log_surv[ni_non_zeros] + beta_urb*ni_urb[ni_non_zeros], disp);
    //

    // isolated
    for (n in 1:i_n_zeros){
    target += log_sum_exp(log_theta,
                          log1m_theta + neg_binomial_2_log_lpmf(0 | beta_0 + i_log_pop[i_zeros[n]]
                          + i_log_surv[i_zeros[n]] + beta_urb*i_urb[i_zeros[n]], disp));
    }
    target += i_n_non_zeros * log1m_theta;
    target += neg_binomial_2_log_lpmf(i_y[i_non_zeros] | beta_0 + i_log_pop[i_non_zeros]
                                      + i_log_surv[i_non_zeros] + beta_urb*i_urb[i_non_zeros], disp);
    //
}