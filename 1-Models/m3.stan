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

    int W_n;
    vector[ni_n_regions] lambda;
    array[2, W_n] int W_sparse;
    vector[ni_n_regions] D_diag;
}
transformed data {
    vector[ni_n_regions] zeros;
    zeros = rep_vector(0, ni_n_regions);
}
parameters {
    real<lower=0> beta_0;
    matrix[ni_n_regions, n_timesteps] phi_raw;
    vector[i_n] p_raw;
    real<lower=0> tau;
    real<lower=-1, upper=1> alpha;
}
transformed parameters {
  matrix[ni_n_regions, n_timesteps] phi;
  phi = phi_raw / sqrt(tau);

  vector[i_n] p;
  p = p_raw / sqrt(tau);
}
model {
    for (t in 1:n_timesteps) {
        phi_raw[,t] ~ sparse_car(alpha, W_sparse, D_diag, lambda, ni_n_regions, W_n);
    }
    p_raw ~ normal(0, 1);
    beta_0 ~ normal(0, 1);
    tau ~ gamma(2, 2);
    
    for (i in 1:ni_n) {
    ni_y[i] ~ poisson_log(beta_0 + ni_log_pop[i] + phi[ni_region_index[i], ni_time_index[i]]);
    }
    i_y ~ poisson_log(beta_0 + i_log_pop + p);
    
}