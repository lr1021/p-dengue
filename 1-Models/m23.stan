functions {
}
data {
    int<lower=1> ni_n;
    int<lower=1> i_n;

    array[ni_n] int<lower=0> ni_y;
    array[i_n] int<lower=0> i_y;

    vector[ni_n] ni_log_pop;
    vector[i_n] i_log_pop;

    vector[ni_n] ni_log_surv;
    vector[i_n] i_log_surv;

    vector[ni_n] ni_urb;
    vector[i_n] i_urb;

    int<lower=1> B_n;
    matrix[ni_n, B_n] B_ni;
    matrix[i_n, B_n] B_i;
}
transformed data {
}
parameters {
    real beta_0;
    real beta_urb;
    vector[B_n] alpha;

    real<lower=1e-12, upper=1e6> disp;
}
transformed parameters {
}
model {
    beta_0 ~ normal(0, 1);
    disp ~ exponential(1);
    beta_urb ~ normal(0, 1);
    alpha ~ normal(0,1);

    ni_y ~ neg_binomial_2_log(beta_0 + ni_log_pop + ni_log_surv + beta_urb*ni_urb + B_ni*alpha, disp);
    i_y ~ neg_binomial_2_log(beta_0 + i_log_pop + i_log_surv + beta_urb*i_urb + B_i*alpha, disp);
}