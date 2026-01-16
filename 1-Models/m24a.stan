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

    // spline data

	int<lower=1> B_n_2;
	matrix[ni_n, B_n_2] B_ni_2;
	matrix[i_n, B_n_2] B_i_2;

	int<lower=1> B_n_1;
	matrix[ni_n, B_n_1] B_ni_1;
	matrix[i_n, B_n_1] B_i_1;
    //
}
transformed data {
}
parameters {
    real beta_0;
    real beta_urb;

    // spline parameters
	vector[B_n_2] alpha_2;
	vector[B_n_1] alpha_1;
    //

    real<lower=1e-12, upper=1e6> disp;
}
transformed parameters {
}
model {
    beta_0 ~ normal(0, 1);
    disp ~ exponential(1);
    beta_urb ~ normal(0, 1);

    vector[ni_n] ni_sadd;
    vector[i_n] i_sadd;
    ni_sadd = rep_vector(0, ni_n);
    i_sadd  = rep_vector(0, i_n);

    // spline priors and calculations

	alpha_2 ~ normal(0,1);
	ni_sadd += B_ni_2*alpha_2;
	i_sadd += B_i_2*alpha_2;

	alpha_1 ~ normal(0,1);
	ni_sadd += B_ni_1*alpha_1;
	i_sadd += B_i_1*alpha_1;
    //

    ni_y ~ neg_binomial_2_log(beta_0 + ni_log_pop + ni_log_surv + beta_urb*ni_urb + ni_sadd, disp);
    i_y ~ neg_binomial_2_log(beta_0 + i_log_pop + i_log_surv + beta_urb*i_urb + i_sadd, disp);
}