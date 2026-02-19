Looked at different possible transformations of the spline basis:

FULL
- B_full: curves / columns sum to 1
- B_full_centred: subtract mean from curves. This removes interaction with intercept across the dataset. In this case the unit vector is a zero vector of the matrix, meaning the curves / columns sum to zero, are not LI and this introduces correlation and unidentifiability
- B_full_centred_unit: this rescales all centred curves to unit range (maintains centering), might equalise curve contributions and make weights more comparable?
- B_full_std: this rescales all curves by their standard deviation, might equalise curve contributions and make weights more comparable?

DROP
- B_drop: drop first curve / columns. 
- B_drop_centred: having removed one curve the columns do not sum to zero, which improves correlations in pair plots
- B_drop_centred_unit:
- B_drop_centred_std:

Q: B=QR
- Q_full_centred: columns of B_full_centred are not LI so the last Q vector is a zero vector, though due to numerical errors this presents spikes / discontinuities
- Q_drop_centred: same curves, without the error zero curve


IMPORTANT:
- At first I used a non centred reparametrisation of sigma_w * w but in this case fitting had divergences at large values of sigma_w, as even small shifts in w would create large jumps (high curvature space). So the problem was not sampling at a funnel (small sigma_w) but rather 'informing' the gradient of w with sigma_w so centred parametrisation removed these divergences.
- centred parametrisation removed intercept correlation, with no need for mean zero constraints as these are analytically always true
- given use of centred parametrisation, B_drop_centred basis had LI curves instead of B_full_centred
- also tried different values of sigma_w, while the likelihood is strong enough so that this doesn't too strongly affect fitting (mean is always similar), tighter / better aligned with likelihood allows faster convergence

Conclusion:
- B_drop_centred (centred parametrisation, sigma_w=0.25) seems to work best compared to all B full/drop centred unit / non_unit centred_parametrisation sigma_w variance = 0.5 / 0.25. Intercept is uncorrelated, sigma_w presents no funnels or difficulties, w correlations are soft
- Q_drop_centred. All pair plots are really round, fitting is faster, higher ESS. Consideration: Given in this model we are defining f = Qw, w standard normal prior, in the previous setting this becomes f = QR R-1w = B R-1w so equal to B setting but with the weights having a N(0, R-1(R-1)^T) prior. Not sure how principled this is, have not found it implemented. The model parameters are basically the same, the curve is the same, but with tighter intervals.

