import numpy as np

def spline_values(input, order, ext_knots):
    if np.isnan(input).sum() > 0:
        raise ValueError("nan values in input")
    knots = ext_knots
    B = np.zeros((input.shape[0], len(knots)-order))
    if order==1:
        for i in range(B.shape[1]):
            B[:, i] = (input >= knots[i])&(input<knots[i+1])
    else:
        B_low = spline_values(input, order-1, knots)
        for i in range(B.shape[1]):
            left_denom = knots[i+order-1] - knots[i]
            right_denom = knots[i+order] - knots[i+1]

            left_term = 0.0
            right_term = 0.0

            if left_denom != 0:
                left_term = (input - knots[i]) / left_denom * B_low[:, i]
            if right_denom != 0:
                right_term = (knots[i+order] - input) / right_denom * B_low[:, i+1]

            B[:, i] = left_term + right_term
    return B