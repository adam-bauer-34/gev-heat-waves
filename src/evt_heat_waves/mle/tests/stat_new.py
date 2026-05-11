"""Testing for new stat fit.

Adam Bauer
UChicago

Last edited: 4/30/2026, 7:37 PM CST
"""

import shutil

import numpy as np
from scipy.stats import genextreme

from evt_heat_waves.mle.mle import _mle_fit
from evt_heat_waves.mle.se import get_standard_errors

width = shutil.get_terminal_size(fallback=(80, 20)).columns


def gen_samples(params, n_samples):
    years = np.arange(0, n_samples, 1) / n_samples  # time
    samples = np.array(
        [
            genextreme.rvs(
                c=-(params[4] + params[5] * t),
                loc=params[0] + params[1] * t,
                scale=params[2] + params[3] * t,
                size=1
            )[0] for t in years
        ]
    )
    return samples

def compute_l2(est, true):
    return np.sqrt(np.sum((est - true) ** 2))

def main():
    # set seed
    np.random.seed(4)

    # set parameters to use to generate the samples
    theta_tr = [
        22.0,  # location
        0.0,  # location trend
        3.0,  # scale
        0.0, # scale trend
        -0.25, # shape
        0.0   # shape trend
    ]

    # take a number of different sample sizes
    # first is satellite era, second is full reanalysis record, final two are sanity checks
    sample_sizes = [2024 - 1979, 2024 - 1950, 100, 1000]
    desc = ["Satellite era", "Full reanalysis record", "Sanity", "Sanity"]

    print("=" * width)
    for i, n in enumerate(sample_sizes):
        print(f"Sample size: {n} | {desc[i]}")
        print("-" * width)
        samples = gen_samples(theta_tr, n)
        est_params = _mle_fit(samples, fit_type='stat_new', SAMPLE_THRES=1)
        se = get_standard_errors(est_params, samples, fit_type='stat_new')

        print(f"Estimated parameters: {est_params}")
        print(f"Standard errors: {se}")
        print(f"L2 distance from true parameters: {compute_l2(est_params, [theta_tr[0], theta_tr[2], theta_tr[4]])}")
        print("-" * width)


if __name__ == "__main__":
    main()