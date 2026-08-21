"""Save cached measurement-variance values in the legacy ``*_variances.p`` format."""

import argparse
import os
import pickle

import numpy as np


METRICS_SUFFIX = "_sampled_graphs_metrics.p"
VARIANCES_SUFFIX = "_variances.p"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Copy cached epsilon^2 M values from <prefix>_sampled_graphs_metrics.p "
            "to <prefix>_variances.p without recalculating them."
        )
    )
    parser.add_argument(
        "prefix",
        help="File prefix used by metrics_histo_pareto.py, for example H2 or BeH2.",
    )
    return parser.parse_args(argv)


def load_cached_measurements(metrics_path):
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(
            "Could not find metrics file '{}'. Run metrics_histo_pareto.py first.".format(
                metrics_path
            )
        )

    with open(metrics_path, "rb") as handle:
        metrics = pickle.load(handle)

    if not isinstance(metrics, dict) or "measurements" not in metrics:
        raise ValueError(
            "Metrics file '{}' does not contain a 'measurements' array.".format(
                metrics_path
            )
        )

    measurements = np.asarray(metrics["measurements"], dtype=float)
    if measurements.ndim != 1:
        raise ValueError(
            "Expected a one-dimensional 'measurements' array in '{}', got shape {}.".format(
                metrics_path,
                measurements.shape,
            )
        )
    return measurements.copy()


def main(argv=None):
    args = parse_args(argv)
    metrics_path = args.prefix + METRICS_SUFFIX
    output_path = args.prefix + VARIANCES_SUFFIX

    all_measurements = load_cached_measurements(metrics_path)
    print(
        "Loaded {} cached measurement values from {}".format(
            len(all_measurements),
            metrics_path,
        )
    )

    with open(output_path, "wb") as handle:
        pickle.dump(all_measurements, handle, pickle.HIGHEST_PROTOCOL)

    print("All Measurement saved to {}".format(output_path))


if __name__ == "__main__":
    main()
