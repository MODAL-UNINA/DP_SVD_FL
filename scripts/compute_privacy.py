#!/usr/bin/env python
"""Nominal (epsilon, delta) of the ideal mechanism for given (q, sigma, T, delta) (paper Sec. IV-C, Supplementary S8).

    python scripts/compute_privacy.py                                   # the main setting: epsilon ~ 10.7948
    python scripts/compute_privacy.py --sigma 0.788 0.95 1.11 1.687     # the privacy sweep of Table V

The value depends only on (q, sigma, T, delta): not on C, the dimension, the ranks or the bases.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dpsvdfl.privacy.accountant import ACCOUNTANT_VERSION, PrivacyAccountantConfig, compute_privacy_accounting  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RDP accountant of the Poisson-subsampled Gaussian mechanism.")
    parser.add_argument("--q", type=float, default=0.1, help="Poisson sampling rate (default 0.1)")
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.95], help="noise multiplier(s) (default 0.95)")
    parser.add_argument("--rounds", type=int, default=300, help="number of rounds T (default 300)")
    parser.add_argument("--num-client-slots", type=int, default=500, help="N; delta defaults to 1/N (default 500)")
    parser.add_argument("--delta", type=float, default=None, help="delta (default 1/N = 0.002)")
    args = parser.parse_args()
    print(f"accountant {ACCOUNTANT_VERSION}: q={args.q} T={args.rounds} N={args.num_client_slots}")
    for sigma in args.sigma:
        config = PrivacyAccountantConfig(sampling_rate=args.q, noise_multiplier=sigma, num_rounds=args.rounds,
                                         num_client_slots=args.num_client_slots, delta=args.delta)
        result = compute_privacy_accounting(config)
        print(f"sigma={sigma:<7g} delta={result.delta:g}  epsilon={result.epsilon:.10f}  "
              f"optimal order={result.optimal_order:g}")


if __name__ == "__main__":
    main()
