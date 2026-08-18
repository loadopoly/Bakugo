"""Monte Carlo validation against synthetic ground truth.

The point of this script is coverage. A tool that reports a 95% confidence
interval is making a falsifiable claim: across many measurements, the truth
should fall inside the interval about 95% of the time. If it does not, the
error bar is decoration and every grade band built on it is misleading.

Run: python validate.py [n_trials]
"""

from __future__ import annotations

import sys
import numpy as np

from cardcenter.centering import measure_centering
from cardcenter.synth import render_capture
from cardcenter.types import CaptureSpec, DetectionError, SLAB_PRESETS


def run(n: int = 120, seed: int = 7, verbose: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    failures = 0

    for i in range(n):
        # Border widths spanning well-centred to badly off-centre.
        base_h = rng.uniform(2.0, 6.0)
        base_v = rng.uniform(2.0, 6.0)
        skew_h = rng.uniform(-0.45, 0.45)
        skew_v = rng.uniform(-0.45, 0.45)
        left = base_h * (1 + skew_h)
        right = base_h * (1 - skew_h)
        top = base_v * (1 + skew_v)
        bottom = base_v * (1 - skew_v)

        slab_name = str(rng.choice(["raw", "raw", "toploader", "psa", "bgs"]))
        tilt = float(rng.uniform(0.0, 32.0))
        az = float(rng.uniform(0.0, 360.0))
        dist = float(rng.uniform(150.0, 400.0))
        focal = 2400.0 * dist / 220.0
        text = bool(rng.random() < 0.5)

        try:
            img, gt, f = render_capture(
                left_mm=left,
                right_mm=right,
                top_mm=top,
                bottom_mm=bottom,
                tilt_deg=tilt,
                azimuth_deg=az,
                distance_mm=dist,
                focal_px=focal,
                slab=slab_name,
                add_border_text=text,
                seed=int(rng.integers(0, 10**6)),
            )
            res = measure_centering(
                img, slab=slab_name, capture=CaptureSpec(focal_px=f)
            )
        except DetectionError as exc:
            failures += 1
            if verbose:
                print(f"  [{i}] detection refused: {str(exc)[:90]}")
            continue

        for name, meas_pair, truth in (
            ("h", res.horizontal, gt.h_ratio),
            ("v", res.vertical, gt.v_ratio),
        ):
            m = meas_pair.ratio_pct
            lo, hi = m.interval(1.96)
            rows.append(
                {
                    "axis": name,
                    "truth": truth,
                    "meas": m.value,
                    "sigma": m.sigma,
                    "err": m.value - truth,
                    "covered": lo <= truth <= hi,
                    "tilt": tilt,
                    "slab": slab_name,
                }
            )

    if not rows:
        raise SystemExit("no successful measurements")

    err = np.array([r["err"] for r in rows])
    sig = np.array([r["sigma"] for r in rows])
    cov = np.array([r["covered"] for r in rows])
    z = err / np.where(sig < 1e-9, 1e-9, sig)

    out = {
        "n_images": n,
        "n_refused": failures,
        "n_measurements": len(rows),
        "bias_pp": float(err.mean()),
        "rms_pp": float(np.sqrt((err**2).mean())),
        "p95_abs_err_pp": float(np.percentile(np.abs(err), 95)),
        "max_abs_err_pp": float(np.abs(err).max()),
        "mean_reported_sigma_pp": float(sig.mean()),
        "actual_sigma_pp": float(err.std()),
        "coverage_95": float(cov.mean()),
        "z_std": float(z.std()),
        "rows": rows,
    }
    return out


def report(out: dict) -> None:
    print(f"images rendered      : {out['n_images']}")
    print(f"detection refusals   : {out['n_refused']}")
    print(f"ratio measurements   : {out['n_measurements']}")
    print()
    print(f"bias                 : {out['bias_pp']:+.3f} pp")
    print(f"RMS error            : {out['rms_pp']:.3f} pp")
    print(f"95th pct |error|     : {out['p95_abs_err_pp']:.3f} pp")
    print(f"max |error|          : {out['max_abs_err_pp']:.3f} pp")
    print()
    print(f"mean reported sigma  : {out['mean_reported_sigma_pp']:.3f} pp")
    print(f"actual error sigma   : {out['actual_sigma_pp']:.3f} pp")
    print(f"z-score std (want ~1): {out['z_std']:.2f}")
    print(f"95% CI coverage      : {out['coverage_95'] * 100:.1f}%  (want >= 95%)")

    if out["coverage_95"] < 0.93:
        print()
        print("FAIL: the reported confidence interval is too narrow. The error")
        print("bar is not honest and must be widened before this ships.")
    elif out["coverage_95"] > 0.995 and out["z_std"] < 0.4:
        print()
        print("NOTE: intervals are very conservative; bands will be wider than")
        print("necessary. Acceptable, but there is precision being left unused.")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    report(run(n))
