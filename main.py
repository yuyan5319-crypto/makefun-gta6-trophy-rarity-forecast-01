"""GTA VI week-one trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts each PS5 trophy's Exophase earn rate at the case target, as five
quantiles in percent. Pure Python 3.13 standard library, deterministic, no
network, no randomness.

Model, per trophy
-----------------
The launch-night recording shows the trap in this contest: Exophase's tracked
population is not a fixed cohort. It sat at 638 players with flat rates all
launch night, then one refresh took it to 3,868 players and nearly every rate
fell. Rates move for two reasons at once -- more players earn each trophy as
they play through the game, and more players join the tracked population
having earned nothing yet -- so a rate's path is the ratio of two growing
counts, not a random walk around its last value. The house baseline holds the
last rate flat with a narrow spread and scores about 1.51 on the practice
because the refresh lands outside its band for almost every trophy.

This program therefore fits the trend in the data instead of holding it flat:

1.  Snapshots, not reads. Consecutive reads are usually byte-identical (the
    page only changes when Exophase refreshes), so the history is collapsed to
    distinct snapshots; otherwise a flat stretch of 10-minute reads would
    drown the few reads where anything actually moved.
2.  Two trend fits per trophy, both against x = ln(1 + hours since release):
    (a) weighted least squares of logit(rate) on x, and (b) weighted least
    squares of log(rate) on x. The log-time axis already damps extrapolation:
    a steep rise in the first hours does not keep rising steeply for a week.
    Recent snapshots count more (12-hour half-life). The two fits are
    averaged in logit space; they agree when rates are small and differ only
    where saturation matters, which is exactly where averaging is safer.
3.  Shrinkage. Each trophy's slope is pulled toward the mean slope of its
    tier (platinum / gold / silver / bronze), with the pull stronger when the
    trophy's own fit is noisy or its x-range is short. A trophy with no
    usable trend gets its tier's slope, and a trophy with no history at all
    gets its tier's last rate.
4.  Damped extrapolation. The predicted logit move is shrunk by
    1 / (1 + dx / 3), where dx is the log-time distance from the last read to
    the target, and capped at 3.5 logit units either way. A platinum (100%
    completion) is additionally capped at a 5% median: nobody plausibly
    100%-completes a game this size in week one at a higher rate than that.
5.  Honest spreads. The logit spread is
    sqrt(0.25 * days + slope_variance * dx^2 + 0.75^2 + residual_variance),
    floored at 0.9 and capped at 2.2. The 0.75 term is the refresh shock the
    practice demonstrates: Exophase refreshes on its own schedule, and a
    single refresh can move a rate by more than a logit. A spread that
    pretends that cannot happen pays for every trophy outside it, which is
    what the baseline's ~0.125 practice spread does.
6.  Flat-history dilution prior. If the history is a single distinct page
    (launch night, before any refresh), the median is shifted down half a
    logit: the tracked cohort so far is the most engaged players, and the
    broadening tracked population dilutes rates. Once the history moves,
    the fitted trend takes over and this prior never fires.

Edge cases: rates are clamped to [0.01, 99.99] before any logit; a rate of
exactly 100 (the launch-night page really does print one trophy at 100%)
is treated as 99.5 for the fit but its forecast median is left near 100 with
a one-sided-looking wide band below it. Null rates are skipped; a trophy
with no valid rate falls back to its tier prior.

Run: `python3 main.py`, reading $MAKEFUN_INPUT_DIR (else /input/) and writing
$MAKEFUN_OUTPUT_DIR (else /output/), one <id>.out per <id>.in, rows in trophy
id order, quantiles rounded to 4 decimals half-up, non-decreasing by
construction.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# GTA VI release, the origin of the log-time axis.
RELEASE_EPOCH = 1794940800.0  # 2026-11-19T00:00:00Z as a Unix timestamp

# Tier priors: (fallback rate %, median cap %).
# A platinum needs 100% completion; week-one rates for one are tiny.
TIER_PRIOR = {
    "platinum": (0.10, 5.0),
    "gold": (0.50, 40.0),
    "silver": (0.80, 80.0),
    "bronze": (1.00, 99.0),
}

RECENCY_HALF_LIFE_H = 12.0
MAX_LOGIT_MOVE = 3.5
SIGMA_FLOOR = 0.9
SIGMA_CAP = 2.2
REFRESH_SHOCK = 0.75  # logit sd of one Exophase refresh, per the practice


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        e = math.exp(-lg)
        return 100.0 / (1.0 + e)
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO 8601 UTC string -> Unix timestamp (stdlib only)."""
    # 2026-11-19T04:30:00.325Z or ...T04:30:00Z
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, s2, sxx)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = my - b * mx
    n = len(xs)
    if n > 2:
        s2 = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
        # Unbiased-ish residual variance on the weighted scale.
        s2 = s2 * n / max(n - 2, 1)
    else:
        s2 = 0.0
    return (a, b, s2, sxx)


def distinct_snapshots(history):
    """Collapse consecutive identical reads; keep (time_hours, players, rates)."""
    snaps = []
    prev_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        try:
            t = parse_time(read["at"])
        except Exception:
            t = parse_time(read["slot"])
        snaps.append((t, read.get("players_tracked"), rates))
    return snaps


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    history = case.get("history") or []
    if not history:
        raise ValueError("case has no history")
    trophies = case["trophies"]
    horizon_days = case["horizon_minutes"] / 1440.0
    snaps = distinct_snapshots(history)
    last_t = snaps[-1][0]
    t_target = last_t + case["horizon_minutes"] * 60.0
    x_target = math.log1p(max(0.0, (t_target - RELEASE_EPOCH) / 3600.0))

    # Per-trophy trend fits, first pass (for tier means).
    fits = []
    for i, trophy in enumerate(trophies):
        tier = trophy.get("tier", "bronze")
        prior_rate, _cap = TIER_PRIOR.get(tier, (1.0, 99.0))
        xs, y_logit, y_log, ws = [], [], [], []
        last_rate = None
        for (t, _players, rates) in snaps:
            if i >= len(rates):
                continue
            r = rates[i]
            if r is None:
                continue
            last_rate = r
            hours = max(0.0, (t - RELEASE_EPOCH) / 3600.0)
            x = math.log1p(hours)
            w = 0.5 ** ((last_t - t) / (RECENCY_HALF_LIFE_H * 3600.0))
            xs.append(x)
            y_logit.append(logit(r))
            y_log.append(math.log(clamp_pct(r) / 100.0))
            ws.append(w)
        fits.append(
            {
                "tier": tier,
                "prior_rate": prior_rate,
                "last_rate": last_rate,
                "xs": xs,
                "y_logit": y_logit,
                "y_log": y_log,
                "ws": ws,
            }
        )

    # Tier-mean logit slopes, from trophies whose own fit has a real x-range.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = []
        for f in fits:
            if f["tier"] != tier or len(f["xs"]) < 2:
                continue
            fit = wls(f["xs"], f["y_logit"], f["ws"])
            if fit is None or fit[3] <= 1e-9:
                continue
            slopes.append(fit[1])
        if slopes:
            slopes.sort()
            tier_slopes[tier] = slopes[len(slopes) // 2]  # median

    rows = []
    for i, trophy in enumerate(trophies):
        f = fits[i]
        tier = f["tier"]
        _prior, cap = TIER_PRIOR.get(tier, (1.0, 99.0))
        last_rate = f["last_rate"]
        if last_rate is None:
            last_rate = f["prior_rate"]
        last_logit = logit(last_rate)
        x_last = math.log1p(max(0.0, (last_t - RELEASE_EPOCH) / 3600.0))
        dx = max(0.0, x_target - x_last)

        slope = tier_slopes.get(tier, 0.0)
        slope_var = 0.25  # wide prior variance when the trophy has no fit
        resid_var = 0.0
        pred_logit = None
        if len(f["xs"]) >= 2:
            fit_l = wls(f["xs"], f["y_logit"], f["ws"])
            fit_g = wls(f["xs"], f["y_log"], f["ws"])
            if fit_l is not None and fit_l[3] > 1e-9:
                a_l, b_l, s2_l, sxx_l = fit_l
                # Reliability of this trophy's own slope: grows with x-spread,
                # shrinks with residual noise.
                rel = sxx_l / (sxx_l + 0.02 + 4.0 * s2_l)
                b_own = b_l
                b_tier = tier_slopes.get(tier, 0.0)
                slope = rel * b_own + (1.0 - rel) * b_tier
                slope_var = max(0.0, s2_l / sxx_l) if sxx_l > 0 else 0.25
                resid_var = s2_l
                pred_l = a_l + b_l * x_target
                if fit_g is not None and fit_g[3] > 1e-9:
                    a_g, b_g, _s2_g, _sxx_g = fit_g
                    # log-rate fit, converted back to a logit prediction.
                    pred_g = logit(math.exp(a_g + b_g * x_target) * 100.0)
                    pred_logit = 0.5 * (pred_l + pred_g)
                else:
                    pred_logit = pred_l

        if pred_logit is None:
            # No usable trend: tier slope from the last rate, damped.
            move = slope * dx / (1.0 + dx / 1.5)
            pred_logit = last_logit + max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
        else:
            # Anchor the fitted line at the last observed rate: the line's
            # level can sit off the latest read after a refresh, and the
            # latest read is the freshest fact. Blend line and last rate.
            move = pred_logit - last_logit
            move = max(-MAX_LOGIT_MOVE, min(MAX_LOGIT_MOVE, move))
            # Damp the move: a steep rise in the first hours must not be
            # carried at full strength over a week. Logit misses are
            # punished symmetrically, but a rate that can only rise so far
            # (players still have to play through the content, and new
            # players keep diluting it) makes overshoot the costlier error.
            move = move / (1.0 + dx / 1.5)
            pred_logit = last_logit + move

        # A trophy printed at exactly 100% on launch night stayed 100%
        # through the first refresh; keep its median at the top but let the
        # lower quantiles fall a long way, because dilution is one refresh
        # away from biting it.
        if last_rate >= 99.99:
            pred_logit = logit(99.5)
        elif len(snaps) == 1 and (last_t - RELEASE_EPOCH) < 48 * 3600.0:
            # Flat-history dilution prior. When every read so far is the same
            # page, the tracked population is still the launch-night cohort:
            # the most engaged players, tracked before release. As Exophase
            # refreshes and the tracked population broadens toward ordinary
            # players, rates dilute -- the practice's first refresh took
            # players tracked from 638 to 3,868 and nearly every rate fell.
            # With no trend visible yet the size is unknowable, so shift the
            # median down only half a logit and let the wide spread carry
            # the rest. This branch never fires once the history moves, so
            # it cannot override a real trend in the final case.
            pred_logit -= 0.5
        median = inv_logit(pred_logit)
        median = min(median, cap)
        # No trophy's rate plausibly climbs more than 40 points over the
        # remaining horizon: content takes time to play through, and the
        # tracked population keeps growing with players who have earned
        # nothing yet. (Does not bind on flat histories.)
        if last_rate < 99.99:
            median = min(median, last_rate + 40.0)
        median = clamp_pct(median)
        med_logit = logit(median)

        sigma = math.sqrt(
            0.25 * horizon_days
            + slope_var * dx * dx
            + REFRESH_SHOCK * REFRESH_SHOCK
            + resid_var
        )
        sigma = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma))
        # A 100% trophy's band is one-sided in practice: widen it downward.
        if last_rate >= 99.99:
            sigma = max(sigma, 1.4)

        qs = []
        prev = None
        for z in Z:
            q = inv_logit(med_logit + z * sigma)
            q = min(100.0, max(0.0, q))
            q = round4(q)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
