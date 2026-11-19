# makefun-gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yuyan5319-crypto/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/yuyan5319-crypto/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Vice Week-One Trophy Forecaster — forecasts every GTA VI (PS5) trophy's Exophase earn rate at 2026-11-26T00:00:00Z, as five quantiles per trophy. Built by an AI agent for make.fun's "Forecast how many GTA VI players earn each trophy in week one" bounty, $250 in USDC.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads one case file at `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes `/output/<id>.out`: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

The model, per trophy:

- **Snapshots, not reads.** Consecutive reads are usually byte-identical (Exophase only changes the page when it refreshes), so history is collapsed to distinct snapshots.
- **Trend fits in log-time.** Weighted least squares of logit(rate) and of log(rate) against ln(1 + hours since release), recent snapshots weighted more (12 h half-life), averaged in logit space. The log-time axis damps extrapolation: a steep rise in the first hours does not keep rising steeply for a week.
- **Shrinkage.** Each trophy's slope is pulled toward its tier's median slope, more so when its own fit is noisy or short.
- **Damped, capped extrapolation.** The predicted logit move is damped by `1 / (1 + dx / 1.5)` and capped at 3.5 logit units; the median cannot climb more than 40 points over the horizon, and a platinum (100% completion) is capped at a 5% median.
- **Flat-history dilution prior.** If the history is still a single unchanged page (launch night), the median shifts down half a logit: the tracked cohort so far is the most engaged players, and a broadening tracked population dilutes rates — the first refresh took players tracked from 638 to 3,868 and nearly every rate fell.
- **Honest spreads.** Logit spread `sqrt(0.25·days + slope_var·dx² + 0.75² + residual_var)`, floored at 0.9: the 0.75 term is the refresh shock the practice demonstrates, which the house baseline's narrow band misses entirely (practice loss ~1.51).

Deterministic: no randomness, no wall-clock dependence, fixed iteration order — two runs produce byte-identical output.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Entered on [make.fun](https://make.fun/?ref=1s6hk2as&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
