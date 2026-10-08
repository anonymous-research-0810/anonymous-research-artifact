# Meta Deep Hedging

This project contains the source code accompanying the paper **“Meta Deep Hedging: Regime-Aware and Data-Augmented Reinforcement Learning Agent for Option Hedging.”** It is prepared as an anonymous research artifact for peer review.

The implementation constructs hedging tasks from training-market regimes, augments each task with shared-path SABR option cohorts, and learns a context-conditioned hedging policy. It also includes conventional RL and Black–Scholes–Merton (BSM) Delta baselines, the two paper ablations, and the scripts for regime plots, latent-task diagnostics, and hyperparameter sensitivity plots.

## Data availability

**The SPX and SX5E datasets required for the paper's experiments cannot be distributed with this project because of data privacy restrictions.** Obtain the required data independently with appropriate access rights, and supply the external data directories through the configuration generator below. No market observations, precomputed results, trained models, or other execution artifacts are included.

For SPX, provide these Parquet files:

- `spx_options_YYYY.parquet`, one file per quote year from 2016 through 2025;
- `spx_spot.parquet`, containing daily index levels;
- `spx_div_yield.parquet`, containing dividend yields;
- `zero_curve.parquet`, containing zero-coupon interest-rate curves.

Annual option files use the OptionMetrics schema: `secid`, `date`, `exdate`, `cp_flag`, `strike_price`, `best_bid`, `best_offer`, `volume`, `open_interest`, `impl_volatility`, `delta`, `gamma`, `vega`, `theta`, `optionid`, and `symbol`. Dates must be parseable calendar dates. `strike_price` is the strike multiplied by 1,000; option bid/ask prices and index levels are in index points. `cp_flag` is `C` or `P`. The SPX identifier is 108105, and the paper uses SPXW PM-settled calls. IV is annualized in decimal units.

The spot table contains `secid`, `date`, and `close`. The dividend table contains `secid`, `date`, and `rate`; the zero curve contains `date`, `days`, and `rate`. Raw dividend and zero rates are percentages and are converted to annualized decimal rates by the loader. Market tables must cover all requested trading dates; curve maturities must be unique and increasing within each date.

For SX5E, provide `sx5e_options_YYYY.parquet`, `sx5e_spot.parquet`, `sx5e_div_yield.parquet`, and `zero_curve.parquet`. Option files contain the same option columns plus `quote_valid`. Spot files additionally contain `open`, `high`, `low`, `volume`, and `return`. Raw spot/dividend identifiers are `secid=-1001`. Strike and rate units follow the same conventions as above.

The SX5E adapter writes a `processed_options/` directory beside the raw data. It retains valid two-sided quotes and converts file and identifier conventions to the shared reader schema. The internal SPXW-compatible symbol and identifier are schema aliases; the experiment still records `data_source=sx5e`, uses EUR, and applies multiplier 10. SX5E settlement uses the paper's standard European-call payoff `max(S_T-K, 0)` based on the terminal closing index, rather than a separate intraday settlement average. Missing IV is recovered by the implemented pricing, same-strike, surface, and past-only fallback rules; missing liquidity fields retain explicit missing-value indicators.

## Installation

Run commands from this project's root directory. Python 3.10 is the reference runtime; a CPU is sufficient for verification, and training can use a compatible PyTorch CUDA installation.

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS. Then install:

```sh
python -m pip install -e ".[test]"
```

Alternatively, install `requirements.txt` and execute the scripts directly from the project root. The optional `wrds` extra enables the SPX acquisition utilities for users with their own WRDS access; it is not required to read externally supplied Parquet files.

## Project layout

| Location | Purpose |
| --- | --- |
| `data_processing/` and `data_source.py` | External-data readers, validation, and the SX5E adapter |
| `datasets/option_dataset/` | Complete option episodes, balanced cohorts, and IV recovery |
| `rl_envs/hedging_env/` | Self-financing hedging environment and training-only normalization |
| `rl_agents.py`, `rl_utils.py` | SAC/TD3 policies, replay, training, and evaluation |
| `segmentation/market_segmentation/` | Training-only RBF-PELT regime detection |
| `simulation/market_simulation/` | Regime-specific and global SABR calibration and simulation |
| `meta_rl/hedging_meta_rl/` | Probabilistic task inference and context-conditioned control |
| `no_simulation_variant/`, `no_meta_variant/` | The two paper ablations |
| `figure_making/` | Plots and latent-task mechanism analyses |
| `tests/` | Executable unit and integration checks |

Source-specific names such as SPXW and the external Parquet filenames identify market inputs. Shared packages, classes, and functions use market-neutral names, including `OptionDataset`, `HedgingEnv`, `StateNormalizer`, `segment_option_dataset`, and `run_market_segmentation`.

## Experimental protocol

The chronological split is training in 2016–2022, validation in 2023, and testing from January 2024 through August 2025. Complete 20-interval episodes remain within their split. State normalization, regime detection, and simulator calibration use training data only. Checkpoints are selected by validation risk-adjusted loss; test evaluation uses real episodes with frozen network parameters.

Each market has eight configurations: SAC or TD3, Direct or Delta-residual actions, and shaped Accounting or Cash Flow rewards. The generated configurations use ten seeds, 0–9. SPX retains three strikes with initial strike/spot in `[0.95, 1.05]`; SX5E retains twenty in `[0.8, 1.2]`. The initial bid–ask spread cap is 10% of the midpoint. Both markets use proportional transaction costs 0.001 and evaluation risk aversion 1.5. Contract multipliers are 100 for SPX and 10 for SX5E.

The policy observes normalized spot, normalized strike, previous holding, remaining maturity, IV, zero rate, and dividend yield. Executed holdings lie in `[0, 1]`; the Delta-residual correction bound is 0.1. Rewards include opening and terminal liquidation costs, dividends, and financing. Accounting and Cash Flow wealth are reconciled at settlement. Monetary outputs use each market's contract currency.

Hedging evaluation reports four metrics: mean terminal loss, population standard deviation of terminal loss, risk-adjusted loss `J = mean_loss + 1.5 * std_loss`, and mean total discounted transaction cost. Counts and configuration fields are recorded separately as execution metadata.

Stage 2 evaluates exactly four simulation features: terminal spot ratio, ATM IV, pre-expiry IV change, and IV skew. ATM uses the available strike nearest the forward price; skew is the least-squares IV slope against log strike-to-forward moneyness. Terminal spot ratios count each cohort once, and IV changes exclude the settlement observation. Wasserstein distances are divided by `max(real_std, abs(real_mean), 1e-12)` and averaged equally across regimes for each feature. The global comparison reuses the regime-specific simulation's anchor plan.

The default augmentation ratio is 1 and the recent context window is 4. Simulated contracts in a cohort share underlying and volatility paths. Only anchors whose complete lifecycles fit within the regime are eligible. Fractional augmentation ratios are rounded to the nearest complete cohort, with half-integers rounded upward and at least one generated cohort; outputs record the realized ratio. At deployment, context contains only episodes completed by the current inception date. Contracts in a new cohort share the inferred task representation. Training context and RL query batches have disjoint real/anchor source roots.

## Running the paper pipeline

Generate all six configuration files for a selected market. Replace the external directory with your own path:

```sh
python configure_experiment.py --data-source spx --data-root /absolute/path/to/spx --output-dir configs
```

For SX5E, use `--data-source sx5e --data-root /absolute/path/to/sx5e`; the generator applies the corresponding strike filters, calibration cross-section size, currency, multiplier, and separate output roots. Substitute `sx5e` for `spx` in the commands below. Use `--seeds 0` when generating configurations for a shorter execution check.

Run regime detection and simulation:

```sh
python run_seg.py --config configs/spx_seg.json --experiment-id paper_seg
python run_sim.py --config configs/spx_sim.json --seg-result segmentation/seg_results/spx/paper_seg.json --experiment-id paper_sim
```

Run the full MDH method:

```sh
python train_meta.py --config configs/spx_meta.json --simulation-result simulation/sim_results/spx/paper_sim --experiment-id paper
python test_meta.py --train-results-root train_results/meta/spx --experiment-id paper
```

Run conventional RL and BSM Delta baselines:

```sh
python train_basis.py --config configs/spx_basis.json --experiment-id paper
python test_basis.py --train-results-root train_results/basis/spx --experiment-id paper
python test_delta.py --config configs/spx_basis.json --output-root test_results/delta/spx --experiment-id paper
```

Run the ablation without simulated samples and the ablation without meta-RL:

```sh
python train_no_simulation.py --config configs/spx_no_simulation.json --segmentation-result segmentation/seg_results/spx/paper_seg.json --experiment-id paper
python test_no_simulation.py --train-results-root train_results/no_simulation/spx --experiment-id paper
python train_no_meta.py --config configs/spx_no_meta.json --simulation-result simulation/sim_results/spx/paper_sim --experiment-id paper
python test_no_meta.py --train-results-root train_results/no_meta/spx --experiment-id paper
```

Use a new experiment name when repeating a run. The stage loaders accept explicit result paths; automatic discovery considers only completed results with the standard timestamp naming convention. The named examples above therefore pass Stage 1 and Stage 2 paths explicitly. `run_seg.py` and `run_sim.py` accept repeated `--set SECTION.KEY=VALUE` overrides. Training scripts also accept algorithm/run filters; see each script's `--help`.

Omitting `--data-source` retains the market declared by the configuration or the completed training record.

Outputs are created at runtime under `segmentation/seg_results/`, `simulation/sim_results/`, `train_results/`, and `test_results/`. They include the run configuration, dataset fingerprints, validation-best checkpoint, pooled metrics, and episode/latent records. These directories, external data, credentials, caches, and generated figures are excluded by `.gitignore` and are not part of the anonymous source release.

## Mechanism and sensitivity analyses

```sh
python figure_making/plot_segmentation.py --result segmentation/seg_results/spx/paper_seg.json --output-dir figures
python figure_making/validate_latent_task_information.py --train-experiment train_results/meta/spx/paper --test-experiment test_results/meta/spx/paper --output figures/spx_latent_diagnostics.json
```

The latent probe uses posterior means and standard deviations, a class-balanced Extra Trees classifier, and five-fold evaluation grouped by episode. It reports accuracy, balanced accuracy, macro-F1, real/simulated subset accuracy, and the majority baseline. Test-time diagnostics report latest-task share, dominant task/share, overlap with the assigned task's 95th-percentile training radius, and the median probability margin. At least five distinct training episodes per task are needed for grouped five-fold evaluation.

For augmentation sensitivity, regenerate configurations with ratios 0.5, 1, 1.5, and 2, assigning unique simulation/training experiment names. The zero-augmentation point uses the no-simulation ablation. For context-window sensitivity, use windows 1, 4, 8, 16, and 32 with augmentation ratio 1. Evaluate the four Delta-residual reward/backbone configurations with matched seeds.

Sensitivity plotting reads measured per-seed results from a CSV with columns `parameter`, `value`, `market`, `reward`, `algorithm`, `seed`, and `j_lambda`. Supported parameter names are `num_simulation_times` and `num_recent_context_episodes`; rewards are `shaped_accounting` and `cash_flow`, and algorithms are `sac` and `td3`. No paper-result values are embedded in the plotting code.

```sh
python figure_making/plot_hyperparameter_sensitivity.py --input /path/to/sensitivity_measurements.csv --output-dir figures
```

## Verification and reproducibility

The data-independent suite checks pricing, episode filtering and schedules, normalization, financial accounting, SAC/TD3 updates and checkpoint reloads, regime detection, SABR calibration/generation, paired comparisons, context/query isolation, causal evaluation, and the paper experiment matrix:

```sh
python -B -m pytest -m "not integration" -q
```

Optional real-data integration tests use `data/` at the project root, or the external SPX directory specified by the `MDH_TEST_DATA_ROOT` environment variable. Run them with `python -B -m pytest -m integration -q`. The full pipeline can instead use an external directory through the generated configurations. Reproducing the reported experiments requires the original data coverage and preprocessing conventions, all ten seeds, and the full training schedules. Short or synthetic execution checks verify functionality and do not reproduce the paper's numerical results. Hardware, PyTorch versions, and nondeterministic GPU kernels can affect numerical outcomes; configurations record seeds and software versions.

This source tree contains one README and no author names, affiliations, personal contacts, private repository links, or original local workspace paths.

The reference verification environment used Python 3.10.19, NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3, PyArrow 25.0.0, PyTorch 2.10.0, Gymnasium 1.2.3, ruptures 1.1.10, scikit-learn 1.7.2, Matplotlib 3.10.8, and pytest 9.1.1. Functional training checks used the CPU.
