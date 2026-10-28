# T1: keadaan tiap unit (run lokal dengan image yang patuh aturan 8/9: know33 ke atas), 7 Okt 2026

Catatan: run sebelum 26 Sep (know13/know15) memakai replay solusi dan tidak dihitung. 65 unit belum pernah dijalankan dengan image yang patuh; sweep know43 atas 65 unit itu sedang berjalan (full43).

units: 86 | diagnosis counts: {'mostly wrong values': 6, 'never run locally': 65, 'half right: method/convention errors': 7, 'passed before, latest fails (flaky)': 1, 'PASSES': 5, 'nearly there: a few value/format checks': 2}

| unit | category | runs | best (passed/total) | latest | diagnosis | failing tests (best run) | last error |
|---|---|---|---|---|---|---|---|
| 13f-amendment-aware-crowding | cross-domain | 2 | 7/51 r=0.0 | 4/51 (n12_37) | mostly wrong values | test_reference_match_resolution_and_holdings_outputs, test_reference_match_manager_analytics_outputs, test_reference_match_crowding_and_audit_outputs, test_summary_matches_reference | stopped early with 30s left rather than risk being killed mid-write |
| alpha-hedge-strategy | cross-domain | 3 | 0/1 r=0.0 | 0/1 (n12_43) | mostly wrong values | test_verification | [examiner] ok: 5 finding(s) not confirmed by execution (flagged) |
| american-option-fd-new | derivatives-pricing | 0 | - | - | never run locally |  |  |
| asian-option-levy-curran | derivatives-pricing | 0 | - | - | never run locally |  |  |
| barone-adesi-whaley | cross-domain | 0 | - | - | never run locally |  |  |
| barrier-garch-var | cross-domain | 0 | - | - | never run locally |  |  |
| binance-btc-participation-tca | execution | 0 | - | - | never run locally |  |  |
| bl-regime-hmm | cross-domain | 0 | - | - | never run locally |  |  |
| bollinger-backtest-aapl | backtesting | 0 | - | - | never run locally |  |  |
| brinson-sector-attribution | performance-attribution | 0 | - | - | never run locally |  |  |
| bs-greeks-pde | derivatives-pricing | 0 | - | - | never run locally |  |  |
| cir-bond-pricing | fixed-income | 0 | - | - | never run locally |  |  |
| cliquet-ratchet-pricing | derivatives-pricing | 0 | - | - | never run locally |  |  |
| cme-hdd-option-pricing | derivatives | 9 | 45/52 r=0.0 | 21/52 (ab35) | half right: method/convention errors | test_mc_price, test_temp_delta_value, test_vega_value, test_kappa_sens_value | stopped early with 91s left rather than risk being killed mid-write |
| compound-option-geske | cross-domain | 26 | 34/34 r=1.0 | 31/34 (ab35) | passed before, latest fails (flaky) |  | [model][fix:crash] AssertionError: max_parity_error=10.590541262903145 --- traceback tail  |
| copula-equity-fitting | risk-management | 0 | - | - | never run locally |  |  |
| copula-sampling-rank-correlation | dependence-modeling | 0 | - | - | never run locally |  |  |
| corporate-action-adjustment | tool-using | 8 | 7/7 r=1.0 | 7/7 (s12_43) | PASSES |  | [examiner] ok: 1 finding(s) not confirmed by execution (flagged) |
| credit-migration-matrix | credit-risk | 9 | 65/107 r=0.0 | 22/107 (s12_43) | half right: method/convention errors | test_shape, test_columns, test_row_labels, test_row_stochastic | stopped early with 109s left rather than risk being killed mid-write |
| credit-portfolio-var-cvar | credit-risk | 9 | 107/130 r=0.0 | 71/130 (s12_43) | half right: method/convention errors | test_t_copula_heavier_tail_at_999, test_pct_contributions_sum_to_100, test_t_copula_heavier_tails, test_var_ratio_positive | stopped early with 49s left rather than risk being killed mid-write |
| credit-spread-decomposition | credit-analysis | 0 | - | - | never run locally |  |  |
| creditmetrics-portfolio-var | risk-management | 9 | 22/26 r=0.0 | 1/26 (s12_43) | half right: method/convention errors | test_aaa_mean_loss, test_upgrade_rate, test_downgrade_rate, test_rates_sum_less_than_one | stopped early with 34s left rather than risk being killed mid-write |
| cross-sectional-momentum | cross-sectional-strategies | 0 | - | - | never run locally |  |  |
| crypto-funding-rate-basis-carry | crypto | 9 | 50/63 r=0.0 | 43/63 (s12_43) | half right: method/convention errors | test_symbol_and_obs, test_ou_parameter[half_life_periods-2.251-0.3], test_ou_parameter[half_life_days-0.75-0.1], test_ou_num_observations_fit | [model][fix:selfcheck] AttributeError: module 'scipy.stats' has no attribute 'adfuller' -- |
| cta-basel-capital | cross-domain | 0 | - | - | never run locally |  |  |
| dcc-garch-portfolio-var | risk-modeling | 0 | - | - | never run locally |  |  |
| delta-hedging-pnl-simulation | derivatives-pricing | 0 | - | - | never run locally |  |  |
| digital-barrier-options | derivatives-pricing | 0 | - | - | never run locally |  |  |
| double-sort | factor-research | 0 | - | - | never run locally |  |  |
| dupire-local-vol | cross-domain | 0 | - | - | never run locally |  |  |
| earnings-surprise-calculator | tool-using | 0 | - | - | never run locally |  |  |
| etf-cross-asset-lead-lag | cross-asset-analysis | 0 | - | - | never run locally |  |  |
| etf-overlap-redemption-pressure | portfolio-analysis | 0 | - | - | never run locally |  |  |
| event-study-earnings | factor-models | 28 | 32/34 r=0.0 | 31/34 (ab35) | nearly there: a few value/format checks | test_st_corrado_z_full_pinned, test_st_corrado_z_day0_pinned | [model][fix:crash] ZeroDivisionError: float division by zero --- traceback tail --- /agent |
| evt-pot-var | risk-management | 0 | - | - | never run locally |  |  |
| ewma-portfolio-risk-decomposition | risk-management | 0 | - | - | never run locally |  |  |
| fama-french-factor-model-new | factor-models | 0 | - | - | never run locally |  |  |
| fft-compound-poisson | risk-management | 9 | 63/63 r=1.0 | 63/63 (ab35) | PASSES |  | [examiner] ok: 1 finding(s) not confirmed by execution (flagged) |
| first-passage-time | cross-domain | 34 | 31/31 r=1.0 | 31/31 (s12_43) | PASSES |  | [examiner] ok: 1 finding(s) not confirmed by execution (flagged) |
| fomc-tone-event-study | fixed-income-nlp | 0 | - | - | never run locally |  |  |
| form4-cross-sectional-sale-pressure | cross-domain | 0 | - | - | never run locally |  |  |
| fx-carry-forward-hedge | fx-strategy | 2 | 1/45 r=0.0 | 1/45 (n12_37) | mostly wrong values | test_required_schema, test_mid_values[EURJPY-157.6565], test_mid_values[CHFJPY-165.1034], test_mid_values[GBPCHF-1.1138] | stopped early with -102s left rather than risk being killed mid-write |
| fx-forward-cross-rate | fx-pricing | 0 | - | - | never run locally |  |  |
| geometric-mean-reverting-jd | fixed-income | 0 | - | - | never run locally |  |  |
| historical-var-data-prep | risk-management | 0 | - | - | never run locally |  |  |
| hull-white-swaption | interest-rate-derivatives | 0 | - | - | never run locally |  |  |
| implied-vol-approximations | derivatives-pricing | 0 | - | - | never run locally |  |  |
| interest-rate-cap-floor | pricing | 0 | - | - | never run locally |  |  |
| intraday-volume-fitting-and-execution-scheduling | execution | 9 | 9/13 r=0.0 | 9/13 (s12_43) | half right: method/convention errors | test_final_schedule_scheduled_quantity_sums_to_order_quantit, test_final_schedule_matches_reference_datetimes_and_quantiti, test_final_schedule_predicted_bar_share_matches_reference, test_final_schedule_cumulative_matches_reference_within_qty_ | [examiner] ok: 1 finding(s) not confirmed by execution (flagged) |
| ipca-latent-factors | cross-domain | 0 | - | - | never run locally |  |  |
| kelly-var-sizing | cross-domain | 0 | - | - | never run locally |  |  |
| lob-pc-signal | predictive-alpha-modeling | 0 | - | - | never run locally |  |  |
| localvol-barrier | derivatives-pricing | 9 | 24/39 r=0.0 | 9/39 (s12_43) | half right: method/convention errors | test_expiry_row_distribution, test_sample_nodes_pinned[25SEP26-66000-75886.17074388417-145, test_sample_call_surface_values[25SEP26-70000-12089.86520219, test_local_vol_shape | [model][fix:crash] TypeError: Addition/subtraction of integers and integer-arrays with Dat |
| lookback-options | derivatives_pricing | 8 | 25/25 r=1.0 | 25/25 (s12_43) | PASSES |  | [examiner] ok: 3 finding(s) not confirmed by execution (flagged) |
| mc-greek-surface-1 | derivatives-pricing | 0 | - | - | never run locally |  |  |
| merton-jump-diffusion | derivatives-pricing | 0 | - | - | never run locally |  |  |
| momentum-backtest | backtesting | 0 | - | - | never run locally |  |  |
| mtm-xccy-basis-desk | cross-currency-rates | 9 | 58/145 r=0.0 | 28/145 (s12_43) | mostly wrong values | test_results_deliverable_matches_reference[trade_pv_usd], test_results_deliverable_matches_reference[usd_leg_pv_usd], test_results_deliverable_matches_reference[gbp_leg_pv_usd], test_results_deliverable_matches_reference[par_basis_bps] | [budget] stopped: 3 House requests left |
| multimodal-alpha-fusion-edgar-cot-gdelt | cross-domain | 0 | - | - | never run locally |  |  |
| ohlc-realized-vol-estimators | volatility-modeling | 0 | - | - | never run locally |  |  |
| option-put-call-parity-forward-audit | derivatives | 0 | - | - | never run locally |  |  |
| ou-jump-commodity | stochastic-processes | 0 | - | - | never run locally |  |  |
| pca-factor-portfolio | strategy | 8 | 8/8 r=1.0 | 8/8 (s12_43) | PASSES |  | [examiner] ok: 3 finding(s) not confirmed by execution (flagged) |
| polars-api-migration | ? | 0 | - | - | never run locally |  |  |
| prediction-markets-cross-venue-dislocation | cross-domain | 0 | - | - | never run locally |  |  |
| realized-vol-estimators | cross-domain | 0 | - | - | never run locally |  |  |
| regime-cta-vol-target | cross-domain | 0 | - | - | never run locally |  |  |
| regime-riskparity-cvar | cross-domain | 0 | - | - | never run locally |  |  |
| residual-momentum | factor-research | 0 | - | - | never run locally |  |  |
| sec-10k-report-long | cross-domain | 0 | - | - | never run locally |  |  |
| sec-8k-event-alpha | event-driven-analysis, data-processing | 0 | - | - | never run locally |  |  |
| sentiment-factor-alpha | cross-domain | 0 | - | - | never run locally |  |  |
| sma-crossover-spy | backtesting | 0 | - | - | never run locally |  |  |
| smith-tail-index | extreme-value-theory | 0 | - | - | never run locally |  |  |
| spread-option-kirk-margrabe | derivatives-pricing | 0 | - | - | never run locally |  |  |
| stable-residual | factor-research | 2 | 1/19 r=0.0 | 1/19 (n12_37) | mostly wrong values | test_summary_schema_and_types, test_summary_values_match_reference, test_monthly_csv_schema_sorting_and_types, test_monthly_csv_matches_reference | stopped early with 112s left rather than risk being killed mid-write |
| standard-var-methods | risk-management | 9 | 41/42 r=0.0 | 40/42 (ab35) | nearly there: a few value/format checks | test_ewma_current_vol | [examiner] ok: 1 finding(s) not confirmed by execution (flagged) |
| stochvol-implied-surface-new | derivatives-pricing | 0 | - | - | never run locally |  |  |
| structured-note-risk | cross-domain | 9 | 0/1 r=0.0 | 0/1 (s12_43) | mostly wrong values | test_verification | stopped early with 295s left rather than risk being killed mid-write |
| swap-curve-bootstrap-ois | fixed-income | 0 | - | - | never run locally |  |  |
| var-es-estimation | risk-management | 0 | - | - | never run locally |  |  |
| variance-swap-replication | derivatives-pricing | 0 | - | - | never run locally |  |  |
| yield-curve-bond-immunization | fixed-income | 0 | - | - | never run locally |  |  |
| yield-curve-bootstrap-immunization | fixed-income | 0 | - | - | never run locally |  |  |
| yield-curve-pca-dynamics | statistical-analysis | 0 | - | - | never run locally |  |  |
| zero-coupon-bootstrapping | pricing | 0 | - | - | never run locally |  |  |
