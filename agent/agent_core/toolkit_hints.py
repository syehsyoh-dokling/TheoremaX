"""Task-specific toolkit recommendations.

29 Sep know15 run: var-es-estimation crashed 9 times re-deriving a kernel-density VaR/ES by hand
(scipy quad over a KDE) while qf_toolkit.kernel_var_es sat in the 12k-character API block under a
"MANDATORY" rule. A long generic menu is not read; a short list of the calls that cover THIS task is.
Keywords of the instruction map to toolkit names, and the API lines for those names are shown at the
top of the plan and code prompts. General vocabulary only - no unit ids, no values."""
from __future__ import annotations

import re

#: (regex on the lower-cased instruction, toolkit names)
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # know78 group 1 (qf_toolkit/helpers78_1.py whole-task bundles; self-tests proven_work/fail72/selftest78_1.py).
    # Placed FIRST on purpose: names_for orders whole-task helpers by rule order, and later generic rules
    # (re-hedge, put-call parity, ois...libor) put other bundles ahead of these for the same instruction.
    (r"mark-to-market cross-currency|cross-currency (basis )?swap|mtm reset|reset[- ]notional",
     ("mtm_xccy_swap_bundle", "stress_scenarios_from_labels")),
    (r"\bfx carry|g10 (fx )?carry|forward[- ]hedge overlay|hedge book",
     ("fx_carry_hedge_bundle", "fx_carry_hedge_backtest", "fx_smile_option_book", "return_risk_metrics")),
    (r"cerep|transition-matrix-\d|default-rates-\d", ("cerep_migration_report",)),
    (r"kernel density|\bkde\b", ("kernel_var_es",)),
    (r"historical simulation|historical var|\bhs var|empirical quantile", ("historical_var_es",)),
    (r"age[- ]weighted|boudoukh|\bbrw\b", ("age_weighted_var_es",)),
    (r"filtered historical|\bfhs\b", ("filtered_hs_var_es",)),
    (r"\bewma\b|riskmetrics|exponentially weighted", ("ewma_var_es", "ewma_vol", "ewma_sigma_series", "ewma_variance")),
    (r"(ewma|exponentially weighted).{0,60}(covariance|variance-covariance|portfolio)|covariance.{0,40}ewma",
     ("ewma_covariance",)),
    (r"backtest|trades\.csv|trade log|daily_portfolio|final capital|whole shares|next (day'?s )?open",
     ("trade_ledger", "sma_crossover_signals", "bollinger_signals", "momentum_signal")),
    (r"corporate action|split ratio|ex-dividend date", ("corporate_action_adjust",)),
    (r"split|dividend adjust|adjustment factor|corporate action", ("cumulative_adjustment_factor",)),
    (r"(student[- ]?t|t-distribution).{0,80}(var|value at risk)|parametric var|variance[- ]covariance|delta[- ]normal|"
     r"cornish", ("parametric_var_es", "cornish_fisher_var")),
    (r"expected shortfall|\bcvar\b|\bvar\b", ("parametric_var_es", "historical_var_es", "scale_horizon")),
    (r"kupiec|proportion of failures", ("kupiec_pof", "exceedances")),
    (r"christoffersen|independence test|conditional coverage", ("christoffersen_independence",
                                                                "christoffersen_cc")),
    (r"traffic[- ]light|basel.*backtest", ("basel_traffic_light",)),
    (r"component var|marginal var|risk contribution|euler", ("component_var",)),
    (r"\bkelly\b", ("kelly_fraction", "fractional_kelly", "var_constrained_fraction")),
    (r"barrier", ("barrier_price", "barrier_hit_prob", "bgk_shift")),
    (r"first[- ]passage|hitting (time|probability)|running (max|min)", (
        "barrier_hit_prob", "first_passage_cdf", "first_passage_pdf", "running_max_cdf",
        "running_min_cdf", "joint_end_max_cdf", "expected_first_passage_time", "mc_hit_probability",
        "first_hit_times", "running_max_path", "running_min_path", "gbm_paths")),
    (r"lookback", ("lookback_floating_price", "lookback_fixed_price")),
    (r"digital|binary option|cash[- ]or[- ]nothing|asset[- ]or[- ]nothing", ("digital_price",)),
    (r"one[- ]touch|no[- ]touch", ("one_touch_price", "no_touch_price")),
    (r"asian", ("asian_geometric_price", "asian_arithmetic_levy", "asian_curran")),
    (r"margrabe|exchange option", ("margrabe_price",)),
    (r"\bkirk\b|spread option", ("kirk_spread_price", "kirk_spread_price_spot")),
    # know71 workflow recipes (whole steps in one tested call)
    (r"calibrat|historical vol|annuali[sz]ed (vol|volatility)|daily log[- ]returns|from (the )?(daily )?(closing )?prices",
     ("calibrate_gbm",)),
    (r"double[- ]sort|two[- ]way sort|dependent sort|independent sort|(quintile|tercile|decile).{0,80}(within|conditional)",
     ("double_sort_portfolios",)),
    (r"(ou|ornstein[- ]uhlenbeck|mean[- ]revert\w*).{0,80}jump|jump.{0,80}(ou\b|ornstein|mean[- ]revert)",
     ("fit_ou_jump", "simulate_ou_jump", "ou_from_ar1")),
    (r"black[- ]scholes|\bbsm\b|greeks", ("bs_price", "bs_greeks")),
    (r"implied vol", ("implied_vol",)),
    (r"american option|early exercise|binomial|\bcrr\b|barone|whaley", ("crr_price", "baw_american")),
    (r"garman[- ]kohlhagen", ("garman_kohlhagen_price",)),
    (r"black[- ]?76|futures option", ("black76_price",)),
    (r"monte carlo|simulat", ("gbm_paths", "running_extremes")),
    (r"covered interest|\bcip\b|forward points|\bpips?\b|outright", (
        "cip_forward", "forward_points", "outright_from_points", "pip_multiplier",
        "cip_forward_bid_ask", "cip_deviation_bps", "money_market_year_fraction")),
    (r"cross rate|triangular", ("cross_rate", "invert_quote", "triangular_arbitrage")),
    (r"mark[- ]to[- ]market|\bmtm\b|net (delta|exposure)|p&l attribution|pnl attribution", (
        "forward_mtm", "forward_pnl_attribution", "net_currency_exposure", "to_usd", "direction_sign")),
    # know87 round 2: Merton (1976) jump-diffusion (helpers87_A) - ahead of the structural-credit Merton line
    (r"poisson[- ]weighted mixture|merton series",
     ("merton_jump_diffusion_report", "merton_jd_calibrate", "merton_jd_loglik", "merton_series_call")),
    (r"merton|structural model|distance to default", ("merton_model", "merton_calibrate")),
    (r"black[- ]cox", ("black_cox_pd",)),
    (r"\bcds\b|hazard rate|credit triangle|survival probabilit", (
        "hazard_from_spread", "cds_bootstrap", "cds_par_spread", "survival_prob")),
    (r"\birb\b|basel.*capital|risk[- ]weight", ("basel_irb_capital",)),
    (r"vasicek.*(loss|portfolio|asrf)|asrf|large pool|one[- ]factor", (
        "vasicek_loss_quantile", "vasicek_loss_distribution", "one_factor_mc_losses")),
    (r"migration|transition matrix|generator matrix|cohort", (
        "matrix_power_migration", "generator_from_matrix", "cohort_matrix", "cumulative_default_probs")),
    (r"vasicek|\bcir\b|cox[- ]ingersoll|hull[- ]white", (
        "vasicek_zcb", "cir_zcb", "hull_white_zcb", "vasicek_zcb_option", "hull_white_zcb_option",
        "vasicek_swaption")),
    (r"caplet|floorlet|interest[- ]rate caps?|caps? and floors?|cap/floor", ("black_caplet",
                                                                               "bachelier_caplet")),
    (r"swaption", ("black_swaption", "bachelier_swaption", "swap_rate_and_annuity")),
    (r"copula", ("copula_param_from_tau", "copula_sample", "fit_copula", "select_copula",
                 "pseudo_observations", "tail_dependence", "copula_logpdf")),
    (r"kendall", ("empirical_kendall_tau", "copula_tau")),
    (r"tail dependence", ("tail_dependence", "empirical_tail_dependence")),
    (r"compound poisson|aggregate loss|panjer|\bfft\b", (
        "severity_dist", "suggest_lattice", "discretize_severity", "compound_poisson_fft", "panjer_poisson", "aggregate_var_es",
        "compound_poisson_moments")),
    (r"bollinger", ("bollinger_bands", "bollinger_signals", "backtest_positions", "performance_summary")),
    (r"moving average|\bsma\b|crossover", ("sma_crossover_signals", "backtest_positions",
                                             "performance_summary")),
    (r"backtest|equity curve|trading strategy", ("signals_to_positions", "backtest_positions",
                                                 "trade_list", "performance_summary")),
    (r"momentum", ("momentum_signal",)),
    (r"sharpe|drawdown|calmar|sortino", ("performance_summary", "sharpe", "max_drawdown")),
    (r"brinson|allocation effect|selection effect|attribution", (
        "brinson_attribution", "link_attribution", "returns_based_attribution")),
    (r"delta[- ]hedg|hedging p&?l|hedging pnl", ("delta_hedge_pnl", "bs_greeks")),
    (r"compound option|call[- ]on[- ]call|put[- ]on[- ]put|put[- ]on[- ]call|call[- ]on[- ]put|geske", (
        "compound_option_price", "geske_critical_price", "compound_parity_rhs", "compound_option_mc")),
    (r"chooser", ("chooser_simple_price", "chooser_complex_price")),
    (r"event study|event[- ]study|abnormal return|\bcar\b|market model|patell|\bbmp\b|boehmer|corrado|"
     r"kolari|pynn", ("market_model", "abnormal_returns", "car", "patell_standardized", "patell_test",
                      "bmp_test", "kolari_pynnonen_adjust", "average_residual_correlation",
                      "corrado_rank_test", "sign_test", "event_study_panel")),
    (r"\bgarch\b", ("fit_garch11",)),
    (r"\bdcc\b|dynamic conditional correlation", ("dcc_correlation", "fit_garch11")),
    (r"duration|convexity|immuniz", ("duration_convexity", "bond_price", "bond_ytm")),
    (r"bootstrap|zero curve|par curve|discount factor", ("par_curve", "bootstrap_par",
                                                         "discount_factor", "zero_rate")),
    (r"ornstein|mean[- ]reversion|half[- ]life|\bar\(1\)", ("ar1_fit", "ou_from_ar1")),
    (r"jump[- ]diffusion", ("simulate_jump_diffusion",)),
    (r"newey|hac", ("newey_west_tstat", "ols")),
    (r"regression|\bols\b|factor model|fama", ("ols",)),
    (r"day count|act/360|act/365|30/360", ("year_fraction",)),
    # know75 group 4 (qf_toolkit/helpers75_4.py; self-tests proven_work/fail72/selftest75_4.py)
    (r"participation[- ]cap|participation rate|implementation_shortfall|\btca\b|child order",
     ("participation_bucket_fills", "epoch_ms_int", "write_fixed_csv", "first_quote_at_or_after")),
    (r"epoch milliseconds|epoch ms|\w_ms\b", ("epoch_ms_int",)),
    (r"volume[- ]share|volume profile|intraday volume|largest (fractional )?remainder",
     ("volume_share_profile", "walk_forward_profile_scores", "share_r2", "largest_remainder_allocation")),
    (r"\bofi\b|order flow imbalance|principal component|\bpca\b", ("rolling_pc1_ols_forecast", "nan_safe_ic")),
    (r"spearman|pearson|information coefficient", ("nan_safe_ic",)),
    (r"spot lag|spot date|broken[- ]date|swap roll|roll case|holiday calendar",
     ("fx_spot_date", "fx_add_tenor", "fx_swap_roll")),
    (r"implied (foreign )?yield|implied foreign|implied local rate", ("implied_foreign_yield",)),
    (r"25[- ]?d(elta)?\b|risk reversal|butterfly|premium[- ]adjusted|delta convention",
     ("fx_smile_25d", "fx_strike_from_delta")),
    (r"cross rate|triangulat|synthetic cross", ("cross_via_usd",)),
    (r"cross[- ]currency|xccy|quote_unit|quote book|normali[sz]\w* quote", ("normalize_quote_book", "loglinear_interp")),
    (r"linear in -?ln|log[- ]linear interp", ("loglinear_interp",)),
    (r"golden cross|death cross|cross(es)? (above|below)", ("crossover_backtest",)),
    (r"escrowed|discrete (cash )?dividend", ("escrowed_dividend_delta_hedge",)),
    (r"pathwise|likelihood ratio|score function|\bipa\b", ("mc_greeks_suite", "bs_delta_vega_surface")),
    (r"geske|compound option|call[- ]on[- ]call|put[- ]on[- ]put", ("geske_compound_grid", "mc_agreement")),
    (r"interpolat\w* (the )?zero rate|zero rates? linearly|par coupon rate",
     ("bootstrap_par_linear_zero", "forward_from_zero")),
    (r"transition matri|migration matri|generator matri|chi[- ]squared|homogeneity|semicolon",
     ("read_delimited_matrix", "collapse_rating_counts", "row_stochastic", "chi2_homogeneity",
      "generator_logm_clip")),
    (r"\bpolars\b", ("polars_0x_to_1x_source", "check_python_source")),
    # know75 group 3 (qf_toolkit/helpers75_3.py; self-tests proven_work/fail72/selftest75_3.py)
    (r"hidden markov|\bhmm\b|baum[- ]welch", ("gaussian_hmm_fit", "gaussian_hmm_posteriors")),
    (r"black[- ]litterman", ("black_litterman", "bl_absolute_views")),
    (r"brinson", ("brinson_fachler_drift", "month_end_returns")),
    (r"\bkelly\b", ("mvn_return_cube", "portfolio_path_returns", "wealth_path_metrics", "kelly_portfolio_var")),
    (r"sma[- ]seeded|\bema\b|exponential moving average|vol(atility)?[ _-]target|\bcta\b",
     ("sma_seeded_ema", "ewma_variance_alpha", "vol_target_weights", "lagged_portfolio_returns",
      "log_return_performance")),
    (r"\bgarch\b", ("garch11_conditional_vol",)),
    (r"percentile.{0,60}regime|regime.{0,120}percentile", ("percentile_regimes",)),
    (r"absorption ratio|marchenko|eigenvalue.{0,80}regime", (
        "rolling_absorption_ratio", "classify_upper_thresholds", "month_end_flags", "inverse_vol_weights",
        "hold_weights_returns", "tail_var_cvar")),
    (r"risk[- ]parity|inverse[- ]vol", ("inverse_vol_weights", "month_end_flags", "hold_weights_returns")),
    (r"implied vol|newton[- ]raphson", ("implied_vol_vec", "newton_iv_vec")),
    (r"\bsvi\b|dupire|local[- ]vol", ("svi_fit", "svi_raw", "svi_derivatives", "gatheral_local_var",
                                       "dupire_local_vol_grid", "local_vol_interpolator", "localvol_mc_barrier")),
    (r"deribit|premiums? (are )?(also )?(quoted )?in btc|btc[- ]quoted|inverse option",
     ("forward_from_coin_parity", "parse_option_instrument")),
    (r"heston|chiarella", ("heston2_price", "heston2_char_func")),
    (r"brenner|corrado|hallerbach|\bli (atm|non)", ("brenner_subrahmanyam_iv", "li_atm_iv", "li_nonatm_iv",
                                                     "cmh_iv", "newton_iv_vec")),
    (r"time_to_expiry|time to (expiry|maturity)|act/365.{0,120}(timestamp|utc)|millisecond|unix ms",
     ("ttm_years_exact",)),
    (r"crank[- ]nicolson|\bpsor\b|finite[- ]difference", ("cn_fd_option",)),
    (r"conditional moments|jump.{0,120}(ornstein|mean[- ]revert)", ("ou_jump_moments",)),
    (r"plug-in euler|funding rate|exact ou transition|exact[- ]discreti",
     ("ou_fit_euler_plugin", "ou_exact_paths", "loss_var_cvar_sorted")),
    (r"\bfred\b|forward-fill|discontinued|last (valid|available) observation in each calendar month",
     ("monthly_last", "align_monthly")),
    (r"newey|\bhac\b|rolling (multivariate )?ols|rolling.{0,30}regression", ("rolling_ols_hac",)),
    (r"index `?ceil|ceil\(", ("loss_var_cvar_sorted",)),
    # know75 group 2 (qf_toolkit/helpers75_2.py; self-tests proven_work/fail72/selftest75_2.py)
    (r"cross-sectional alpha signal|long top-?n|short bottom-?n|start of each calendar month|per-stock weight",
     ("cs_zscore", "month_start_indices", "topn_long_short", "periodic_ls_backtest", "strategy_metrics",
      "factor_regression_stats")),
    (r"newey[- ]west|gibbons|\bgrs\b|variance inflation|\bvif\b|rolling (window )?betas?|durbin",
     ("factor_regression_stats", "newey_west_lag_rule", "grs_test", "vif", "rolling_ols_coefs")),
    (r"lead[- ]lag|lagged correlation|fisher z", ("lead_lag_table", "lagged_corr", "fisher_gate",
                                                    "factor_regression_stats")),
    (r"\bipca\b|instrumented principal", ("panel_to_array", "ipca_als", "grs_test", "factor_regression_stats")),
    (r"committee|model[- ]selection layer|martingale.{0,120}\bgbm\b|cross[- ]sectional median",
     ("cs_normalize", "fit_return_processes")),
    (r"retweets|engagement multiplier|ex-ante|vol-target scaling|rank ic\b|spearman rank ic",
     ("daily_sum_count", "ex_ante_vol_scaler", "rank_ic_series", "factor_regression_stats")),
    (r"winsori[sz]|beta-neutral|turnover cap|sign agreement|moore-penrose",
     ("winsorize_columns", "residualize_zscore", "cs_ols_slopes", "sign_stable_median", "rank_long_short",
      "beta_neutral_project", "turnover_capped_weights")),
    (r"accrued interest|30/360 us|dated cash[- ]flows?|not-a-knot|\bfrn\b|callable bond",
     ("bootstrap_par_g_spline", "dated_bond_cashflows", "coupon_schedule", "thirty360_us", "par_bump_krd")),
    (r"key rate durations?|\bkrds?\b|immuniz|svensson|z-spread",
     ("spot_bump_krd", "immunize_pv_krd", "fit_svensson", "svensson_rate", "par_bump_krd")),
    (r"copula.{0,300}\bcvar\b|euler (allocation|decomposition)|granularity|gordy|wrong[- ]way|obligor",
     ("copula_default_mc", "loss_var_es", "euler_es_contributions", "vasicek_portfolio_var",
      "gordy_granularity_adjustment", "hhi")),
    (r"creditmetrics|rating transition|transition thresholds|rating migrations?",
     ("migration_thresholds", "simulate_migrations", "loss_var_es")),
    (r"empirical tail dependence|tail dependence coefficients|lambda_upper|p\(u2 [<>]",
     ("empirical_tail_table", "empirical_rank_correlations")),
    (r"\bpot\b|peaks? over threshold|genpareto|\bgpd\b|garch[- ]evt|hill (tail|estimator)|smith (estimator|tail|gpd|1987)",
     ("pot_fit", "pot_var_es", "gpd_mle_se", "hill_xi", "hist_var_es", "normal_var_es", "garch11_zero_mean",
      "garch_evt_var_es", "rolling_tail_backtest", "level_key")),
    (r"hull[- ]white|jamshidian|trinomial|bermudan", ("hw_caplet_price", "hw_jamshidian_swaption",
                                                       "hw_trinomial_tree", "hw_tree_bermudan")),
    (r"\bhdd\b|heating degree|ghcn|degree[- ]day", ("ghcn_month_temperatures", "fahrenheit_to_celsius",
                                                     "fit_ou_within_year", "ou_hdd_mc")),
    # know75 group 1 (qf_toolkit/helpers75_1.py; self-tests proven_work/fail72/selftest75_1.py)
    (r"\b13f\b|infotable|coverpage|summarypage", ("sec13f_load", "sec13f_filings", "sec13f_resolve",
                                                  "sec13f_clean_rows", "holdings_book", "book_concentration",
                                                  "book_turnover", "book_overlap")),
    (r"\.xlsx|workbook", ("read_banner_table", "parse_money", "parse_as_of_date", "top_n_by_weight")),
    (r"form 4|form4|insider|10b5-1", ("parse_form4", "form4_entity_name", "form4_price_range")),
    (r"xbrl|10-k\b", ("ixbrl_main_html", "ixbrl_facts", "ixbrl_fact", "ixbrl_bool", "ixbrl_by_context",
                      "html_tables", "find_table_rows", "table_row_numbers")),
    (r"\b8-k\b|date of report", ("filing_plain_text", "date_of_report", "item_numbers", "guidance_midpoint_musd",
                                 "debt_initial_principal_musd", "departing_officer", "money_musd")),
    (r"prediction[- ]market|kalshi|polymarket|locked[- ]edge|cross[- ]venue", (
        "clean_venue_snapshots", "synchronized_snapshots", "locked_edge_backtest")),
    (r"market[- ]model|estimation window", ("event_study_tables",)),
    (r"lexicon|tf-?idf|novelty", ("normalize_tokens", "lexicon_counts", "rolling_prior_mean", "tfidf_novelty")),
    (r"durbin|leave-one-out|cook'?s|sign-flip|wild sign", (
        "ols_hac_fit", "ols_influence", "leave_one_out_coef", "wild_sign_flip_pvalue")),
    (r"first valid .{0,60}on or after", ("next_valid_index", "ar1_fit_window")),
    (r"\bgarch\b", ("garch11_forecast",)),
    (r"central finite differences?|finite[- ]difference greeks", ("fd_greeks",)),
    (r"(delta|gamma|vega).{0,80}(var|variance) decomposition|var decomposition|vol(atility)?[- ]of[- ]vol",
     ("garch_dgv_var_report", "delta_gamma_vega_var")),
    (r"structured note|zero-coupon bond.{0,40}(option|participation)|capital[- ]protected", ("capital_protected_note",)),
    (r"stressed var|capital charge|internal models approach", ("basel_ima_garch_capital", "garch_rolling_max_var")),
    (r"\bdcc\b|dynamic conditional correlation|stationary bootstrap", (
        "dcc_garch_portfolio", "dcc_fit", "stationary_bootstrap_indices", "fhs_portfolio_var_es")),
    (r"student[- ]?t", ("student_t_mle_var_es",)),
    (r"compound poisson|aggregate loss", ("compound_poisson_risk_table",)),
    (r"\bois\b|dual[- ]curve|projection curve|3m libor", (
        "dual_curve_bootstrap", "dual_curve_swap_value", "add_business_days", "backward_schedule")),
    # know76 group 3 (qf_toolkit/helpers76_3.py whole-task reports; self-tests proven_work/fail72/selftest76_3.py)
    (r"geske|call-on-call|compound option", ("geske_compound_deliverables",)),
    (r"cash-or-nothing|asset-or-nothing|gap option|barrier digital|binary pricing",
     ("digital_barrier_suite", "cash_digital_barrier")),
    (r"structured note|zero-coupon bond.{0,40}(option|participation)", ("structured_note_report",)),
    (r"first passage|running (maximum|minimum|max|min)|reflection principle", ("gbm_first_passage_report",)),
    (r"jump detection|detected jumps|poisson[- ]driven jumps|ou process with (poisson )?jumps",
     ("ou_jump_report", "ou_jump_threshold_fit")),
    (r"funding rate|basis carry|perpetual futures", ("funding_carry_report",)),
    (r"\bkelly\b", ("kelly_var_sizing_report",)),
    (r"composite vol|regime.{0,80}(cta|vol(atility)?[ _-]target)", ("cta_regime_report",)),
    (r"absorption ratio", ("absorption_riskparity_report",)),
    (r"participation[- ]capped|participation tca|child quantity|bucket-by-bucket", ("tca_participation_report",)),
    (r"volume[- ]share|intraday volume|execution schedul", ("intraday_volume_schedule",)),
    (r"crank[- ]nicolson|\bpsor\b", ("american_fd_report",)),
    (r"pathwise|likelihood ratio|score function", ("mc_greeks_report",)),
    (r"spread decomposition|tedrate|t10yie|baa.{0,40}treasury", ("credit_spread_report",)),
    # know76 group 1 (qf_toolkit/helpers76_1.py whole-step calls; self-tests proven_work/fail72/selftest76_1.py)
    (r"t-12 through t-2|formation window|skips the most recent month|cross-sectional momentum",
     ("skip_month_momentum_ls", "compounded_performance")),
    (r"jensen|treynor|gibbons|variance inflation", ("factor_model_table", "grs_summary", "vif_named")),
    (r"fama.{0,3}macbeth|expanding-window oos|q5.q1|quintile sorts", ("fama_macbeth_oos", "nw_mean_tstat",
                                                                     "quantile_buckets")),
    (r"largest absolute daily change|nelson-siegel|butterfly 2s5s10s|2s10s",
     ("largest_abs_change", "covariance_pca", "nelson_siegel_fixed_lambda")),
    (r"expanding window with a minimum|ewma.{0,80}individual asset|current conditional covariance",
     ("expanding_hs_var_backtest", "ewma_portfolio_vol")),
    (r"gaussian_kde|kernel density estimation|contaminated normal", ("loss_var_es_table", "analytic_loss_var_es",
                                                                     "var_es_method_comparison", "loss_sample_var_es")),
    (r"\bois\b.{0,200}(libor|ibor)|dual-curve|swap_to_value", ("dual_curve_deliverables",)),
    (r"turnover cap|sign agreement|beta-neutral projection", ("stable_residual_backtest",)),
    (r"lead[- ]lag|lagged correlation", ("lead_lag_report", "fisher_gate_vec")),
    (r"\bipca\b|instrumented principal", ("ipca_report", "ipca_als_normalized")),
    (r"committee|vendor bulletin|reference vintage", ("trailing_event_features", "vintage_mismatch_features",
                                                     "committee_agent_score", "ranked_long_short_selection",
                                                     "weights_pnl_with_turnover")),
    (r"retweets|sentiment word|bullish", ("lexicon_post_scores", "single_factor_alpha_beta")),
    (r"ema[- ]crossover|trend-following", ("cta_ema_vol_target_returns",)),
    (r"cross-sectional alpha signal|per-stock weight", ("alpha_signal_backtest",)),
    (r"\bdcc\b|dynamic conditional correlation", ("dcc_var_deliverables",)),
    # know76 group 4 (helpers76_4 + in-place fixes in helpers75_*)
    (r"spot_bidoffer|deposit_bidoffer|pair_conventions|holiday_calendar|broken[- ]date|fx swap roll|implied foreign yield|\bndf\b",
     ("fx_snapshot_desk", "fx_smile_25d", "fx_strike_from_delta", "garman_kohlhagen_price")),
    (r"forward points|synthetic cross|triangulat|pnl attribution|net delta exposure", ("fx_forward_book",)),
    (r"quote_adjustment|libor ?_?3m|mtm reset|cross[- ]currency basis|basis swap", (
        "select_curve_quotes", "fx_forward_nodes", "normalize_quote_book")),
    (r"\bsvi\b|dupire", ("svi_dupire_surface", "select_liquid_expiries", "svi_fit", "newton_iv_vec")),
    (r"deribit|btc[- ]quoted|premiums? (are )?(also )?quoted in btc|coin[- ]margined|inverse option", (
        "coin_option_surface", "forward_from_coin_parity")),
    (r"two[- ]factor (heston|stochastic vol)|chiarella|christoffersen|double heston", ("heston2_surface_tables", "heston2_price")),
    (r"bollinger", ("bollinger_risk_stop_backtest",)),
    (r"golden cross|death cross|sma\(\d+\)|moving average crossover|sma crossover", ("crossover_backtest",)),
    (r"copula famil|best copula|copula fitting|lowest aic", ("pairwise_copula_table", "fit_copula", "pseudo_observations")),
    (r"trading calendar|non-trading|listwise", ("clean_returns_with_calendar", "iso_date_strings")),
    (r"wrong[- ]way|euler allocation|marginal (risk )?contribution|granularity adjustment|multi[- ]factor merton", (
        "copula_default_mc", "loss_var_es", "euler_es_contributions", "vasicek_portfolio_var",
        "gordy_granularity_adjustment", "hhi")),
    (r"creditmetrics|rating migrations? .{0,60}(simulation|monte carlo)", ("creditmetrics_default_sim", "simulate_migrations")),
    (r"cerep|transition-matrix-|default-rates-|bop\\\\?eop", ("cerep_migration_study", "read_delimited_matrix")),
    (r"\bpolars\b", ("polars_0x_to_1x_source", "check_python_source")),
    # know76 group 2 (qf_toolkit/helpers76_2.py whole-task bundles -> {file: obj}; self-tests proven_work/fail72/selftest76_2.py)
    (r"\b13f\b|infotable|coverpage|summarypage", ("sec13f_crowding_bundle",)),
    (r"redemption.{0,80}(etf|fund)|(etf|fund).{0,80}redemption|product[- ]data workbook", (
        "etf_redemption_pressure_bundle", "header_table")),
    (r"market[- ]model|estimation window|\bcaar\b", ("event_study_bundle",)),
    (r"lexicon|tf-?idf|tone surprise", ("fomc_tone_event_study",)),
    (r"form 4|form4|insider sale", ("form4_sale_pressure_bundle",)),
    (r"prediction[- ]market|kalshi|polymarket|locked[- ]edge", ("prediction_market_bundle",)),
    (r"xbrl|10-k\b|__ina\b|__bovera\b", ("sec_report_pair_results", "sec_query", "sec_find")),
    (r"\b8-k\b|date of report", ("sec8k_event_alpha_bundle",)),
    (r"\bfrn\b|callable|between reset|stale hedge|re-?hedge|pension liability", (
        "bond_hedge_immunization_bundle", "dated_instrument_price")),
    (r"immuniz|key rate durations?|z-spread|svensson", ("par_curve_immunization_bundle", "annual_par_bootstrap")),
    (r"brenner|corrado-miller|hallerbach|li (non-)?atm", ("iv_approximation_bundle",)),
    (r"put-call parity|synthetic forward|parity audit", ("parity_forward_audit_bundle",)),
    (r"cliquet|ratchet|forward[- ]start", ("cliquet_bundle", "forward_start_atm_call")),
    (r"hull[- ]white|jamshidian|bermudan", ("hw_swaption_bundle", "loglinear_df_curve")),
    (r"\bhdd\b|heating degree|ghcn", ("cme_hdd_option_bundle",)),
    # know77 group 1 (qf_toolkit/helpers77_1.py pattern bundles; self-tests proven_work/fail72/selftest77_1.py)
    (r"t-12 through t-2|skips? the most recent month|formation window|cross-sectional momentum",
     ("momentum_ls_deliverables",)),
    (r"committee|disagreement penalty|stochastic-process (filter|model-selection)|process[- ]selection layer",
     ("committee_process_alpha_bundle", "build_base_panel", "eval_feature")),
    (r"cartesian product", ("build_base_panel",)),
    (r"engagement multiplier|bullish.{0,60}bearish|sentiment.{0,300}(spearman|rank ic|vol[- ]target)",
     ("sentiment_momentum_report",)),
    (r"\bkirk\b|margrabe|spread options?\b|exchange options?\b", ("spread_option_report", "correlated_terminal_gbm")),
    (r"exactly these keys|keys in exactly|exactly this structure|expected keys", ("conform_to_keys",)),
    (r"fixed[- ]point|fixed six decimal|exactly 6 decimals|is `true` iff|lowercase true", ("fixed_text_frame",)),
    # know77 group 4 (qf_toolkit/helpers77_4.py + in-place fixes; self-tests proven_work/fail72/selftest77_4.py)
    (r"spot lag|broken[- ]date|swap roll|\bndf\b|carry trade|_comment",
     ("fx_pair_info", "first_iso_date", "money_market_basis", "fx_snapshot_desk")),
    (r"forward points|cross rates?|triangulat|pnl attribution|net delta|portfolio mtm", ("fx_forward_book", "money_market_basis")),
    (r"\bois\b.{0,200}(fra|libor)|quote_adjustment|projection curves?|fra adjustments?",
     ("bootstrap_df_curve", "normalize_quote_book", "select_curve_quotes", "fx_forward_nodes")),
    (r"funding rate|perpetual futures|analysis_end", ("funding_carry_report", "between_inclusive")),
    (r"geometric mean[- ]reverting|log[- ]space|stationary (mean|variance|distribution)", ("log_ou_jump_report", "ou_jump_stationary")),
    (r"mc validation|validate via monte carlo|analytical formulas", ("ou_jump_report", "ou_jump_stationary")),
    (r"immuniz\w*.{0,300}(frn|callable)|hedge_instruments|re-?hedge", ("bond_hedge_immunization_bundle",)),
    (r"immuniz|key rate durations?|z-spread|svensson|stress test", ("par_curve_immunization_bundle", "immunize_pv_krd")),
    (r"wrong[- ]way|euler allocation|marginal (risk )?contributions?|granularity adjustment|multi[- ]factor merton",
     ("copula_credit_portfolio_report",)),
    (r"creditmetrics|rating transition matrix|rating migrations?|transition thresholds",
     ("creditmetrics_report", "creditmetrics_default_sim")),
    (r"kupiec|christoffersen|conditional coverage|chi-squared p-values?",
     ("var_coverage_tests", "kupiec_pof", "christoffersen_independence")),
    (r"ema[- ]crossover|stressed var|internal models approach|capital charge", ("cta_ima_capital_report", "basel_ima_garch_capital")),
    (r"\bhdd\b|heating degree|ghcn|burn analysis", ("cme_hdd_option_bundle",)),
    # know77 group 2 (qf_toolkit/helpers77_2.py + in-place fixes in helpers75_1/76_1/76_2/76_3; self-tests
    # proven_work/fail72/selftest77_2.py)
    (r"form 4|form4|insider sale|ownership filings?", ("form4_sale_pressure_bundle", "bundle_values", "collect_input_files")),
    (r"__in[ab]\b|__bovera\b|10-k\b.{0,400}(xbrl|report)", ("sec_pair_results_auto", "parse_json_template",
                                                          "fill_template", "sec_auto_spec")),
    (r"(ewma|riskmetrics).{0,300}(euler|marginal risk|component risk|risk contribution|risk decomposition)"
     r"|(euler|risk decomposition).{0,300}(ewma|riskmetrics)", ("ewma_risk_report", "ewma_cov_recursive",
                                                               "euler_vol_decomposition")),
    (r"(historical simulation|variance-covariance|age-weighted|ewma).{0,600}(10-day|ten-day|overlapping)"
     r"|(five|several|standard) (standard )?(var )?methods", ("var_methods_comparison_bundle", "var_es_methods_table",
                                                             "var_es_one", "overlapping_sums")),
    (r"contaminated normal|true (analytical )?(var|risk measures)|simulated loss|estimation methods?.{0,120}rmse",
     ("loss_estimator_study_bundle", "simulate_loss_samples")),
    (r"bandi[- ]russell|bipower|volatility signature|microstructure[- ]noise|realized variance",
     ("realized_measures_by_day",)),
    (r"parkinson|garman[- ]klass|rogers[- ]satchell|yang[- ]zhang|range[- ]based (vol|estimator)",
     ("ohlc_vol_estimator_bundle", "ohlc_variances", "rolling_ohlc_vols")),
    (r"structured note|capital[- ]protected|zero-coupon bond.{0,80}(option|participation)", ("structured_note_report",)),
    (r"\bpolars\b.{0,400}(migrat|0\.x|1\.x)|migrat\w*.{0,300}\bpolars\b", ("polars_migrate_and_run",)),
    (r"\b8-k\b|date of report", ("sec8k_event_alpha_bundle",)),
    # know77 group 3 (qf_toolkit/helpers77_3.py whole-pattern bundles + in-place fixes in helpers75_1/2/3/4,
    # helpers76_1/3/4, risk.kupiec_pof; self-tests proven_work/fail72/selftest77_3.py)
    (r"brinson|allocation effect.{0,120}selection effect|beginning of (the )?month weights|weights? drift",
     ("brinson_fachler_report", "month_label_values")),
    (r"\bsvi\b|stochastic volatility inspired|dupire.{0,300}(market|quotes|option prices|smooth)",
     ("svi_dupire_report", "select_expiries_by_target")),
    (r"deribit|premiums? (are )?(also )?quoted in (btc|eth|the (underlying|coin))|discretely monitored|"
     r"local[- ]vol(atility)? monte carlo", ("localvol_barrier_report",)),
    (r"two[- ]factor (heston|stochastic vol)|chiarella|double heston|gauss[- ]legendre.{0,200}quad",
     ("heston2_surface_deliverables",)),
    (r"2s10s|butterfly|nelson[- ]siegel|par yields?.{0,200}(pca|principal comp)|(pca|principal comp).{0,200}yield",
     ("par_yield_curve_report", "tenor_years")),
    (r"kendall'?s tau.{0,300}(sampl|simulat|draw)|copulas?.{0,120}(sampl|simulat)|"
     r"tail dependence.{0,200}(gumbel|clayton|frank)", ("copula_family_study",)),
    (r"copula famil|best copula|copula fitting|lowest aic|pseudo[- ]observations", ("equity_pair_copula_report",)),
    (r"cerep|transition[- ]matri(x|ces)|rating migration|generator matrix|withdrawals?.{0,80}censor",
     ("cerep_migration_report", "sp_rating_mapping")),
    (r"projection curve|discount curve.{0,200}(libor|ibor|projection)|dual[- ]curve", ("dual_curve_deliverables",)),
    (r"\bdcc\b|dynamic conditional correlation|stationary bootstrap", ("dcc_var_deliverables",)),
    # know78 group 2 (qf_toolkit/helpers78_2.py; self-tests proven_work/fail72/selftest78_2.py)
    (r"flat[- ]vol(atility)? (test|check)|constant (black-scholes )?implied vol(atility)?.{0,300}(dupire|local vol)",
     ("dupire_flat_vol_check",)),
    (r"forward variance|calendar[- ]arbitrage|total variance between consecutive expir", ("forward_variance_table",)),
    (r"dupire|local[- ]vol|\bsvi\b|heston|put-call parity|\bhdd\b|heating degree|\bdcc\b|implied vol(atility)? "
     r"approximation", ("bundle_files",)),
    # know78 group 3 (qf_toolkit/helpers78_3.py; self-tests proven_work/fail72/selftest78_3.py)
    (r"update_id|sequence (number|id)|best[- ]bid[- ]best[- ]ask|first quote at or after|aggtrade",
     ("event_order",)),
    (r"bullish|bearish|sentiment (word|lexicon|score)|word list|lexicon", ("alpha_tokens",)),
    (r"instrumented principal|\bipca\b|latent factor.{0,200}characteristic", ("kps_ipca_als",)),
    (r"gibbons[- ]ross[- ]shanken|\bgrs\b", ("grs_statistic",)),
    # know78 group 4 (qf_toolkit/helpers78_4.py + in-place fixes in helpers75_1/76_2/77_1/77_2; self-tests
    # proven_work/fail72/selftest78_4.py)
    (r"__in[ab]\b|__bovera\b|report a\b.{0,300}report b\b|year[- ]over[- ]year comparison.{0,300}(10-k|report)",
     ("sec_pair_results_auto", "sec_table_lookup", "report_pair_sources", "instruction_text")),
    (r"(agent|committee|ensemble).{0,200}(score|signal).{0,600}normali[sz]ed versions? of|"
     r"(disagreement penalty|committee_score|agreement_multiplier)",
     ("committee_alpha_auto", "signed_component_lists", "committee_process_alpha_bundle")),
    (r"daily_return|daily returns?.{0,120}(annuali[sz]|sharpe)|equity curve.{0,200}(annuali[sz]|sharpe|volatility)",
     ("equity_curve_returns",)),
    (r"migrat\w*.{0,400}(preserve|outputs|api)|api.{0,40}(0\.x|1\.x|2\.x)", ("polars_migrate_and_run",)),
    (r"inline[- ]?xbrl|\bixbrl\b|\bdei\b|xbrl", ("ixt_canonical",)),
    # know79 group 2 (qf_toolkit/helpers79_2.py; self-tests proven_work/fail72/selftest79_2.py)
    (r"(dupire|local vol).{0,600}(finite[- ]difference|central difference|second (strike )?derivative|"
     r"d2c/dk2|∂²c/∂k²)|(finite[- ]difference|central difference).{0,600}(dupire|local vol)",
     ("fd_second_derivative", "dupire_local_vol_grid")),
    (r"local[- ]vol(atility)? monte carlo|local[- ]vol(atility)? (mc|simulation)|"
     r"(barrier|exotic).{0,300}local[- ]vol(atility)? (surface|grid)", ("local_vol_callable",)),
    # know79 group 3 (qf_toolkit/helpers79_3.py; self-tests proven_work/fail72/selftest79_3.py)
    (r"\"intermediates\"\s*:|intermediate checkpoints|reported as the `?\w+`? intermediate|"
     r"report(ed)? as (a )?positive magnitude|is a positive magnitude",
     ("json_deliverables", "output_json_templates")),
    (r"instrumented principal|\bipca\b|alternating least squares|\bals\b.{0,200}(converge|iteration)",
     ("ipca_sweep_sign_rule", "per_sweep_sign_rule")),   # know87 C: sign follows each eigen-rotation of a KPS sweep
    (r"last (trading|business) day of (each|the|every) (calendar )?month", ("month_end_sample_rule",)),
    (r"fama[- ]french|\bsmb\b|\bhml\b|mkt[_-]rf|residual (factor )?betas?", ("canonical_factor_name",)),
    # know79 group 1 (qf_toolkit/helpers79_1.py; self-tests proven_work/fail72/selftest79_1.py)
    (r"garch.{0,400}(native units|do not rescale|rescale\s*=\s*false|raw daily|without rescal|not rescale)|"
     r"(native units|do not rescale|rescale\s*=\s*false).{0,400}garch", ("garch11_mle",)),
    # know79 group 4 (qf_toolkit/helpers79_4.py; self-tests proven_work/fail72/selftest79_4.py)
    (r"form 4|form4|insider (sale|transaction|filing)|ownership (bucket|footnote)|nonderivativeholding",
     ("form4_ownership_footnote_ids", "form4_direct_label")),
    (r"polars.{0,400}(migrat|0\.x|1\.x|upgrade|deprecat)|(migrat|upgrade).{0,400}polars",
     ("polars_join_semantics_0x", "polars_fetch_to_head")),
    # know80 group 1 (qf_toolkit/helpers80_1.py; self-tests proven_work/fail72/selftest80_1.py)
    (r"(spot )?hedge (amount|size|trade)s?|delta[- ]hedg(e|ing) (amount|size|trade|notional)|hedge-relevant greeks",
     ("delta_hedge_trade",)),
    (r"structured note|capital[- ]protected|zero-coupon bond.{0,80}(option|participation)",
     ("note_conventions_from_text",)),
    # know80 group 2 (qf_toolkit/helpers80_2.py; self-tests proven_work/fail72/selftest80_2.py)
    (r"instrumented principal|\bipca\b|(alternating least squares|\bals\b).{0,600}(rotat|eigendecomposition|rescal)",
     ("ipca_identification_rule",)),
    (r"(cross[- ]sectional|time[- ]series|total|average|mean) r(²|\^?2|-squared| squared).{0,200}"
     r"(un-?)?cent(ere|re)d|(un-?)?cent(ere|re)d.{0,200}r(²|\^?2|-squared| squared)",
     ("r2_centering_rule", "row_r2")),
    # know80 group 3 (qf_toolkit/helpers80_3.py; self-tests proven_work/fail72/selftest80_3.py)
    (r"garch\s*\(\s*1\s*,\s*1\s*\)[^#]{0,600}(per[- ]asset|composite|regime|conditional vol)|"
     r"(per[- ]asset|composite)[^#]{0,300}garch", ("garch_mean_rule",)),
    # know81 group 2 (qf_toolkit/helpers81_2.py; self-tests proven_work/fail72/selftest81_2.py): a task that names
    # the Kelly-Pruitt-Su estimator gets that estimator's conventions (identification, R2 against zero)
    (r"kelly[,\s-]+pruitt|instrumented\s+principal\s+comp|\bipca\b", ("kps_ipca_named", "ipca_r2_rule")),
)
MAX_NAMES = 32   # know76: 24 -> 32 (bundles + their parts); task-pattern helpers first


def _h75_names() -> set:
    try:
        import qf_toolkit
        return {n for m in getattr(qf_toolkit, "_H75", []) for n in dir(m) if not n.startswith("_")}
    except Exception:  # noqa: BLE001
        return set()


def names_for(instruction: str) -> list[str]:
    text = (instruction or "").lower()
    out: list[str] = []
    for pattern, names in _RULES:
        if re.search(pattern, text):
            out.extend(n for n in names if n not in out)
    h75 = _h75_names()
    # know77: whole-task helpers (*_bundle / *_report / *_deliverables) first - they return every
    # deliverable and remove the hand-wiring the model kept getting wrong - then the other task-pattern
    # helpers, then the general toolkit
    whole = re.compile(r"_(?:bundle|report|deliverables|study|book|suite)$")
    out = ([n for n in out if n in h75 and whole.search(n)] + [n for n in out if n in h75 and not whole.search(n)]
           + [n for n in out if n not in h75])
    # know78: the frame matched to this task (frame_bank, by its wording) names the helpers of ITS pattern; those
    # go first, in the frame's order - a broad rule ("forward", "parity", "var") no longer puts another
    # pattern's bundle ahead of the right one
    try:
        from . import frame_bank
        top = frame_bank.select(instruction or "", k=1)
        if top:
            f = top[0]
            body = " ".join([str(f.get("frame", ""))] + [str(s) for s in f.get("steps", [])]
                            + [str(s) for s in f.get("formulas", [])])
            named = []
            for m in re.finditer(r"(?<![A-Za-z0-9_])([a-z][a-z0-9_]{3,})\(", body):
                n = m.group(1)
                if n in h75 and n not in named:
                    named.append(n)
            if named:
                out = named + [n for n in out if n not in named]
    except Exception:  # noqa: BLE001 - ordering aid only
        pass
    return out[:MAX_NAMES]


def block(instruction: str, api: str) -> str:
    """The API lines that mention the recommended names, as a short prompt block ('' if none)."""
    names = names_for(instruction)
    if not names or not api:
        return ""
    lines = api.splitlines()
    picked: list[str] = []
    for name in names:
        for line in lines:
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\(", line) and line not in picked:
                picked.append(line)
                break
    if not picked:
        return ""
    return ("TOOLKIT CALLS THAT COVER THIS TASK - use these (from qf_toolkit import ...) instead of "
            "re-deriving them; pass the task's conventions through their keywords:\n"
            + "\n".join(line.strip() for line in picked))
