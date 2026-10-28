# T1: masalah per unit (know43, run House, 8 Okt 2026)

Sumber: nama tes yang gagal (bukan isi checks), pesan crash tiap upaya, audit internal, dan catatan agen.
Jenis: **P** = bisa diselesaikan Python secara deterministik (parser/konversi/toolkit); **K** = konvensi atau rumus
spesifik yang perlu ditetapkan; **M** = metode/model keuangan yang salah atau belum ada; **T** = tugas teks/filing tanpa toolkit;
**W** = kehabisan waktu/kuota.

## Lulus (19): asian-levy-curran, barone-adesi, bs-greeks, cliquet, corporate-action, earnings-surprise, fft-compound-poisson,
first-passage, geometric-mr-jd, historical-var-data-prep, interest-rate-cap-floor, lob-pc-signal, lookback, ou-jump-commodity,
pca-factor, smith-tail, spread-option-kirk-margrabe, variance-swap, yield-curve-pca. (compound-geske lulus di run lain; terakhir 31/34.)

## Hampir lulus (1-3 tes gagal)

| Unit | Tes | Masalah | Jenis |
|---|---|---|---|
| cir-bond-pricing | 34/35 | log-likelihood tidak finite: estimasi CIR memberi parameter yang melanggar Feller lalu LL = -inf; perlu lantai parameter dan LL dihitung pada hasil akhir | K |
| copula-sampling-rank-correlation | 50/51 | nilai tail dependence Student-t: rumus 2·t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ))) salah satu konvensi (ν+1 vs ν) | K |
| standard-var-methods | 41/42 | ewma_current_vol: EWMA dipakai pada return portofolio, soal minta kovarians EWMA per aset lalu w'Σw, lalu anualisasi | K |
| implied-vol-approximations | 9/10 | summary.json nilai oracle: satu dari pendekatan (Brenner-Subrahmanyam / Corrado-Miller / Li) memakai varian rumus yang berbeda; program juga crash 3x karena kurung tidak ditutup (balasan terpotong) | K |
| event-study-earnings | 32/34 | corrado_z_full dan day0: definisi rank Corrado 1989 (rank gabungan estimasi+jendela, skala multi-hari) | K |
| digital-barrier-options | 32/35 | cash digital down-and-out call dan up-and-out put: rumus Reiner-Rubinstein untuk digital barrier salah suku; monotonicity ikut gagal | K |
| ohlc-realized-vol-estimators | 31/33 | "range estimators more efficient / lower variance": varians estimator Parkinson/GK dihitung pada skala berbeda dari close-to-close | K |
| var-es-estimation | 44/47 | best_for_student_t, RMSE historis dan kernel: pemilihan metode terbaik dan bandwidth kernel (Silverman) berbeda dari yang diminta | K |
| evt-pot-var | 50/55 | VaR/ES historis 99/99.5/99.9: kuantil empiris memakai interpolasi pandas, soal minta kuantil empiris diskret (atau sebaliknya) | K |
| fama-french-factor-model-new | 93/101 | t-stat dan p-value alpha Newey-West: lag HAC atau small-sample correction berbeda; crash awal `pd.column_stack` dan broadcast (2515,) vs (2515,4) | K + P |
| realized-vol-estimators | 129/141 | noise_var_est per hari dan noise_positive: estimator varians mikrostruktur (autokovarians lag-1 negatif → ambil −γ1) tanda/skala salah | K |

## Setengah benar (metode atau konvensi)

| Unit | Tes | Masalah | Jenis |
|---|---|---|---|
| american-option-fd-new | 39/54 | premi early-exercise call = −1,6e48: grid PDE meledak (skema eksplisit tidak stabil / dividen diskrit salah), call Amerika < Eropa, interpolasi grid di S0 | M |
| bollinger-backtest-aapl | 22/35 | semua metrik return/PnL/trades salah: eksekusi next-open dan whole shares tidak diikuti, kolom band 'middle'/'lower' hilang (KeyError) | K |
| momentum-backtest | 15/26 | sama: pnl seluruhnya null (kolom trades 'type, signal_date, exec_date, shares, pnl' tidak dibuat), html plotly tanpa trace | K + P |
| sma-crossover-spy | 14/20 | sama: pnl null, total_pnl/win_rate/final_capital salah; cast NaN ke integer crash | K + P |
| cme-hdd-option-pricing | 45/52 | harga MC, delta suhu, vega, sensitivitas kappa: bump-and-revalue satuan/skala; mean & std HDD bulanan | K |
| copula-equity-fitting | 21/28 | rho Gaussian JPM-GS, theta Gumbel, LL Student-t < Gaussian, pasangan terbaik: estimasi via tau-inversion vs MLE tertukar; Gumbel butuh tau≥0 (crash) | K |
| credit-migration-matrix | 65/107 (terbaik) / 22/107 (know43) | bentuk/label matriks (state D, NR, WD), baris stokastik, default absorbing; know43 crash di load ('method' not iterable, Length mismatch 0 vs 14 kolom) | P + K |
| credit-portfolio-var-cvar | 107/130 (terbaik) / 71/130 | kontribusi % tidak = 100, kunci summary hilang (n_obligors, total_exposure_usd, VaR Gaussian/t), HHI, top-5; broadcast (990,1)×(100000,990) | K + P |
| creditmetrics-portfolio-var | 22/26 (terbaik) / 1/26 | mean/std loss per rating, laju upgrade/downgrade (definisi stay/up/down/default); know43 crash broadcast (200000,100)×(1,6) dan string tak tertutup | K + P |
| crypto-funding-rate-basis-carry | 50/63 (terbaik) / 43/63 | half-life OU dalam periode (8 jam) vs hari, jumlah observasi fit, regime stats, VaR/CVaR MC; crash 'skew' unbound, KeyError 'r2'/'beta' | K + P |
| cross-sectional-momentum | 24/37 | num_signal_months, tanggal batas portofolio, baris pertama: konvensi skip-month (12-1) dan tanggal akhir bulan | K |
| delta-hedging-pnl-simulation | 15/27 | delta t0, total biaya transaksi, dividen, PnL per frekuensi rebalance: dividen kas diskrit dan biaya per rebalance dihitung berbeda | K |
| intraday-volume-fitting | 9/13 | jadwal akhir: jumlah saham tidak = order qty (pembulatan sisa terbesar), datetime bar acuan, bar share | P |
| localvol-barrier | 24/39 (terbaik) / 9/39 | distribusi baris per expiry, node sampel, label OTM: parsing instrumen Deribit (expiry_timestamp, 25SEP26) dan aritmetika tanggal; know43 crash KeyError 'expiration_timestamp' | P |
| mc-greek-surface-1 | 26/49 | FD/pathwise/LR greeks melenceng >5%, SE: bump terlalu kecil tanpa common random numbers; crash 'S' not defined, pd.random, greeks.json dibaca sebelum ditulis | M + P |
| yield-curve-bootstrap-immunization | 45/52 | forward rate negatif, KRD 2y/5y, Nelson-Siegel beta0/lambda2, efektivitas imunisasi: fitting NS (bounds/initial) dan KRD bump | K |
| zero-coupon-bootstrapping | 4/6 | DF tidak konsisten dengan zero rate, par bond tidak reprice: bootstrap_par menolak grid tak teratur → model mengarang; harus par_curve atau bootstrap manual | P + K |
| dcc-garch-portfolio-var | 24/63 | parameter DCC alpha/beta/LL, kolom dan baris VaR, anualisasi vol: einsum/ndim crash lalu DCC tidak dihitung | M + P |
| hull-white-swaption | 17/81 | forward negatif, RMSE kalibrasi, caplet diffs, 3 swaption hilang: brentq tanpa bracket (f(a), f(b) sama tanda), index 47 di pohon trinomial | M |
| form4-cross-sectional-sale-pressure | 26/54 | rollforward kepemilikan, band harga, entitas: parsing Form 4 (transaction_shares, is_static_holding_only) → semua nilai 0 | T |

## Sebagian besar salah

| Unit | Tes | Masalah | Jenis |
|---|---|---|---|
| alpha-hedge-strategy | 0/1 | verifikasi tunggal gagal; tidak ada crash → hasil ada tapi salah (backtest hedge beta, aturan lag sinyal) | M |
| bl-regime-hmm | 0/1 | Black-Litterman + HMM Baum-Welch: hasil ada tapi salah (tau, omega, prior ekuilibrium) | M |
| cta-basel-capital | 0/1 | Basel IMA: concat dimensi (5 vs ..) dan weights_lagged undefined → sebagian stub | M + P |
| kelly-var-sizing | 0/1 | SVD tidak konvergen, einsum salah, broadcast (3,3)×(10000,252,3): simulasi wealth path salah bentuk | P |
| regime-cta-vol-target | 0/1 | 'Data must be 1-dimensional (1260,5)' dan ARCHModelResult.alpha: GARCH per aset vs matriks | P |
| regime-riskparity-cvar | 0/1 | KeyError tanggal level, eigenvalues tidak konvergen, array read-only | P + M |
| structured-note-risk | 0/1 | tuple indices (hasil fungsi toolkit tuple dipakai sebagai dict), lalu berhenti 295 s sebelum batas: deliverable kosong | P + W |
| sentiment-factor-alpha | 0/1 | model minta 'posts.jsonl' yang tidak ada (input sebenarnya sentiment_*.csv), reindex(columns) pada Series | T + P |
| credit-spread-decomposition | 12/39 | rentang tanggal, NaN, nilai sampel, HAC t-stat: index 47 out of bounds (rolling window 48 bulan off-by-one), dekomposisi varians tidak jumlah 1 | K + P |
| double-sort | 4/19 | net return dan corner portfolio: bobot equal-weight per corner dan tanggal; parquet 36 MB | K |
| dupire-local-vol | 15/68 | kalibrasi S0/r/d, expiry: `Cannot convert input [... 2024-04-30` → kolom tanggal tz/format gagal diparse (prep belum menangkap bentuk ini) | P |
| merton-jump-diffusion | 7/28 | kalibrasi MLE Merton timeout 240 s dua kali (loop Python), lalu berhenti: parameter dan harga ATM tidak ada | M + W |
| multimodal-alpha-fusion | 2/26 | KeyError 'asof_date', 'article_ratio': skema panel dan nama kolom input tidak dibaca dari evidence | T + P |
| residual-momentum | 3/33 | datetime64.strftime, truth value Series ambiguous, lalu berhenti 202 s sebelum batas: deliverable hampir kosong | P + W |
| sec-10k-report-long | 1/35 | ekstraksi field XBRL/dei (DocumentType, PeriodEndDate, EntityRegistrantName, 12b titles): tidak ada parser; plan gagal | T |
| yield-curve-bond-immunization | 3/41 | bootstrap_par menolak grid 0.5..5.0 (perlu par_curve), basis '30/360/US' tidak dikenal, broadcast (60,) vs (5,); 1130 s → timeout | P + K + W |
| stable-residual | 2/19 | kolom monthly_results tidak dibuat (KeyError daftar kolom), file hilang, kurung tak cocok baris 300: program terlalu panjang untuk 4.000 token | P + W |
| 13f-amendment-aware-crowding | 14/51 | NaT.strftime, merge str × datetime pada REPORTCALENDARORQUARTER: parsing tanggal 13F; aturan amendment same-day | T + P |

## Crash di kode model (deliverable tidak terbentuk atau kosong)

| Unit | Tes | Masalah | Jenis |
|---|---|---|---|
| barrier-garch-var | 0/1 | `'module' object is not callable` 5x: model memanggil modul (arch/riskx) sebagai fungsi, tidak pernah lewat | P |
| binance-btc-participation-tca | 0/5 | perbandingan datetime64[ms] dengan int (epoch ms vs Timestamp) di load_inputs, tidak pernah lewat | P |
| brinson-sector-attribution | 0/42 | KeyError 'Date', pembagian DatetimeArray, identitas atribusi gagal (jumlah bobot 10.9 vs 12 → bobot bulanan tidak dinormalisasi) | P + K |
| etf-cross-asset-lead-lag | 3/24 | 'kernel reply unreadable' 3x (sel menghasilkan output terlalu besar/biner), market_model 0 observasi | P |
| etf-overlap-redemption-pressure | 7/40 | `'str'.iloc` (INPUTS teks dipakai sebagai DataFrame), ParserError 'Expected 2 fields saw 3' (file campuran) | P |
| ewma-portfolio-risk-decomposition | 0/27 | matmul dengan operand 0-dim: bobot dibaca sebagai float bukan vektor (params.json) | P |
| fomc-tone-event-study | 9/20 | KeyError 'sigma_e'/'sigma', concat dimensi 48: regresi state-adjusted dan leave-one-out tidak selesai | M + P |
| fx-carry-forward-hedge | 1/45 | KeyError 'carry'/'signal_lag' (config nested), list indices str; 13 deliverable, plan tidak selesai | P + W |
| fx-forward-cross-rate | 0/37 | KeyError 'EUR/CHF', 'tenor_6M', forward_rates_dict undefined: struktur quotes nested tidak dipetakan; cross rate dan MTM tidak dihitung | P |
| ipca-latent-factors | 0/1 | kolom 'size_', 'value_' … tidak ada (nama kolom ditebak), matmul 120 vs 50 | P + M |
| mtm-xccy-basis-desk | 58/145 (terbaik) / 28/145 | cip_deviation_bps dan bootstrap_par dipanggil dengan argumen salah, KeyError 0/'spot_fx', year_frac multi-kolom: 11 deliverable, PV per kaki tidak pernah benar | P + K |
| option-put-call-parity-forward-audit | 0/5 | tz-naive vs tz-aware, 'Cannot convert input [2025-03-21 16:00:00+00:00', KeyError time_to_expiry_years | P |
| prediction-markets-cross-venue | 10/25 | boolean indexer tidak selaras, Timestamp vs date, variabel belum didefinisikan di sel 'solution' | P |
| sec-8k-event-alpha | 0/1 | ModuleNotFoundError 'html_to_text', path relatif 'environment/data/filings', UnicodeDecodeError 0x95: parser HTML filing tidak ada | T + P |
| stochvol-implied-surface-new | 4/64 | 9 file (csv+png) hampir tidak ada: price_quad negatif, pd.zeros_like, 0-dim array; Heston/SVI surface tidak selesai | M + P |
| swap-curve-bootstrap-ois | 0/19 | `int / NoneType`: konvensi (day count / freq) dari conventions.json dibaca None; tidak pernah menulis file | P |

## Rekap jenis (62 unit tidak lulus)
- P (deterministik, bisa ditutup Python): muncul di 36 unit. Pola berulang: tanggal (tz-aware, epoch ms, NaT, datetime64.strftime), config nested JSON dibaca salah, toolkit dipanggil dengan signature salah, array 0-dim/float vs vektor, kolom hasil tidak dibuat sesuai kontrak, membaca deliverable sebelum ditulis.
- K (konvensi/rumus spesifik): 28 unit; 11 di antaranya "hampir lulus" hanya butuh satu definisi (EWMA kovarians, Corrado, digital barrier, NW HAC, tail dependence t, kuantil empiris, bandwidth kernel).
- M (metode keuangan): 12 unit (PDE stabil, Hull-White tree + brentq, DCC, Heston surface, Merton MLE, HMM/BL, IMA).
- T (teks/filing tanpa toolkit): 7 unit (10-K, 8-K, 13F, Form 4, FOMC tone, sentiment, multimodal).
- W (waktu habis): 6 unit, semuanya 7-13 deliverable atau MLE loop Python.
