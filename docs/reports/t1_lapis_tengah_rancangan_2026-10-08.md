# Rancangan: Python lapis tengah T1 ("penerjemah soal") - 8 Okt 2026

Tujuan: Python berdiri di antara soal dan model. Python membaca soal, mengenali keluarga dan kerangka,
mengisi SEMUA variabel ke dalam kerangka jawaban, memutuskan bagian mana yang dihitung Python sendiri
(toolkit/resep) dan bagian mana yang dihitung model, lalu memberi model prompt PENDEK (≤ 1.500 token)
per sel. Model tidak lagi membaca teori 30 ribu karakter, tidak menebak nama kolom, tidak memilih rumus.

Dasar: peta masalah 62 unit (t1_masalah_per_unit_2026-10-08.md). 36 unit gagal karena hal deterministik,
28 karena satu konvensi/rumus, 12 karena metode, 7 karena tugas teks, 6 karena waktu.

## 1. Alur

```
instruction.md + data        ─► [A] INTAKE (Python)        : graf tugas: deliverable, kuantitas, kalimat aturan,
                                                               parameter+satuan, tanggal, config diratakan, profil data
                             ─► [B] PENGENAL (Python)      : keluarga teori + kerangka terdekat (frame_bank) +
                                                               referensi rumus; slot kerangka diisi dari graf tugas
                             ─► [C] PERENCANA (Python)     : per deliverable: rute = toolkit | resep | model-cell;
                                                               urutan sel; anggaran waktu; kolom kontrak
                             ─► [D] PROMPT PENDEK (Python) : per model-cell: kerangka terisi, nama variabel runtime,
                                                               kolom wajib, 1-3 rumus, 1-2 jebakan. Tanpa teori panjang
                             ─► [E] EKSEKUSI (Python)      : kernel sel, prep input, adapter toolkit, auto-fix error,
                                                               emitter kontrak, audit hitung-ulang, pengatur waktu
                             ─► [F] PERBAIKAN (Python)     : error → perbaikan dari kamus; hanya sel yang gagal
                                                               dikirim balik ke model dengan diff yang persis
```

Prinsip: setiap keputusan yang bisa diambil dari teks soal + data diambil Python. Model hanya mengisi
perhitungan yang belum ada resepnya. Jika semua deliverable punya rute toolkit/resep, model tidak dipanggil
sama sekali untuk kode (hanya untuk konfirmasi pemetaan slot bila ada yang ambigu).

## 2. Modul

### A. INTAKE - `agent_core/intake.py`
- `task_graph(spec)` → {deliverables (dari contracts), quantities (nama, satuan, sumber kalimat), rules
  (kalimat must/only/exactly/before/after/in this order), params (angka+satuan: %, bps, hari, tahun),
  dates (as-of, window, horizon), examples (JSON contoh di soal → skema literal)}.
- `config_flatten(obj, prefix)` → variabel datar `cfg.signal_lag`, `cfg.quotes["EUR/CHF"].tenor_6M`,
  daftar jalur yang tersedia. Menutup: fx-forward-cross-rate (KeyError 'EUR/CHF', 'tenor_6M'), fx-carry
  ('carry', 'signal_lag'), swap-curve-ois (konvensi None), ewma-portfolio (bobot float vs vektor),
  ipca (nama kolom ditebak), sentiment (nama file input salah).
- `data_profile_typed()` dari prep: tiap kolom → tipe terdeteksi (tanggal tz/epoch, numerik-teks, kode,
  id), frekuensi seri, gap. Menutup: binance (epoch ms vs int), parity-audit (tz-naive vs aware),
  dupire/swap-ois ("Cannot convert input [... 2024-04-30"), 13f (NaT), residual-momentum/bollinger
  (datetime64.strftime).

### B. PENGENAL - `agent_core/recogniser.py`
- Keluarga (theory.select_library) + kerangka (frame_bank.select, generic frame diturunkan bobotnya).
- `fill_slots(frame, graph, profile)`: slot kerangka (S, K, r, q, sigma, T, dates, window, lambda, nu,
  freq, basis, notional, ...) diisi dari params/profil; sumber tiap slot dicatat (kalimat/kolom).
- Slot kosong → ditanya ke model dalam satu prompt pendek "pilih salah satu" (bukan kode).
- Referensi: baris rumus kartu teori yang menyebut slot itu saja (bukan seluruh kartu).

### C. PERENCANA - `agent_core/router.py`
- Untuk tiap deliverable: cari rute.
  1. **toolkit**: ada fungsi qf_toolkit yang menghasilkan kolom wajib → Python memanggil sendiri lewat
     `toolkit_adapter` (pemetaan slot → kwargs berdasar signature; argumen asing dibuang; `bootstrap_par`
     grid tak teratur dialihkan ke `par_curve`; alias basis "30/360/US", "USD"; modul vs fungsi).
     Menutup: mtm-xccy (cip_deviation_bps, bootstrap_par), yield-curve-bond, zero-coupon, barrier-garch
     ('module' not callable).
  2. **resep** (`recipes/`): komposit deterministik yang sudah teruji pada nilai buku teks:
     - `backtest_ledger`: sinyal close t → eksekusi open t+1, whole shares, trade log dengan kolom
       type/signal_date/exec_date/price/shares/pnl, daily_portfolio, metrik (total return, final capital,
       CAGR, Sharpe, MDD, Calmar, win rate). Menutup bollinger, momentum, sma (3 unit, pola sama).
     - `allocate_integer` (sisa terbesar) + `session_bars`: intraday.
     - `deribit_instrument` parser + `expiry_grid`: localvol.
     - `migration_matrix(states=[..., D, NR], absorbing=True)` + `migration_rates`: credit-migration,
       creditmetrics.
     - `euler_contributions` (jumlah = 100%), `one_factor_t_losses`: credit-portfolio.
     - `ewma_covariance`, `newey_west_alpha`, `corrado_1989`, `digital_barrier_price`,
       `t_tail_dependence`, `empirical_quantile(mode)`, `kernel_var(bandwidth=silverman)`,
       `noise_variance(gamma1)`, `cir_loglik_floor`: 9 unit "hampir lulus".
     - `bump_sensitivity` (beda terpusat, satuan eksplisit, CRN): cme-hdd, mc-greek.
     - `pde_cn_american` (Crank-Nicolson + PSOR, dividen diskrit): american-fd.
     - `hw_tree` + `bracket_brentq`: hull-white. `merton_mle_vectorised`: merton. `ns_fit(bounds)`:
       yield-curve-bootstrap. `dcc_fit` via arch: dcc-garch.
  3. **model-cell**: sisanya. Model menerima kerangka terisi + variabel runtime.
- Urutan sel: input → deliverable termurah dulu; `time_governor`: sisa waktu dibagi per deliverable,
  jumlah path MC diturunkan otomatis, deliverable yang belum ada ditulis versi minimum 60 s sebelum batas.
  Menutup 6 unit W.

### D. PROMPT PENDEK - `agent_core/brief.py`
Format per sel (contoh untuk `var_1day.csv` pada standard-var):
```
CELL var_1day (1 of 5). Variables now: ret [T x 2 float], w [2], cfg.lambda=0.94, cfg.level=[0.95,0.99]
WRITE emit('var_1day.csv', DataFrame[method, alpha, var_dollar, es_dollar]) - 10 rows, methods HS,
Normal, StudentT, AgeWeightedHS, EWMA; dollar amounts positive.
RULES: EWMA = covariance per asset (ewma_covariance(ret, 0.94)) then w'Σw; scale 10-day by sqrt(10).
TOOLKIT: var_es_historical(r, level), var_es_normal(r, level), var_es_student_t(r, level, nu),
age_weighted_var_es(r, level, lam), ewma_covariance(R, lam) -> Σ.
PITFALL: alpha column holds 0.95/0.99 as floats; es_dollar >= var_dollar.
```
≈ 250 token. Teori, playbook, contoh program, bank kerangka tidak dikirim lagi; semuanya sudah dipakai
Python untuk menyusun baris RULES/TOOLKIT/PITFALL.

### E. EKSEKUSI - sudah ada sebagian (know43): kernel sel, prep, toolkit toleran, kamus error.
Tambahan:
- `contract_emitter`: Python membangun DataFrame kontrak (kolom persis, urutan, dtype) dari variabel yang
  dihasilkan sel; kolom hilang → diisi dari resep atau diminta ke model sebagai sel 1 kolom. Menutup
  momentum/sma (pnl null), stable-residual (kolom monthly_results), credit-migration (shape/label).
- `recompute_audit`: Python membaca deliverable yang ditulis dan menghitung ulang identitas murah dengan
  toolkit (paritas, jumlah kontribusi, DF↔zero rate, KO+KI=vanilla, dirty=clean+accrued); selisih →
  permintaan perbaikan sel tertentu.
- `read_guard`: `read_json/read_table/find_input` menerima objek yang sudah dimuat (Colab: 26 crash).

### F. PERBAIKAN
Kamus error (29 pola) diperluas dengan yang baru ditemukan: NaT.strftime, datetime64.strftime,
tz-naive vs tz-aware, 'module' not callable, 'str'.iloc, einsum subscripts, matmul 0-dim, SVD/eigen
tidak konvergen (ridge), brentq tanpa bracket, IntCastingNaN, boolean indexer tidak selaras,
Timestamp vs date, read-only array, kernel reply unreadable (output sel terlalu besar → batasi print).

## 3. Pemetaan 62 unit → modul yang menutupnya

| Modul | Unit yang ditutup |
|---|---|
| config_flatten + data_profile_typed | fx-forward-cross-rate, fx-carry, swap-ois, ewma-portfolio, ipca, sentiment, binance, parity-audit, dupire, 13f, residual-momentum, prediction-markets, multimodal, etf-overlap, stable-residual |
| toolkit_adapter | mtm-xccy, yield-curve-bond, zero-coupon, barrier-garch, regime-cta (arch result), structured-note (tuple) |
| resep backtest_ledger | bollinger, momentum, sma |
| resep K (9 rumus) | standard-var, event-study, digital-barrier, fama-french, copula-sampling, evt-pot, var-es, cir, realized-vol, ohlc-vol |
| resep lain | intraday, localvol, credit-migration, creditmetrics, credit-portfolio, cme-hdd, mc-greek, copula-equity, delta-hedging, cross-sectional-momentum, credit-spread, double-sort, brinson, yield-curve-bootstrap |
| metode M | american-fd, hull-white, dcc-garch, merton, stochvol, kelly, regime-riskparity, bl-regime, cta-basel, alpha-hedge |
| time_governor + contract_emitter | stable-residual, residual-momentum, structured-note, merton, yield-curve-bond, fx-carry, mtm-xccy |
| teks T (tahap akhir) | sec-10k (dei/XBRL), sec-8k (html→teks), 13f, form4, fomc-tone, sentiment, multimodal |

## 4. Tahapan dan ukuran keberhasilan (semua diuji di VPS dengan model House, 12 unit sampel + 12 never-passed)
1. **know44a** (deterministik): intake + config_flatten + profil tipe + toolkit_adapter + contract_emitter +
   time_governor + prompt pendek. Target: crash < 25%, unit P yang tidak pernah menulis file menjadi menulis.
2. **know44b** (resep K + backtest_ledger): target 11 unit "hampir" dan 3 backtest lulus di lokal.
3. **know44c** (metode M): PDE, HW, DCC, Merton, NS.
4. **know44d** (teks): parser dei/XBRL, html→teks, 13F/Form 4.
Unggah hanya bila batch sampel menunjukkan kenaikan lulus yang jelas pada dua run.
