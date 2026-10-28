"""Theory cards: the formulas, conventions and identities of each Track 1 category, plus the
procedure for a task whose inputs are incomplete.

Where the playbook holds lessons EARNED from failures, this holds the textbook the model is
expected to know - stated compactly, so the model derives a result instead of recalling one.
Nothing here names a task, a dataset or an expected value: it must be as useful on a sealed task
as on a public one.
"""
from __future__ import annotations

MAX_CHARS = 9000

MISSING_INPUTS = """\
WHEN AN INPUT IS MISSING, AMBIGUOUS OR INCOMPLETE (do this, never invent a number silently):
1. List every quantity the method needs; mark each as GIVEN (file/instruction), DERIVABLE, or
   CONVENTION. Print the list in your reading program before writing the solution.
2. Derive before assuming: rates from discount factors (r = -ln(DF)/T), forwards from spot and
   carry (F = S e^{(r-q)T}), volatility from prices (invert Black-Scholes), a missing leg from
   an identity (parity, CIP, attribution closure, rows summing to 1).
3. If still undetermined, enumerate the standard candidates - day count (ACT/365, ACT/360,
   30/360), compounding (continuous, annual, semi-annual), annualisation (252, 365, 12, 52),
   quantile method, ddof (0/1), log vs simple returns, percent vs decimal - and COMPUTE each.
4. Keep the candidate that satisfies every check the instruction states or implies (tolerances,
   identities, bounds, reprice-to-par, row counts, monotonicity). If several survive, prefer the
   convention the instruction's wording or the data's units point to; if none survives, report
   the closest and say which check failed.
5. Record the chosen assumptions in the deliverable wherever a notes/assumptions field exists.
"""

THEORY = {
    "derivatives-pricing": """\
- Black-Scholes-Merton with yield q: d1 = [ln(S/K) + (r - q + s^2/2)T]/(s sqrt T), d2 = d1 - s sqrt T;
  C = S e^{-qT} N(d1) - K e^{-rT} N(d2); P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1).
- Greeks: delta_C = e^{-qT}N(d1); gamma = e^{-qT} n(d1)/(S s sqrt T); vega = S e^{-qT} n(d1) sqrt T;
  theta per year (divide by 365 for per-day unless told); rho_C = K T e^{-rT} N(d2).
- Parity C - P = S e^{-qT} - K e^{-rT}; bounds max(0, S e^{-qT}-K e^{-rT}) <= C <= S e^{-qT}.
- Implied vol: bracketed root-find (Brent) on price(s) - market, vega-free; reject arbitrage-violating quotes.
- American: CRR/LR binomial or PDE with early-exercise max(hold, intrinsic); American call on
  non-dividend stock = European. Finite differences: Crank-Nicolson, check convergence in grid.
- Monte Carlo: exact GBM step S e^{(r-q-s^2/2)dt + s sqrt(dt) Z}; antithetic/control variates;
  report standard error; barriers need Brownian-bridge crossing correction on discrete grids.
- Exotics: Asian geometric closed form (s_G = s/sqrt 3, adjusted drift); Levy/Curran for
  arithmetic; digital cash-or-nothing = e^{-rT} N(d2); knock-in + knock-out = vanilla;
  cliquet = sum of forward-start options with local floors/caps; lookback via running extremum.
- Local vol (Dupire): s_loc^2 = [dC/dT + (r-q)K dC/dK + qC] / [0.5 K^2 d2C/dK2]; smooth the surface first.
- Heston: characteristic function pricing; Feller 2 kappa theta > xi^2; Merton jump-diffusion =
  Poisson-weighted sum of BS prices.""",
    "fixed-income": """\
- DF(t) = e^{-z t} (continuous) or (1+z/m)^{-m t}; forward f(t1,t2) = [ln DF(t1) - ln DF(t2)]/(t2-t1).
- Bootstrap sequentially: each par instrument prices to par given earlier DFs; solve the new DF
  (closed form for annual coupons, root-find otherwise); verify every input reprices.
- Bond price = sum c DF(t_i) + 100 DF(T); dirty = clean + accrued; YTM by root-find.
- Macaulay D = sum t PV / P; modified D = D_mac/(1+y/m); convexity = sum t(t+1/m) PV/(P(1+y/m)^2);
  dP/P ~ -D_mod dy + 0.5 C dy^2. Key-rate durations bump one node, re-price.
- Swaps: par rate = (1 - DF_n)/sum(tau_i DF_i); OIS discounting separate from the projection curve.
- Short-rate: Vasicek/CIR closed-form zero prices; Hull-White trinomial tree calibrated to the curve.
- Nelson-Siegel(-Svensson) curves fit by least squares on yields; PCA on curve changes:
  level/slope/curvature dominate.""",
    "credit": """\
- Hazard lambda: survival Q(t) = exp(-integral lambda); PD(0,t) = 1 - Q(t); constant lambda ~ spread/(1-R).
- CDS: premium leg = s sum tau_i DF_i Q_i (+ accrual on default); protection = (1-R) sum DF_i (Q_{i-1}-Q_i);
  par spread equates the legs; bootstrap hazards tenor by tenor.
- Merton: E = V N(d1) - D e^{-rT} N(d2), sigma_E E = N(d1) sigma_V V; solve both for V, sigma_V;
  distance to default = d2; PD = N(-d2).
- Rating transitions: rows sum to 1; multi-period = matrix power; generator via log for continuous time.
- Portfolio credit: one-factor Gaussian copula, default if sqrt(rho) M + sqrt(1-rho) e < N^{-1}(PD);
  Vasicek LHP loss distribution; CreditMetrics revalues by rating migration.
- Expected loss = EAD x PD x LGD; CVA = (1-R) sum EE(t_i) DF_i PD(t_{i-1}, t_i).""",
    "factor-research": """\
- Time-series regression r_i - r_f = a + b'F + e (OLS, HAC/Newey-West if stated); alpha = intercept.
- Fama-MacBeth: cross-sectional regression each period, then mean and t = mean / (sd/sqrt T).
- Portfolio sorts: rank at t on information known at t, hold t+1; double sorts independent vs
  conditional as stated; long-short = top - bottom.
- IC = corr(signal_t, return_{t+1}) (Spearman for rank IC); IR = mean IC / sd IC x sqrt(periods).
- Event study: abnormal return = r - (a + b r_m) from an estimation window disjoint from the event
  window; CAR sums ARs; test with the cited statistic.
- Momentum: cumulative return over lookback skipping the most recent month when specified.""",
    "backtesting": """\
- Positions from signals known at t trade at t+1 (or the stated lag); P&L_t = w_{t-1} . r_t.
- Costs: turnover sum |w_t - w_{t-1}| x cost per unit traded; net = gross - costs.
- Metrics: annual return (geometric from compounded wealth), volatility x sqrt(252), Sharpe =
  (mean - rf)/sd x sqrt(252), max drawdown = min(W/cummax(W) - 1), hit rate, turnover.
- Indicators: SMA/EMA (alpha = 2/(n+1)), Bollinger mean +/- k sd over n (population vs sample as
  stated), RSI with Wilder smoothing.
- Walk-forward: fit on window, test on next block, roll; never refit on test data.""",
    "risk-management": """\
- Historical VaR_a = -quantile(returns, 1-a); ES = -mean(returns <= that quantile).
- Parametric normal: VaR = -(mu + s z_{1-a}); ES = -mu + s n(z_a)/(1-a). Student-t: scale by sqrt((nu-2)/nu).
- Monte Carlo VaR: simulate factor returns (Cholesky of the covariance), revalue, take quantile.
- EWMA: s_t^2 = lambda s_{t-1}^2 + (1-lambda) r_{t-1}^2 (lambda 0.94 daily RiskMetrics).
- GARCH(1,1): s_t^2 = w + a r_{t-1}^2 + b s_{t-1}^2, MLE; persistence a+b < 1.
- EVT peaks-over-threshold: fit GPD (xi, beta) to exceedances over u; VaR = u + beta/xi [((n/N_u)(1-a))^{-xi} - 1].
- Backtests: Kupiec POF (LR with chi2(1)), Christoffersen independence; component VaR via Euler.
- Scaling: sqrt(h) rule only under i.i.d.; stress = revalue under the named shock set.""",
    "microstructure": """\
- VWAP = sum p v / sum v; implementation shortfall = side x (exec price - decision price)/decision price.
- Almgren-Chriss: minimise E[cost] + lambda Var[cost]; trajectory x_j = X sinh(k(T-t_j))/sinh(kT),
  k = arccosh(0.5 k~^2 tau^2 + 1)/tau with k~^2 = lambda s^2/eta~.
- Square-root impact: cost ~ Y s sqrt(Q/V). Participation schedules follow the intraday volume curve.
- Roll spread = 2 sqrt(-cov(dp_t, dp_{t-1})); Kyle lambda = slope of price change on signed volume.
- Realised variance = sum r^2 over intraday returns; Parkinson/Garman-Klass/Rogers-Satchell use OHLC.
- Order imbalance = (bid vol - ask vol)/(bid vol + ask vol); VPIN from volume buckets with BVC.""",
    "fx": """\
- Covered interest parity: F = S (1 + i_d tau)/(1 + i_f tau) (simple) or S e^{(r_d - r_f)T};
  forward points = F - S (quote convention, often x1e4).
- Cross rates: A/C = A/B x B/C; mind which currency is base; triangular arbitrage = product != 1.
- Garman-Kohlhagen: BSM with q = r_f: C = S e^{-r_f T} N(d1) - K e^{-r_d T} N(d2).
- Smile quotes: sigma_25RR = s_25C - s_25P; sigma_25BF = (s_25C + s_25P)/2 - s_ATM.
- Carry trade return = spot change + interest differential; hedged return removes the spot term via forwards.""",
    "nlp-on-finance": """\
- Tokenise consistently (lowercase, strip punctuation) before counting; state the dictionary used.
- Sentiment score = (pos - neg)/(pos + neg) or /(total words) as the task defines; handle zero denominators.
- Section extraction from filings: anchor on item headings (e.g. "Item 7"), take text to the next
  heading; entity resolution by normalised names plus identifiers.
- Event linkage: align document timestamps to trading days (after-close -> next day).""",
    "portfolio": """\
- Mean-variance: w* proportional to Sigma^{-1} mu; minimum variance w = Sigma^{-1}1 / (1'Sigma^{-1}1).
- Risk parity: w_i (Sigma w)_i equal across i; solve iteratively; weights sum to 1.
- Brinson: allocation (w_p - w_b)(R_b,i - R_b), selection w_b (R_p,i - R_b,i), interaction
  (w_p - w_b)(R_p,i - R_b,i); totals reconcile to active return.
- Black-Litterman: posterior mu = [(tau Sigma)^{-1} + P'O^{-1}P]^{-1}[(tau Sigma)^{-1} Pi + P'O^{-1}Q].""",
}

KEYWORDS = {
    "derivatives-pricing": ("option", "strike", "greek", "implied vol", "black-scholes", "barrier"),
    "fixed-income": ("yield", "bond", "discount factor", "bootstrap", "swap", "duration"),
    "credit": ("default", "hazard", "cds", "credit", "rating", "migration"),
    "factor-research": ("factor", "fama", "alpha", "information coefficient", "sort", "event study"),
    "backtesting": ("backtest", "signal", "sharpe", "drawdown", "moving average", "turnover"),
    "risk-management": ("var", "expected shortfall", "cvar", "garch", "stress", "quantile"),
    "microstructure": ("vwap", "execution", "order book", "spread", "participation", "intraday"),
    "fx": ("fx", "forward", "currency", "cross rate", "carry", "usd"),
    "nlp-on-finance": ("sentiment", "10-k", "filing", "10-q", "entity", "lexicon"),
    "portfolio": ("portfolio", "weights", "attribution", "risk parity", "mean-variance", "brinson"),
}


#: The family library (../theory_library/<family>.md, 26 Sep): per family the core formulas, the
#: conventions a task leaves implicit and how to detect them, how to complete missing inputs, the
#: invariants checkers test, and pitfalls - distilled from standard references. Loaded lazily;
#: a missing or unreadable library leaves the category cards above as they were.
LIBRARY_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "theory_library"
LIBRARY_CHARS = 29500   # 2 Oct: +3k for the Worked frame at the top of each card (3-4 cards per prompt)
_LIBRARY: list | None = None


def _library() -> list:
    """[(family, category, keywords tuple, body)] from the library files; [] if absent."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = []
        try:
            for path in sorted(LIBRARY_DIR.glob("*.md")):
                text = path.read_text(encoding="utf-8", errors="replace")
                head, sep, body = text.partition("\n---\n")
                if not sep:
                    continue
                meta = {}
                for line in head.splitlines():
                    k, _, v = line.partition(":")
                    meta[k.strip().lower()] = v.strip()
                kws = tuple(k.strip().lower() for k in meta.get("keywords", "").split(",") if k.strip())
                _LIBRARY.append((meta.get("family", path.stem), meta.get("category", ""), kws, body.strip()))
        except OSError:
            _LIBRARY = []
        _DF.clear()
        for _f, _c, kws, _b in _LIBRARY:
            for k in set(kws):
                _DF[k] = _DF.get(k, 0) + 1
    return _LIBRARY


_DF: dict = {}          # keyword -> number of cards that list it


def select_library(category: str | None, instruction: str, budget: int = LIBRARY_CHARS) -> str:
    """The family cards that match this task, best first, within `budget` characters.

    Score = keyword hits in the instruction (a phrase counts once), +2 when the card's category is
    the task's category. A card needs at least two keyword hits, or one plus the category match,
    so a stray word never drags an unrelated family in."""
    import math
    import re
    text = (instruction or "").lower()
    scored, weak = [], []
    for family, cat, kws, body in _library():
        # whole-word / whole-phrase matches only: as substrings "es" matched every text, "car"
        # matched "carry", "ask" matched "task", "var" matched "variance" (26 Sep QA)
        counts = {k: len(re.findall(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", text))
                  for k in kws if k}
        matched = {k: n for k, n in counts.items() if n}
        hits = len(matched)
        same_cat = bool(cat) and cat == (category or "")
        # rank by specificity x prominence: a keyword only one card carries ("lookback") outweighs
        # one five cards carry ("monte carlo"), and the task's subject word, repeated throughout
        # its text, outweighs a passing mention - the ranking decides which cards fit the budget
        score = sum((1.0 + math.log(n)) / _DF.get(k, 1) for k, n in matched.items()) \
            + (1.0 if same_cat else 0.0)
        if hits >= 3 or (hits >= 1 and same_cat):
            scored.append((-score, family, body))
        elif hits:
            weak.append((-score, family, body))
    if not scored and weak:
        # nothing clears the bar (an unusual category, a short text): the single best card with
        # any whole-word hit is still better than no family theory at all
        scored = sorted(weak)[:1]
    out, used = [], 0
    for _neg, family, body in sorted(scored):
        block = f"[family: {family}]\n{body}"
        if used + len(block) > budget:
            continue
        out.append(block)
        used += len(block)
    # know74: method cards (theory_library/methods/*.md) - one named method each (formulas, the
    # decisions it leaves open with their textbook defaults, pitfalls), recognised by keyword and
    # added on top of the family cards within their own budget.
    meth = select_methods(instruction)
    if meth:
        out.append(meth)
    return "\n\n".join(out)


METHODS_DIR = LIBRARY_DIR / "methods"
METHOD_CHARS = 7000
_METHODS: list | None = None


def _methods() -> list:
    """[(name, family, keywords tuple, body)] from theory_library/methods/*.md; [] if absent."""
    global _METHODS
    if _METHODS is None:
        _METHODS = []
        try:
            for path in sorted(METHODS_DIR.glob("*.md")):
                text = path.read_text(encoding="utf-8", errors="replace")
                head, sep, body = text.partition("\n---\n")
                if not sep:
                    continue
                meta = {}
                for line in head.splitlines():
                    k, _, v = line.partition(":")
                    meta[k.strip().lower()] = v.strip()
                kws = tuple(k.strip().lower() for k in meta.get("keywords", "").split(",") if k.strip())
                _METHODS.append((meta.get("method", path.stem), meta.get("family", ""), kws, body.strip()))
        except OSError:
            _METHODS = []
    return _METHODS


def select_methods(instruction: str, budget: int = METHOD_CHARS, limit: int = 4) -> str:
    """Method cards whose keywords the task text names: a multi-word or hyphenated keyword is enough
    on its own (it names the method), single words need two hits. Best first, within `budget`."""
    import re
    text = (instruction or "").lower()
    ranked = []
    for name, family, kws, body in _methods():
        hits, strong = 0, 0
        for k in kws:
            if k and re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", text):
                hits += 1
                strong += 1 if (" " in k or "-" in k or len(k) >= 9) else 0
        if strong >= 1 or hits >= 2:
            ranked.append((-(2 * strong + hits), name, family, body))
    out, used = [], 0
    for _s, name, family, body in sorted(ranked)[:limit]:
        block = f"[method: {name} | family: {family}]\n{body}"
        if used + len(block) > budget:
            continue
        out.append(block)
        used += len(block)
    return "\n\n".join(out)


def select(category: str | None, instruction: str) -> str:
    """The missing-input procedure, the card for the task's category, cards its text names, then
    the matching family cards of the theory library."""
    lib = select_library(category, instruction)
    base = _select_cards(category, instruction)
    return base + ("\n\nFAMILY THEORY (formulas, conventions, how to complete missing inputs, checks):\n"
                   + lib if lib else "")


def _select_cards(category: str | None, instruction: str) -> str:
    """The missing-input procedure, the card for the task's category, then cards its text names."""
    parts = [MISSING_INPUTS]
    text = (instruction or "").lower()
    ranked = []
    for slug, card in THEORY.items():
        if slug == (category or ""):
            ranked.append((0, 99, slug, card))
            continue
        hits = sum(1 for k in KEYWORDS[slug] if k in text)
        if hits >= 2:
            ranked.append((1, -hits, slug, card))
    extra = 0
    for rank, _, slug, card in sorted(ranked):
        if rank == 1:
            if extra >= 2:
                break
            extra += 1
        candidate = "\n".join(parts + [f"[{slug}]", card])
        if len(candidate) > MAX_CHARS:
            continue
        parts += [f"[{slug}]", card]
    return "\n".join(parts)
