"""know87 group C: the sign convention of an IPCA ALS sweep that performs its own eigen-rotation.

Task pattern: an IPCA / bilinear ALS task whose identification sits INSIDE the sweep (Kelly-Pruitt-Su: Gamma'Gamma = I
by Cholesky, then the eigenvectors of F'F/T in descending order, every sweep) and that states a sign convention for
eigendecompositions ('Eigendecompositions leave the sign of each eigenvector undetermined ... after the rotation flip
the sign of each factor ... whenever the factor's time-series mean is negative').

Decision. The sign rule resolves the sign freedom of an eigendecomposition, so it belongs to EVERY eigen-rotation the
estimator performs. A KPS sweep performs one per sweep; the reference implementation of the estimator (ipca package,
bkelly-lab, ALS step: Gamma orthonormalised, F rotated to a diagonal F'F, then 'enforce sign convention' mean(F_k) > 0)
applies the sign right after each of those rotations. Without it LAPACK's arbitrary eigenvector sign alternates
between sweeps, max|dGamma| stays O(1) and the loop's convergence flag reports that sign artefact, not the estimator
(the identified Gamma, F converge). Every factor quantity is identical either way (the final rotation + sign flip);
only als_converged / als_iterations differ. A plain ALS (identification once after convergence) has no eigen step
inside the sweep, so the question does not arise there (ipca_als ignores the switch).
Sources: Kelly, Pruitt, Su (2019) JFE sec. 2 (identification); ipca package (bkelly-lab) ALS step sign convention;
conventions table entry 'ipca_sign'.

API lines (one per function):
     ipca_sweep_sign_rule(text=None, identification=None) -> bool: True when the sweep carries the KPS identification (the sign convention follows every eigen-rotation) unless the task forbids a per-sweep sign; plain ALS -> helpers79_3.per_sweep_sign_rule(text)

No task names, no data values. Self-tests: proven_work/fail72/selftest87_C.py.
"""
from __future__ import annotations

import re

_NO_SWEEP_SIGN = (r"(?:do not|don't|never|must not|should not)\s+(?:flip|fix|apply|enforce|normali[sz]e)[^.]{0,60}"
                  r"sign[^.]{0,60}(?:each|every|inside|within)\s+(?:the\s+)?(?:als\s+)?(?:iteration|sweep|step)|"
                  r"sign[^.]{0,40}only once\b|only once[^.]{0,40}sign")


def _text(text):
    if text is not None:
        return str(text)
    try:
        from .helpers78_4 import instruction_text
        return instruction_text() or ""
    except Exception:  # noqa: BLE001
        return ""


def ipca_sweep_sign_rule(text=None, identification=None):
    """Whether the IPCA ALS sweep applies the factor sign convention (mean(F_k) >= 0) after its own eigen-rotation.

    identification: 'kps' (Gamma'Gamma = I + eigen-rotation inside every sweep) | 'post' (plain ALS, one rotation
    after convergence) | None = helpers80_2.ipca_identification_rule(text).
    'kps' -> True (the sign convention is attached to every eigendecomposition; ipca package ALS step), unless the
    task explicitly forbids a per-sweep sign ('do not flip the sign ... each iteration', 'sign ... only once').
    'post' -> helpers79_3.per_sweep_sign_rule(text) (no eigen step in a plain sweep; the switch is inert there)."""
    t = _text(text)
    if identification is None:
        from .helpers80_2 import ipca_identification_rule
        identification = ipca_identification_rule(t if t else None)
    if str(identification).lower() == "kps":
        return not re.search(_NO_SWEEP_SIGN, t.lower())
    from .helpers79_3 import per_sweep_sign_rule
    return per_sweep_sign_rule(t if t else None)
