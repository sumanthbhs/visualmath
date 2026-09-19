(function(){let e=null;async function t(){let{loadPyodide:e}=await import(`https://cdn.jsdelivr.net/pyodide/v314.0.2/full/pyodide.mjs`),t=await e();return await t.loadPackage(`sympy`),t.runPython(`"""Symbolic backend for the Linear Algebra domain (runs under Pyodide in the browser).

Design rule carried over from the calculus prototype (math-viz): SymPy is
authoritative when it succeeds and DANGEROUS when it half-succeeds. Every verdict
is therefore one of
  'proved'   — an exact symbolic result (rational/integer matrix), trustworthy
  'refuted'  — a symbolic counterexample, trustworthy
  'numeric'  — floating-point input; SymPy's algorithms are tolerance-blind on
               floats (see the matrix traps below), so a float matrix is graded
               approximately and MUST be labelled, never silently promoted
  'unknown'  — the engine abstains (e.g. a hang would occur, or the input is
               genuinely indeterminate)
Never silently promote 'numeric' or 'unknown' to 'proved'.

Known SymPy matrix traps (carried into every analyse_* below):
  - \`Matrix.rank()\` on a matrix of Python floats is tolerance-blind: it can report
    a different rank than the same matrix read as exact rationals purely from
    round-off in the pivot search. ALWAYS route floats through \`parse_matrix\`,
    which distinguishes "exact" (every entry an exact rational/integer) from
    "numeric" (any entry a decimal), and NEVER call \`.rank()\`/\`.rref()\` on raw
    Python floats without that distinction attached to the result's provenance.
  - \`eigenvects()\` on a defective (non-diagonalizable) matrix with float entries
    can mis-report multiplicities or hang under near-repeated eigenvalues — the
    watchdog (symbolicSingleton.js on the JS side; the selftest's subprocess
    watchdog on the Python side) covers the hang; provenance covers the rest.
  - \`nsimplify\` can mis-fire on a float that is not actually a "nice" rational
    (e.g. 0.1 -> 1/10 is fine and intended, but a genuinely irrational-looking
    float should NOT be forced into a rational reading). \`parse_matrix\` only
    calls \`nsimplify\` on strings that look like decimals the UI itself produced
    (fixed precision), never on arbitrary floats — if in doubt, abstain to
    'numeric' rather than guess exact.

This file has NO __main__ self-test — see symbolicEngine.selftest.py for that.
This file is ?raw-imported into symbolic.worker.js and pyodide.runPython()'d
there, which runs it with __name__ == "__main__"; an inline self-test block
would silently re-run on every worker init, adding many seconds before the
worker could service its first real request.
"""
from sympy import (
    Matrix, Rational, nsimplify, simplify, S, Symbol, sympify, eye, zeros,
    ImmutableMatrix, Float, sqrt, symbols, linsolve, diag,
    Abs, Interval, oo, sign, floor, ceiling, Piecewise, E, pi, Sum,
)
from sympy.calculus.util import continuous_domain
from sympy.calculus.singularities import singularities
from sympy.series.limitseq import limit_seq
from sympy.solvers.inequalities import solve_univariate_inequality
import sympy as sp
import mpmath
import operator
from itertools import combinations, product

Q = Rational  # shorthand used throughout


def parse_entry(s):
    """Parse one matrix/vector entry. Accepts integers ('3', '-2'), fractions
    ('1/2', '-3/4'), and decimals ('0.5'). Returns (value, is_exact) — is_exact
    is False the moment any entry arrives as a decimal, which infects the whole
    matrix's provenance (see parse_matrix)."""
    s = str(s).strip()
    if s == "":
        raise ValueError("empty matrix entry")
    if "." in s or "e" in s.lower():
        # A decimal string — nsimplify is safe here because the string ITSELF
        # is already a fixed-precision decimal (the UI's own text field), not an
        # arbitrary float suspected to be irrational.
        return nsimplify(Float(s), rational=True), False
    # '1/2', '-3/4', or a bare integer — sympify handles all three exactly.
    val = sympify(s)
    if not val.is_rational:
        raise ValueError(f"unsupported entry {s!r}: expected integer, fraction, or decimal")
    return val, True


def parse_matrix(rows):
    """rows: list[list[str]] (or list[str] for a vector, treated as one column).
    Returns (M, exact) where M is a sympy Matrix of exact Rationals (decimals are
    nsimplify'd to their rational reading) and exact is False if ANY entry arrived
    as a decimal — that flag becomes the result's provenance ('proved'/'refuted'
    for exact input, 'numeric' for float input), per the module-level docstring."""
    if rows and not isinstance(rows[0], (list, tuple)):
        rows = [[r] for r in rows]  # bare vector -> column matrix
    exact = True
    parsed = []
    for row in rows:
        prow = []
        for cell in row:
            val, is_exact = parse_entry(cell)
            exact = exact and is_exact
            prow.append(val)
        parsed.append(prow)
    return Matrix(parsed), exact


def provenance_for(exact, *, refuted=False):
    """Central place every analyse_* asks "what do I call this verdict?" — keeps
    the exact/numeric/refuted vocabulary consistent across the whole domain."""
    if refuted:
        return "refuted"
    return "proved" if exact else "numeric"


# --- analyse_<id> functions land here, one per module, added by module-builder
#     agents during Phase 3 (symbolic tier) of each module's build. Each MUST:
#       - call parse_matrix (never touch raw floats without it)
#       - return a JSON-serialisable dict via pyodide's toJs(), with a
#         provenance field using provenance_for()
#       - abstain to 'unknown' rather than guess when genuinely undecidable
#     See docs/skills.md and .claude/skills/math-module/SKILL.md (Linear-algebra
#     addendum) for the per-module contract.


def analyse_rowreduce(rows):
    """rows: list[list[str]] — the raw matrix entries as typed by the user.
    Returns RREF, pivot columns (0-indexed) and rank, all via SymPy's own
    Matrix.rref() (an independent implementation from the JS engine's own
    Gauss-Jordan sweep — deliberately not re-derived from the JS algorithm).

    Provenance: 'proved' when every entry parsed as an exact integer/fraction;
    a single decimal entry anywhere flips the WHOLE matrix to 'numeric' (see
    parse_matrix/parse_entry) even though nsimplify still substitutes SOME
    rational value for that entry — a decimal is never 'proved' by this
    analysis. This function never returns 'refuted' (RREF has no true/false
    verdict to refute) and only abstains to
    'unknown' if the input fails to parse as a matrix at all (caught by the
    caller — parse_matrix raising ValueError propagates as an error message,
    which the worker/singleton already reports honestly as an error, not a
    silently-promoted verdict)."""
    M, exact = parse_matrix(rows)
    rref_m, pivots = M.rref()
    return {
        "rref": [[str(c) for c in rref_m.row(i)] for i in range(rref_m.rows)],
        "pivotCols": list(pivots),
        "rank": len(pivots),
        "rowsCount": M.rows,
        "cols": M.cols,
        "provenance": provenance_for(exact),
    }


def analyse_fundspaces(rows):
    """rows: list[list[str]] — the raw matrix entries as typed by the user.

    Fundamental subspaces via SymPy's OWN independent routines (Matrix.rref(),
    .rowspace(), .nullspace(), .columnspace(), .T.nullspace()) — deliberately not
    re-derived from the JS engine's free-variable parametrization, so this is a
    genuine second opinion, per CLAUDE.md's non-negotiable #1.

    Column-space convention: SymPy's own \`.columnspace()\` returns a basis drawn
    from the ORIGINAL matrix's pivot columns (matching the JS engine's documented
    convention in fundspacesEngine.js) — NOT columns of the RREF.

    Rank-Nullity has no breakable hypothesis over R^n (Linear-algebra addendum):
    rank + nullity == cols and Row space perp Null space hold UNCONDITIONALLY for
    every matrix, so both identities are verified here and returned as booleans
    rather than treated as a pass/fail verdict on the input itself. Provenance
    still follows exact/numeric per parse_matrix — a float input is never
    promoted to 'proved' even though the identity itself cannot fail."""
    M, exact = parse_matrix(rows)
    cols = M.cols

    rref_m, pivots = M.rref()
    pivots = list(pivots)
    rank = len(pivots)

    row_space = M.rowspace()
    null_space = M.nullspace()
    col_space = M.columnspace()
    left_null_space = M.T.nullspace()

    nullity = len(null_space)
    rank_nullity_holds = (rank + nullity == cols)

    orthogonality_holds = True
    for rv in row_space:
        rv_list = list(rv)
        for nv in null_space:
            nv_list = list(nv)
            dp = sum(a * b for a, b in zip(rv_list, nv_list))
            if simplify(dp) != 0:
                orthogonality_holds = False

    reduces_to_invertible = (M.rows == M.cols and rank == M.cols)

    return {
        "provenance": provenance_for(exact),
        "rank": rank,
        "nullity": nullity,
        "pivotCols": pivots,
        "rref": [[str(c) for c in rref_m.row(i)] for i in range(rref_m.rows)],
        "rowSpaceBasis": [[str(c) for c in v] for v in row_space],
        "nullSpaceBasis": [[str(c) for c in v] for v in null_space],
        "columnSpaceBasis": [[str(c) for c in v] for v in col_space],
        "leftNullSpaceBasis": [[str(c) for c in v] for v in left_null_space],
        "leftNullity": len(left_null_space),
        "rowsCount": M.rows,
        "cols": cols,
        "rankNullityHolds": rank_nullity_holds,
        "orthogonalityHolds": orthogonality_holds,
        "reducesToInvertible": reduces_to_invertible,
    }


def _is_orthogonal(M):
    """M^T M == I, checked entrywise after simplify (exact for rational M)."""
    n = M.rows
    MtM = simplify(M.T * M)
    In = eye(n)
    for i in range(n):
        for j in range(n):
            diff = simplify(MtM[i, j] - In[i, j])
            if diff != 0:
                return False
    return True


def _classify_2x2(M, det):
    """rotation / reflection / scaling / shear / singular / general — mirrors the
    JS engine's classify() (lineartransformEngine.js) but computed symbolically/
    exactly rather than with an epsilon tolerance. Order of checks matters, see
    the JS docstring for the reasoning."""
    if det == 0:
        return "singular"
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    if _is_orthogonal(M):
        return "rotation" if det > 0 else "reflection"
    if b == 0 and c == 0:
        return "scaling"
    one_zero = (b == 0) != (c == 0)  # exactly one off-diagonal entry is zero
    unit_diag = (a == d) and (a == 1 or a == -1)
    if one_zero and unit_diag:
        return "shear"
    return "general"


def _classify_3x3(M, det):
    """rotation / reflection / scaling / shear / singular / general — 3×3 analogue
    of _classify_2x2; mirrors the JS classify3() heuristics."""
    if simplify(det) == 0:
        return "singular"
    n = M.rows
    is_orth = _is_orthogonal(M)
    is_diag = all(M[i, j] == 0 for i in range(n) for j in range(n) if i != j)
    if is_orth:
        return "rotation" if simplify(det - 1) == 0 else "reflection"
    if is_diag:
        return "scaling"
    if simplify(det - 1) == 0:
        return "shear"
    return "general"


def analyse_lineartransform(matrix):
    """A 2×2 or 3×3 matrix as a linear map: exact determinant, a classification
    (rotation/reflection/scaling/shear/singular/general), trace, and — for 2×2
    only — a trace/det-based eigenvalue hint via the quadratic formula on the
    characteristic polynomial (NOT a full eigenvects() call; that's the separate
    eigen module's job). Abstains to 'unknown' on non-square or unsupported size."""
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {
            "provenance": "unknown",
            "note": "expects a 2×2 or 3×3 square matrix",
        }

    det = simplify(M.det())
    tr = simplify(M.trace())
    classification = _classify_2x2(M, det) if M.rows == 2 else _classify_3x3(M, det)

    # Eigenvalue hint — 2×2 only; quadratic formula on tr^2 - 4*det.
    eigen_hint = None
    if M.rows == 2:
        try:
            disc = simplify(tr * tr - 4 * det)
            if disc.is_number:
                if disc >= 0:
                    sq = sqrt(disc)
                    l1 = simplify((tr + sq) / 2)
                    l2 = simplify((tr - sq) / 2)
                    eigen_hint = {"kind": "real", "values": [str(l1), str(l2)]}
                else:
                    sq = sqrt(-disc)
                    re = simplify(tr / 2)
                    im = simplify(sq / 2)
                    eigen_hint = {
                        "kind": "complex",
                        "values": [f"{re} + {im}*I", f"{re} - {im}*I"],
                    }
            # else: disc did not resolve to a definite number — leave eigen_hint as None (abstain).
        except TypeError:
            # SymPy raised on an indeterminate inequality — abstain rather than guess.
            eigen_hint = None

    return {
        "provenance": provenance_for(exact),
        "det": str(det),
        "trace": str(tr),
        "classification": classification,
        "eigen_hint": eigen_hint,
    }


def analyse_eigen(matrix):
    """2×2 or 3×3 real matrix: eigenvalue/eigenvector/diagonalizability analysis via
    SymPy's own Matrix.eigenvals()/.eigenvects()/.is_diagonalizable() — an
    INDEPENDENT implementation from the JS engine's own quadratic-formula/
    nullspace code (eigenEngine.js), used as the module's symbolic second
    opinion, not a re-derivation.

    Provenance:
      'proved'  — exact (rational) input AND every eigenvalue found is exactly
                  rational (SymPy returns Rational/Integer, not an irrational
                  sqrt() expression) — includes the case of exact COMPLEX
                  eigenvalues (an exact negative discriminant is itself a valid
                  proof that no real eigenvector exists).
      'numeric' — decimal input anywhere (float mode; see the module docstring's
                  float-eigenvects trap — 'eigenvects()' on a defective float
                  matrix can mis-report multiplicity or hang), OR exact input
                  whose eigenvalues are genuinely irrational (SymPy's answer
                  contains an unevaluated sqrt/Pow) — never silently promoted.

    For 2×2: reports diagonalizability over the REALS explicitly
    (diagonalizable_over_r), which is what the UI actually demonstrates via
    PlaneStage. is_diagonalizable() is SymPy's own notion, defined over the
    COMPLEX numbers: a rotation matrix reports is_diagonalizable() == True
    in SymPy, but has NO real eigenvectors — do not confuse the two.

    For 3×3: uses M.eigenvects() which SymPy handles symbolically; reports
    kind, eigenvalues (as strings), algebraic/geometric multiplicities.
    """
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_eigen expects a 2×2 or 3×3 square matrix"}

    # ── 3×3 branch ──────────────────────────────────────────────────────────
    if M.rows == 3:
        if not exact:
            # Float input: eigenvects() near repeated eigenvalues can hang or
            # mis-report multiplicities — use eigenvals() only, labelled numeric.
            try:
                evals = M.eigenvals()
            except Exception as e:
                return {"provenance": "unknown", "note": f"eigenvals() failed on float 3×3: {e}"}
            has_complex = any(not ev.is_real for ev in evals.keys())
            if has_complex:
                kind = "complex"
            elif len(evals) < 3:
                kind = "repeated"
            else:
                kind = "distinct"
            return {
                "provenance": "numeric",
                "kind": kind,
                "eigenvalues": [str(ev) for ev in evals.keys()],
                "algebraic_mults": [mult for mult in evals.values()],
                "note": "float 3×3 input — multiplicities approximate, not certified",
            }

        # Exact (rational) 3×3 input.
        try:
            evs = M.eigenvects()  # list of (eigenvalue, alg_mult, [eigenvectors])
        except Exception as e:
            return {"provenance": "unknown", "note": f"eigenvects() failed on 3×3: {e}"}

        has_complex = any(not ev[0].is_real for ev in evs)
        any_defective = any(ev[1] > len(ev[2]) for ev in evs)
        any_repeated = any(ev[1] > 1 for ev in evs)

        if has_complex:
            kind = "complex"
        elif any_defective:
            kind = "defective"
        elif any_repeated:
            kind = "repeated"
        else:
            kind = "distinct"

        # Provenance: 'proved' when input is exact — SymPy's eigenvects() on a
        # rational 3×3 matrix always returns exact algebraic numbers (integer,
        # rational, radical, or complex algebraic). The cubic formula guarantees
        # a closed-form answer exists; SymPy uses it. So 'proved' = exact input
        # + eigenvects() succeeded (not just "rational eigenvalues").
        provenance = provenance_for(exact)

        return {
            "provenance": provenance,
            "kind": kind,
            "eigenvalues": [str(ev[0]) for ev in evs],
            "algebraic_mults": [ev[1] for ev in evs],
            "geometric_mults": [len(ev[2]) for ev in evs],
            "note": "3×3 analysis via SymPy eigenvects()",
        }

    # ── 2×2 branch (original logic, kept exactly) ───────────────────────────
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    tr = simplify(a + d)
    det = simplify(a * d - b * c)
    is_symmetric = simplify(b - c) == 0
    disc = simplify(tr * tr - 4 * det)

    # Float input: eigenvects() on a defective float matrix can mis-report
    # multiplicity or hang near repeated eigenvalues (the watchdog covers the
    # hang; provenance covers the rest) — NEVER promote a float-derived result
    # to 'proved', regardless of how "clean" SymPy's answer looks.
    if not exact:
        try:
            evals = M.eigenvals()
        except Exception:
            return {"provenance": "unknown", "note": "eigenvals() failed on float input"}
        kind = "complex" if any(not e.is_real for e in evals.keys()) else (
            "repeated" if len(evals) == 1 else "distinct"
        )
        return {
            "provenance": "numeric",
            "trace": str(tr),
            "det": str(det),
            "disc": str(disc),
            "isSymmetric": bool(is_symmetric),
            "kind": kind,
            "eigenvalues": [str(complex(e)) if not e.is_real else str(float(e)) for e in evals.keys()],
            "note": "float input — eigenvects()/multiplicities are tolerance-blind near repeated eigenvalues; treat as approximate",
        }

    # Exact (rational) input from here on.
    if disc.is_number and disc < 0:
        # Provably complex eigenvalues — an exact proof, even though there is
        # no real eigenvector to report.
        return {
            "provenance": "proved",
            "trace": str(tr),
            "det": str(det),
            "disc": str(disc),
            "isSymmetric": bool(is_symmetric),
            "kind": "complex",
            "eigenvalues": [str(e) for e in M.eigenvals().keys()],
            "diagonalizableOverR": False,
            "note": "complex eigenvalues -> no real eigenvectors exist",
        }

    evals = M.eigenvals()
    # An eigenvalue is "exactly rational" if SymPy's own value .is_rational is
    # True — a genuinely irrational discriminant produces an unevaluated
    # sqrt(...)-bearing expression here, which is NOT rational.
    all_rational = all(e.is_rational for e in evals.keys())
    provenance = "proved" if all_rational else "numeric"

    eigvects = M.eigenvects()
    pairs = []
    for ev, mult, vs in eigvects:
        for v in vs:
            Av = M * v
            lv = ev * v
            residual_zero = simplify(Av - lv) == Matrix([0, 0])
            pairs.append({
                "lambda": str(ev),
                "v": [str(v[0]), str(v[1])],
                "algebraicMult": mult,
                "residualZero": bool(residual_zero),
            })

    kind = "repeated" if len(evals) == 1 else "distinct"
    is_diag_over_c = M.is_diagonalizable()
    # Over the reals: diagonalizable iff (already excluded complex above) every
    # eigenvalue's geometric multiplicity equals its algebraic multiplicity —
    # SymPy's own is_diagonalizable() already encodes exactly this (since we
    # have already excluded the complex-eigenvalue branch above, is_diagonalizable()
    # here IS the real-diagonalizability answer).
    diagonalizable_over_r = bool(is_diag_over_c)

    return {
        "provenance": provenance,
        "trace": str(tr),
        "det": str(det),
        "disc": str(disc),
        "isSymmetric": bool(is_symmetric),
        "kind": kind,
        "eigenvalues": [str(e) for e in evals.keys()],
        "eigenvectors": pairs,
        "diagonalizableOverR": diagonalizable_over_r,
        "reducesTo": (
            "Spectral Theorem (symmetric matrix): real eigenvalues guaranteed, eigenvectors orthogonal"
            if is_symmetric else None
        ),
    }
def analyse_determinants(matrix):
    """Square matrix (2×2 or 3×3): determinant via SymPy's own Matrix.det() — an
    INDEPENDENT implementation from the JS engine's cofactor-expansion code
    (determinantsEngine.js), used as this module's symbolic second opinion, not
    a re-derivation. Also cross-checks a cofactor expansion along row 0 built
    from SymPy's OWN .cofactor() method against that same M.det() value (two
    different SymPy code paths, not just the JS vs Python boundary), and
    demonstrates — LIVE, on this matrix, not asserted in prose — the three
    breakable definition properties: swapping two rows negates det, a repeated
    row zeroes it, and scaling one row by k scales det by k.

    Provenance: 'proved' when every entry parsed exact (rational/integer);
    'numeric' the moment any entry is a decimal. Cofactor expansion is pure
    +,-,* over the parsed entries, so an exact input ALWAYS yields an exact
    rational determinant — no irrational-root trap the way analyse_eigen's
    discriminant has. This function never abstains to 'unknown' except on a
    non-square or unsupported-size matrix."""
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_determinants expects a 2×2 or 3×3 square matrix"}
    n = M.rows
    det = simplify(M.det())

    # Cofactor expansion along row 0, via SymPy's OWN .cofactor() — a DIFFERENT
    # code path from M.det()'s own algorithm — cross-checked against M.det().
    terms = []
    running = S.Zero
    for j in range(n):
        entry = M[0, j]
        cof = M.cofactor(0, j)
        term = simplify(entry * cof)
        running += term
        sign = 1 if j % 2 == 0 else -1
        terms.append({
            "i": 0, "j": j, "sign": sign,
            "entry": str(entry), "cofactor": str(simplify(cof)), "term": str(term),
        })
    cofactor_matches_det = simplify(running - det) == 0

    # Row swap: swapping rows 0 and 1 (always valid, n >= 2) negates det.
    swapped = M.copy()
    swapped.row_swap(0, 1)
    row_swap_flips_sign = simplify(swapped.det() + det) == 0

    # Repeated row: replacing row 1 with a copy of row 0 makes det = 0.
    repeated = M.copy()
    for j in range(n):
        repeated[1, j] = M[0, j]
    repeated_row_zero = simplify(repeated.det()) == 0

    # Scaled row: scaling row 0 by k=3 scales det by 3.
    k = Rational(3)
    scaled = M.copy()
    for j in range(n):
        scaled[0, j] = k * M[0, j]
    scaled_row_scales = simplify(scaled.det() - k * det) == 0

    return {
        "provenance": provenance_for(exact),
        "n": n,
        "det": str(det),
        "terms": terms,
        "cofactorMatchesDet": bool(cofactor_matches_det),
        "rowSwapFlipsSign": bool(row_swap_flips_sign),
        "repeatedRowZero": bool(repeated_row_zero),
        "scaledRowScales": bool(scaled_row_scales),
    }


def analyse_inverses(matrix, b=None):
    """Square matrix (2×2 or 3×3): invertibility verdict + A⁻¹ via SymPy's own
    Matrix.inv() — an INDEPENDENT implementation from the JS engine's two paths
    (inversesEngine.js's adjugate division and its rrefEngine-shared Gauss-Jordan
    sweep on [A|I]), used as this module's symbolic second opinion. Cross-checks,
    LIVE on this matrix: the defining identity A·A⁻¹ = I, and the adjugate route
    A⁻¹ = adj(A)/det(A) built from SymPy's OWN .cofactor() (adj(A)[i,j] =
    cofactor(A)[j,i]) against Matrix.inv(). When b is supplied, also computes
    Cramer's rule per variable: xᵢ = det(Aᵢ)/det(A) with Aᵢ = A with column i
    replaced by b, and verifies A·x = b.

    Singular case (det = 0): reports invertible = False with the rank (SymPy's
    rank on the EXACT rational parse — safe per the matrix-traps note above) and
    the first column the [A|I] story visibly fails on, mirroring the JS engine's
    failedAtCol. No 'refuted' verdict: the theorem is an iff, not a boolean
    claim being disproved — the hypothesis det ≠ 0 is simply unmet, certified
    exactly (an exact 0 determinant IS a proof of singularity).

    Provenance: 'proved' when every entry parsed exact (rational/integer) — the
    inverse, when it exists, is an exact rational matrix (Gauss-Jordan on exact
    rationals stays rational); 'numeric' the moment any entry is a decimal.
    Never abstains to 'unknown' except on non-square / unsupported size.
    NOTE (known asymmetry, deliberate): SymPy's det != 0 test on FLOAT input is
    an exact comparison while the JS engine applies its 1e-6 pivot tolerance —
    a barely-invertible float matrix can therefore read invertible here but
    rank-deficient there. Both sides stay 'numeric' and the UI surfaces the
    disagreement instead of hiding it (mirrors the rowreduce nsimplify trap)."""
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_inverses expects a 2×2 or 3×3 square matrix"}
    n = M.rows
    det = simplify(M.det())
    rank = M.rank()
    invertible = simplify(det) != 0

    result = {
        "provenance": provenance_for(exact),
        "n": n,
        "det": str(det),
        "rank": int(rank),
        "invertible": bool(invertible),
        "inverse": None,
        "inverseByAdjugate": None,
        "adjugate": None,
        "methodsAgree": None,
        "residualIdentity": None,
        "cramer": None,
        "singularReason": None,
    }

    if not invertible:
        # The visible failure point: the first column of A carrying no pivot in
        # the [A|I] sweep (rationalise SymPy's own rref pivot list).
        _, pivot_tuple = M.rref()
        pivot_cols = set(pivot_tuple)
        failed_at = next((c for c in range(n) if c not in pivot_cols), None)
        result["singularReason"] = (
            f"det(A) = 0 (rank {rank} < {n}): no matrix B satisfies A·B = I — "
            f"column {failed_at} cannot be rescued by any row swap"
            if failed_at is not None else
            f"det(A) = 0 (rank {rank} < {n}): no matrix B satisfies A·B = I"
        )
        return result

    # Method 1: SymPy's own inv() (its internal Gauss-Jordan / adjugate choice).
    inv = M.inv()
    inv = simplify(inv)

    # Method 2: adjugate / Cramer — adj(A)[i,j] = cofactor(A)[j,i], A⁻¹ = adj/det.
    # Built from SymPy's OWN .cofactor(), a different code path from .inv().
    adj = Matrix(n, n, lambda i, j: simplify(M.cofactor(j, i)))
    inv_adj = simplify(adj / det)

    methods_agree = all(
        simplify(inv[i, j] - inv_adj[i, j]) == 0
        for i in range(n) for j in range(n)
    )

    # Defining identity, verified LIVE: A · A⁻¹ = I (entrywise simplify).
    product = simplify(M * inv)
    residual = all(
        simplify(product[i, j] - (1 if i == j else 0)) == 0
        for i in range(n) for j in range(n)
    )

    result["inverse"] = [[str(inv[i, j]) for j in range(n)] for i in range(n)]
    result["inverseByAdjugate"] = [[str(inv_adj[i, j]) for j in range(n)] for i in range(n)]
    result["adjugate"] = [[str(adj[i, j]) for j in range(n)] for i in range(n)]
    result["methodsAgree"] = bool(methods_agree)
    result["residualIdentity"] = bool(residual)

    # Cramer's rule (optional — only when the UI supplies a b vector).
    if b is not None:
        b_vals = [parse_entry(s)[0] for s in b]
        if len(b_vals) != n:
            raise ValueError(f"Cramer's rule needs a length-{n} b vector")
        dets = []
        xs = []
        for i in range(n):
            Ai = M.copy()
            Ai[:, i] = b_vals
            di = simplify(Ai.det())
            dets.append(str(di))
            xs.append(str(simplify(di / det)))
        x_vec = Matrix([sympify(s) for s in xs])
        ax_equals_b = simplify(M * x_vec - Matrix(b_vals)) == zeros(n, 1)
        result["cramer"] = {
            "columnDets": dets,
            "solution": xs,
            "residual": bool(ax_equals_b),
        }

    return result


def analyse_vectorspaces(vectors, target=None):
    """vectors: list[list[str]] — each inner list is ONE vector's components
    (all vectors share the same ambient dimension), matching the JS engine's
    own convention. Independently recomputes, via SymPy's own Matrix.rank()
    and Matrix.nullspace() (never re-derived from the JS RREF sweep):
      - rank of the vector set (columns of the transpose of the row-per-vector
        matrix) -> independent iff rank == count
      - when dependent, an actual nontrivial dependency relation via
        nullspace() of that same column matrix (SHOW the relation, not just a
        verdict, per the concept-module Shape)
      - spansAmbient iff rank == dimension; isBasis iff independent AND spansAmbient

    If \`target\` (list[str], one vector) is given, ALSO solves the span-
    membership question sum(ci*vi) == target via linsolve on the exact same
    column matrix, reporting whether it is consistent and, if so, one actual
    combination (particular solution: every free coefficient set to 0).

    Provenance: 'proved' iff every vector entry (and the target's entries, if
    given) parsed as an exact integer/fraction — a single decimal anywhere
    flips the WHOLE computation to 'numeric' (parse_matrix's whole-matrix
    provenance rule). Never returns 'refuted' (there is no boolean theorem to
    refute here, only a definition to satisfy or fail) and only abstains to
    'unknown' if a genuinely different-dimensioned vector set is passed."""
    M, exact = parse_matrix(vectors)  # M: rows = vectors, cols = ambient dimension
    count = M.rows
    dimension = M.cols
    A = M.T  # n x k column matrix: column i is vector i

    rank = A.rank()
    independent = rank == count
    dependency = None
    if not independent:
        ns = A.nullspace()
        if ns:
            v = ns[0]
            dependency = [str(v[i]) for i in range(count)]

    spans_ambient = rank == dimension
    is_basis = independent and spans_ambient

    exact_all = exact
    result = {
        "dimension": dimension,
        "count": count,
        "rank": rank,
        "independent": independent,
        "dependency": dependency,
        "spansAmbient": spans_ambient,
        "isBasis": is_basis,
    }

    if target is not None:
        tM, texact = parse_matrix(target)  # column vector, n x 1
        exact_all = exact and texact
        if tM.rows != dimension:
            result["target"] = {
                "provenance": "unknown",
                "note": "target vector dimension does not match the vector set's ambient dimension",
            }
        else:
            aug = A.row_join(tM)
            rank_A = A.rank()
            rank_aug = aug.rank()
            consistent = rank_A == rank_aug
            combination = None
            if consistent:
                cs = symbols(f"c0:{count}")
                sol = linsolve((A, tM), *cs)
                if sol:
                    sol_tuple = next(iter(sol))
                    free_syms = set()
                    for expr in sol_tuple:
                        free_syms |= expr.free_symbols
                    subs = {s: 0 for s in free_syms}
                    combination = [str(expr.subs(subs)) for expr in sol_tuple]
            result["target"] = {
                "consistent": consistent,
                "combination": combination,
                "provenance": provenance_for(exact_all),
            }

    result["provenance"] = provenance_for(exact_all)
    return result


def analyse_orthogonality(vectors):
    """vectors: list[list[str]] — each inner list ONE vector's components (the
    same convention as analyse_vectorspaces and the JS engine's
    orthogonalityEngine.js). SymPy's independent second opinion on the
    Gram-Schmidt process, computed by a hand-rolled exact walk (SymPy's
    Matrix.GramSchmidt REFUSES dependent input rather than reporting the halt
    step — this module's whole point is showing WHERE u_k = 0, so the walk is
    explicit):
      - per step k: every projection coefficient (v_k.u_j)/(u_j.u_j) as an
        exact rational string, the residual u_k, and its squared norm
      - the process HALTS at the first k with u_k exactly zero — that zero is
        the dependence certificate; nothing after it exists
      - input definition clauses: pairwise dots v_i.v_j, norms squared,
        orthogonal / orthonormal verdicts
      - the orthonormal set e_k = u_k/sqrt(u_k.u_k) kept SYMBOLIC (exact
        irrationals like 'sqrt(2)/2' — the one thing the JS engine cannot do,
        since its normalization is float by mathematical necessity)
      - span equivalence on the PROCESSED prefix (the theorem's actual claim —
        see the JS engine's header note): rank{v_1..v_r} == rank{u_1..u_r},
        plus the full input rank so the UI can show the halt gap
      - the Gram determinant det(V V^T): zero EXACTLY when the set is
        dependent (the textbook independence certificate, one number)

    Provenance: 'proved' iff every entry parsed exact; a single decimal flips
    the whole result to 'numeric'. Known deliberate asymmetry with the JS
    engine (same trap as inverses' FLOAT_SINGULAR_TOL): a decimal entry like
    '1e-7' parses here to its exact rational reading 1/10^7, so the
    near-parallel float preset [[1,0],[1,1e-7]] reads INDEPENDENT in SymPy
    while the JS engine's 1e-6 tolerance reads it dependent. Both stay
    'numeric'; the UI surfaces the disagreement with a warning, never hides
    it, and never promotes either side. Never returns 'refuted' (a definition
    has no counterexample verdict) and never abstains to 'unknown' for
    well-formed input."""
    M, exact = parse_matrix(vectors)  # rows = vectors, cols = ambient dimension
    count = M.rows
    n = M.cols
    vs = [M.row(k) for k in range(count)]

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(n))

    # -- input definition clauses ------------------------------------------
    input_dots = []
    for i in range(count):
        for j in range(i + 1, count):
            input_dots.append({"i": i, "j": j, "value": str(dot(vs[i], vs[j]))})
    input_norms_sq = [str(dot(vs[k], vs[k])) for k in range(count)]
    input_orthogonal = all(d["value"] == "0" for d in input_dots)
    input_orthonormal = input_orthogonal and all(s == "1" for s in input_norms_sq)

    # -- the exact Gram-Schmidt walk (halts at the first zero residual) -----
    us = []
    steps = []
    halt_step = None
    for k in range(count):
        projections = []
        u = vs[k]
        for j, uj in enumerate(us):
            c = dot(vs[k], uj) / dot(uj, uj)
            projections.append({"j": j, "coeff": str(c)})
            u = u - c * uj
        norm_sq = dot(u, u)
        is_zero = norm_sq == 0
        steps.append({
            "k": k,
            "coefficients": projections,
            "u": [str(u[i]) for i in range(n)],
            "normSq": str(norm_sq),
            "isZero": bool(is_zero),
        })
        if is_zero:
            halt_step = k
            break
        us.append(u)

    pairwise_dots_u = []
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            pairwise_dots_u.append({"i": i, "j": j, "value": str(dot(us[i], us[j]))})

    # e_k = u_k / ||u_k|| kept SYMBOLIC — exact irrationals, SymPy's edge
    es = []
    for u in us:
        norm = sqrt(dot(u, u))
        es.append([str(simplify(u[i] / norm)) for i in range(n)])

    # -- ranks: full input, processed prefix, u-set --------------------------
    if count > 0:
        A = M.T  # n x k column matrix, column i is vector i
        input_rank = A.rank()
        prefix = Matrix.hstack(*[vs[i].T for i in range(len(us))]) if us else zeros(n, 0)
        processed_rank = prefix.rank() if us else 0
    u_rank = len(us)

    if halt_step is None:
        reason = None
    elif all(vs[halt_step][i] == 0 for i in range(n)):
        reason = (
            f"v{halt_step + 1} is the zero vector — it lies in every span, so "
            f"u{halt_step + 1} = 0 and the process halts"
        )
    else:
        reason = (
            f"u{halt_step + 1} = 0 exactly: v{halt_step + 1} was already in the "
            f"span of v1..v{halt_step}"
        )

    gram = M * M.T
    gram_det = gram.det() if count > 0 else S.Zero

    return {
        "provenance": provenance_for(exact),
        "dimension": n,
        "count": count,
        "inputDots": input_dots,
        "inputNormsSq": input_norms_sq,
        "inputOrthogonal": bool(input_orthogonal),
        "inputOrthonormal": bool(input_orthonormal),
        "steps": steps,
        "haltStep": halt_step,
        "dependentReason": reason,
        "us": [[str(u[i]) for i in range(n)] for u in us],
        "orthonormal": es,
        "pairwiseDotsU": pairwise_dots_u,
        "inputRank": input_rank,
        "processedRank": processed_rank,
        "uRank": u_rank,
        "spansMatch": processed_rank == u_rank,
        "gramDet": str(gram_det),
        "gramSingular": bool(gram_det == 0),
    }


def analyse_linsystems(rows):
    """rows: list[list[str]] — the AUGMENTED matrix [A|b], last column is b
    (same input shape as the JS engine's linsystemsEngine.js — see that file's
    docstring for the "why one matrix, not two" design note). Classifies the
    system via the Rouche-Capelli theorem, read off SymPy's OWN Matrix.rref()
    (an independent implementation from the JS engine's Gauss-Jordan sweep,
    mirroring analyse_rowreduce's own "deliberately not re-derived" contract):
      - a pivot landing in the b column itself (col index == nVars) means a
        row reduced to 0 = nonzero -> 'inconsistent' (rank(A) < rank([A|b]))
      - rank(A) == nVars -> 'unique' solution, read straight off the pivot
        rows' last column
      - rank(A) < nVars (and consistent) -> 'infinite' solutions: a
        particular solution (every free variable = 0) plus one nullspace
        basis vector per free column, exactly as linsystemsEngine.js's
        extractSolution() derives them (same formula, independent code path).

    Provenance: 'proved' when every entry parsed as an exact integer/fraction;
    any decimal entry flips the WHOLE system to 'numeric' (see parse_matrix).
    Never returns 'refuted' — there is no true/false claim to refute here,
    only a three-way classification. Abstains are not needed: parse_matrix
    raising ValueError on unparsable input propagates as an error, handled
    honestly by the worker/singleton (never silently promoted)."""
    M, exact = parse_matrix(rows)
    n_vars = M.cols - 1
    rref_m, pivots = M.rref()
    pivots = list(pivots)
    inconsistent = n_vars in pivots
    var_pivots = [p for p in pivots if p < n_vars]
    rank_a = len(var_pivots)
    rank_aug = len(pivots)

    solution = None
    if inconsistent:
        classification = "inconsistent"
    elif rank_a == n_vars:
        classification = "unique"
        x = [S.Zero] * n_vars
        for i, col in enumerate(var_pivots):
            x[col] = rref_m[i, n_vars]
        solution = {"x": [str(v) for v in x]}
    else:
        classification = "infinite"
        free_cols = [c for c in range(n_vars) if c not in var_pivots]
        particular = [S.Zero] * n_vars
        for i, col in enumerate(var_pivots):
            particular[col] = rref_m[i, n_vars]
        basis = []
        for f in free_cols:
            v = [S.Zero] * n_vars
            v[f] = S.One
            for i, col in enumerate(var_pivots):
                v[col] = -rref_m[i, f]
            basis.append([str(c) for c in v])
        solution = {
            "particular": [str(v) for v in particular],
            "basis": basis,
            "freeVarCols": free_cols,
        }

    is_invertible_case = classification == "unique" and M.rows == n_vars and rank_a == M.rows

    return {
        "provenance": provenance_for(exact),
        "nVars": n_vars,
        "rowsCount": M.rows,
        "rank": rank_a,
        "rankAugmented": rank_aug,
        "classification": classification,
        "solution": solution,
        "isInvertibleCase": is_invertible_case,
    }


def analyse_leastsquares(matrix, b):
    """matrix: list[list[str]] — A, m rows (equations/data points) x n columns
    (unknowns). b: list[str] — the length-m target vector (parse_matrix treats
    a bare list as a column matrix — the same convention analyse_vectorspaces'
    \`target\` and analyse_inverses' \`b\` use).

    Theorem: the LEAST-SQUARES solution to (generally inconsistent) Ax = b
    minimizes ‖Ax − b‖ and is characterized by the NORMAL EQUATIONS
    AᵀA x̂ = Aᵀb; when AᵀA is invertible (A's columns linearly independent),
    x̂ = (AᵀA)⁻¹Aᵀb is UNIQUE. Solved here via SymPy's OWN Matrix.solve() on
    AᵀA — an INDEPENDENT code path from the JS engine's Gauss-Jordan sweep on
    the augmented [AᵀA | Aᵀb] (rrefEngine.mjs's rrefTwoPhase). Cross-checks,
    LIVE on this input, never merely asserted:
      - the normal equations themselves: AᵀA x̂ = Aᵀb, entrywise
      - the money-shot geometric claim: residual = b − Ax̂ is ORTHOGONAL to
        EVERY column of A (residual · columnⱼ = 0 for every j)

    Singular case (det(AᵀA) = 0, equivalently A's columns are linearly
    dependent): reports independent = False with the rank and the first
    column of AᵀA carrying no pivot in SymPy's own rref (mirrors
    analyse_inverses' failedAtCol convention) — there is no unique x̂, so
    xHat/p/residual all stay None. No 'refuted' verdict: H2 (AᵀA invertible)
    is simply met or unmet, certified exactly for exact input — the same
    iff-hypothesis shape as analyse_inverses' det(A) != 0.

    Provenance: 'proved' when every entry of BOTH A and b parsed exact
    (rational/integer) — the normal-equations solve, the projection, and the
    orthogonality certificate are then all exact rational facts; 'numeric'
    the moment any entry (in A or b) is a decimal. Abstains to 'unknown' only
    when b's length does not match A's row count."""
    M, exact_m = parse_matrix(matrix)
    bM, exact_b = parse_matrix(b)
    exact = exact_m and exact_b
    m, n = M.rows, M.cols
    if bM.rows != m:
        return {"provenance": "unknown", "note": f"b must have length {m} to match A's {m} rows"}

    AT = M.T
    ATA = simplify(AT * M)
    ATb = simplify(AT * bM)
    det_ata = simplify(ATA.det())
    rank = M.rank()
    independent = rank == n

    result = {
        "provenance": provenance_for(exact),
        "m": m,
        "n": n,
        "det": str(det_ata),
        "rank": int(rank),
        "independent": bool(independent),
        "xHat": None,
        "p": None,
        "residual": None,
        "residualZero": None,
        "normalEquationHolds": None,
        "orthogonalityChecks": None,
        "orthogonalityHolds": None,
        "singularReason": None,
    }

    if not independent:
        _, pivot_tuple = ATA.rref()
        pivot_cols = set(pivot_tuple)
        failed_at = next((c for c in range(n) if c not in pivot_cols), None)
        result["singularReason"] = (
            f"det(AᵀA) = 0 (rank {rank} < {n}): the columns of A are linearly dependent — "
            f"no unique least-squares solution"
            + (f" (column {failed_at} of AᵀA carries no pivot)" if failed_at is not None else "")
        )
        return result

    xhat = simplify(ATA.solve(ATb))
    p = simplify(M * xhat)
    residual = simplify(bM - p)
    residual_zero = all(simplify(residual[i, 0]) == 0 for i in range(m))

    ortho_checks = []
    orth_holds = True
    for j in range(n):
        col = M[:, j]
        d = simplify((residual.T * col)[0, 0])
        is_zero = d == 0
        orth_holds = orth_holds and is_zero
        ortho_checks.append({"j": j, "value": str(d), "isZero": bool(is_zero)})

    normal_eq_holds = all(
        simplify((ATA * xhat)[i, 0] - ATb[i, 0]) == 0 for i in range(n)
    )

    result["xHat"] = [str(xhat[i, 0]) for i in range(n)]
    result["p"] = [str(p[i, 0]) for i in range(m)]
    result["residual"] = [str(residual[i, 0]) for i in range(m)]
    result["residualZero"] = bool(residual_zero)
    result["normalEquationHolds"] = bool(normal_eq_holds)
    result["orthogonalityChecks"] = ortho_checks
    result["orthogonalityHolds"] = bool(orth_holds)
    return result


def analyse_spectral(matrix):
    """2×2 or 3×3 real matrix: the Spectral Theorem — a REAL SYMMETRIC matrix A
    (Aᵀ = A) is orthogonally diagonalizable, A = QDQᵀ with Q orthogonal
    (QᵀQ = I, columns an orthonormal eigenbasis) and D diagonal (the real
    eigenvalues). Built on SymPy's OWN Matrix.eigenvects() — an INDEPENDENT
    implementation from the JS engine's quadratic-formula/nullspace code
    (spectralEngine.js, which itself reuses eigenEngine.js/orthogonalityEngine.js)
    — used as this module's symbolic second opinion, not a re-derivation. Where
    an eigenvalue's eigenspace has geometric multiplicity > 1, orthogonalizes
    it with the SAME hand-rolled exact Gram-Schmidt walk as analyse_orthogonality
    (never Matrix.GramSchmidt, which raises on dependent input rather than
    reporting the halt step — not needed here since eigenvects() already
    returns a linearly independent basis per eigenspace, but the walk itself is
    reused for consistency and because it is the one that keeps intermediate
    dot-product certificates around for the UI).

    Two guarantees a general (non-symmetric) matrix does NOT have, both
    verified LIVE on this input rather than merely asserted:
      1. every eigenvalue is real (a genuine complex pair is possible for a
         non-symmetric matrix; provably impossible for a symmetric one)
      2. eigenvectors from DIFFERENT eigenvalues are automatically orthogonal
         (crossEigenvalueOrthogonal) — checked via a direct dot-product on the
         (now orthonormal) eigenbasis, not assumed from symmetry.

    Provenance mirrors analyse_eigen exactly: 'proved' requires BOTH exact
    (rational) input AND every eigenvalue exactly rational (SymPy returns
    Rational/Integer, not an unevaluated sqrt/Pow expression) — an exact input
    whose eigenvalues are genuinely irrational abstains to 'numeric' rather
    than attempting a heavy nested-radical Q/QᵀQ/A=QDQᵀ simplify that could be
    slow or produce an unsimplified (but truthful) non-zero residual; this is
    an honest abstention per CLAUDE.md's non-negotiable #2, not a bug. Float
    input abstains the same way (and skips eigenvects() entirely — the known
    trap of eigenvects() on a defective float matrix mis-reporting
    multiplicity or hanging near repeated eigenvalues) — never silently
    promoted. Complex eigenvalues and defective (geometric < algebraic
    multiplicity) matrices are exact 'proved' verdicts that the theorem's
    conclusion (spectralTheoremHolds) is FALSE, since 'no real eigenbasis
    exists at all' is itself an exact fact for rational input."""
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_spectral expects a 2×2 or 3×3 square matrix"}
    n = M.rows
    is_symmetric = simplify(M - M.T) == zeros(n, n)

    if not exact:
        # Float input: eigenvects() on a defective float matrix can mis-report
        # multiplicity or hang near repeated eigenvalues — use eigenvals() only.
        try:
            evals = M.eigenvals()
        except Exception as e:
            return {"provenance": "unknown", "note": f"eigenvals() failed on float input: {e}"}
        has_complex = any(not ev.is_real for ev in evals.keys())
        kind = "complex" if has_complex else ("repeated" if len(evals) < n else "distinct")
        return {
            "provenance": "numeric",
            "dim": n,
            "isSymmetric": bool(is_symmetric),
            "kind": kind,
            "eigenvalues": [str(ev) for ev in evals.keys()],
            "allEigenvaluesReal": not has_complex,
            "spectralTheoremHolds": None,
            "note": "float input — eigen decomposition is tolerance-blind here; treat as approximate, not certified",
        }

    # Exact (rational) input from here on.
    try:
        evs = M.eigenvects()  # [(eigenvalue, alg_mult, [eigenvectors])]
    except Exception as e:
        return {"provenance": "unknown", "note": f"eigenvects() failed: {e}"}

    has_complex = any(not ev[0].is_real for ev in evs)
    if has_complex:
        return {
            "provenance": "proved",
            "dim": n,
            "isSymmetric": bool(is_symmetric),
            "kind": "complex",
            "eigenvalues": [str(ev[0]) for ev in evs],
            "allEigenvaluesReal": False,
            "fullEigenbasis": False,
            "spectralTheoremHolds": False,
            "reason": "Eigenvalues are complex — no real eigenvectors exist, so no real orthogonal diagonalization is possible."
            + (" (This cannot happen for a genuinely symmetric matrix — the Spectral Theorem guarantees real eigenvalues.)" if is_symmetric else ""),
        }

    any_defective = any(ev[1] > len(ev[2]) for ev in evs)
    if any_defective:
        defective = next(ev for ev in evs if ev[1] > len(ev[2]))
        return {
            "provenance": "proved",
            "dim": n,
            "isSymmetric": bool(is_symmetric),
            "kind": "defective",
            "eigenvalues": [str(ev[0]) for ev in evs],
            "allEigenvaluesReal": True,
            "fullEigenbasis": False,
            "spectralTheoremHolds": False,
            "reason": f"eigenvalue {defective[0]} has geometric multiplicity {len(defective[2])} < algebraic multiplicity {defective[1]} (DEFECTIVE) — no eigenbasis at all can be built."
            + (" (This cannot happen for a genuinely symmetric matrix — the Spectral Theorem guarantees full eigenspaces.)" if is_symmetric else ""),
        }

    all_rational = all(ev[0].is_rational for ev in evs)
    if not all_rational:
        kind = "repeated" if any(ev[1] > 1 for ev in evs) else "distinct"
        return {
            "provenance": "numeric",
            "dim": n,
            "isSymmetric": bool(is_symmetric),
            "kind": kind,
            "eigenvalues": [str(ev[0]) for ev in evs],
            "allEigenvaluesReal": True,
            "fullEigenbasis": True,
            "spectralTheoremHolds": None,
            "note": "eigenvalues are irrational (unevaluated radical) — exact Q construction abstained; verify numerically instead",
        }

    # Exact rational eigenvalues, full eigenbasis (algebraic == geometric for
    # every eigenvalue). Build the orthonormal eigenbasis: within any
    # eigenspace of dimension > 1, orthogonalize with the same hand-rolled
    # exact Gram-Schmidt walk analyse_orthogonality uses (never
    # Matrix.GramSchmidt, which raises rather than reports the halt step —
    # not load-bearing here since eigenvects() already returns a linearly
    # independent basis, but kept for one consistent exact-walk code path).
    def dot(a, b):
        return sum(a[i] * b[i] for i in range(n))

    q_cols = []  # [(lambda, orthonormal eigenvector)]
    gram_schmidt_used = False
    for ev, _mult, vs in evs:
        if len(vs) > 1:
            gram_schmidt_used = True
            us = []
            for v in vs:
                u = v
                for uj in us:
                    c = dot(v, uj) / dot(uj, uj)
                    u = u - c * uj
                us.append(u)
            basis = us
        else:
            basis = vs
        for u in basis:
            norm = sqrt(dot(u, u))
            e = simplify(u / norm)
            q_cols.append((ev, e))

    kind = "repeated" if any(m > 1 for _, m, _ in evs) else "distinct"
    full_eigenbasis = len(q_cols) == n

    Qm = Matrix.hstack(*[col for _, col in q_cols])
    Dm = diag(*[lam for lam, _ in q_cols])
    Qt = Qm.T
    qtq_diff = simplify(Qt * Qm - eye(n))
    qtq_identity = qtq_diff == zeros(n, n)
    recon_diff = simplify(Qm * Dm * Qt - M)
    reconstructs = recon_diff == zeros(n, n)

    pairwise_dots = []
    for i in range(len(q_cols)):
        for j in range(i + 1, len(q_cols)):
            same_eigenvalue = simplify(q_cols[i][0] - q_cols[j][0]) == 0
            value = simplify(dot(q_cols[i][1], q_cols[j][1]))
            pairwise_dots.append({"i": i, "j": j, "sameEigenvalue": bool(same_eigenvalue), "value": str(value)})
    cross_eigenvalue_orthogonal = all(
        d["value"] == "0" for d in pairwise_dots if not d["sameEigenvalue"]
    )

    orthogonal_diagonalizable = bool(cross_eigenvalue_orthogonal and qtq_identity and reconstructs)
    spectral_theorem_holds = bool(is_symmetric) and orthogonal_diagonalizable

    return {
        "provenance": "proved",
        "dim": n,
        "isSymmetric": bool(is_symmetric),
        "kind": kind,
        "eigenvalues": [str(ev) for ev, _, _ in evs],
        "allEigenvaluesReal": True,
        "fullEigenbasis": full_eigenbasis,
        "gramSchmidtUsed": gram_schmidt_used,
        "Q": [[str(Qm[i, j]) for j in range(Qm.cols)] for i in range(Qm.rows)],
        "D": [str(lam) for lam, _ in q_cols],
        "qtqIdentity": bool(qtq_identity),
        "reconstructs": bool(reconstructs),
        "pairwiseDots": pairwise_dots,
        "crossEigenvalueOrthogonal": bool(cross_eigenvalue_orthogonal),
        "orthogonalDiagonalizable": orthogonal_diagonalizable,
        "spectralTheoremHolds": spectral_theorem_holds,
    }


def analyse_svd(matrix):
    """rows: list[list[str]], m ∈ {2,3} rows, n ∈ {2,3} columns. CONCEPT module
    (unconditional — every real matrix has an SVD, no hypothesis to break):
    A = UΣVᵀ, U (m×m) and V (n×n) orthogonal, Σ (m×n) diagonal with singular
    values σ₁≥σ₂≥…≥0. Built directly on the Spectral Theorem applied to AᵀA
    (ALWAYS symmetric, positive semi-definite) — the same exact-Gram-Schmidt-
    within-a-repeated-eigenspace machinery analyse_spectral uses, applied
    here to AᵀA instead of A itself; SVD is the direct structural sequel to
    the Spectral Theorem (same relationship spectralEngine.js/svdEngine.js
    have on the JS side). Cross-checked independently against SymPy's OWN
    \`Matrix.singular_value_decomposition()\`/\`singular_values()\` in the
    selftest (a genuinely different SymPy code path, not this function's own
    arithmetic grading itself).

    parse_matrix nsimplifies every entry (decimal or not) to an exact
    Rational before this function ever sees it (see parse_matrix's
    docstring) — so unlike analyse_spectral's own float branch, there is no
    separate "cheap path to dodge a literal float matrix" here: the ONLY
    extra cost a genuinely irrational AᵀA eigenvalue can cause is a nested-
    radical simplify, which is exactly the case short-circuited below
    (return early on \`not all_rational\`, mirroring analyse_spectral's own
    "abstain rather than pay for a heavy nested-radical simplify" choice).

    Provenance: a singular value σᵢ = √λᵢ for an eigenvalue λᵢ of AᵀA is
    generically IRRATIONAL even when λᵢ itself is an exact rational (e.g.
    λ=2 ⟹ σ=√2) — so 'proved' requires BOTH exact (rational) input AND
    every σᵢ ITSELF exactly rational (sqrt(λᵢ).is_rational), not merely
    every λᵢ rational. This is a strictly MORE PRECISE test than the JS
    engine's own (documented, deliberately conservative) integer-perfect-
    square-only check — svdEngine.js avoids reconstructing a Fraction from
    a float λ (the nsimplify "nice rational" trap CLAUDE.md warns about) by
    restricting its own 'proved' tier to integer λ; this symbolic tier has
    no such float-reconstruction risk (λ is already an exact sympy Rational
    throughout), so it can safely recognise λ=9/4 ⟹ σ=3/2 as exact where the
    JS engine's conservative rule reports 'numeric' for the same case. This
    is a deliberate, documented asymmetry between the two backends (see
    CLAUDE.md: "do not assume they'll fail identically"), not a bug —
    neither ever promotes a genuinely-irrational or float result to
    'proved'."""
    M, exact = parse_matrix(matrix)
    m, n = M.rows, M.cols
    if m not in (2, 3) or n not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_svd expects 2 or 3 rows and 2 or 3 columns"}
    rmin = min(m, n)
    AtA = simplify(M.T * M)

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(len(a)))

    try:
        evs = AtA.eigenvects()  # [(eigenvalue, alg_mult, [eigenvectors])]
    except Exception as e:
        return {"provenance": "unknown", "m": m, "n": n, "note": f"eigenvects() on AᵀA failed: {e}"}

    has_complex = any(not ev[0].is_real for ev in evs)
    if has_complex:
        return {
            "provenance": "unknown", "m": m, "n": n,
            "note": "AᵀA reported a complex eigenvalue — impossible for a genuinely symmetric matrix; engine abstains.",
        }
    any_defective = any(ev[1] > len(ev[2]) for ev in evs)
    if any_defective:
        return {
            "provenance": "unknown", "m": m, "n": n,
            "note": "AᵀA reported a defective eigenspace — impossible for a genuinely symmetric matrix; engine abstains.",
        }

    q_cols = []  # [(lambda, orthonormal eigenvector)] — same hand-rolled exact
    # Gram-Schmidt walk analyse_spectral uses within a repeated eigenspace.
    gram_schmidt_used = False
    for ev, _mult, vs in evs:
        if len(vs) > 1:
            gram_schmidt_used = True
            us = []
            for v in vs:
                u = v
                for uj in us:
                    c = dot(v, uj) / dot(uj, uj)
                    u = u - c * uj
                us.append(u)
            basis = us
        else:
            basis = vs
        for u in basis:
            norm = sqrt(dot(u, u))
            e = simplify(u / norm)
            q_cols.append((ev, e))

    if len(q_cols) != n:
        return {
            "provenance": "unknown", "m": m, "n": n,
            "note": "AᵀA did not yield a full eigenbasis — impossible for a genuinely symmetric matrix; engine abstains.",
        }

    all_rational = all(lam.is_rational for lam, _ in q_cols)
    if not all_rational:
        sorted_lams = sorted(q_cols, key=lambda p: p[0], reverse=True)
        return {
            "provenance": "numeric",
            "m": m, "n": n,
            "sigmas": [str(simplify(sqrt(lam))) for lam, _ in sorted_lams],
            "allEigenvaluesRational": False,
            "note": "AᵀA's eigenvalues are irrational (unevaluated radical) — exact U/Σ/V construction abstained; verify numerically instead",
        }

    q_cols.sort(key=lambda p: p[0], reverse=True)  # descending eigenvalue -> descending singular value
    sigmas = [simplify(sqrt(lam)) for lam, _ in q_cols]
    all_sigma_rational = all(s.is_rational for s in sigmas)
    provenance = "proved" if (exact and all_sigma_rational) else "numeric"

    Vm = Matrix.hstack(*[col for _, col in q_cols])

    u_cols = [None] * rmin
    for i in range(rmin):
        if simplify(sigmas[i]) != 0:
            u_cols[i] = simplify((M * q_cols[i][1]) / sigmas[i])

    def complete_basis(existing, dim):
        """Gram-Schmidt-extend \`existing\` (already orthonormal) to a full
        dim-dimensional orthonormal basis by projecting the standard basis
        against it and keeping whichever candidates survive with nonzero
        residual — SVD's own "complete U past the nonzero singular values"
        step (mirrors svdEngine.js's completeOrthonormalBasis exactly);
        distinct from analyse_orthogonality's gramSchmidt, which halts
        entirely at the first dependent vector by design."""
        vecs = list(existing)
        for e in range(dim):
            if len(vecs) >= dim:
                break
            cand = zeros(dim, 1)
            cand[e, 0] = 1
            u = cand
            for v in vecs:
                c = dot(u, v)
                u = u - c * v
            nsq = simplify(dot(u, u))
            if nsq != 0:
                vecs.append(simplify(u / sqrt(nsq)))
        return vecs

    known_u = [u for u in u_cols if u is not None]
    completed = complete_basis(known_u, m)
    ci = len(known_u)
    final_u = []
    for u in u_cols:
        if u is not None:
            final_u.append(u)
        else:
            final_u.append(completed[ci]); ci += 1
    while len(final_u) < m:
        final_u.append(completed[ci]); ci += 1

    Um = Matrix.hstack(*final_u)
    Sigma = zeros(m, n)
    for i in range(rmin):
        Sigma[i, i] = sigmas[i]

    UtU = simplify(Um.T * Um)
    VtV = simplify(Vm.T * Vm)
    recon = simplify(Um * Sigma * Vm.T)
    u_orthogonal = UtU == eye(m)
    v_orthogonal = VtV == eye(n)
    reconstructs = recon == M

    rank = sum(1 for s in sigmas if simplify(s) != 0)
    is_symmetric_input = (m == n) and simplify(M - M.T) == zeros(n, n)
    is_orthogonal_input = (m == n) and simplify(M.T * M - eye(n)) == zeros(n, n)

    return {
        "provenance": provenance,
        "m": m, "n": n,
        "sigmas": [str(s) for s in sigmas],
        "rank": rank,
        "U": [[str(Um[i, j]) for j in range(Um.cols)] for i in range(Um.rows)],
        "V": [[str(Vm[i, j]) for j in range(Vm.cols)] for i in range(Vm.rows)],
        "gramSchmidtUsed": gram_schmidt_used,
        "uOrthogonal": bool(u_orthogonal),
        "vOrthogonal": bool(v_orthogonal),
        "reconstructs": bool(reconstructs),
        "svdHolds": bool(u_orthogonal and v_orthogonal and reconstructs),
        "isSymmetricInput": bool(is_symmetric_input),
        "isOrthogonalInput": bool(is_orthogonal_input),
    }


def analyse_changeofbasis(basis, vector, other_basis=None):
    """basis: list[list[str]] — n vectors (each n components), the SAME
    row-per-vector convention as analyse_vectorspaces. vector: list[str] — the
    target v, in STANDARD coordinates. other_basis: optional list[list[str]],
    a second basis C, for the transition-matrix case.

    Independently recomputes, via SymPy's own Matrix.rank()/.det()/.inv()
    (never re-derived from the JS engine's RREF-solve + adjugate-inverse
    cross-check):
      - P_B (columns = basis vectors), det(P_B), rank(P_B) -> isBasis iff
        det(P_B) != 0 (equivalently rank == dimension, for n vectors in R^n)
      - when B is a genuine basis: [v]_B = P_B^{-1} v (SymPy's own .inv()),
        plus the live reconstruction P_B[v]_B == v
      - when other_basis (C) is supplied AND both B and C are genuine bases:
        the transition matrix P_{C<-B} = P_C^{-1} P_B, and [v]_C computed TWO
        ways — directly (P_C^{-1} v) and via the transition matrix
        (P_{C<-B} [v]_B) — cross-checked as transitionHolds

    THE GENUINE BREAKABLE CONDITION (mirrors analyse_orthogonality's halt):
    a dependent candidate is not a basis, so [v]_B does not exist — the
    result reports isBasis: False with the rank certificate rather than a
    bogus coordinate vector. No 'refuted' verdict: "is this set a basis?" is
    a definition being satisfied or not, not a theorem being disproved.

    Provenance: 'proved' iff every basis entry AND the target vector's
    entries (and, when supplied, every entry of the second basis) parsed as
    an exact integer/fraction — a single decimal anywhere flips the WHOLE
    computation to 'numeric' (parse_matrix's whole-matrix provenance rule,
    applied here across basis + vector + other_basis together). Only
    abstains to 'unknown' for a malformed shape (wrong vector count/
    dimension mismatch) — never guesses."""
    M, exact = parse_matrix(basis)  # rows = basis vectors
    n_vecs = M.rows
    dim = M.cols
    if n_vecs != dim or dim not in (2, 3):
        return {
            "provenance": "unknown",
            "note": f"analyse_changeofbasis expects {dim if dim in (2, 3) else 'a 2- or 3-dimensional'} basis vectors (got {n_vecs} vectors of dimension {dim})",
        }
    P = M.T  # columns = basis vectors
    det_p = simplify(P.det())
    rank_p = P.rank()
    is_basis = simplify(det_p) != 0

    vM, vexact = parse_matrix(vector)  # column vector, dim x 1
    exact_all = exact and vexact

    result = {
        "dimension": dim,
        "detP": str(det_p),
        "rank": int(rank_p),
        "isBasis": bool(is_basis),
        "transition": None,
    }

    if not is_basis:
        result["provenance"] = provenance_for(exact_all)
        result["note"] = (
            f"The given vectors are linearly DEPENDENT (rank {rank_p} < {dim}) — "
            f"not a basis; P_B is singular, so [v]_B does not exist."
        )
        return result

    if vM.rows != dim:
        return {"provenance": "unknown", "note": "target vector dimension does not match the basis's ambient dimension"}

    p_inv = simplify(P.inv())
    coords = simplify(p_inv * vM)
    reconstructs = simplify(P * coords - vM) == zeros(dim, 1)

    result["coordinates"] = [str(coords[i]) for i in range(dim)]
    result["reconstructs"] = bool(reconstructs)
    result["provenance"] = provenance_for(exact_all)

    if other_basis is not None:
        C, cexact = parse_matrix(other_basis)
        if C.rows != dim or C.cols != dim:
            result["transition"] = {"provenance": "unknown", "note": "second basis has the wrong shape for this dimension"}
            return result
        Pc = C.T
        det_c = simplify(Pc.det())
        is_basis_c = simplify(det_c) != 0
        exact_t = exact_all and cexact
        if not is_basis_c:
            rank_c = Pc.rank()
            result["transition"] = {
                "provenance": provenance_for(exact_t),
                "isBasis": False,
                "note": f"The second basis C is linearly DEPENDENT (rank {rank_c} < {dim}) — not a basis; P_C is singular.",
            }
        else:
            pc_inv = simplify(Pc.inv())
            p_c_b = simplify(pc_inv * P)  # P_{C<-B}
            coords_c_direct = simplify(pc_inv * vM)
            coords_c_via = simplify(p_c_b * coords)
            transition_holds = simplify(coords_c_direct - coords_c_via) == zeros(dim, 1)
            result["transition"] = {
                "provenance": provenance_for(exact_t),
                "isBasis": True,
                "PcB": [[str(p_c_b[i, j]) for j in range(dim)] for i in range(dim)],
                "coordsC": [str(coords_c_direct[i]) for i in range(dim)],
                "transitionHolds": bool(transition_holds),
            }

    return result


def _qf_level_kind(signs, dim):
    """Classify the level set q(x) = 1 (as opposed to q(x) = classification,
    which is a different question) from the SIGN multiset of the symmetric
    S's eigenvalues — mirrors quadraticformsEngine.js's classifyLevelSet
    exactly (same kind vocabulary), reimplemented natively in Python rather
    than imported, per this file's own-code-path convention (analyse_svd/
    analyse_spectral never import JS; every analyse_* is a from-scratch
    SymPy computation used as an INDEPENDENT second opinion)."""
    pos = signs.count("+")
    neg = signs.count("-")
    zero = len(signs) - pos - neg
    if pos == 0:
        return "empty", False
    if neg == 0:
        if zero == 0:
            return ("ellipse" if dim == 2 else "ellipsoid"), True
        if pos == 1 and zero == dim - 1:
            return ("twoParallelLines" if dim == 2 else "twoParallelPlanes"), True
        return ("twoParallelLines" if dim == 2 else "ellipticCylinder"), True
    if zero == 0:
        if dim == 2:
            return "hyperbola", True
        return ("hyperboloidOneSheet" if pos == 2 else "hyperboloidTwoSheets"), True
    return "hyperbolicCylinder", True


def analyse_quadraticforms(matrix):
    """2×2 or 3×3 real matrix A: the CONCEPT of a quadratic form q(x) = xᵀAx
    and its definiteness classification (positive definite / positive
    semidefinite / indefinite / negative semidefinite / negative definite).
    CONCEPT module (unconditional given a symmetric matrix — like svd/
    fundspaces' Rank-Nullity, there is no theorem promise to verify once the
    input is symmetric; see the Linear-algebra addendum's Rank-Nullity
    exception in .claude/skills/math-module/SKILL.md).

    SYMMETRIZE-AND-SAY: only A's symmetric part S = (A+Aᵀ)/2 ever appears in
    q, because xᵀ(A−Aᵀ)/2·x ≡ 0 for every x (the skew part swaps each xᵢxⱼ
    term against its negative). A non-symmetric input is therefore analyzed
    as S, and the skew part / the fact that it was symmetrized is reported —
    never silently discarded, never silently accepted as-is.

    TWO INDEPENDENT SymPy code paths, cross-checked (mirrors
    quadraticformsEngine.js's own eigen-vs-Sylvester cross-check, done here
    with SymPy's OWN eigenvects()/det() rather than reused from the JS side —
    same "independent second opinion" relationship every analyse_* in this
    file has to its JS sibling):
      1. EIGEN-SIGNS: S.eigenvects() — the principal-axes theorem rewrites
         q(x) = Σλᵢyᵢ², so the eigenvalue signs ARE the classification.
      2. SYLVESTER'S CRITERION: the n leading principal minors Δ₁..Δₙ of S,
         via SymPy's own .det() on each leading block — pure polynomial
         arithmetic, exact whenever S's entries are exact rationals
         regardless of whether the eigenvalues themselves are rational. All
         Δₖ > 0 ⟺ positive definite; when every Δₖ ≠ 0, the sign changes in
         (1, Δ₁, …, Δₙ) count the negative eigenvalues exactly (Jacobi's
         inertia theorem, LDLᵀ pivots) — cross-checked against path 1's own
         negative-eigenvalue count.
      A disagreement between the two EXACT paths is an engine-bug guard
      (mirrors changeofbasis's cross-check pattern): demoted to 'unknown'
      rather than picking a side.

    Provenance: 'proved' requires BOTH exact (rational) input AND every
    eigenvalue of S exactly rational (SymPy returns Rational/Integer, not an
    unevaluated radical) — a 2×2 with irrational eigenvalues (e.g.
    [[5,2],[2,1]]) is 'numeric' even on integer input, exactly like
    analyse_eigen/analyse_spectral's own documented rule. The classification
    ITSELF is still corroborated by the exact Sylvester minors regardless of
    eigenvalue rationality (minors are pure +,-,× — no root extraction), so
    a 'numeric' verdict still carries an exact-minor cross-check whenever the
    input itself was exact; that corroboration is surfaced via \`sylvester\`.
    Unlike analyse_spectral's own float branch, eigenvects() is always run
    here (even on float-nsimplify'd input, exactly as analyse_svd does):
    there is no "hang near a repeated eigenvalue" trap for THIS matrix,
    because S=(A+Aᵀ)/2 is always genuinely symmetric, guaranteeing real
    eigenvalues and a full eigenbasis by the Spectral Theorem regardless of
    whether the original A was. 'numeric' provenance is assigned whenever the
    original input carried a decimal or an eigenvalue came out irrational —
    never promoted to 'proved'."""
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_quadraticforms expects a 2×2 or 3×3 square matrix"}
    n = M.rows

    skew = simplify((M - M.T) / 2)
    is_symmetric = skew == zeros(n, n)
    S = simplify((M + M.T) / 2)

    try:
        evs = S.eigenvects()  # [(eigenvalue, alg_mult, [eigenvectors])] — S is
        # always genuinely symmetric, so this can never hang/mis-report the
        # way analyse_spectral's float-defective-matrix trap warns about.
    except Exception as e:
        return {"provenance": "unknown", "note": f"eigenvects() on S=(A+Aᵀ)/2 failed: {e}"}

    has_complex = any(not ev[0].is_real for ev in evs)
    any_defective = any(ev[1] > len(ev[2]) for ev in evs)
    if has_complex or any_defective:
        # Cannot happen for a genuinely symmetric S — the Spectral Theorem
        # guarantees real eigenvalues and a full eigenbasis. Abstain rather
        # than fabricate a classification.
        return {
            "provenance": "unknown",
            "note": "S=(A+Aᵀ)/2 failed to produce a full real eigenbasis — impossible for a symmetric matrix; engine abstains.",
        }

    signs = []
    eigenvalues = []
    all_rational = True
    for ev, mult, _vs in evs:
        if not ev.is_rational:
            all_rational = False
        s = "0" if ev == 0 else ("+" if ev.is_positive else "-")
        for _ in range(mult):
            signs.append(s)
            eigenvalues.append({"value": str(ev), "sign": s})

    pos = signs.count("+")
    neg = signs.count("-")
    zero = signs.count("0")
    if pos == len(signs):
        classification = "positiveDefinite"
    elif neg == len(signs):
        classification = "negativeDefinite"
    elif pos > 0 and neg > 0:
        classification = "indefinite"
    elif pos > 0 and zero > 0:
        classification = "positiveSemidefinite"
    elif neg > 0 and zero > 0:
        classification = "negativeSemidefinite"
    else:
        classification = "positiveSemidefinite"  # the zero form (also NSD)
    zero_form = pos == 0 and neg == 0

    # Sylvester's leading principal minors — an INDEPENDENT SymPy code path
    # (Matrix.det() on each leading block), not derived from evs above.
    minors = [simplify(S[:k, :k].det()) for k in range(1, n + 1)]
    says_pd = all(m > 0 for m in minors)
    all_nonzero = all(m != 0 for m in minors)
    all_nonnegative = all(m >= 0 for m in minors)
    if all_nonzero:
        seq = [1] + [1 if m > 0 else -1 for m in minors]
        neg_from_minors = sum(1 for i in range(1, len(seq)) if seq[i - 1] * seq[i] < 0)
        inertia = {"neg": neg_from_minors, "pos": n - neg_from_minors}
    else:
        inertia = None
    rank = S.rank()
    nullity = n - rank
    # Sylvester's criterion is a PD test only: all leading minors >= 0 does
    # NOT imply PSD (diag(0,-1): Δ1=Δ2=0, q<=0) — flag that trap live.
    psd_trap = all_nonnegative and classification in (
        "indefinite", "negativeSemidefinite", "negativeDefinite",
    )

    agrees = says_pd == (classification == "positiveDefinite")
    if inertia is not None:
        agrees = agrees and inertia["neg"] == neg and inertia["pos"] == pos
    else:
        agrees = agrees and nullity == zero

    provenance = "proved" if (exact and all_rational) else "numeric"
    if not agrees and exact and all_rational:
        provenance = "unknown"  # two exact paths disagree — engine-bug guard

    level_kind, level_nonempty = _qf_level_kind(signs, n)
    neg_signs = ["-" if s == "+" else "+" if s == "-" else "0" for s in signs]
    neg_level_kind, _neg_nonempty = _qf_level_kind(neg_signs, n)

    result = {
        "provenance": provenance,
        "dim": n,
        "isSymmetric": bool(is_symmetric),
        "symmetrized": not bool(is_symmetric),
        "classification": classification,
        "zeroForm": bool(zero_form),
        "eigenvalues": eigenvalues,
        "allEigenvaluesRational": bool(all_rational),
        "sylvester": {
            "minors": [str(m) for m in minors],
            "saysPD": bool(says_pd),
            "allNonzero": bool(all_nonzero),
            "allNonnegative": bool(all_nonnegative),
            "inertia": inertia,
            "rank": int(rank),
            "nullity": int(nullity),
            "agrees": bool(agrees),
            "psdTrap": bool(psd_trap),
        },
        "levelKind": level_kind,
        "levelNonempty": bool(level_nonempty),
        "negLevelKind": neg_level_kind,
    }
    if not is_symmetric:
        result["skew"] = [[str(skew[i, j]) for j in range(n)] for i in range(n)]
        result["note"] = (
            "Input A is not symmetric — classified S = (A+Aᵀ)/2 instead; "
            "the skew part (A−Aᵀ)/2 contributes exactly 0 to xᵀAx for every x."
        )
    return result

    return result


def analyse_factorizations(matrix, mode):
    """2×2 or 3×3 real SQUARE matrix A. mode: 'lu' or 'qr' — one payload, two
    independent factorizations behind the JS module's own mode toggle (same
    two-branch shape as inverses' Gauss-Jordan/Cramer split).

    LU (THEOREM-shaped, genuine breakable hypothesis): A = LU, L unit-lower-
    triangular, U upper-triangular, via SymPy's OWN \`Matrix.LUdecomposition()\`
    — a genuinely different code path from the JS engine's rrefTwoPhase-built
    L/U (SymPy's own pivoting internals, not reused, not re-derived). H1 ("no
    pivoting required") is the MATHEMATICAL criterion — every leading
    principal minor D_1,…,D_{n-1} nonzero — computed here via SymPy's own
    \`.det()\` on each leading block (pure polynomial arithmetic, exact
    whenever the input is exact), independent of whichever pivot path
    LUdecomposition() itself happens to choose (see factorizationsEngine.js's
    header for why "the hypothesis" and "did this concrete run swap" are
    DELIBERATELY kept as two separate, honestly-reported facts rather than
    conflated — SymPy's own pivoting strategy need not agree with the JS
    engine's, and neither is "more correct": both are valid Gaussian-
    elimination pivot choices for a genuinely singular leading minor).
    \`perm\` is SymPy's own permutation (as row-swap pairs) converted to the
    same "final row i came from original row perm[i]" convention the JS
    engine uses, so PA = LU is checked identically. Never 'refuted' — H1 is
    simply met or unmet, certified exactly.

    QR (CONCEPT-shaped, unconditional given independent columns): a
    hand-rolled EXACT Gram-Schmidt walk on A's columns (SymPy's own
    \`Matrix.QRdecomposition()\` does NOT halt on dependent columns — it
    silently returns a reduced-rank Q/R with the dependent column dropped,
    which is the wrong shape for this module's "does a square, invertible-R
    QR exist" question — confirmed directly against SymPy on
    [[1,2],[2,4]]: QRdecomposition() returns a 2×1 Q / 1×2 R instead of
    halting, so this function walks Gram-Schmidt itself, exactly mirroring
    analyse_orthogonality's own documented reason for the same choice). Q's
    entries are kept SYMBOLIC (exact irrational expressions like
    'sqrt(2)/2') — the one thing the JS engine's float normalization cannot
    do — and R = simplify(Qᵀ·A), upper-triangular by construction (same
    argument as factorizationsEngine.js's own R-construction comment).

    Provenance:
      LU — 'proved' for exact (rational) input (pure +,−,×,÷ throughout);
           'numeric' for any decimal entry.
      QR — 'proved' requires BOTH exact input AND every ‖u_i‖² a PERFECT
           SQUARE RATIONAL (sqrt(normSq).is_rational) — strictly MORE
           PRECISE than the JS engine's own conservative integer-perfect-
           square-only carve-out (svdEngine.js's documented asymmetry
           pattern: SymPy never risks the nsimplify "nice rational" trap
           here because normSq is already an exact Rational throughout, so
           it can recognise e.g. normSq=9/4 -> norm=3/2 as exact where the
           JS engine's conservative rule would report 'numeric' for the same
           case). Never promoted past this table.
    'unknown' is reserved for malformed/unsupported shapes.
    """
    M, exact = parse_matrix(matrix)
    if not M.is_square or M.rows not in (2, 3):
        return {"provenance": "unknown", "note": "analyse_factorizations expects a 2×2 or 3×3 square matrix"}
    n = M.rows

    if mode == "lu":
        minors = [M[:k, :k].det() for k in range(1, n)]
        no_pivoting_required = all(d != 0 for d in minors)

        L, U, perm_pairs = M.LUdecomposition()
        idx = list(range(n))
        for (i, j) in perm_pairs:
            idx[i], idx[j] = idx[j], idx[i]
        pivoting_occurred = idx != list(range(n))

        PA = Matrix.hstack(*[M.row(idx[i]).T for i in range(n)]).T
        LU = L * U
        reconstructs = simplify(PA - LU) == zeros(n, n)
        reconstructs_without_p = (not pivoting_occurred) and (simplify(M - LU) == zeros(n, n))

        l_unit_lower = all(L[i, i] == 1 for i in range(n)) and all(
            L[i, j] == 0 for i in range(n) for j in range(i + 1, n)
        )
        u_upper = all(U[i, j] == 0 for i in range(n) for j in range(i))
        rank = M.rank()

        return {
            "kind": "lu",
            "n": n,
            "provenance": provenance_for(exact),
            "L": [[str(L[i, j]) for j in range(n)] for i in range(n)],
            "U": [[str(U[i, j]) for j in range(n)] for i in range(n)],
            "perm": idx,
            "pivotingOccurred": bool(pivoting_occurred),
            "noPivotingRequired": bool(no_pivoting_required),
            "minors": [str(d) for d in minors],
            "rank": int(rank),
            "singular": bool(rank < n),
            "reconstructs": bool(reconstructs),
            "reconstructsWithoutP": bool(reconstructs_without_p),
            "lIsUnitLower": bool(l_unit_lower),
            "uIsUpperTriangular": bool(u_upper),
            "luHolds": bool(reconstructs and l_unit_lower and u_upper),
        }

    if mode == "qr":
        cols = [M.col(k) for k in range(n)]

        def dot(a, b):
            return sum(a[i] * b[i] for i in range(n))

        us = []
        norms_sq = []
        dependent_at = None
        for k in range(n):
            u = cols[k]
            for uj in us:
                c = dot(cols[k], uj) / dot(uj, uj)
                u = u - c * uj
            nsq = simplify(dot(u, u))
            if nsq == 0:
                dependent_at = k
                break
            us.append(u)
            norms_sq.append(nsq)

        if dependent_at is not None:
            return {
                "kind": "qr",
                "n": n,
                "provenance": provenance_for(exact),
                "qrExists": False,
                "dependentAt": dependent_at,
                "reason": (
                    f"column {dependent_at + 1} lies in the span of the previous columns — "
                    f"Gram-Schmidt's own projection subtracts it away to the zero vector, so no "
                    f"orthonormal basis (and hence no QR with an invertible R) exists for these columns."
                ),
            }

        es = [simplify(u / sqrt(nsq)) for u, nsq in zip(us, norms_sq)]
        Qm = Matrix.hstack(*es)
        R = simplify(Qm.T * M)
        QtQ = simplify(Qm.T * Qm)
        q_orthonormal = QtQ == eye(n)
        reconstructed = simplify(Qm * R)
        reconstructs = reconstructed == M
        r_upper = all(R[i, j] == 0 for i in range(n) for j in range(i))

        all_perfect_square_rational = all(sqrt(nsq).is_rational for nsq in norms_sq)
        provenance = "proved" if (exact and all_perfect_square_rational) else "numeric"

        return {
            "kind": "qr",
            "n": n,
            "provenance": provenance,
            "qrExists": True,
            "Q": [[str(Qm[i, j]) for j in range(n)] for i in range(n)],
            "R": [[str(R[i, j]) for j in range(n)] for i in range(n)],
            "qOrthonormal": bool(q_orthonormal),
            "reconstructs": bool(reconstructs),
            "rUpperTriangular": bool(r_upper),
            "qrHolds": bool(q_orthonormal and reconstructs and r_upper),
        }

    return {"provenance": "unknown", "note": f"analyse_factorizations: unknown mode {mode!r} (expected 'lu' or 'qr')"}


# =============================================================================
# ── Calculus domain (DOMAINS[1]) ──
#
# Ported from a calculus-domain drop's src/workers/symbolic/{common,calculus}.py
# (see mathviz-calculus-execution-plan.md), merged into this single file per
# CLAUDE.md's Calculus integration notes: symbolicEngine.py stays ONE file (the
# checker hardcodes its path), rather than the drop's own multi-file split.
#
# Same provenance vocabulary as the Linear-Algebra section above, with one
# addition: 'unknown' is common here (undecidable continuity/differentiability
# for floor/ceiling/sign/Piecewise), whereas in Linear Algebra the common
# non-proved state is 'numeric'. Provenance for an expression is always exact
# by construction (the JS expr.mjs printer emits a canonical SymPy string);
# only the ENDPOINTS a/b can be float-derived (\`numericOf()\` in expr.mjs, used
# by a future slider-driven module), which caps the whole result at 'numeric'
# via \`cap()\` below — this module (rolle) sends only Rational/pi/integer
# endpoints, so \`exact\` is effectively always True for it today.
# =============================================================================

x = Symbol('x', real=True)

# Restricted namespace for the canonical SymPy string produced by expr.mjs::toSympy().
CALC_NAMESPACE = {
    'x': x, 'pi': pi, 'E': E, 'Rational': Rational, 'Float': Float,
    'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
    'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh, 'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt,
    'Abs': Abs, 'floor': floor, 'ceiling': ceiling, 'sign': sign,
    # 'factorial' added for powerseries (topic 2.12) — see expr.mjs's own
    # matching addition; sp.factorial handles a symbolic/positive-integer
    # argument exactly (no overflow, unlike the JS numeric engine's own
    # IEEE-754 double — see powerseriesEngine.js's header).
    'factorial': sp.factorial,
}

CALC_ABSTAIN_FUNCS = (floor, ceiling, sign, Piecewise)


class Abstain(Exception):
    """Raised when the calculus engine cannot decide honestly. Callers turn this into
    provenance 'unknown' rather than guessing (see the module-level docstring's
    provenance vocabulary)."""


def calc_parse(expr_str):
    """Parse the canonical string from expr.mjs::toSympy(). Raises ValueError on
    anything outside CALC_NAMESPACE."""
    try:
        f = sp.sympify(expr_str, locals=CALC_NAMESPACE)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'cannot parse: {e}') from e
    bad = f.free_symbols - {x}
    if bad:
        raise ValueError(f'unexpected symbols {sorted(map(str, bad))}')
    return f


def calc_parse_scalar(v):
    """Endpoint / parameter (already a SymPy-syntax string from expr.mjs::parseNumber,
    e.g. 'Rational(1, 2)', 'pi', '3'). Returns (sympy value, exact: bool)."""
    if isinstance(v, bool):
        raise ValueError('bad scalar')
    if isinstance(v, int):
        return sp.Integer(v), True
    if isinstance(v, float):
        return Float(v), False
    s = str(v).strip()
    val = sp.sympify(s, locals=CALC_NAMESPACE)
    if val.free_symbols:
        raise ValueError(f'scalar has free symbols: {s}')
    exact = not val.has(Float)
    return (sp.nsimplify(val) if exact else val), exact


def calc_has_abstain_funcs(f) -> bool:
    return any(f.has(fn) for fn in CALC_ABSTAIN_FUNCS)


def calc_continuous_on(f, a, b):
    """{'pass': bool, 'issues': [...], 'provenance': 'proved'|'refuted'|'unknown'} on
    the CLOSED [a,b]. Trap: floor/ceiling/sign/Piecewise defeat continuous_domain, so
    the engine abstains rather than guess (see the module docstring's matrix-traps
    analogue for this domain)."""
    if calc_has_abstain_funcs(f):
        raise Abstain('continuous_domain is unreliable with floor/ceiling/sign/Piecewise')
    try:
        dom = continuous_domain(f, x, Interval(a, b))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'continuous_domain failed: {e}') from e
    target = Interval(a, b)
    if dom == target:
        return {'pass': True, 'issues': [], 'provenance': 'proved'}
    missing = target - dom
    pts = _calc_sample_points(missing)
    issues = [{'kind': 'discontinuity', 'x': _calc_num(p), 'xTex': sp.latex(p),
               'detail': 'f is not continuous here'} for p in pts]
    return {'pass': False, 'issues': issues, 'provenance': 'refuted'}


def calc_suspect_points(f, a, b):
    """Interior points where diff() must not be trusted: zeros of Abs arguments and
    singularities of f' (see safe_diff_at — the |x| -> sign(x) trap the module
    docstring warns about, applied to calculus instead of matrices)."""
    pts = set()
    for sub in f.atoms(Abs):
        arg = sub.args[0]
        try:
            sols = sp.solveset(arg, x, Interval.open(a, b))
        except Exception:  # noqa: BLE001
            raise Abstain('cannot locate Abs kinks')
        if not sols.is_FiniteSet:
            raise Abstain('Abs kink set is not finite')
        pts |= set(sols)
    try:
        fp = sp.diff(f, x)
        from sympy.calculus.singularities import singularities
        sing = singularities(fp, x, Interval.open(a, b))
        if not sing.is_FiniteSet:
            raise Abstain("singularity set of f' is not finite")
        pts |= set(sing)
    except Abstain:
        raise
    except Exception:  # noqa: BLE001
        raise Abstain("cannot compute singularities of f'")
    return sorted(pts, key=lambda p: float(p))


def calc_safe_diff_at(f, x0):
    """One-sided limits of the ACTUAL difference quotient at x0 (never sp.diff at a
    suspect point — see the module docstring). Returns (left, right); None if a side
    has no limit."""
    h = Symbol('h', positive=True)
    q_right = (f.subs(x, x0 + h) - f.subs(x, x0)) / h
    q_left = (f.subs(x, x0) - f.subs(x, x0 - h)) / h
    try:
        r = sp.limit(q_right, h, 0, '+')
        l = sp.limit(q_left, h, 0, '+')
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'one-sided limit failed at {x0}: {e}') from e
    return l, r


def calc_differentiable_on(f, a, b):
    """Differentiability on the OPEN (a,b). Assumes continuity already established."""
    if calc_has_abstain_funcs(f):
        raise Abstain('differentiability with floor/ceiling/sign/Piecewise')
    for p in calc_suspect_points(f, a, b):
        l, r = calc_safe_diff_at(f, p)
        finite = lambda v: v is not None and v.is_finite and v.is_real  # noqa: E731
        if not (finite(l) and finite(r) and sp.simplify(l - r) == 0):
            issue = {'kind': 'corner' if finite(l) and finite(r) else 'cusp', 'x': _calc_num(p), 'xTex': sp.latex(p),
                     'left': _calc_num(l) if l is not None else None, 'right': _calc_num(r) if r is not None else None,
                     'leftTex': sp.latex(l), 'rightTex': sp.latex(r),
                     'detail': 'one-sided slopes disagree' if finite(l) and finite(r) else 'one-sided slope is infinite'}
            return {'pass': False, 'issues': [issue], 'provenance': 'refuted'}
    return {'pass': True, 'issues': [], 'provenance': 'proved'}


def calc_stationary_points(f, a, b, max_points=12):
    """Exact zeros of f' in (a,b) when solveset can; otherwise certified sign-change
    enclosures via mpmath.iv (Darboux: f' has the intermediate value property, so a
    certified sign change certifies a zero). Returns (list of dicts, certificate) with
    certificate in {'exact', 'enclosure'}."""
    fp = sp.diff(f, x)
    try:
        sols = sp.solveset(sp.simplify(fp), x, Interval.open(a, b))
    except Exception:  # noqa: BLE001
        sols = None
    if sols is not None and sols.is_FiniteSet:
        # TRAP: diff(Abs(x)) -> sign(x) and solveset(sign(x)) -> {0} would wrongly
        # certify |x| as satisfying Rolle. Any candidate sitting on a suspect point is
        # re-checked with the actual one-sided difference quotient.
        try:
            suspects = set(calc_suspect_points(f, a, b))
        except Abstain:
            suspects = None
        out = []
        for c in sorted(sols, key=lambda v: float(v)):
            if not c.is_real:
                continue
            if suspects is None:
                raise Abstain('cannot validate witnesses at suspect points')
            if c in suspects:
                l, r = calc_safe_diff_at(f, c)
                ok = (l is not None and r is not None and l.is_finite and r.is_finite and sp.simplify(l) == 0 and sp.simplify(r) == 0)
                if not ok:
                    continue
            out.append({'c': _calc_num(c), 'cTex': sp.latex(c), 'fpc': 0, 'certificate': 'exact'})
        return out[:max_points], 'exact'
    if sols is not None and sols == S.EmptySet:
        return [], 'exact'
    if isinstance(sols, sp.Interval) or (sols is not None and sols.is_Interval):
        return [{'c': _calc_num((a + b) / 2), 'cTex': sp.latex((a + b) / 2), 'fpc': 0, 'certificate': 'exact', 'kind': 'constant'}], 'exact'
    return calc_enclose_roots(fp, a, b, max_points), 'enclosure'


def calc_enclose_roots(g, a, b, max_points=12, depth=14):
    """Certified sign-change enclosures of zeros of g on (a,b) using interval
    arithmetic (mpmath.iv) — the fallback when solveset cannot solve g = 0 exactly."""
    try:
        gf = sp.lambdify(x, g, modules=[{'Abs': abs}, 'mpmath'])
    except Exception:  # noqa: BLE001
        raise Abstain('cannot lambdify for enclosure')
    iv = mpmath.iv
    iv.dps = 30
    out = []
    n = 512
    lo_f, hi_f = float(a), float(b)
    xs = [lo_f + (hi_f - lo_f) * i / n for i in range(n + 1)]

    def val(t):
        try:
            v = gf(iv.mpf(t))
            return iv.mpf(v)
        except Exception:  # noqa: BLE001
            return None

    prev = val(xs[1])
    for i in range(2, n):
        cur = val(xs[i])
        if prev is None or cur is None:
            prev = cur
            continue
        if (prev.b < 0 and cur.a > 0) or (prev.a > 0 and cur.b < 0):
            lo, hi, vlo = xs[i - 1], xs[i], prev
            for _ in range(depth):
                m = (lo + hi) / 2
                vm = val(m)
                if vm is None or not (vm.a > 0 or vm.b < 0):
                    break
                if (vlo.b < 0 and vm.a > 0) or (vlo.a > 0 and vm.b < 0):
                    hi = m
                else:
                    lo, vlo = m, vm
            out.append({'c': (lo + hi) / 2, 'enclosure': [lo, hi], 'certificate': 'enclosure'})
            if len(out) >= max_points:
                break
        prev = cur
    return out


def calc_cap(provenance, exact_inputs: bool):
    """Float endpoints cap proved/refuted at numeric (module docstring's provenance
    vocabulary, applied to the Calculus domain)."""
    if not exact_inputs and provenance in ('proved', 'refuted'):
        return 'numeric'
    return provenance


def calc_unknown(reason: str, **extra):
    return {'provenance': 'unknown', 'reason': reason, **extra}


def _calc_num(v):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return None


def calc_interior_extremum(f, a, b, candidates):
    """Does f attain its max or min on [a,b] at an interior point? Compares f at the
    candidate interior points against the endpoints (numerically, high precision).
    Returns {'interior': bool, 'argmax', 'argmin', 'maxInterior', 'minInterior'}."""
    pts = [(sp.N(f.subs(x, a), 30), a, False), (sp.N(f.subs(x, b), 30), b, False)]
    for c in candidates:
        cv = sp.nsimplify(c) if not isinstance(c, sp.Basic) else c
        try:
            v = sp.N(f.subs(x, cv), 30)
        except Exception:  # noqa: BLE001
            continue
        if v.is_real:
            pts.append((v, cv, True))
    mx = max(pts, key=lambda t: t[0])
    mn = min(pts, key=lambda t: t[0])
    return {'interior': bool(mx[2] or mn[2]), 'argmax': _calc_num(mx[1]), 'argmin': _calc_num(mn[1]),
            'maxInterior': bool(mx[2]), 'minInterior': bool(mn[2])}


def _calc_sample_points(s, limit=3):
    """A few representative points of a sympy Set (finite set, interval, or union)."""
    pts = []
    if s.is_FiniteSet:
        pts = list(s)
    elif isinstance(s, sp.Union):
        for part in s.args:
            pts += _calc_sample_points(part, 1)
    elif isinstance(s, sp.Interval):
        pts = [s.start if s.start.is_finite else s.end]
    return pts[:limit]


def _calc_prepare(expr_str, a_str, b_str):
    f = calc_parse(expr_str)
    a, ea = calc_parse_scalar(a_str)
    b, eb = calc_parse_scalar(b_str)
    if not (a.is_real and b.is_real) or not (a < b):
        raise ValueError('need real a < b')
    return f, a, b, (ea and eb)


def _calc_hypotheses_block(f, a, b, exact):
    """Continuity on [a,b] then differentiability on (a,b). Each hypothesis carries
    its own provenance."""
    hyps = {}
    try:
        c = calc_continuous_on(f, a, b)
        c['provenance'] = calc_cap(c['provenance'], exact)
    except Abstain as e:
        c = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    hyps['continuous'] = c
    if c['pass'] is True:
        try:
            d = calc_differentiable_on(f, a, b)
            d['provenance'] = calc_cap(d['provenance'], exact)
        except Abstain as e:
            d = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    elif c['pass'] is False:
        d = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'not continuous'}], 'provenance': c['provenance']}
    else:
        d = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': 'continuity undecided'}
    hyps['differentiable'] = d
    return hyps


def _calc_witness_block(g, a, b, exact, target_slope):
    """Zeros of g' in (a,b) as witnesses. 'exact' certificate -> proved; 'enclosure'
    -> proved (Darboux) but flagged; failure -> unknown (never 'refuted': existence is
    what the theorem guarantees, so there is nothing to refute here)."""
    try:
        pts, cert = calc_stationary_points(g, a, b)
    except Abstain as e:
        return {'witnesses': [], 'provenance': 'unknown', 'reason': str(e), 'certificate': None}
    ws = []
    for p in pts:
        w = dict(p)
        w['slope'] = _calc_num(target_slope)
        w['slopeTex'] = sp.latex(target_slope)
        ws.append(w)
    prov = 'proved' if ws else 'unknown'
    return {'witnesses': ws, 'provenance': calc_cap(prov, exact), 'certificate': cert}


def _calc_overall(hyps, wb, any_fail):
    """Module-level provenance: the weakest link among what the verdict depends on."""
    provs = [h['provenance'] for h in hyps.values()] + [wb['provenance']]
    if any_fail:
        hp = [h['provenance'] for h in hyps.values() if h['pass'] is False]
        return 'numeric' if 'numeric' in hp else 'refuted'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    return 'proved'


def analyse_rolle(expr_str, a_str, b_str):
    """expr_str/a_str/b_str: SymPy-syntax strings from expr.mjs (toSympy()/.sympy).
    Rolle = MVT with the extra endpoint hypothesis f(a) = f(b); witnesses are zeros of
    f' itself. Returns VisualMath's flat {provenance, ...} shape (never c-wrapped — this
    module has a genuine witness LIST, unlike Linear Algebra's single-result modules,
    so 'conclusion.witnesses' is itself an array, still inside the one flat dict)."""
    try:
        f, a, b, exact = _calc_prepare(expr_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    fa, fb = f.subs(x, a), f.subs(x, b)
    diff_ends = sp.simplify(fb - fa)
    ends_pass = bool(diff_ends == 0)
    hyps = _calc_hypotheses_block(f, a, b, exact)
    hyps['endpoints'] = {
        'pass': ends_pass,
        'issues': [] if ends_pass else [{'kind': 'endpoints', 'detail': f'f(a) = {sp.latex(sp.simplify(fa))}, f(b) = {sp.latex(sp.simplify(fb))}'}],
        'provenance': calc_cap('proved' if ends_pass else 'refuted', exact),
    }
    all_pass = all(h['pass'] is True for h in hyps.values())
    any_fail = any(h['pass'] is False for h in hyps.values())
    wb = _calc_witness_block(f, a, b, exact, sp.Integer(0))
    fp = sp.diff(f, x)
    # Which proof act dies? (non-negotiable #4) Computed from the proof STATE, not
    # from the hypothesis table alone: unequal endpoints only kill the interiority
    # act if both extrema sit at the endpoints — mirrors rolleEngine.js's own
    # numericBlockedAct so the two tiers agree by construction.
    blocked = None
    extremum = None
    if hyps['continuous']['pass'] is False:
        blocked = 'evt'
    else:
        cands = [w['c'] for w in wb['witnesses']]
        try:
            cands += [float(p) for p in calc_suspect_points(f, a, b)]
        except Abstain:
            pass
        try:
            extremum = calc_interior_extremum(f, a, b, cands)
        except Exception:  # noqa: BLE001
            extremum = None
        if extremum is not None and not extremum['interior'] and not hyps['endpoints']['pass']:
            blocked = 'interior'
        elif hyps['differentiable']['pass'] is False:
            blocked = 'fermat'
    return {
        'id': 'rolle',
        'provenance': _calc_overall(hyps, wb, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f), 'fpTex': sp.latex(sp.simplify(fp))},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'values': {'fa': _calc_num(fa), 'fb': _calc_num(fb), 'faTex': sp.latex(sp.simplify(fa)), 'fbTex': sp.latex(sp.simplify(fb))},
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'extremum': extremum,
        'conclusion': wb,
        'reducesTo': {'id': 'constant', 'active': bool(sp.simplify(fp) == 0), 'detail': 'f is constant: every point is a witness'},
    }


def analyse_mvt(expr_str, a_str, b_str):
    """expr_str/a_str/b_str: SymPy-syntax strings from expr.mjs. Lagrange's Mean Value
    Theorem: hypotheses (continuity, differentiability) and witnesses c with
    f'(c) = (f(b) - f(a)) / (b - a) — Rolle applied to the auxiliary
    g(x) = f(x) - (secant of f). Wired to MvtModule.jsx (module 2.6); returns the same
    flat shape as analyse_rolle plus the secant slope, g, and the proof-state fields
    the module's theatre reads (blockedAct / extremum / degenerate)."""
    try:
        f, a, b, exact = _calc_prepare(expr_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    fa, fb = f.subs(x, a), f.subs(x, b)
    slope = sp.simplify((fb - fa) / (b - a))
    g = f - fa - slope * (x - a)  # auxiliary function: g(a) = g(b) = 0
    hyps = _calc_hypotheses_block(f, a, b, exact)
    all_pass = all(h['pass'] is True for h in hyps.values())
    any_fail = any(h['pass'] is False for h in hyps.values())
    wb = _calc_witness_block(g, a, b, exact, slope)
    fp = sp.diff(f, x)
    reduces_to = {'id': 'rolle', 'active': bool(sp.simplify(fb - fa) == 0),
                  'detail': 'f(a) = f(b): the secant is horizontal — this input is exactly Rolle'}
    # Proof state for MvtModule's theatre (module 2.6). The proof is Rolle's, run on g:
    # g(a) = g(b) = 0 by construction, so the interiority step can never die — only
    # continuity (EVT on g) or differentiability (Fermat on g) can. Mirrors
    # src/modules/mvtEngine.js's numericBlockedAct so the two tiers agree by construction.
    g_is_zero = bool(sp.simplify(g) == 0)
    blocked = None
    extremum = None
    if hyps['continuous']['pass'] is False:
        blocked = 'evt'
    else:
        cands = [w['c'] for w in wb['witnesses']]
        try:
            cands += [float(p) for p in calc_suspect_points(f, a, b)]
        except Abstain:
            pass
        try:
            extremum = calc_interior_extremum(g, a, b, cands)
        except Exception:  # noqa: BLE001
            extremum = None
        if hyps['differentiable']['pass'] is False:
            blocked = 'fermat'
    return {
        'id': 'mvt',
        'provenance': _calc_overall(hyps, wb, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f), 'fpTex': sp.latex(sp.simplify(fp))},
        'g': {'tex': sp.latex(sp.expand(g)), 'isZero': g_is_zero},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'values': {'fa': _calc_num(fa), 'fb': _calc_num(fb), 'faTex': sp.latex(sp.simplify(fa)), 'fbTex': sp.latex(sp.simplify(fb))},
        'secantSlope': {'value': _calc_num(slope), 'tex': sp.latex(slope)},
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'extremum': extremum,
        'conclusion': wb,
        'reducesTo': reduces_to,
        'degenerate': {'linear': g_is_zero, 'detail': 'f is linear: g ≡ 0 and every point is a witness'},
    }


def calc_limit_at(f, c):
    """Two-sided limit of f as x -> c: {'left', 'right', 'exists', 'value'}. Abstains on
    the same floor/ceiling/sign/Piecewise traps calc_continuous_on already abstains on
    (topic 2.1, limits/continuity/differentiability at a point)."""
    if calc_has_abstain_funcs(f):
        raise Abstain('limit computation is unreliable with floor/ceiling/sign/Piecewise')
    try:
        l = sp.limit(f, x, c, '-')
        r = sp.limit(f, x, c, '+')
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'limit failed at {c}: {e}') from e
    finite = lambda v: v is not None and v.is_finite and v.is_real  # noqa: E731
    exists = bool(finite(l) and finite(r) and sp.simplify(l - r) == 0)
    return {'left': l, 'right': r, 'exists': exists, 'value': l if exists else None}


def _calc_overall_limits(hyps, any_fail):
    """Module-level provenance for analyse_limits: the weakest link among the four
    clauses (no witness block here, unlike rolle/mvt's _calc_overall)."""
    provs = [h['provenance'] for h in hyps.values()]
    if any_fail:
        return 'numeric' if 'numeric' in provs else 'refuted'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    return 'proved'


def analyse_limits(expr_str, c_str, override_str=None):
    """expr_str/c_str: SymPy-syntax strings from expr.mjs (toSympy()/parseNumber().sympy).
    override_str: None (no override — the JS side sends null, which pyodide/plain-Python
    both read as None) or a SymPy-syntax scalar string for the module-owned "redefine f(c)"
    demonstration (see src/modules/limitsEngine.js's header) — parsed exactly like c_str via
    calc_parse_scalar. This is a definitions module, not an existence theorem: three nested
    definitions (limit, continuity, differentiability) checked at a single point c, in the
    SAFE EVALUATION ORDER limit exists -> f(c) defined -> they agree [continuity] -> f'(c)
    exists [differentiability], each clause GATING the next exactly like analyse_rolle's own
    \`blocked\` computation, so the JS (limitsEngine.js's numericBlockedAct) and Python tiers
    agree by construction. Flat {provenance, ...} shape (never c-wrapped, per CLAUDE.md)."""
    try:
        f = calc_parse(expr_str)
        c, exact_c = calc_parse_scalar(c_str)
        if not c.is_real:
            raise ValueError('c must be real')
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')

    override = None
    exact_override = True
    if override_str is not None and str(override_str).strip().lower() not in ('', 'none'):
        try:
            override, exact_override = calc_parse_scalar(override_str)
            if not override.is_real:
                raise ValueError('override must be real')
        except ValueError as e:
            return calc_unknown(f'bad override: {e}')
    exact = exact_c and exact_override
    finite_real = lambda v: v is not None and v.is_finite and v.is_real  # noqa: E731

    # Clause 1: the limit.
    try:
        lim = calc_limit_at(f, c)
        limit_clause = {
            'pass': lim['exists'],
            'value': _calc_num(lim['value']) if lim['exists'] else None,
            'valueTex': sp.latex(lim['value']) if lim['exists'] else None,
            'leftTex': sp.latex(lim['left']), 'rightTex': sp.latex(lim['right']),
            'issues': [] if lim['exists'] else [{'kind': 'no-limit', 'detail': f"left = {sp.latex(lim['left'])}, right = {sp.latex(lim['right'])}"}],
            'provenance': calc_cap('proved' if lim['exists'] else 'refuted', exact),
        }
    except Abstain as e:
        lim = {'value': None}
        limit_clause = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    # Clause 2: f(c) defined (or explicitly overridden — see the module docstring above).
    try:
        raw_fc = f.subs(x, c)
    except Exception:  # noqa: BLE001
        raw_fc = None
    if override is not None:
        fc_value, defined_pass, overridden = override, True, True
    else:
        fc_value = raw_fc
        defined_pass = bool(finite_real(raw_fc))
        overridden = False
    defined_clause = {
        'pass': defined_pass,
        'value': _calc_num(fc_value) if defined_pass else None,
        'valueTex': sp.latex(fc_value) if defined_pass else None,
        'overridden': overridden,
        'issues': [] if defined_pass else [{'kind': 'undefined', 'detail': 'f(c) is not a finite real number'}],
        'provenance': calc_cap('proved' if defined_pass else 'refuted', exact),
    }

    # Clause 3: continuity — the limit exists, f(c) is defined, and they agree.
    if limit_clause['pass'] is None:
        continuous_clause = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': 'the limit is undecided'}
    elif limit_clause['pass'] is False or not defined_clause['pass']:
        continuous_clause = {
            'pass': False,
            'issues': [{'kind': 'skipped', 'detail': 'the limit does not exist' if limit_clause['pass'] is False else 'f(c) is undefined'}],
            'provenance': limit_clause['provenance'] if limit_clause['pass'] is False else defined_clause['provenance'],
        }
    else:
        agree = bool(sp.simplify(lim['value'] - fc_value) == 0)
        continuous_clause = {
            'pass': agree,
            'issues': [] if agree else [{'kind': 'disagree', 'detail': f"the limit is {sp.latex(lim['value'])} but f(c) = {sp.latex(fc_value)}"}],
            'provenance': calc_cap('proved' if agree else 'refuted', exact),
        }

    # Clause 4: differentiability — only meaningful once continuity holds. Uses
    # calc_safe_diff_at (the ACTUAL one-sided difference-quotient limit, never sp.diff at a
    # suspect point — the same |x| -> sign(x) trap the module docstring warns about).
    if continuous_clause['pass'] is None:
        diff_clause = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': 'continuity is undecided'}
    elif continuous_clause['pass'] is False:
        diff_clause = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'not continuous at c'}], 'provenance': continuous_clause['provenance']}
    else:
        try:
            l, r = calc_safe_diff_at(f, c)
            ok = finite_real(l) and finite_real(r) and sp.simplify(l - r) == 0
            diff_clause = {
                'pass': bool(ok),
                'value': _calc_num(l) if ok else None,
                'leftTex': sp.latex(l) if l is not None else None, 'rightTex': sp.latex(r) if r is not None else None,
                'issues': [] if ok else [{
                    'kind': 'corner' if finite_real(l) and finite_real(r) else 'cusp',
                    'detail': 'one-sided slopes disagree' if finite_real(l) and finite_real(r) else 'one-sided slope is infinite',
                }],
                'provenance': calc_cap('proved' if ok else 'refuted', exact),
            }
        except Abstain as e:
            diff_clause = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    hyps = {'limitExists': limit_clause, 'defined': defined_clause, 'continuous': continuous_clause, 'differentiable': diff_clause}
    blocked = None
    if limit_clause['pass'] is False:
        blocked = 'limit'
    elif defined_clause['pass'] is False:
        blocked = 'defined'
    elif continuous_clause['pass'] is False:
        blocked = 'continuous'
    elif diff_clause['pass'] is False:
        blocked = 'differentiable'
    all_pass = all(h.get('pass') is True for h in hyps.values())
    any_fail = blocked is not None
    return {
        'id': 'limits',
        'provenance': _calc_overall_limits(hyps, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)},
        'c': {'value': _calc_num(c), 'tex': sp.latex(c)},
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
    }


def calc_value_roots(f, k, a, b, max_points=12):
    """Exact zeros of g(x) = f(x) - k on the CLOSED [a,b] when solveset can
    solve them (a witness may legitimately sit at an endpoint here, unlike
    calc_stationary_points's open-interval derivative search); otherwise
    certified sign-change enclosures via calc_enclose_roots (Darboux applies
    to g itself here, not to a derivative — g is continuous by hypothesis).
    Returns (list of dicts, certificate) with certificate in
    {'exact', 'enclosure'}. Mirrors calc_stationary_points's shape, but hunts
    zeros of g DIRECTLY rather than of g' — genuinely different machinery,
    since IVT's witness is a value of f, not a stationary point."""
    if calc_has_abstain_funcs(f):
        raise Abstain('root-finding is unreliable with floor/ceiling/sign/Piecewise')
    g = sp.simplify(f - k)
    try:
        sols = sp.solveset(g, x, Interval(a, b))
    except Exception:  # noqa: BLE001
        sols = None
    if sols is not None and sols.is_FiniteSet:
        out = []
        for c in sorted(sols, key=lambda v: float(v)):
            if not c.is_real:
                continue
            out.append({'c': _calc_num(c), 'cTex': sp.latex(c), 'certificate': 'exact'})
        return out[:max_points], 'exact'
    if sols is not None and sols == S.EmptySet:
        return [], 'exact'
    if isinstance(sols, sp.Interval) or (sols is not None and sols.is_Interval):
        return [{'c': _calc_num((a + b) / 2), 'cTex': sp.latex((a + b) / 2), 'certificate': 'exact', 'kind': 'constant'}], 'exact'
    return calc_enclose_roots(g, a, b, max_points), 'enclosure'


def _calc_ivt_witness_block(f, k, a, b, exact):
    """Zeros of f - k on [a,b] as witnesses. 'exact' certificate -> proved;
    'enclosure' -> proved (Darboux) but flagged; failure -> unknown (never
    'refuted': existence is what the theorem guarantees, so there is nothing
    to refute here) — mirrors _calc_witness_block's own contract."""
    try:
        pts, cert = calc_value_roots(f, k, a, b)
    except Abstain as e:
        return {'witnesses': [], 'provenance': 'unknown', 'reason': str(e), 'certificate': None}
    ws = []
    for p in pts:
        w = dict(p)
        w['value'] = _calc_num(k)
        w['valueTex'] = sp.latex(k)
        ws.append(w)
    prov = 'proved' if ws else 'unknown'
    return {'witnesses': ws, 'provenance': calc_cap(prov, exact), 'certificate': cert}


def analyse_ivt(expr_str, a_str, b_str, k_str):
    """expr_str/a_str/b_str/k_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy). Intermediate Value Theorem: f continuous
    on [a,b], and k lies between f(a) and f(b) (inclusive) ⟹ ∃ c ∈ [a,b] with
    f(c) = k. Two hypotheses (continuity, the k-between-f(a)-f(b) bracket),
    each consumed by one proof step; the witness search hunts zeros of
    f(x) - k directly (calc_value_roots), never a derivative — genuinely
    different machinery from analyse_rolle/analyse_mvt's stationary-point
    search, so it is not routed through _calc_witness_block. Returns
    VisualMath's flat {provenance, ...} shape (never c-wrapped)."""
    try:
        f, a, b, exact_ab = _calc_prepare(expr_str, a_str, b_str)
        k, exact_k = calc_parse_scalar(k_str)
        if not k.is_real:
            raise ValueError('k must be a real number')
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    exact = exact_ab and exact_k
    fa, fb = f.subs(x, a), f.subs(x, b)

    try:
        c = calc_continuous_on(f, a, b)
        c['provenance'] = calc_cap(c['provenance'], exact)
    except Abstain as e:
        c = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    # The bracket hypothesis: k between f(a) and f(b) inclusive. fa/fb/k are
    # all concrete numbers once substituted (no free symbols), so >= and <=
    # decide outright rather than returning an undecided relational.
    lo, hi = sp.Min(fa, fb), sp.Max(fa, fb)
    bracket_pass = bool(k >= lo) and bool(k <= hi)
    hyps = {
        'continuous': c,
        'bracket': {
            'pass': bracket_pass,
            'issues': [] if bracket_pass else [{'kind': 'bracket', 'detail': f'k = {sp.latex(k)} is not between f(a) = {sp.latex(sp.simplify(fa))} and f(b) = {sp.latex(sp.simplify(fb))}'}],
            'provenance': calc_cap('proved' if bracket_pass else 'refuted', exact),
        },
    }
    all_pass = all(h['pass'] is True for h in hyps.values())
    any_fail = any(h['pass'] is False for h in hyps.values())

    wb = _calc_ivt_witness_block(f, k, a, b, exact)

    # Which proof act dies (non-negotiable #4)? Continuity failing ALWAYS
    # blocks (mirrors analyse_rolle's own 'evt': without it there is no
    # argument at all); the bracket hypothesis failing only blocks when no
    # witness turns up anyway — mirrors src/modules/ivtEngine.js's own
    # numericBlockedAct so the two tiers agree by construction.
    blocked = None
    if hyps['continuous']['pass'] is False:
        blocked = 'continuity'
    elif hyps['bracket']['pass'] is False and not wb['witnesses']:
        blocked = 'bracket'

    g_is_zero = bool(sp.simplify(f - k) == 0)
    return {
        'id': 'ivt',
        'provenance': _calc_overall(hyps, wb, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'k': {'value': _calc_num(k), 'tex': sp.latex(k)},
        'values': {'fa': _calc_num(fa), 'fb': _calc_num(fb), 'faTex': sp.latex(sp.simplify(fa)), 'fbTex': sp.latex(sp.simplify(fb))},
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'conclusion': wb,
        'degenerate': {'constant': g_is_zero, 'detail': 'f is constant and equal to k: every point is a witness'},
    }


def calc_bounded_on(f, a, b):
    """Boundedness of f on the CLOSED [a,b] — the one genuinely breakable
    clause of topic 2.3 (Integration as a limit of Riemann sums; see
    src/modules/riemannEngine.js's header for why boundedness, not
    continuity, is the definitional gate here — a bounded function with
    finitely many jump discontinuities, e.g. floor(x), is still Riemann
    integrable). f is unbounded near a point exactly when a one-sided limit
    there is infinite; reuses the same singularities()/limit machinery as
    calc_continuous_on / calc_differentiable_on and abstains on the same
    floor/ceiling/sign/Piecewise trap those already document — NOTE this
    means the symbolic tier abstains on floor(x) even though floor(x) is
    numerically perfectly well-behaved for boundedness; the numeric tier
    (riemannEngine.js's checkBounded) is the one that actually demonstrates
    the "looks risky, still integrable" staircase case."""
    if calc_has_abstain_funcs(f):
        raise Abstain('boundedness is unreliable with floor/ceiling/sign/Piecewise')
    try:
        sing = singularities(f, x, Interval(a, b))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot compute singularities: {e}') from e
    if not sing.is_FiniteSet:
        raise Abstain('singularity set is not finite')
    issues = []
    for p in sorted(sing, key=lambda v: float(v)):
        try:
            l = sp.limit(f, x, p, '-') if p > a else None
            r = sp.limit(f, x, p, '+') if p < b else None
        except Exception as e:  # noqa: BLE001
            raise Abstain(f'limit failed at {p}: {e}') from e
        for v in (l, r):
            if v is not None and not (v.is_finite and v.is_real):
                issues.append({'kind': 'unbounded', 'x': _calc_num(p), 'xTex': sp.latex(p),
                                'detail': f'f is unbounded near x = {sp.latex(p)}'})
                break
    pass_ = len(issues) == 0
    return {'pass': pass_, 'issues': issues, 'provenance': 'proved' if pass_ else 'refuted'}


def calc_definite_integral(f, a, b):
    """Exact definite integral via SymPy's integrate() — the GROUND TRUTH the
    numeric tier's Riemann sums are shown converging toward. Returns
    (value_or_None, certificate): certificate 'exact' for a finite closed-form
    value, 'divergent' when SymPy reports +-oo/zoo (a genuinely non-integrable
    unbounded case, e.g. 1/x on [0,1]); raises Abstain when SymPy leaves the
    integral unevaluated (no closed form found — e.g. a case beyond what
    solveset/integrate can certify) rather than guessing."""
    try:
        val = sp.integrate(f, (x, a, b))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'integration failed: {e}') from e
    if val.has(sp.Integral):
        raise Abstain('SymPy could not evaluate this integral in closed form')
    if val in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        return None, 'divergent'
    if val.is_real is False:
        raise Abstain('integral did not evaluate to a real number')
    return val, 'exact'


def analyse_riemann(expr_str, a_str, b_str):
    """expr_str/a_str/b_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy). Topic 2.3: integration as a limit of
    Riemann sums — a CONCEPT module, not a theorem (mirrors analyse_limits's
    shape more than analyse_rolle/analyse_ivt's). Two clauses, weakest to
    strongest, in the same gating-chain style as analyse_limits: (1) f is
    bounded on [a,b] — genuinely breakable, since an unbounded function can
    always be forced to blow up a Riemann sum by choosing a tag near the
    singularity; (2) the Riemann sums converge to a single value (Riemann's
    criterion) — supplied here by the EXACT definite integral, which is the
    ground truth src/modules/riemannEngine.js's numeric sums are shown
    converging toward. Flat {provenance, ...} shape (never c-wrapped, per
    CLAUDE.md)."""
    try:
        f, a, b, exact = _calc_prepare(expr_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')

    try:
        bounded = calc_bounded_on(f, a, b)
        bounded['provenance'] = calc_cap(bounded['provenance'], exact)
    except Abstain as e:
        bounded = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    integral_value, cert = None, None
    if bounded['pass'] is True:
        try:
            integral_value, cert = calc_definite_integral(f, a, b)
            if cert == 'divergent':
                convergent = {'pass': False, 'issues': [{'kind': 'divergent', 'detail': 'the definite integral diverges'}],
                              'provenance': calc_cap('refuted', exact)}
            else:
                convergent = {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact)}
        except Abstain as e:
            convergent = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    elif bounded['pass'] is False:
        convergent = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f is not bounded on [a, b]'}], 'provenance': bounded['provenance']}
    else:
        convergent = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': 'boundedness is undecided'}

    hyps = {'bounded': bounded, 'convergent': convergent}
    all_pass = all(h.get('pass') is True for h in hyps.values())
    blocked = None
    if bounded['pass'] is False:
        blocked = 'bounded'
    elif convergent['pass'] is False:
        blocked = 'convergent'
    any_fail = blocked is not None

    is_constant = bool(sp.diff(f, x) == 0)
    return {
        'id': 'riemann',
        'provenance': _calc_overall_limits(hyps, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'integral': {
            'value': _calc_num(integral_value) if integral_value is not None else None,
            'tex': sp.latex(integral_value) if integral_value is not None else None,
            'certificate': cert,
        },
        'degenerate': {'constant': is_constant, 'detail': 'f is constant: every Riemann sum equals f times (b - a)'},
    }


def _calc_overall_netchange(hyps, blocked):
    """Module-level provenance for analyse_netchange: the weakest link among
    the three clauses. UNLIKE _calc_overall_limits, a clause's own 'pass'
    being False does not by itself force 'refuted'/'numeric' here -- only
    \`blocked\` (set from continuous/bounded alone, mirroring
    netchangeEngine.js's numericBlockedAct) does. Differentiability failing
    on its own (the module's documented fails-yet-holds exception) must NOT
    demote a provenance that is otherwise 'proved' down to 'refuted'."""
    provs = [h['provenance'] for h in hyps.values()]
    if blocked is not None:
        return 'numeric' if 'numeric' in provs else 'refuted'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    return 'proved'


def analyse_netchange(expr_str, a_str, b_str):
    """expr_str/a_str/b_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy). Topic 2.4: the Net Change Theorem --
    if F is continuous on [a,b], differentiable on (a,b), and F' is bounded
    on [a,b] (hence Riemann-integrable in the ordinary sense), then
    integral_a^b F'(x) dx = F(b) - F(a). THREE hypotheses, unlike
    Rolle/MVT/IVT's single extra endpoint-style hypothesis and unlike
    riemann's single-breakable-clause chain: continuity and boundedness of F'
    EACH independently block the guarantee (mirrored in \`blockedAct\`);
    differentiability is REPORT-ONLY and never blocks -- see
    netchangeEngine.js's own header for why a single interior kink still
    leaves F' bounded almost everywhere, so the identity survives (this
    module's documented "hypothesis fails yet conclusion holds" exception,
    the same shape as the Linear-algebra addendum's Rank-Nullity exception).
    Returns VisualMath's flat {provenance, ...} shape (never c-wrapped).

    Reuses calc_continuous_on (clause 1), calc_differentiable_on (clause 2,
    report-only), and calc_bounded_on -- APPLIED TO F', not to F itself --
    for clause 3: the SAME boundedness check topic 2.3 (riemann) already
    performs on the ORIGINAL function is here performed one derivative down,
    exactly the "applied to the derivative rather than to F itself" framing
    this topic teaches.

    TWO genuinely new traps, discovered building this module (added to the
    running list CLAUDE.md's "Known matrix/calculus traps" already keeps):

    (1) sp.diff(Abs(g(x)), x) introduces sign(g(x)) into F' even when F
    itself (e.g. Abs(x-1)) has NO abstain functions and its own continuity
    and differentiability are fully decidable exactly -- so the
    BOUNDEDNESS-of-F' clause can abstain via CALC_ABSTAIN_FUNCS on an F whose
    other two clauses were JUST exactly decided. Combined with this module's
    own odd-root preset (x^(1/3) reaches SymPy as sign(x)*Abs(x)**Rational(1,3),
    abstaining already at the continuity check -- the same trap rolle/mvt
    already document) and the floor(x) preset (abstains immediately, same as
    riemann's own floor(x) case), the symbolic tier abstains on most of this
    module's more interesting presets; the NUMERIC tier (netchangeEngine.js)
    is what actually demonstrates them, exactly as CLAUDE.md's "provenance is
    rendered, never hidden" already establishes for riemann's floor(x) case.

    (2) sp.integrate() of an expression built from an UNEVALUATED
    Derivative(...) (e.g. diff(floor(x), x), which SymPy cannot resolve) can
    silently perform the Fundamental Theorem's own round-trip
    (integral of dF/dx dx = F(b)-F(a) BY CONSTRUCTION) rather than the
    ORDINARY (pointwise, measure-zero-blind) Riemann integral the hypothesis
    "F' bounded" is actually about -- independently confirmed while building
    this module: sp.integrate(Derivative(floor(x), x), (x, 0, 5/2)) returns
    2 (= F(b)-F(a) exactly), not the pointwise-a.e. value 0 a genuine
    Riemann sum of floor(x)'s a.e. derivative converges to. This module
    therefore NEVER treats a successful sp.integrate(F', (x,a,b)) as
    confirming the identity by itself -- \`identityHolds\`'s provenance is
    governed ENTIRELY by the exact hypothesis checks (continuity,
    boundedness), exactly mirroring netchangeEngine.js's own identityHolds
    (never true while blockedAct is set, regardless of what the raw computed
    numbers happen to show). In practice this trap never actually fires for
    this module's own six presets, because every case where it WOULD matter
    (floor, the odd-root rewrite) already abstains earlier via
    CALC_ABSTAIN_FUNCS -- but the guard below is structural, not
    coincidental on that being true forever."""
    try:
        F, a, b, exact = _calc_prepare(expr_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')

    fa, fb = F.subs(x, a), F.subs(x, b)
    lhs = sp.simplify(fb - fa)

    try:
        continuous = calc_continuous_on(F, a, b)
        continuous['provenance'] = calc_cap(continuous['provenance'], exact)
    except Abstain as e:
        continuous = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    if continuous['pass'] is True:
        try:
            differentiable = calc_differentiable_on(F, a, b)
            differentiable['provenance'] = calc_cap(differentiable['provenance'], exact)
        except Abstain as e:
            differentiable = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    elif continuous['pass'] is False:
        differentiable = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'not continuous'}], 'provenance': continuous['provenance']}
    else:
        differentiable = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': 'continuity undecided'}

    fp = sp.diff(F, x)
    try:
        bounded = calc_bounded_on(fp, a, b)
        bounded['provenance'] = calc_cap(bounded['provenance'], exact)
    except Abstain as e:
        bounded = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    hyps = {'continuous': continuous, 'differentiable': differentiable, 'bounded': bounded}
    all_pass = all(h.get('pass') is True for h in hyps.values())

    # Which act halts the proof (non-negotiable #4)? Continuity failing
    # ALWAYS blocks; boundedness failing halts the ordinary Riemann-sum
    # construction. Differentiability failing NEVER blocks -- see the
    # docstring above. Mirrors netchangeEngine.js's numericBlockedAct so the
    # two tiers agree by construction.
    blocked = None
    if continuous['pass'] is False:
        blocked = 'continuous'
    elif bounded['pass'] is False:
        blocked = 'bounded'

    # The RHS is only ever attempted once boundedness is EXACTLY certified
    # (never merely abstained/unknown) -- see trap (2) above for why a
    # successful sp.integrate() on an undecided F' would prove nothing.
    integral_value, cert = None, None
    if blocked is None and bounded['pass'] is True:
        try:
            integral_value, cert = calc_definite_integral(fp, a, b)
        except Abstain:
            integral_value, cert = None, None

    identity_holds = bool(
        blocked is None and integral_value is not None
        and sp.simplify(integral_value - lhs) == 0
    )

    return {
        'id': 'netchange',
        'provenance': _calc_overall_netchange(hyps, blocked),
        'exactInputs': exact,
        'f': {'tex': sp.latex(F), 'fpTex': sp.latex(fp)},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'lhs': {'value': _calc_num(lhs), 'tex': sp.latex(lhs)},
        'integral': {
            'value': _calc_num(integral_value) if integral_value is not None else None,
            'tex': sp.latex(integral_value) if integral_value is not None else None,
            'certificate': cert,
        },
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'identityHolds': identity_holds,
    }


def _calc_prepare_ftc(f_str, F_str, a_str, b_str):
    """_calc_prepare's own three-string signature (f, a, b) is shared by
    rolle/mvt/limits/ivt/riemann/netchange -- their call sites take exactly
    that shape, so it is extended ADDITIVELY here (a fresh sibling helper,
    never a changed signature on the existing one) rather than widened
    in place. FTC is the first Calculus module whose input genuinely has a
    SECOND expression (the student's candidate antiderivative F, alongside
    f) -- CLAUDE.md: "if a new theorem or domain strains the contract,
    change the contract, do not bend the theorem to fit". Returns
    (f, F, a, b, exact_inputs)."""
    f = calc_parse(f_str)
    F = calc_parse(F_str)
    a, ea = calc_parse_scalar(a_str)
    b, eb = calc_parse_scalar(b_str)
    if not (a.is_real and b.is_real) or not (a < b):
        raise ValueError('need real a < b')
    return f, F, a, b, (ea and eb)


def _calc_overall_ftc(continuous, part2, blocked):
    """Module-level provenance for analyse_ftc: the weakest link among
    continuity and Part 2's own two exact checks (F really is an
    antiderivative of f; the evaluation identity itself). Mirrors
    _calc_overall_netchange's shape: any_fail collects every clause that
    is DEFINITELY false (never merely abstained) and reports the weakest
    provenance among THOSE; otherwise the weakest provenance among
    everything that was actually decided."""
    provs = [continuous['provenance'], part2['antiderivativeProvenance'], part2['provenance']]
    any_fail = blocked is not None or part2.get('isAntiderivative') is False or part2.get('identityHolds') is False
    if any_fail:
        fail_provs = [p for p in provs if p in ('refuted', 'numeric')]
        return 'numeric' if 'numeric' in fail_provs else 'refuted'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    return 'proved'


def analyse_ftc(f_str, F_str, a_str, b_str):
    """f_str/F_str/a_str/b_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy) -- FOUR strings, not three (see
    _calc_prepare_ftc's own docstring for why this is an additive contract
    change, not a modification to the shared three-arg _calc_prepare).
    Topic 2.7: the Fundamental Theorem of Integral Calculus, both halves:

      Part 1: f continuous on [a,b] => G(x) = int_a^x f(t)dt is
              differentiable on [a,b] with G'(x) = f(x) everywhere.
      Part 2: for ANY antiderivative F of f (F' = f on [a,b]),
              int_a^b f(x)dx = F(b) - F(a).

    ONE hypothesis (continuity of f) -- see ftcEngine.js's own header for
    why there is deliberately no differentiability-of-f hypothesis (FTC's
    whole point) and no "hypothesis fails yet conclusion holds" preset
    group (continuity is NECESSARY here, by Darboux's theorem: a
    derivative cannot itself have a jump, so no F with F'=f can exist on
    an interval where f jumps -- Part 2's own premise has nothing to be
    true of). Part 1 itself is not independently re-verified symbolically
    at a SPECIFIC point the way the numeric tier's checkPart1 inspects
    x0's -- G(x) as a genuinely new symbolic object (a Piecewise-free
    accumulation function) is exactly what Part 2's antiderivative check
    already certifies for the STUDENT's own F, which is the more useful
    verdict here: "is what the student typed actually right", not merely
    "does some primitive exist" (SymPy could always construct one via
    sp.integrate(f, x), but that does not confirm the student's own
    candidate). Part 1 is instead reported qualitatively: whenever
    continuity holds, Part 1 is GUARANTEED by the theorem itself (a proof
    fact, not a per-input computation), exactly as ftcEngine.js's own
    checkPart1 confirms numerically at inspection points -- the symbolic
    tier's job is to certify the ONE thing that genuinely varies per
    input and per student answer: whether F is exactly f's antiderivative,
    and whether the evaluation identity holds for it.

    Reuses calc_continuous_on (the one hypothesis) and calc_definite_integral
    (the ground-truth integral value) verbatim -- the SAME primitives
    riemann/netchange already use. Two checks or the module's own:
    (1) sp.simplify(diff(F,x) - f) == 0 -- is F genuinely an antiderivative
        of f? A DIRECT symbolic equality test, not routed through
        calc_bounded_on/CALC_ABSTAIN_FUNCS -- confirmed while building this
        module that this succeeds even when diff(F,x) itself CONTAINS
        sign() (e.g. F = x*Abs(x)/2, f = Abs(x): diff(F,x) = x*sign(x)/2 +
        Abs(x)/2, which sp.simplify correctly reduces to 0 against f,
        because x is declared real=True at module scope -- x*sign(x) =
        Abs(x) for real x is exactly the identity SymPy's simplify already
        knows). This is a GENUINE positive finding, not a trap: unlike
        netchange's own boundedness-of-F' check (which must abstain via
        CALC_ABSTAIN_FUNCS the instant sign()/floor()/etc. appear, because
        that machinery needs continuous_domain/singularities to reason
        about an INTERVAL), a direct pointwise symbolic equality test has
        no such dependency and can decide exactly through a sign()-bearing
        expression.
    (2) sp.simplify((F(b)-F(a)) - integral_value) == 0 -- the evaluation
        identity itself, using calc_definite_integral's exact ground truth
        (never a coincidental match promoted to proved -- see
        analyse_netchange's own documented trap (2) for why a successful
        sp.integrate() must never stand in for a hypothesis check by
        itself: here it is gated on continuity holding AND F genuinely
        being an antiderivative, never on its own).

    Returns VisualMath's flat {provenance, ...} shape (never c-wrapped)."""
    try:
        f, F, a, b, exact = _calc_prepare_ftc(f_str, F_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')

    Fa, Fb = F.subs(x, a), F.subs(x, b)
    lhs = sp.simplify(Fb - Fa)

    try:
        continuous = calc_continuous_on(f, a, b)
        continuous['provenance'] = calc_cap(continuous['provenance'], exact)
    except Abstain as e:
        continuous = {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}

    # THREE-WAY gate on continuity, not a binary "blocked or not": calc_
    # continuous_on abstains OUTRIGHT (pass=None) on floor/ceiling/sign/
    # Piecewise -- the SAME trap riemann's/netchange's own floor(x) presets
    # already document -- and an abstained hypothesis must NOT be silently
    # treated as "not blocked, proceed" (that bug produced a false
    # 'refuted' for this module's own floor(x) preset during development,
    # instead of the correct 'unknown' every sibling module's floor(x)
    # preset reports symbolically). Only pass=True proceeds to Part 2.
    blocked = 'continuous' if continuous['pass'] is False else None

    is_antideriv = None
    antideriv_provenance = 'unknown'
    integral_value, cert = None, None
    identity_holds = None
    identity_provenance = 'unknown'

    if continuous['pass'] is True:
        try:
            Fp = sp.diff(F, x)
            is_antideriv = bool(sp.simplify(Fp - f) == 0)
            antideriv_provenance = calc_cap('proved' if is_antideriv else 'refuted', exact)
        except Exception:  # noqa: BLE001
            is_antideriv = None
            antideriv_provenance = 'unknown'

        try:
            integral_value, cert = calc_definite_integral(f, a, b)
        except Abstain:
            integral_value, cert = None, None

        if is_antideriv is True and integral_value is not None:
            try:
                identity_holds = bool(sp.simplify(lhs - integral_value) == 0)
                identity_provenance = calc_cap('proved' if identity_holds else 'refuted', exact)
            except Exception:  # noqa: BLE001
                identity_holds = None
                identity_provenance = 'unknown'
        elif is_antideriv is False:
            identity_holds = False
            identity_provenance = antideriv_provenance
    elif continuous['pass'] is False:
        # blocked: Part 2's own premise has nothing to be true of (Darboux
        # -- no F can have F'=f on an interval where f jumps), mirrored
        # exactly as ftcEngine.js's own 'skipped' issue.
        is_antideriv = False
        antideriv_provenance = continuous['provenance']
        identity_holds = False
        identity_provenance = continuous['provenance']
    # else: continuity itself abstained (pass=None) -- Part 2 stays
    # entirely 'unknown', never guessed at.

    part2 = {
        'isAntiderivative': is_antideriv,
        'antiderivativeProvenance': antideriv_provenance,
        'identityHolds': identity_holds,
        'provenance': identity_provenance,
    }

    all_pass = bool(continuous.get('pass') is True and is_antideriv is True and identity_holds is True)

    return {
        'id': 'ftc',
        'provenance': _calc_overall_ftc(continuous, part2, blocked),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)},
        'F': {'tex': sp.latex(F)},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'hypotheses': {'continuous': continuous},
        'blockedAct': blocked,
        'lhs': {'value': _calc_num(lhs), 'tex': sp.latex(lhs)},
        'integral': {
            'value': _calc_num(integral_value) if integral_value is not None else None,
            'tex': sp.latex(integral_value) if integral_value is not None else None,
            'certificate': cert,
        },
        'part2': part2,
        'allHypothesesPass': all_pass,
    }


def _calc_prepare_improper(expr_str, a_str, b_str):
    """_calc_prepare's own three-string signature, additively widened: a
    bound may be an INFINITE extended-real (oo/-oo), which fails a plain
    \`.is_real\` check (oo.is_real is False; oo.is_extended_real is True) --
    every other Calculus module's [a,b] is always finite, so this is the
    module's own small extension, not a change to the shared three-arg
    _calc_prepare every other analyse_* still calls unmodified."""
    f = calc_parse(expr_str)
    a, ea = calc_parse_scalar(a_str)
    b, eb = calc_parse_scalar(b_str)
    if not (a.is_extended_real and b.is_extended_real) or not (a < b):
        raise ValueError('need extended-real a < b (a bound may be oo or -oo)')
    return f, a, b, (ea and eb)


def _improper_blows_up(f, p, direction) -> bool:
    """Does f fail to have a finite real limit approaching p from
    \`direction\` ('+' or '-')? Raises Abstain if the limit itself cannot be
    decided (mirrors calc_bounded_on's own limit-failure handling)."""
    try:
        lim = sp.limit(f, x, p, direction)
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'limit failed at {p}: {e}') from e
    return not (lim.is_finite and lim.is_real)


def analyse_improper(expr_str, a_str, b_str):
    """expr_str/a_str/b_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy, PLUS this module's own parseBound, which
    prints 'oo'/'-oo' for an infinite endpoint -- sympify recognises both
    natively, no CALC_NAMESPACE change needed). Topic 2.8: improper
    integrals -- classification & convergence. A CONCEPT module (mirrors
    analyse_limits/analyse_riemann's shape, not analyse_rolle/analyse_ivt's),
    on the SAME chained-clause structure as improperEngine.js's own header:

      (1) classification (Type I / Type II / both / neither) -- INFORMATION,
          not a pass/fail clause.
      (2) wellBehaved -- no UNACCOUNTED singularity strictly inside (a,b),
          beyond the declared endpoint(s). The genuinely breakable clause.
      (3) convergent -- the defining limit exists and is finite. Unlike the
          NUMERIC engine (which must manually split at an interior point and
          march two independent ladders when BOTH ends are improper), the
          symbolic tier hands the WHOLE (possibly doubly-improper) interval
          straight to sp.integrate(f, (x, a, b)) -- SymPy's own integration
          machinery already performs the equivalent split-and-limit
          internally (confirmed by hand: integrate(exp(-x**2), (x,-oo,oo))
          returns sqrt(pi) directly, with no manual split needed on this
          side, unlike improperEngine.js's own marching pieces).

    SymPy trap, confirmed by hand while building this module and documented
    as this module's OWN addition to the floor/ceiling/sign/Piecewise
    abstain family: an integral that genuinely OSCILLATES without settling
    (e.g. integrate(cos(x), (x, 0, oo))) returns an \`AccumBounds\` object, not
    oo/-oo/zoo and not an unevaluated Integral -- a THIRD shape genuine
    divergence can take here, distinct from both. Caught explicitly below
    (never silently coerced to a number, never abstained on either -- an
    AccumBounds result IS a definitive answer: the limit does not exist).

    A second trap: Abs()-wrapped integrands are frequently where
    sp.integrate gives up -- the flagship conditional-convergence preset
    (sin(x)/x on [1,oo), absolute value diverging) is exactly this shape:
    integrate(sin(x)/x, (x,1,oo)) evaluates cleanly in closed form, but
    integrate(Abs(sin(x)/x), (x,1,oo)) returns an unevaluated Integral. This
    is NOT treated as an overall abstention (the convergence verdict itself
    is already decided and exact) -- only the absolute-convergence LAYER
    (non-gating, exactly like improperEngine.js's own absoluteConvergence)
    reports 'unknown' for that one sub-question, an honest, narrow
    abstention rather than one that drags down the whole result.
    """
    try:
        f, a, b, exact = _calc_prepare_improper(expr_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')

    left_inf = (a == -oo)
    right_inf = (b == oo)
    try:
        left_singular = (not left_inf) and _improper_blows_up(f, a, '+')
        right_singular = (not right_inf) and _improper_blows_up(f, b, '-')
    except Abstain as e:
        return calc_unknown(str(e))

    left_kind = 'neg-inf' if left_inf else ('singular' if left_singular else 'none')
    right_kind = 'pos-inf' if right_inf else ('singular' if right_singular else 'none')

    try:
        interior = singularities(f, x, Interval.open(a, b))
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot compute interior singularities: {e}')
    if not interior.is_FiniteSet:
        return calc_unknown('interior singularity set is not finite')
    interior_pts = sorted((p for p in interior if p.is_real), key=lambda p: float(p))

    well_behaved_pass = len(interior_pts) == 0
    well_behaved = {
        'pass': well_behaved_pass,
        'issues': [] if well_behaved_pass else [
            {'kind': 'unaccounted-singularity', 'x': _calc_num(p), 'xTex': sp.latex(p),
             'detail': 'f is unbounded here, strictly inside (a, b) -- unaccounted for by the declared endpoint(s)'}
            for p in interior_pts],
        'provenance': calc_cap('proved' if well_behaved_pass else 'refuted', exact),
    }

    n_improper = (left_kind != 'none') + (right_kind != 'none')
    itype = 'neither' if n_improper == 0 else ('one' if n_improper == 1 else 'both')
    parts = []
    if left_kind == 'neg-inf' or right_kind == 'pos-inf':
        parts.append('Type I (infinite bound)')
    if left_kind == 'singular' or right_kind == 'singular':
        parts.append('Type II (unbounded near an endpoint)')
    type_label = ' and '.join(parts) if parts else 'neither — an ordinary proper integral'
    classification = {
        'leftKind': left_kind, 'rightKind': right_kind, 'type': itype, 'typeLabel': type_label,
        'interiorSingularities': [_calc_num(p) for p in interior_pts],
    }

    blocked = None
    value = None
    value_tex = None
    abs_conv = {'pass': None, 'applicable': False, 'label': None}

    if not well_behaved_pass:
        blocked = 'wellBehaved'
        convergent = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f is not well-behaved away from the declared improper point(s)'}],
                      'provenance': well_behaved['provenance']}
    else:
        try:
            val = sp.integrate(f, (x, a, b))
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'integration failed: {e}')
        if val.has(sp.Integral):
            return calc_unknown('SymPy could not evaluate this integral in closed form')
        is_accum = isinstance(val, sp.calculus.accumulationbounds.AccumulationBounds)
        if is_accum:
            convergent = {'pass': False, 'issues': [{'kind': 'oscillates', 'detail': 'the integral oscillates without settling (SymPy returns an accumulation range, not a single value)'}],
                          'provenance': calc_cap('refuted', exact)}
            value = None
        elif val in (oo, -oo, sp.zoo, sp.nan):
            convergent = {'pass': False, 'issues': [{'kind': 'divergent', 'detail': 'the integral diverges to infinity'}],
                          'provenance': calc_cap('refuted', exact)}
            value = None
        elif val.is_real is False:
            return calc_unknown('integral did not evaluate to a real number')
        else:
            convergent = {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact)}
            value = val
            value_tex = sp.latex(val)

        if convergent['pass'] is True:
            if itype == 'neither':
                abs_conv = {'pass': True, 'applicable': True, 'trivial': True, 'label': 'absolute'}
            else:
                try:
                    aval = sp.integrate(Abs(f), (x, a, b))
                except Exception as e:  # noqa: BLE001
                    abs_conv = {'pass': None, 'applicable': True, 'label': None, 'reason': f'absolute-convergence check failed: {e}'}
                else:
                    if aval.has(sp.Integral):
                        abs_conv = {'pass': None, 'applicable': True, 'label': None,
                                    'reason': 'SymPy could not decide absolute convergence in closed form'}
                    elif isinstance(aval, sp.calculus.accumulationbounds.AccumulationBounds) or aval in (oo, -oo, sp.zoo, sp.nan):
                        abs_conv = {'pass': False, 'applicable': True, 'label': 'conditional'}
                    else:
                        abs_conv = {'pass': True, 'applicable': True, 'label': 'absolute', 'value': _calc_num(aval)}
        else:
            blocked = 'convergent'

    all_pass = bool(blocked is None and convergent.get('pass') is True)
    hyps = {'wellBehaved': well_behaved, 'convergent': convergent}
    return {
        'id': 'improper',
        'provenance': _calc_overall_limits(hyps, blocked is not None),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)},
        'interval': {
            'a': _calc_num(a), 'b': _calc_num(b),
            'aTex': sp.latex(a), 'bTex': sp.latex(b),
        },
        'classification': classification,
        'hypotheses': hyps,
        'blockedAct': blocked,
        'allHypothesesPass': all_pass,
        'integral': {
            'value': _calc_num(value) if (blocked is None and value is not None) else None,
            'tex': value_tex,
        },
        'absoluteConvergence': abs_conv,
    }


_GB_X = Symbol('xg', positive=True)


def _gb_domain_issue(label, val):
    """One domain-check issue dict. \`label\` is '' for Gamma's sole parameter
    s, or 'p'/'q' for Beta's two independent parameters."""
    exponent = 1 - val
    name = label or 's'
    return {
        'kind': 'domain', 'side': (label or None),
        'detail': f'{name} = {sp.nsimplify(val)} <= 0: the Type II p-test exponent 1-{name} = {sp.nsimplify(exponent)} >= 1 -- the defining integral does not converge (a statement about the plain defining integral, never about evaluating a pole of the analytically-continued function).',
    }


def _gb_integrate_or_none(expr, bounds):
    """sp.integrate on the RAW defining integral, never sp.gamma()/sp.beta()
    as the primary evaluator -- see analyse_gammabeta's own docstring for the
    two confirmed-by-hand traps this sidesteps. Returns (value, None) or
    (None, reason), mirroring analyse_improper's own AccumBounds/±∞/zoo/
    unevaluated-Integral handling verbatim (this integral family can hit the
    SAME shapes analyse_improper's own docstring already documents)."""
    try:
        v = sp.integrate(expr, bounds)
    except Exception as e:  # noqa: BLE001
        return None, f'integration failed: {e}'
    if v.has(sp.Integral):
        return None, 'SymPy could not evaluate this integral in closed form'
    if isinstance(v, sp.calculus.accumulationbounds.AccumulationBounds):
        return None, 'the integral oscillates without settling'
    if v in (oo, -oo, sp.zoo, sp.nan):
        return None, 'the integral diverges to infinity'
    if v.is_real is False:
        return None, 'integral did not evaluate to a real number'
    return v, None


def analyse_gammabeta(kind, p_str, q_str):
    """kind: 'gamma' | 'beta'. Topic 2.9 (Gamma & Beta functions). For
    'gamma', p_str carries s (the sole parameter) and q_str is UNUSED (the JS
    side still sends a uniform three-argument payload -- the same "extend
    additively, keep the call shape uniform" precedent as analyse_ftc's own
    four-string signature, rather than two differently-shaped DISPATCH
    entries for one module); for 'beta', p_str/q_str are p and q.

    Domain (s > 0, or p > 0 AND q > 0) is checked FIRST and independently of
    any integration -- this is the one genuinely breakable clause (see
    gammabetaEngine.js's own header for the full rationale, identical here).
    Once domain holds, sp.integrate runs ON THE RAW DEFINING INTEGRAL
    directly (mirroring analyse_improper's own established choice), NOT
    sp.gamma()/sp.beta() as the primary evaluator -- two traps confirmed by
    hand while building this module, both avoided by evaluating the domain
    gate before ever calling sp.integrate:
      1. sp.gamma(Rational(-1,2)) returns -2*sqrt(pi) -- a FINITE value via
         analytic continuation. Trusting it for s<=0 would silently answer
         with the analytically-continued Gamma function (a different, later
         object), not this course's plain defining integral, which does not
         converge there at all.
      2. sp.simplify(sp.beta(-1, 2)) returns 0 directly (confirmed by hand)
         -- silently misleading, since the true defining integral diverges
         (oo). Only sp.beta(...).rewrite(sp.gamma) reveals the correct zoo.
    Because domain already gates every s<=0 / p<=0 / q<=0 input before any
    evaluation is attempted, neither trap can actually fire here -- but it is
    exactly why sp.integrate on the raw integral, not the closed-form
    special function, is this module's primary evaluator, the same choice
    analyse_improper already made and for the identical reason.

    Once the value itself is established, the flagship IDENTITIES are
    verified independently (never merely asserted from the theorem):
    Gamma's functional equation Gamma(s+1) = s*Gamma(s) via a SECOND
    sp.integrate call, and Beta's symmetry B(p,q)=B(q,p) plus the
    Beta-Gamma relation B(p,q) = Gamma(p)Gamma(q)/Gamma(p+q) via THREE
    further separate sp.integrate calls -- genuinely independent routes to
    the same claim, not one formula echoing itself. Non-negotiable #5 (the
    conclusion is an iterator where existence is being taught) is N/A here,
    the same documented call analyse_netchange/analyse_ftc made for their
    own identities: these are identities between already-computed numbers,
    not existence claims.
    """
    try:
        p, ep = calc_parse_scalar(p_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    q = None
    if kind == 'beta':
        try:
            q, eq = calc_parse_scalar(q_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad input: {e}')
        exact = ep and eq
    else:
        exact = ep

    if kind == 'gamma':
        domain_pass = bool(p > 0)
        issues = [] if domain_pass else [_gb_domain_issue('', p)]
    elif kind == 'beta':
        issues = []
        if not bool(p > 0):
            issues.append(_gb_domain_issue('p', p))
        if not bool(q > 0):
            issues.append(_gb_domain_issue('q', q))
        domain_pass = len(issues) == 0
    else:
        return calc_unknown(f"unknown kind '{kind}' (expected 'gamma' or 'beta')")

    domain = {'pass': domain_pass, 'issues': issues, 'provenance': calc_cap('proved' if domain_pass else 'refuted', exact)}

    if not domain_pass:
        return {
            'id': 'gammabeta', 'kind': kind, 'provenance': domain['provenance'], 'exactInputs': exact,
            'domain': domain, 'blockedAct': 'domain',
            'convergent': {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'the domain requirement fails -- there is no legitimate convergence question to even ask yet'}], 'provenance': domain['provenance']},
            'allHypothesesPass': False, 'value': {'value': None, 'tex': None}, 'identity': None,
        }

    x = _GB_X
    if kind == 'gamma':
        val, err = _gb_integrate_or_none(x ** (p - 1) * sp.exp(-x), (x, 0, oo))
    else:
        val, err = _gb_integrate_or_none(x ** (p - 1) * (1 - x) ** (q - 1), (x, 0, 1))

    if val is None:
        convergent = {'pass': False, 'issues': [{'kind': 'divergent', 'detail': err}], 'provenance': calc_cap('refuted', exact)}
        blocked = 'convergent'
        value_block = {'value': None, 'tex': None}
        identity = None
    else:
        convergent = {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact)}
        blocked = None
        value_block = {'value': _calc_num(val), 'tex': sp.latex(val)}
        identity = None
        if kind == 'gamma':
            val2, err2 = _gb_integrate_or_none(x ** p * sp.exp(-x), (x, 0, oo))
            if val2 is not None:
                rhs = sp.simplify(p * val)
                identity = {
                    'kind': 'functional-equation',
                    'lhs': _calc_num(val2), 'rhs': _calc_num(rhs),
                    'lhsTex': sp.latex(val2), 'rhsTex': sp.latex(rhs),
                    'holds': bool(sp.simplify(val2 - rhs) == 0),
                }
            else:
                identity = {'kind': 'functional-equation', 'lhs': None, 'rhs': None, 'holds': None, 'reason': err2}
        else:
            val_swapped, err_sw = _gb_integrate_or_none(x ** (q - 1) * (1 - x) ** (p - 1), (x, 0, 1))
            symmetry = None
            if val_swapped is not None:
                symmetry = {'a': _calc_num(val), 'b': _calc_num(val_swapped), 'holds': bool(sp.simplify(val - val_swapped) == 0)}
            gp, egp = _gb_integrate_or_none(x ** (p - 1) * sp.exp(-x), (x, 0, oo))
            gq, egq = _gb_integrate_or_none(x ** (q - 1) * sp.exp(-x), (x, 0, oo))
            gpq, egpq = _gb_integrate_or_none(x ** (p + q - 1) * sp.exp(-x), (x, 0, oo))
            beta_gamma = None
            if gp is not None and gq is not None and gpq is not None:
                rhs = sp.simplify((gp * gq) / gpq)
                beta_gamma = {
                    'lhs': _calc_num(val), 'rhs': _calc_num(rhs),
                    'lhsTex': sp.latex(val), 'rhsTex': sp.latex(rhs),
                    'holds': bool(sp.simplify(val - rhs) == 0),
                }
            identity = {'kind': 'beta-gamma', 'symmetry': symmetry, 'betaGamma': beta_gamma}

    provs = [domain['provenance'], convergent['provenance']]
    overall = 'unknown' if 'unknown' in provs else ('refuted' if blocked else ('numeric' if 'numeric' in provs else 'proved'))

    return {
        'id': 'gammabeta', 'kind': kind, 'provenance': overall, 'exactInputs': exact,
        'domain': domain, 'convergent': convergent, 'blockedAct': blocked,
        'allHypothesesPass': blocked is None, 'value': value_block, 'identity': identity,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — sequences (topic 2.10) ──
# =============================================================================
# n is its OWN symbol -- integer, positive -- distinct from the shared
# continuous-x symbol \`x\` every other Calculus analyse_* uses. A discrete
# sequence a_n = f(n) is genuinely a different object from a continuous
# function evaluated at a point, so this module gets its own small
# parse/limit helpers rather than routing through calc_parse/_calc_prepare
# (which are hardwired to the shared \`x\` symbol) -- the same "a new object
# gets its own small helpers" precedent gammabeta's own classifyGamma/
# classifyBeta (built analytically, not via calc_parse) already set.
_SEQ_N = Symbol('n', integer=True, positive=True)
_SEQ_NAMESPACE = {**CALC_NAMESPACE, 'n': _SEQ_N}


def _seq_parse(expr_str):
    """Parse a_n = f(n) from expr.mjs::toSympy()'s output, called with n (not
    x) as the free variable (src/modules/sequencesEngine.js's own parseExpr
    wrapper calls parseExpr(text, ['n']))."""
    try:
        f = sp.sympify(expr_str, locals=_SEQ_NAMESPACE)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'cannot parse: {e}') from e
    bad = f.free_symbols - {_SEQ_N}
    if bad:
        raise ValueError(f'unexpected symbols {sorted(map(str, bad))}')
    return f


def _seq_limit(f):
    """The PRIMARY evaluator is \`limit_seq\` (sympy.series.limitseq), built
    specifically for discrete n->oo SEQUENCE limits (Poincare-type asymptotic
    analysis) -- confirmed by hand to correctly resolve alternating-sign
    forms the generic CONTINUOUS \`sp.limit\` cannot: \`sp.limit((-1)**n, n,
    oo)\` raises "Result depends on the sign of I" (it tries to route
    (-1)**n through a complex-exponential representation and gets stuck
    resolving the branch), while \`limit_seq((-1)**n, n)\` correctly returns
    \`AccumBounds(-1, 1)\`. Likewise \`sp.limit((-1)**n/n, n, oo)\` and
    \`sp.limit((-1)**n*n, n, oo)\` both raise the SAME error, while
    \`limit_seq\` resolves them cleanly (0, and AccumBounds(-oo, oo)
    respectively -- the latter is a genuinely UNBOUNDED oscillation, see
    below). \`limit_seq\` itself sometimes DECLINES (returns None -- confirmed
    by hand for cos(pi*n/2), a clean period-4 oscillator) -- in that case,
    fall back to the plain continuous \`sp.limit\`, which DOES resolve that
    one (also AccumBounds(-1, 1), confirmed by hand). Either stage may still
    raise on unusual input; the caller (analyse_sequences) turns any
    exception into an honest 'unknown' abstain rather than letting it
    propagate -- this alternating-sign family is this module's own addition
    to the domain's running list of documented SymPy traps (floor/ceiling/
    sign/Piecewise, AccumBounds-shaped divergence, Abs-wrapped integrands)."""
    lim = limit_seq(f, _SEQ_N)
    if lim is not None:
        return lim
    return sp.limit(f, _SEQ_N, oo)


def analyse_sequences(expr_str, startN_str):
    """expr_str: SymPy-syntax string from expr.mjs's toSympy(), parsed with n
    (not x) as the free variable. startN_str: SymPy-syntax string from
    expr.mjs's parseNumber().sympy (e.g. 'Rational(1, 1)').

    Topic 2.10: Sequences of real numbers. A CONCEPT module, the fourth
    built on the chained-clause shape limits (2.1) / riemann (2.3) /
    improper (2.8) introduced -- see sequencesEngine.js's own header for the
    full clause rationale (bounded gates convergent; a non-gating monotonic
    layer is numeric-only, see below).

    startN validity is checked FIRST and independently of any limit call --
    the same "domain gates before evaluation" discipline analyse_gammabeta's
    own docstring documents.

    \`_seq_limit\` returns one of four shapes, each handled explicitly (never
    a silent catch-all):
      1. A finite real value -> convergent=True, provenance='proved'.
         Bounded is then also True, reported as an ENTAILED consequence
         (a convergent sequence is bounded) rather than re-derived
         separately -- the same choice analyse_gammabeta made reporting
         \`identity\` fields as derived-not-reprobed facts.
      2. +oo/-oo/zoo/nan -> convergent=False, bounded=False,
         provenance='refuted': definite divergence to infinity (nan is
         included here -- confirmed by hand: sp.limit((-1)**n, n, oo)
         itself, when it does not raise, can return \`nan\` rather than a
         genuine AccumBounds, e.g. for some ill-conditioned intermediate
         forms; nan carries no usable envelope, so it is treated the same
         as an outright divergence rather than routed through the
         AccumBounds branch below).
      3. An AccumBounds object -- genuine bounded-or-unbounded OSCILLATION,
         never settling. Its .min/.max are extracted: if BOTH are finite,
         report bounded=True with the actual envelope (e.g. sin(n) ->
         AccumBounds(-1, 1) -> bounded True, envelope [-1, 1]); if either
         bound is infinite (e.g. (-1)**n * n -> AccumBounds(-oo, oo),
         confirmed by hand via limit_seq), the oscillation is ITSELF
         unbounded, so bounded=False instead -- the symbolic-tier mirror of
         sequencesEngine.js's own magnitude-based checkBounded, which does
         not care whether growth is monotone or sign-flipping.
      4. Anything else (a raised exception from either limit attempt, or an
         unevaluated Limit) -> abstain, provenance='unknown', matching this
         domain's universal abstention contract.

    The \`monotonic\` classification sequencesEngine.js reports is NUMERIC
    ONLY (informational, non-gating even there) -- there is no symbolic
    monotonicity check here; the UI simply does not display a symbolic
    monotonic field.
    """
    try:
        f = _seq_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        startN_val, exact = calc_parse_scalar(startN_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    if not (startN_val.is_integer and bool(startN_val > 0)):
        return calc_unknown('startN must be a positive integer')
    startN = int(startN_val)

    try:
        lim = _seq_limit(f)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'limit failed: {e}')

    if lim is None or lim.has(sp.Limit):
        return calc_unknown('SymPy could not evaluate this limit in closed form')

    if isinstance(lim, sp.calculus.accumulationbounds.AccumulationBounds):
        lo, hi = lim.min, lim.max
        finite_env = bool(lo.is_finite) and bool(hi.is_finite)
        if finite_env:
            return {
                'id': 'sequences', 'provenance': calc_cap('refuted', exact), 'exactInputs': exact,
                'startN': startN, 'boundedPass': True, 'boundedEnvelope': [_calc_num(lo), _calc_num(hi)],
                'convergentPass': False, 'blockedAct': 'convergent',
                'limit': None, 'limitTex': None, 'kind': 'oscillates', 'allHypothesesPass': False,
            }
        return {
            'id': 'sequences', 'provenance': calc_cap('refuted', exact), 'exactInputs': exact,
            'startN': startN, 'boundedPass': False, 'boundedEnvelope': None,
            'convergentPass': False, 'blockedAct': 'bounded',
            'limit': None, 'limitTex': None, 'kind': 'diverges', 'allHypothesesPass': False,
        }

    if lim in (oo, -oo, sp.zoo, sp.nan):
        return {
            'id': 'sequences', 'provenance': calc_cap('refuted', exact), 'exactInputs': exact,
            'startN': startN, 'boundedPass': False, 'boundedEnvelope': None,
            'convergentPass': False, 'blockedAct': 'bounded',
            'limit': None, 'limitTex': None, 'kind': 'diverges', 'allHypothesesPass': False,
        }
    if lim.is_real is False:
        return calc_unknown('limit did not evaluate to a real number')

    return {
        'id': 'sequences', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
        'startN': startN, 'boundedPass': True, 'boundedEnvelope': None,
        'convergentPass': True, 'blockedAct': None,
        'limit': _calc_num(lim), 'limitTex': sp.latex(lim), 'kind': 'converges', 'allHypothesesPass': True,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — series: ratio and root tests (topic 2.11) ──
# =============================================================================
# THEOREM module, the direct sequel to sequences (2.10) — see
# src/modules/seriesEngine.js's own header for the full design rationale.
# ONE hypothesis: limitExists — does lim_{n->infinity} b_n exist as a single
# value L (a finite number, or +infinity as a definite, usable degenerate
# case), where b_n is the RATIO |a_{n+1}/a_n| or the ROOT |a_n|^(1/n) of the
# user's a_n = f(n)? Given L, the CONCLUSION is a three-way classification,
# never existence: L < 1 -> converges; L > 1 (or L = +oo) -> diverges;
# L = 1 -> inconclusive (a mathematically EXACT fact once L is known
# exactly, not this module abstaining — see the \`verdict\` field below,
# orthogonal to \`provenance\`, exactly mirroring seriesEngine.js's own
# result shape).
#
# ── Sanctioned narrow reuse ──────────────────────────────────────────────
# This calls the SAME \`_seq_limit\` helper analyse_sequences already defined
# (built on SymPy's own \`limit_seq\`, with a fallback to the generic
# continuous \`limit\` — see \`_seq_limit\`'s own docstring for why both stages
# are needed), applied to the ratio/root expression built from f, rather
# than writing a second symbolic limit-finding routine. Confirmed by hand
# for every one of this module's own presets (see seriesPresets.mjs) before
# writing this function: \`limit_seq\` resolves EVERY one of them EXACTLY,
# including the flagship oscillating case
# (\`Abs(((2+(-1)**(n+1))/2**(n+1)) / ((2+(-1)**n)/2**n))\` ->
# \`AccumBounds(1/6, 3/2)\`, confirming the ratio test's hypothesis genuinely
# fails there) and the root-natural case
# (\`Abs((n/(n+1))**(n**2))**(1/n)\` -> \`exp(-1)\` exactly) — unlike
# seriesEngine.js's own JS numeric tier, which needed a deliberately
# narrowed, overflow-avoiding search budget (see that file's own
# SAFE_BOUNDED_OPTS header) to sample the SAME auxiliary sequences without
# running into IEEE-754 double overflow/underflow; SymPy's symbolic
# machinery has no such limitation, since it never evaluates a_n at an
# astronomically large n as a floating-point number at all.
def analyse_series(expr_str, startN_str, test_str):
    """expr_str/startN_str: SymPy-syntax strings from expr.mjs, parsed with n
    (not x) as the free variable — the SAME \`_seq_parse\`/\`_SEQ_N\` helpers
    analyse_sequences uses. test_str: 'ratio' or 'root' (anything else falls
    back to 'ratio').

    startN validity is checked FIRST and independently of any limit call,
    the same "domain gates before evaluation" discipline every other
    analyse_* in this module follows.

    Builds the auxiliary expression b_n directly from f:
      ratio: Abs(f(n+1) / f(n))
      root:  Abs(f(n)) ** (1/n)
    then calls \`_seq_limit\` on it and classifies EXACTLY like
    \`analyse_sequences\` classifies its own bounded/convergent chain, with
    ONE structural difference: an AccumBounds result here means
    limitExists is definitively FALSE (this test's ONE hypothesis fails,
    provenance='refuted') regardless of whether the envelope itself is
    finite or infinite — unlike sequences' own bounded/unbounded split on
    an AccumBounds envelope, series has no second, independent clause to
    fall back on, so ANY non-settling oscillation is the same outcome here.
    A finite +oo limit is instead a POSITIVE, 'proved' establishment of
    limitExists=True with L=+infinity (verdict='diverges') — this is the
    one case where this module's OWN classification differs in spirit from
    sequences' analogous branch (which treats +-oo/zoo/nan as a \`refuted\`
    divergence, since sequences' hypothesis is boundedness itself; here,
    the hypothesis is merely that A limit exists, and +infinity is a
    genuine limit in the extended-real sense)."""
    try:
        f = _seq_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        startN_val, exact = calc_parse_scalar(startN_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    if not (startN_val.is_integer and bool(startN_val > 0)):
        return calc_unknown('startN must be a positive integer')
    startN = int(startN_val)
    test_mode = 'root' if str(test_str) == 'root' else 'ratio'

    try:
        if test_mode == 'root':
            aux = sp.Abs(f) ** (sp.Integer(1) / _SEQ_N)
        else:
            aux = sp.Abs(f.subs(_SEQ_N, _SEQ_N + 1) / f)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'could not build the auxiliary sequence: {e}')

    try:
        lim = _seq_limit(aux)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'limit failed: {e}')

    if lim is None or lim.has(sp.Limit):
        return calc_unknown('SymPy could not evaluate this limit in closed form')

    if isinstance(lim, sp.calculus.accumulationbounds.AccumulationBounds):
        lo, hi = lim.min, lim.max
        envelope = [_calc_num(lo), _calc_num(hi)] if bool(lo.is_finite) and bool(hi.is_finite) else None
        return {
            'id': 'series', 'provenance': calc_cap('refuted', exact), 'exactInputs': exact,
            'testMode': test_mode, 'startN': startN,
            'limitExists': False, 'L': None, 'Ltex': None, 'envelope': envelope,
            'verdict': None, 'blockedAct': 'limitExists', 'allHypothesesPass': False,
        }

    if lim == oo:
        return {
            'id': 'series', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
            'testMode': test_mode, 'startN': startN,
            'limitExists': True, 'L': None, 'Ltex': '\\\\infty', 'envelope': None,
            'verdict': 'diverges', 'blockedAct': None, 'allHypothesesPass': True,
        }
    if lim in (-oo, sp.zoo, sp.nan):
        return calc_unknown(f'the auxiliary sequence limit evaluated to {lim}, not a genuine real or +infinity value')
    if lim.is_real is False:
        return calc_unknown('limit did not evaluate to a real number')

    try:
        below = bool(lim < 1)
        above = bool(lim > 1)
    except TypeError:
        return calc_unknown('could not compare the limit to 1 exactly')
    verdict = 'converges' if below else ('diverges' if above else 'inconclusive')

    return {
        'id': 'series', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
        'testMode': test_mode, 'startN': startN,
        'limitExists': True, 'L': _calc_num(lim), 'Ltex': sp.latex(lim), 'envelope': None,
        'verdict': verdict, 'blockedAct': None, 'allHypothesesPass': True,
    }

# =============================================================================
# ── Calculus domain (DOMAINS[1]) — power series & radius of convergence
#    (topic 2.12) ──
# =============================================================================
# THEOREM module, the direct sequel to series (2.11) — see
# src/modules/powerseriesEngine.js's own header for the full design
# rationale. ONE hypothesis: limitExists — does
# L = lim_{n->infinity} |c_{n+1}/c_n| (ratio) or |c_n|^(1/n) (root) exist
# (finite, or +infinity), where c_n is the power series' COEFFICIENT
# sequence (NOT a fixed series' own term — the js engine's own key design
# insight, reused here unmodified: evaluating a term formula at the fixed
# offset x0 = center+1 collapses to c_n itself, so this function is built
# EXACTLY like analyse_series, just applied to c_n directly with no \`x\` or
# \`center\` substitution at all). Given L, Cauchy-Hadamard gives the radius
# directly: R = 1/L (R = +infinity when L = 0; R = 0 when L = +infinity —
# both definite, USABLE outcomes, not errors).
#
# Unlike the JS numeric engine (which needs a SECOND, narrower search-budget
# profile for a factorial-flagged coefficient formula — see
# powerseriesEngine.js's header for the full "confirmed by hand" overflow
# writeup), SymPy has NO such limitation here: it never evaluates c_n at an
# astronomically large n as a floating-point number, so a single code path
# handles c_n = 1/n! exactly the same way as any other coefficient family —
# confirmed by hand: limit_seq correctly resolves
# Abs(factorial(n)/factorial(n+1)) -> 0 exactly (giving R = +infinity), with
# no special-casing needed.
#
# startN validity is checked FIRST and independently of any limit call, the
# same "domain gates before evaluation" discipline every other analyse_* in
# this module follows — but UNLIKE analyse_sequences/analyse_series, startN
# here must be a NONNEGATIVE integer (power series conventionally start at
# n=0), not strictly positive.
def analyse_powerseries(expr_str, startN_str, test_str):
    """expr_str/startN_str: SymPy-syntax strings from expr.mjs, parsed with n
    (not x) as the free variable — the SAME \`_seq_parse\`/\`_SEQ_N\` helpers
    analyse_sequences/analyse_series use. test_str: 'ratio' or 'root'
    (anything else falls back to 'ratio').

    Builds the auxiliary expression b_n directly from the COEFFICIENT c_n:
      ratio: Abs(c(n+1) / c(n))
      root:  Abs(c(n)) ** (1/n)
    then calls \`_seq_limit\` on it and classifies EXACTLY like
    \`analyse_series\` classifies its own limitExists/L, with the FINAL step
    replaced by Cauchy-Hadamard's R = 1/L in place of series' own L-vs-1
    verdict classification (a genuinely different question: series asks
    whether Sigma c_n itself converges; this asks for the RADIUS of
    Sigma c_n (x-a)^n, a strictly more general object).

    An AccumBounds result means limitExists is definitively FALSE (this
    module's ONE hypothesis fails, provenance='refuted', no radius at all)
    regardless of whether the envelope itself is finite or infinite — the
    SAME classification analyse_series already uses for the identical
    reason (no second, independent clause to fall back on). A finite +oo
    limit is a POSITIVE, 'proved' establishment of limitExists=True with
    R=0 (the series converges only at the single point x=center) — the
    'pointOnly' case. lim=0 is a POSITIVE establishment of R=+infinity (the
    'everywhere' case) — L_ZERO_TOL is not needed here (SymPy proves lim=0
    exactly, rather than merely landing numerically close to 0)."""
    try:
        f = _seq_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        startN_val, exact = calc_parse_scalar(startN_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    if not (startN_val.is_integer and bool(startN_val >= 0)):
        return calc_unknown('startN must be a nonnegative integer')
    startN = int(startN_val)
    test_mode = 'root' if str(test_str) == 'root' else 'ratio'

    try:
        if test_mode == 'root':
            aux = sp.Abs(f) ** (sp.Integer(1) / _SEQ_N)
        else:
            aux = sp.Abs(f.subs(_SEQ_N, _SEQ_N + 1) / f)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'could not build the auxiliary sequence: {e}')

    try:
        lim = _seq_limit(aux)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'limit failed: {e}')

    if lim is None or lim.has(sp.Limit):
        return calc_unknown('SymPy could not evaluate this limit in closed form')

    if isinstance(lim, sp.calculus.accumulationbounds.AccumulationBounds):
        lo, hi = lim.min, lim.max
        envelope = [_calc_num(lo), _calc_num(hi)] if bool(lo.is_finite) and bool(hi.is_finite) else None
        return {
            'id': 'powerseries', 'provenance': calc_cap('refuted', exact), 'exactInputs': exact,
            'testMode': test_mode, 'startN': startN,
            'limitExists': False, 'L': None, 'Ltex': None, 'envelope': envelope,
            'R': None, 'Rtex': None, 'Rkind': None, 'blockedAct': 'limitExists', 'allHypothesesPass': False,
        }

    if lim == oo:
        return {
            'id': 'powerseries', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
            'testMode': test_mode, 'startN': startN,
            'limitExists': True, 'L': None, 'Ltex': '\\\\infty', 'envelope': None,
            'R': 0.0, 'Rtex': '0', 'Rkind': 'pointOnly', 'blockedAct': None, 'allHypothesesPass': True,
        }
    if lim in (-oo, sp.zoo, sp.nan):
        return calc_unknown(f'the auxiliary sequence limit evaluated to {lim}, not a genuine real or +infinity value')
    if lim.is_real is False:
        return calc_unknown('limit did not evaluate to a real number')

    if lim == 0:
        return {
            'id': 'powerseries', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
            'testMode': test_mode, 'startN': startN,
            'limitExists': True, 'L': 0.0, 'Ltex': '0', 'envelope': None,
            'R': None, 'Rtex': '\\\\infty', 'Rkind': 'everywhere', 'blockedAct': None, 'allHypothesesPass': True,
        }

    R = 1 / lim
    return {
        'id': 'powerseries', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
        'testMode': test_mode, 'startN': startN,
        'limitExists': True, 'L': _calc_num(lim), 'Ltex': sp.latex(lim), 'envelope': None,
        'R': _calc_num(R), 'Rtex': sp.latex(R), 'Rkind': 'finite', 'blockedAct': None, 'allHypothesesPass': True,
    }


def _calc_prepare_cauchy(f_str, g_str, a_str, b_str):
    """A fresh sibling of _calc_prepare (rolle/mvt/limits/ivt/riemann/
    netchange's own three-string shape) and _calc_prepare_ftc (f, F, a, b) —
    cauchymvt is the SECOND Calculus module whose input genuinely needs a
    second expression, but here BOTH f and g are on equal footing (unlike
    ftc's f/candidate-F asymmetry): the theorem's own conclusion is a ratio
    of the two functions' rates. Additive only, per CLAUDE.md's own
    governing line: no existing three-arg call site is touched. Returns
    (f, g, a, b, exact_inputs)."""
    f = calc_parse(f_str)
    g = calc_parse(g_str)
    a, ea = calc_parse_scalar(a_str)
    b, eb = calc_parse_scalar(b_str)
    if not (a.is_real and b.is_real) or not (a < b):
        raise ValueError('need real a < b')
    return f, g, a, b, (ea and eb)


def _calc_check_gprime_nonzero(g, a, b, exact):
    """Hypothesis 5: g'(x) != 0 for EVERY x in (a,b). Reuses
    calc_stationary_points(g, a, b) verbatim — exactly the same zero-of-the-
    derivative search analyse_rolle/analyse_mvt already use to hunt f' = 0,
    now applied to g instead: a found zero IS a counterexample to hypothesis
    5 (certified 'refuted' when exact, since an actual zero was exhibited),
    the SAME 'exact zero found -> refuted' shape calc_continuous_on/
    calc_differentiable_on already use elsewhere in this file.

    Absence is handled asymmetrically, mirroring _calc_witness_block's own
    'no witness found -> unknown, never refuted' rule (existence is what a
    theorem's CONCLUSION promises, so absence-by-sampling is never a proof of
    absence there) — but hypothesis 5 is the opposite direction: it CLAIMS an
    absence (no zero anywhere). solveset succeeding with an EMPTY finite set
    ('exact' certificate) genuinely PROVES there is no zero — the complete
    solution set was computed and is empty, not merely unsearched. The
    'enclosure' fallback (a certified SIGN-CHANGE search) can only ever
    CERTIFY a zero it actually finds; finding none does not certify their
    absence (a double root that never changes sign, or one sitting between
    two sampled grid points, could be missed) — that case abstains rather
    than promoting silence to a proof, honestly weaker than the 'exact'
    branch above it."""
    try:
        pts, cert = calc_stationary_points(g, a, b)
    except Abstain as e:
        return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    if pts:
        p0 = pts[0]
        issue = {'kind': 'stationary', 'x': p0.get('c'), 'xTex': p0.get('cTex'),
                  'detail': "g'(x) = 0 here — hypothesis 5 fails"}
        return {'pass': False, 'issues': [issue], 'provenance': calc_cap('refuted', exact), 'witness': p0}
    if cert == 'exact':
        return {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact)}
    return {'pass': None, 'issues': [], 'provenance': 'unknown',
            'reason': 'no sign change of g\\' found by certified sampling; absence of a zero is not thereby certified'}


def _calc_overall_cauchy(hyps, gprime, wb, any_fail):
    """Module-level provenance: the weakest link among the four ordinary
    hypotheses, hypothesis 5, and the witness block — same shape as
    _calc_overall (rolle/mvt) and _calc_overall_ftc, extended to a fifth
    clause."""
    provs = [h['provenance'] for h in hyps.values()] + [gprime['provenance'], wb['provenance']]
    if any_fail:
        fail_provs = [h['provenance'] for h in hyps.values() if h['pass'] is False]
        if gprime['pass'] is False:
            fail_provs.append(gprime['provenance'])
        return 'numeric' if 'numeric' in fail_provs else 'refuted'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    return 'proved'


def analyse_cauchymvt(f_str, g_str, a_str, b_str):
    """f_str/g_str/a_str/b_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy) -- FOUR strings, the same shape
    analyse_ftc introduced (see _calc_prepare_cauchy's own docstring).

    Topic 2.13: Cauchy's (Generalized) Mean Value Theorem. f, g continuous on
    [a,b], differentiable on (a,b), and g'(x) != 0 for every x in (a,b) =>
    exists c in (a,b) with (f(b)-f(a))/(g(b)-g(a)) = f'(c)/g'(c).

    FIVE hypotheses -- more than any other single-variable Calculus module
    (netchange has three) -- but see cauchymvtEngine.js's own header for why
    they fold into only THREE proof-halt states: the classical proof runs
    Rolle's theorem on h(x) = (f(b)-f(a))*g(x) - (g(b)-g(a))*f(x), and h(a) =
    h(b) is an ALGEBRAIC IDENTITY needing no hypothesis at all (mirrored
    here exactly: both endpoint values are simplify()'d to confirm equality
    stays exact, never merely assumed). f/g continuity failing kills EVT on
    h; f/g differentiability failing kills Fermat's step on h; hypothesis 5
    failing kills only the FINAL division by g'(c) -- Rolle can still find a
    witness for h'(c) = 0 (see _calc_check_gprime_nonzero's own docstring
    for the exact/enclosure asymmetry in how that hypothesis's ABSENCE claim
    is certified, the opposite shape from every other hypothesis in this
    file). Reuses _calc_hypotheses_block verbatim on f AND on g independently
    (the SAME continuity/differentiability primitives analyse_rolle/
    analyse_mvt already use), and _calc_witness_block verbatim on the
    auxiliary h (the SAME zero-of-the-derivative search rolle/mvt already
    run, just on a different auxiliary expression) -- no new search
    machinery, only new bookkeeping for the second function and the fifth
    hypothesis.

    reducesTo mvt: g(x) = x, checked by the SAME sp.simplify(g - x) == 0
    exact-equality test analyse_mvt's own reducesTo-rolle detection uses
    (there: f(a) = f(b); an equation between two closed-form quantities, not
    a search). degenerate affine: f = k*g + m for constants k, m, detected as
    h being IDENTICALLY CONSTANT -- sp.diff(h, x) == 0 exactly, mirroring
    analyse_mvt's own g_is_zero check (there: g itself is zero; here: h's
    DERIVATIVE is zero, since h itself is a nonzero constant -Δg*m in
    general, not zero -- see cauchymvtEngine.js's header for the algebra).

    Returns VisualMath's flat {provenance, ...} shape (never c-wrapped)."""
    try:
        f, g, a, b, exact = _calc_prepare_cauchy(f_str, g_str, a_str, b_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')

    fa, fb = f.subs(x, a), f.subs(x, b)
    ga, gb = g.subs(x, a), g.subs(x, b)
    df = sp.simplify(fb - fa)
    dg = sp.simplify(gb - ga)
    dg_zero = bool(dg == 0)

    hyps_f = _calc_hypotheses_block(f, a, b, exact)
    hyps_g = _calc_hypotheses_block(g, a, b, exact)
    hyps = {
        'fContinuous': hyps_f['continuous'],
        'gContinuous': hyps_g['continuous'],
        'fDifferentiable': hyps_f['differentiable'],
        'gDifferentiable': hyps_g['differentiable'],
    }

    if hyps['gContinuous']['pass'] is True and hyps['gDifferentiable']['pass'] is True:
        gprime = _calc_check_gprime_nonzero(g, a, b, exact)
    else:
        gprime = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'g not established continuous and differentiable'}],
                  'provenance': hyps['gDifferentiable']['provenance']}
    hyps['gPrimeNonzero'] = gprime

    all_pass = all(h['pass'] is True for h in hyps.values())
    any_fail = any(h['pass'] is False for h in hyps.values())

    h_expr = df * g - dg * f
    ratio = None if dg_zero else sp.simplify(df / dg)
    wb = _calc_witness_block(h_expr, a, b, exact, ratio if ratio is not None else sp.nan)

    blocked = None
    if hyps['fContinuous']['pass'] is False or hyps['gContinuous']['pass'] is False:
        blocked = 'evt'
    elif hyps['fDifferentiable']['pass'] is False or hyps['gDifferentiable']['pass'] is False:
        blocked = 'fermat'
    elif gprime['pass'] is False:
        blocked = 'divide'

    reduces_to = {'id': 'mvt', 'active': bool(sp.simplify(g - x) == 0),
                  'detail': 'g(x) = x: this is exactly Lagrange’s Mean Value Theorem (2.6)'}
    h_prime = sp.diff(h_expr, x)
    affine = bool(sp.simplify(h_prime) == 0)

    return {
        'id': 'cauchymvt',
        'provenance': _calc_overall_cauchy(hyps, gprime, wb, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)},
        'g': {'tex': sp.latex(g)},
        'h': {'tex': sp.latex(sp.expand(h_expr))},
        'interval': {'a': _calc_num(a), 'b': _calc_num(b), 'aTex': sp.latex(a), 'bTex': sp.latex(b)},
        'values': {'fa': _calc_num(fa), 'fb': _calc_num(fb), 'ga': _calc_num(ga), 'gb': _calc_num(gb)},
        'delta': {'df': _calc_num(df), 'dg': _calc_num(dg), 'dfTex': sp.latex(df), 'dgTex': sp.latex(dg), 'dgZero': dg_zero},
        'ratio': None if ratio is None else {'value': _calc_num(ratio), 'tex': sp.latex(ratio)},
        'hypotheses': hyps,
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'conclusion': wb,
        'reducesTo': reduces_to,
        'degenerate': {'affine': affine, 'detail': 'f = k·g + m: f′(c)/g′(c) = k at every c'},
    }


# ── Calculus domain (DOMAINS[1]) — Taylor's Theorem with Lagrange Remainder
# (topic 2.14, the final step of the rolle -> mvt -> cauchymvt -> taylor
# chain) ──
def _taylor_check_nth_continuous(f, lo, hi, n, exact):
    """H1: f^(n) continuous on the CLOSED interval [lo, hi]. n=0 checks f
    itself; n>=1 differentiates first. Reuses calc_continuous_on verbatim —
    the SAME primitive rolle/mvt/cauchymvt already use — on the DERIVED
    expression f^(n), exactly the "new auxiliary, same shared checker"
    pattern taylorEngine.js's own numeric tier already establishes."""
    try:
        fn_expr = sp.diff(f, x, n) if n > 0 else f
        c = calc_continuous_on(fn_expr, lo, hi)
        c['provenance'] = calc_cap(c['provenance'], exact)
        return c
    except Abstain as e:
        return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}


def _taylor_check_nplus1_exists(f, lo, hi, n, exact):
    """H2: f^(n+1) exists on the OPEN interval (lo, hi).

    n=0 is special-cased to calc_differentiable_on(f, lo, hi) DIRECTLY — the
    exact same well-tested primitive rolle/mvt/cauchymvt already use for
    "does f' exist here", via calc_suspect_points + calc_safe_diff_at's
    one-sided limits. This matters for a reason confirmed by hand while
    building this module: a plain JUMP discontinuity in a derivative (e.g.
    |x|'s corner, where f'=sign(x) jumps but never blows up) has no
    "singularity" in SymPy's singularities() sense — singularities() hunts
    poles/branch points, not generic corners — so the n>=1 fallback below
    would silently read a corner as "no singularity found -> holds", a false
    positive. calc_differentiable_on's one-sided-limit machinery has no such
    blind spot (this is exactly why it exists as a separate primitive from
    continuity/singularity checking in the first place).

    n>=1 has no general exact primitive available for "does an order-(n+1)
    derivative exist pointwise", so it falls back to asking whether f^(n+1)
    has any singularity (pole/branch point) in the open interval via
    singularities() — this DOES catch a genuine blow-up exactly (e.g. this
    module's own fails-yet-holds preset's x^(-2/3) blow-up, when the input
    is clean enough to differentiate at all) but, like calc_suspect_points
    elsewhere in this file, cannot certify a jump/oscillation with no pole.
    Every curated 'break H2' preset uses n=0 (the case WITH an exact
    primitive), matching taylorEngine.js's own numeric-tier scope decision
    for the identical reason."""
    if n == 0:
        try:
            d = calc_differentiable_on(f, lo, hi)
            d['provenance'] = calc_cap(d['provenance'], exact)
            return d
        except Abstain as e:
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    try:
        fnp1 = sp.diff(f, x, n + 1)
        if calc_has_abstain_funcs(fnp1):
            raise Abstain('f^(n+1) contains floor/ceiling/sign/Piecewise')
        sing = singularities(fnp1, x, Interval.open(lo, hi))
        if not sing.is_FiniteSet:
            raise Abstain('singularity set of f^(n+1) is not finite')
        pts = sorted(sing, key=lambda p: float(p))
    except Abstain as e:
        return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
    except Exception as e:  # noqa: BLE001
        return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': f'singularities failed: {e}'}
    if pts:
        p0 = pts[0]
        issue = {'kind': 'blowup', 'x': _calc_num(p0), 'xTex': sp.latex(p0), 'detail': 'f^(n+1) is not defined here'}
        return {'pass': False, 'issues': [issue], 'provenance': calc_cap('refuted', exact)}
    return {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact)}


def _calc_overall_taylor(h1, h2, wb, any_fail):
    """Module-level provenance: the weakest link among H1, H2, and the
    witness block — same shape as _calc_overall (rolle/mvt)."""
    provs = [h1['provenance'], h2['provenance'], wb['provenance']]
    if any_fail:
        fail_provs = [h['provenance'] for h in (h1, h2) if h['pass'] is False]
        return 'numeric' if 'numeric' in fail_provs else 'refuted'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    return 'proved'


def analyse_taylor(f_str, a_str, x_str, n):
    """f_str/a_str/x_str: SymPy-syntax strings from expr.mjs
    (toSympy()/parseNumber().sympy). n: a plain nonnegative Python int (the
    UI caps it at taylorEngine.js's own MAX_ORDER) — auto-converted by
    pyodide on a bare call like every other scalar DISPATCH argument, so no
    pyodide.toPy() wrapping is needed (the same reasoning as every other
    Calculus DISPATCH entry).

    Topic 2.14: Taylor's Theorem with Lagrange Remainder. Two hypotheses,
    stated at order n/n+1 rather than a fixed order like every prior
    MVT-family module: H1 = f^(n) continuous on the closed interval between
    a and x; H2 = f^(n+1) exists on the open interval between a and x. The
    conclusion is f(x) = P_n(x) + R_n(x), R_n(x) = f^(n+1)(c)/(n+1)!*(x-a)^(n+1)
    for some c strictly between a and x.

    Mirrors taylorEngine.js's own numeric-tier design throughout: P_n(x) and
    R_n(x) = f(x) - P_n(x) are computed directly (R_n is KNOWN before any
    search), target = (n+1)!*R_n(x)/(x-a)^(n+1), and the witness search
    reuses _calc_witness_block VERBATIM (the SAME zero-of-the-derivative
    search rolle/mvt/cauchymvt already run) on the auxiliary
    G(t) = f^(n)(t) - target*t: a zero of G' IS exactly a c with
    f^(n+1)(c) = target — see taylorEngine.js's own header for why this
    single auxiliary needs no second function or Cauchy-style ratio.

    reducesTo mvt is UNCONDITIONAL on n=0 alone (not an input condition to
    check, unlike every other reducesTo detection in this file) — at n=0,
    P_0(x) = f(a) and the conclusion is exactly mvt's own f(x) = f(a) +
    f'(c)(x-a).

    KNOWN LIMITATION, confirmed by hand and shared with rolle/mvt: an odd
    fractional power like x^(7/3) reaches this function as
    sign(x)*Abs(x)**Rational(7,3) (expr.mjs's own real-branch rewrite for
    negative x), so calc_has_abstain_funcs(f) is True and every hypothesis
    below abstains ('unknown') even on an interval like [1,3] that never
    touches x=0 where the sign() ambiguity actually lives — the SAME
    documented rolle/mvt limitation ("the calculus tier abstains on sign"),
    not new to this module. The numeric tier is unaffected."""
    try:
        f = calc_parse(f_str)
        a, ea = calc_parse_scalar(a_str)
        xe, ex = calc_parse_scalar(x_str)
        if not (a.is_real and xe.is_real):
            raise ValueError('need real a, x')
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ValueError('need integer n >= 0')
        if calc_has_abstain_funcs(f):
            raise Abstain('taylor hypotheses are unreliable with floor/ceiling/sign/Piecewise in f')
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    except Abstain as e:
        return calc_unknown(str(e))

    exact = ea and ex
    reduces_to = {'id': 'mvt', 'active': n == 0,
                  'detail': 'n = 0: P_0(x) = f(a), so the conclusion is exactly mvt’s own f(x) = f(a) + f′(c)(x−a)'}

    degenerate = bool(a == xe)
    if degenerate:
        fa = f.subs(x, a)
        return {
            'id': 'taylor', 'provenance': calc_cap('proved', exact), 'exactInputs': exact,
            'f': {'tex': sp.latex(f)}, 'order': n,
            'center': {'value': _calc_num(a), 'tex': sp.latex(a)},
            'evalPoint': {'value': _calc_num(xe), 'tex': sp.latex(xe)},
            'degenerate': True,
            'Pn': {'value': _calc_num(fa), 'tex': sp.latex(sp.simplify(fa))},
            'Rn': {'value': 0.0, 'tex': '0'},
            'target': None,
            'hypotheses': {
                'nthContinuous': {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact), 'skipped': True},
                'nPlus1Exists': {'pass': True, 'issues': [], 'provenance': calc_cap('proved', exact), 'skipped': True},
            },
            'allHypothesesPass': True, 'blockedAct': None,
            'conclusion': {'witnesses': [], 'provenance': calc_cap('proved', exact), 'certificate': None},
            'reducesTo': reduces_to,
        }

    lo, hi = (a, xe) if bool(a < xe) else (xe, a)

    h1 = _taylor_check_nth_continuous(f, lo, hi, n, exact)
    if h1['pass'] is True:
        h2 = _taylor_check_nplus1_exists(f, lo, hi, n, exact)
    else:
        h2 = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f⁽ⁿ⁾ is not continuous'}], 'provenance': h1['provenance']}
    all_pass = h1['pass'] is True and h2['pass'] is True
    any_fail = h1['pass'] is False or h2['pass'] is False
    blocked = 'evt' if h1['pass'] is False else ('fermat' if h2['pass'] is False else None)

    Pn_at_xe = sum((sp.diff(f, x, k).subs(x, a) / sp.factorial(k)) * (xe - a) ** k for k in range(n + 1))
    Rn = sp.simplify(f.subs(x, xe) - Pn_at_xe)
    fact_n1 = sp.factorial(n + 1)
    target = sp.simplify(fact_n1 * Rn / (xe - a) ** (n + 1))

    fn_expr = sp.diff(f, x, n) if n > 0 else f
    G = fn_expr - target * x
    wb = _calc_witness_block(G, lo, hi, exact, target)

    return {
        'id': 'taylor',
        'provenance': _calc_overall_taylor(h1, h2, wb, any_fail),
        'exactInputs': exact,
        'f': {'tex': sp.latex(f)}, 'order': n,
        'center': {'value': _calc_num(a), 'tex': sp.latex(a)},
        'evalPoint': {'value': _calc_num(xe), 'tex': sp.latex(xe)},
        'degenerate': False,
        'Pn': {'value': _calc_num(Pn_at_xe), 'tex': sp.latex(sp.nsimplify(Pn_at_xe) if exact else Pn_at_xe)},
        'Rn': {'value': _calc_num(Rn), 'tex': sp.latex(sp.nsimplify(Rn) if exact else Rn)},
        'target': {'value': _calc_num(target), 'tex': sp.latex(sp.nsimplify(target) if exact else target)},
        'hypotheses': {'nthContinuous': h1, 'nPlus1Exists': h2},
        'allHypothesesPass': all_pass,
        'blockedAct': blocked,
        'conclusion': wb,
        'reducesTo': reduces_to,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — partials: functions of two variables and
#    partial differentiation (topic 2.15) ──
# =============================================================================
# CONCEPT module, the FIRST Calculus module to leave single-variable
# territory — see src/modules/partialsEngine.js's own header for the full
# design rationale (four clauses at a point (a,b): defined, d f/dx exists,
# d f/dy exists, continuous — clauses 2/3/4 are LOGICALLY INDEPENDENT of one
# another, only \`defined\` gates them; this is a deliberate departure from
# every prior chained-clause concept module, documented at length there).
#
# \`y\` is a SECOND free symbol alongside the module-global \`x\` used
# everywhere else in this file; \`_PART_NAMESPACE\` extends \`CALC_NAMESPACE\`
# additively (never widening CALC_NAMESPACE's own single-variable contract,
# since rolle/mvt/limits/ivt/riemann/netchange/ftc/improper/gammabeta all
# sympify through it assuming only \`x\` is free).
y = Symbol('y', real=True)
_PART_NAMESPACE = dict(CALC_NAMESPACE)
_PART_NAMESPACE['y'] = y


def _part_parse(expr_str):
    """Parse f(x,y) from expr.mjs's parseExpr(text, ['x','y']).toSympy()."""
    try:
        f = sp.sympify(expr_str, locals=_PART_NAMESPACE)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'cannot parse: {e}') from e
    bad = f.free_symbols - {x, y}
    if bad:
        raise ValueError(f'unexpected symbols {sorted(map(str, bad))}')
    return f


def _part_eval(expr, a_val, b_val):
    """Evaluate expr at (a_val, b_val); returns (value, ok). ok is False on a
    division-by-zero/complex-infinity/non-real result or a residual free
    symbol (never treated as a finite real number).

    TRAP, confirmed by hand while building this module (a new one, joining
    the floor/sign/nsimplify family this file's own docstring already
    documents): plain \`expr.subs({x: a_val, y: b_val})\` can SILENTLY paper
    over a genuine 0/0 singularity. \`x*y/(x**2+y**2)\` at (0,0) is a textbook
    example — SymPy's substitution walks the expression tree and, the
    moment x is replaced by 0, \`Mul(0, y)\` auto-evaluates to the literal
    integer \`0\` (an eager arithmetic simplification, done BEFORE y is even
    substituted), which then divides cleanly to a clean \`0\` once y is also
    replaced — never producing \`zoo\`/\`nan\` at all, even though the point is
    genuinely undefined (numerator and denominator vanish TOGETHER). Fixed
    by checking the DENOMINATOR of the expression written as a single
    fraction SEPARATELY and FIRST: if it vanishes at (a_val, b_val), the
    point is undefined regardless of what naive whole-expression
    substitution produces."""
    try:
        _, den = sp.together(expr).as_numer_denom()
        den_at = sp.simplify(den.subs({x: a_val, y: b_val}))
        if den_at == 0:
            return None, False
    except Exception:  # noqa: BLE001
        pass  # no denominator to speak of (e.g. a plain polynomial) — fall through
    try:
        v = sp.simplify(expr.subs({x: a_val, y: b_val}))
    except Exception:  # noqa: BLE001
        return None, False
    if v.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or v.free_symbols:
        return None, False
    try:
        vn = sp.N(v, 30)
    except Exception:  # noqa: BLE001
        return None, False
    if not vn.is_real:
        return None, False
    return v, True


def _part_partial(f, var, a_val, b_val):
    """One partial derivative at (a_val, b_val), decided EXACTLY via a
    one-sided-limit check at the pivot — never a bare sp.diff().subs(), the
    same 'never trust diff() at a suspect point' discipline every other
    hypothesis check in this file uses (see calc_safe_diff_at's own
    docstring). Reuses calc_safe_diff_at VERBATIM by first reducing f to a
    SLICE expression purely in the shared symbol \`x\` (substituting the OTHER
    variable's value, and renaming y->x for the y-slice) — the sanctioned
    narrow-reuse shape this domain already established (series/sequences,
    powerseries/series): the x-slice g(t)=f(t,b) is already purely in x; the
    y-slice h(t)=f(a,t) is purely in y, renamed to x so calc_safe_diff_at's
    own hardcoded \`x\` substitutions apply unchanged."""
    if var == 'x':
        slice_expr = f.subs(y, b_val)
        x0 = a_val
    else:
        slice_expr = f.subs(x, a_val).subs(y, x)
        x0 = b_val
    if calc_has_abstain_funcs(slice_expr):
        raise Abstain(f'd f/d{var} via one-sided limits is unreliable with floor/ceiling/sign/Piecewise')
    l, r = calc_safe_diff_at(slice_expr, x0)
    finite = lambda v: v is not None and v.is_finite and v.is_real  # noqa: E731
    if finite(l) and finite(r) and sp.simplify(l - r) == 0:
        # \`exactValue\` (the raw SymPy value, alongside the existing
        # float-rounded \`value\`) is an ADDITIVE field, added for
        # analyse_totaldiff (topic 2.16), which needs the EXACT partial
        # derivative to substitute into its own error-ratio expression —
        # \`_calc_num(l)\` alone is a lossy float round-trip that would
        # silently downgrade an exact result. analyse_partials (topic
        # 2.15) never reads this key, so its own return shape is unchanged.
        return {'pass': True, 'value': _calc_num(l), 'exactValue': l, 'valueTex': sp.latex(sp.nsimplify(l)), 'issues': [], 'provenance': 'proved'}
    issue = {
        'kind': 'corner' if finite(l) and finite(r) else 'cusp',
        'left': _calc_num(l) if l is not None else None, 'right': _calc_num(r) if r is not None else None,
        'detail': 'one-sided slopes disagree' if finite(l) and finite(r) else 'one-sided slope is infinite',
    }
    return {'pass': False, 'value': None, 'issues': [issue], 'provenance': 'refuted'}


def _part_continuity(f, a_val, b_val, fab_val):
    """Continuity at (a_val, b_val) via polar substitution about the point,
    PLUS a mandatory parabolic cross-check — see partialsEngine.js's
    continuityProbe2D docstring for the numeric-tier analogue of this exact
    two-stage strategy, and the module's own completion report for how this
    was confirmed by hand.

    STRATEGY. Substitute x = a + r*cos(theta), y = b + r*sin(theta) (r>0,
    theta a free real parameter) and take lim r->0+. Three outcomes:
      1. The limit does not evaluate in closed form (or raises) -> abstain.
      2. The limit genuinely DEPENDS on theta (confirmed by evaluating at
         two different concrete angles, not merely by an unsimplified
         theta appearing syntactically) -> a CERTAIN discontinuity, the
         SAME way an AccumBounds result elsewhere in this file is treated
         as a definite finding, never an abstention: every fixed
         straight-line direction already disagrees with every other, so no
         two-sided limit can possibly exist. Refuted, decisively.
      3. The limit is independent of theta (a genuine constant): compare it
         against f(a,b). If it disagrees, refuted (radial approach alone
         already disproves continuity). If it AGREES, do NOT yet report
         'proved continuous' — polar substitution with theta held FIXED as
         r->0 is BLIND to a real, well-known class of counterexample: for
         f(x,y) = x^2*y/(x^4+y^2), f(0,0)=0, every FIXED-ANGLE ray through
         the origin gives limit 0 (confirmed by hand: substituting x=r*cos,
         y=r*sin gives r*cos^2(theta)*sin(theta)/(r^2*cos^4(theta) +
         sin^2(theta)) -> 0 for EVERY fixed theta, including theta=0),
         while the function is genuinely discontinuous along the parabola
         y=x^2 (value identically 1/2). A radial-only version of this
         function was written and confirmed to WRONGLY return 'continuous'
         on exactly this input before the parabola cross-check below was
         added — this is not hypothetical, it is the reason the
         cross-check exists at all. So: only after the polar limit agrees
         with f(a,b) do we ALSO evaluate the limit along y = b+m(x-a)^2 for
         several m and both signs of (x-a); any disagreement there refutes
         continuity (documented as 'parabola-disagreement', distinct from
         'depends-on-theta' so the two failure MODES stay distinguishable
         in the UI). Only surviving every check reports 'proved'.
    """
    if calc_has_abstain_funcs(f):
        raise Abstain('continuity via polar substitution is unreliable with floor/ceiling/sign/Piecewise')
    r = Symbol('r', positive=True)
    theta = Symbol('theta', real=True)
    expr_polar = f.subs({x: a_val + r * sp.cos(theta), y: b_val + r * sp.sin(theta)})
    try:
        lim_polar = sp.limit(expr_polar, r, 0, '+')
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'polar limit failed: {e}') from e
    lim_polar_s = sp.simplify(lim_polar)
    if lim_polar_s.has(sp.Limit):
        raise Abstain('polar limit did not evaluate in closed form')

    if theta in lim_polar_s.free_symbols:
        try:
            v1 = sp.N(lim_polar_s.subs(theta, sp.Rational(1, 7)), 30)
            v2 = sp.N(lim_polar_s.subs(theta, sp.Rational(2, 7)), 30)
        except Exception as e:  # noqa: BLE001
            raise Abstain(f'could not confirm theta-dependence: {e}') from e
        if not (v1.is_real and v2.is_real) or abs(complex(v1) - complex(v2)) > 1e-9:
            return {'pass': False, 'kind': 'depends-on-theta',
                    'detail': 'the polar limit genuinely depends on the approach angle theta — every straight-line direction disagrees with some other, a certain discontinuity',
                    'limit': None, 'limitTex': None, 'provenance': 'refuted'}
        # Syntactically theta-dependent but numerically constant at these
        # two sample angles — trust the number, not the unsimplified form.
        lim_polar_s = sp.nsimplify(v1)

    if lim_polar_s.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or not lim_polar_s.is_real:
        raise Abstain('polar limit is not a finite real constant')

    if sp.simplify(lim_polar_s - fab_val) != 0:
        return {'pass': False, 'kind': 'limit-value-mismatch',
                'detail': f'the polar (radial) limit is {sp.latex(lim_polar_s)}, not f(a,b)',
                'limit': _calc_num(lim_polar_s), 'limitTex': sp.latex(lim_polar_s), 'provenance': 'refuted'}

    # Radial approach agrees with f(a,b) — NOT sufficient on its own (see
    # docstring). Cross-check several parabolic paths before certifying.
    t = Symbol('t', positive=True)
    for m in (sp.Integer(1), sp.Integer(-1), sp.Integer(2), sp.Rational(1, 2)):
        for sgn in (1, -1):
            expr_par = f.subs({x: a_val + sgn * t, y: b_val + m * t ** 2})
            try:
                lim_par = sp.limit(expr_par, t, 0, '+')
            except Exception as e:  # noqa: BLE001
                raise Abstain(f'parabolic cross-check limit failed: {e}') from e
            lim_par_s = sp.simplify(lim_par)
            if lim_par_s.has(sp.Limit):
                raise Abstain('parabolic cross-check did not evaluate in closed form')
            if lim_par_s.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or not lim_par_s.is_real:
                raise Abstain('parabolic cross-check limit is not a finite real constant')
            if sp.simplify(lim_par_s - fab_val) != 0:
                return {'pass': False, 'kind': 'parabola-disagreement',
                        'detail': f'the radial limit agrees with f(a,b), but along y=b+({sp.latex(m)})(x-a)^2 the limit is {sp.latex(lim_par_s)} — straight-line sampling alone is not enough',
                        'limit': _calc_num(lim_par_s), 'limitTex': sp.latex(lim_par_s), 'provenance': 'refuted'}

    return {'pass': True, 'kind': None, 'detail': None,
            'limit': _calc_num(lim_polar_s), 'limitTex': sp.latex(lim_polar_s), 'provenance': 'proved'}


def analyse_partials(expr_str, a_str, b_str):
    """expr_str: SymPy-syntax string from expr.mjs's parseExpr(text,['x','y']).toSympy().
    a_str/b_str: SymPy-syntax scalar strings from expr.mjs's parseNumber().sympy.

    Topic 2.15: Functions of two variables & partial differentiation. Four
    clauses at the point (a,b) — see this file's own section header above
    and partialsEngine.js's header for the full design rationale. Unlike
    every prior chained-clause concept module, only \`defined\` gates the
    other three; \`partialX\`/\`partialY\`/\`continuous\` are each decided
    independently and none is skipped because another failed.

    Overall \`provenance\` is the WEAKEST tier among every clause actually
    attempted (unknown < numeric < {proved, refuted} — the same 'weakest
    link' reduction _calc_overall uses elsewhere in this file), since this
    module reports four independent facts rather than one theorem's single
    verdict: 'unknown' if ANY attempted clause abstained; else 'numeric' if
    any input was a float; else 'proved' if defined holds and every
    attempted clause (partialX/partialY/continuous) also holds; else
    'refuted' (defined holds but at least one clause is decisively false).

    Confirmed by hand which presets this decides exactly vs abstains on
    (see partialsPresets.mjs and the module's completion report): the
    paraboloid/sin*cos 'holds' presets, both |x|/|y|-corner 'breakPartial'
    presets (Abs is not in CALC_ABSTAIN_FUNCS, so calc_safe_diff_at's
    one-sided-limit machinery decides the corner exactly), the constant and
    symmetric-minimum degenerate presets, and the f(a,b)-undefined preset
    are ALL decided exactly. \`_part_continuity\`'s own theta-dependence and
    parabola-disagreement branches are ALSO each confirmed exact on the two
    flagship counterexamples (see this file's own selftest, which calls
    \`_part_continuity\` directly to pin this) — BUT the two flagship PRESETS
    THEMSELVES (xy/(x^2+y^2), x^2y/(x^4+y^2), both centered at their own
    origin) report \`defined: False\` when sent through this function AS
    WRITTEN, for a genuine and DOCUMENTED reason, not a defect: the raw
    string, with no piecewise support (expr.mjs's grammar cannot express
    "f(0,0):=0" as a separate clause), is a literal 0/0 at exactly (0,0) —
    see \`_part_eval\`'s own docstring for the \`.subs()\` trap this module
    found and fixed for THAT question. The numeric tier's own \`override\`
    input (partialsEngine.js's \`analyzeNumeric\`) supplies the DEFINED value
    directly and so can demonstrate the full counterexample; the symbolic
    tier honestly reports the raw formula's own undefined-ness instead of
    silently guessing what the override would have been. A student clicking
    "check with symbolic engine" on one of these two presets sees this
    exact, correct disagreement — the module's own UI states it explicitly
    (see PartialsModule.jsx) rather than presenting it as a contradiction.
    """
    try:
        f = _part_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        a_val, ea = calc_parse_scalar(a_str)
        b_val, eb = calc_parse_scalar(b_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    exact = ea and eb

    fab_val, defined_ok = _part_eval(f, a_val, b_val)
    defined = {
        'pass': defined_ok,
        'value': _calc_num(fab_val) if defined_ok else None,
        'valueTex': sp.latex(sp.nsimplify(fab_val)) if defined_ok else None,
        'issues': [] if defined_ok else [{'kind': 'undefined', 'detail': 'f(a,b) is not a finite real number'}],
        'provenance': calc_cap('proved' if defined_ok else 'refuted', exact),
    }

    if not defined_ok:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f(a,b) is undefined'}], 'provenance': defined['provenance']}
        return {
            'id': 'partials', 'provenance': defined['provenance'], 'exactInputs': exact,
            'defined': defined, 'partialX': skipped, 'partialY': skipped, 'continuous': skipped,
            'partialsExist': False, 'counterexample': False, 'blockedAct': 'defined',
        }

    def _safe(fn):
        try:
            return fn()
        except Abstain as e:
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
        except Exception as e:  # noqa: BLE001
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': f'unexpected error: {e}'}

    partialX = _safe(lambda: _part_partial(f, 'x', a_val, b_val))
    partialY = _safe(lambda: _part_partial(f, 'y', a_val, b_val))
    continuous = _safe(lambda: _part_continuity(f, a_val, b_val, fab_val))
    for c in (partialX, partialY, continuous):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)

    partials_exist = partialX.get('pass') is True and partialY.get('pass') is True
    counterexample = bool(partials_exist and continuous.get('pass') is False)

    provs = [defined['provenance'], partialX['provenance'], partialY['provenance'], continuous['provenance']]
    if 'unknown' in provs:
        overall = 'unknown'
    elif 'numeric' in provs:
        overall = 'numeric'
    elif partialX.get('pass') is True and partialY.get('pass') is True and continuous.get('pass') is True:
        overall = calc_cap('proved', exact)
    else:
        overall = calc_cap('refuted', exact)

    return {
        'id': 'partials', 'provenance': overall, 'exactInputs': exact,
        'defined': defined, 'partialX': partialX, 'partialY': partialY, 'continuous': continuous,
        'partialsExist': partials_exist, 'counterexample': counterexample, 'blockedAct': None,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — totaldiff: total differential & tangent
#    plane (topic 2.16), the direct sequel to partials (topic 2.15) ──
# =============================================================================
# CONCEPT module. Two clauses at a point (a,b), genuinely CHAINED (unlike
# partials' own three independent facts — see totaldiffEngine.js's own
# header for the full design rationale):
#   (1) partialsExist — f_x(a,b) AND f_y(a,b) both exist. Reuses
#       \`_part_partial\` VERBATIM (the same one-sided-limit machinery
#       already built for partials' own clauses 2/3) — this is EXACTLY the
#       same question, asked again because total differentiability's own
#       definition needs both numbers to even state the candidate plane.
#   (2) totallyDifferentiable — does the error ratio
#         [f(a+h,b+k) - f(a,b) - f_x(a,b)h - f_y(a,b)k] / sqrt(h^2+k^2)
#       vanish as (h,k) -> (0,0) along EVERY path? Gated by (1): without
#       both constants the ratio cannot even be formed.
#
# \`_td_differentiable_at\` mirrors \`_part_continuity\`'s own two-stage
# strategy almost exactly (polar substitution with theta held free, then a
# parabolic cross-check) — the SAME "a curved path can hide a failure
# invisible to every straight line" risk applies here, for the identical
# reason. The one structural difference: \`_part_continuity\` compares the
# polar limit against f(a,b) (a nonzero target in general); here the
# target is always 0 (the ratio itself, not f), which makes the
# theta-dependence branch slightly different — a THETA-DEPENDENT ratio
# limit does not need a SECOND confirming sample angle to certify failure
# the way continuity's did (there, two disagreeing angles certify that no
# single limit can exist at all); here, since the target is fixed at 0,
# a single sample angle returning a nonzero value already REFUTES total
# differentiability outright (one direction failing to vanish is already
# enough), so only one confirming sample is needed, not two.
def _td_differentiable_at(f, fx_val, fy_val, a_val, b_val, fab_val):
    """f: the ORIGINAL (unshifted) SymPy expression in x, y (from
    \`_part_parse\`). fx_val/fy_val: the ALREADY-DECIDED partial derivatives
    at (a_val, b_val) (SymPy exact values, from \`_part_partial\`).

    Confirmed by hand (see totaldiffEngine.js's own header and the
    module's completion report): the flagship counterexample
    xy/sqrt(x^2+y^2) is caught here via THETA-DEPENDENCE (the ratio's
    polar limit, sqrt-free after cancellation, is a genuine nonconstant
    function of theta whose value at a sample angle is nonzero); the
    second flagship sqrt(|xy|) is caught the SAME way (its own ratio limit
    sqrt(|cos(theta)sin(theta)|) is also theta-dependent and nonzero at a
    sample angle) — NEITHER shipped preset in this module needs the
    parabola cross-check to be decided correctly, exactly mirroring the
    numeric tier's own documented finding. The parabola check is kept as a
    genuine extra safety net (same defensive shape as \`_part_continuity\`),
    not because any shipped preset requires it.
    """
    if calc_has_abstain_funcs(f):
        raise Abstain('total differentiability via polar substitution is unreliable with floor/ceiling/sign/Piecewise')
    r = Symbol('r', positive=True)
    theta = Symbol('theta', real=True)
    h_polar = r * sp.cos(theta)
    k_polar = r * sp.sin(theta)
    g = f.subs({x: a_val + h_polar, y: b_val + k_polar}) - fab_val - fx_val * h_polar - fy_val * k_polar
    ratio_expr = g / r
    try:
        lim_polar = sp.limit(ratio_expr, r, 0, '+')
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'polar ratio limit failed: {e}') from e
    lim_polar_s = sp.simplify(lim_polar)
    if lim_polar_s.has(sp.Limit):
        raise Abstain('polar ratio limit did not evaluate in closed form')

    if theta in lim_polar_s.free_symbols:
        try:
            v1 = sp.N(lim_polar_s.subs(theta, sp.Rational(1, 7)), 30)
        except Exception as e:  # noqa: BLE001
            raise Abstain(f'could not evaluate theta-dependent ratio limit: {e}') from e
        if not v1.is_real:
            raise Abstain('theta-dependent ratio limit is not real at a sample angle')
        if abs(complex(v1)) > 1e-9:
            return {'pass': False, 'kind': 'depends-on-theta',
                    'detail': 'the polar error-ratio limit genuinely depends on the approach angle theta and is nonzero at a sample angle — at least one straight-line direction has a non-vanishing error ratio',
                    'provenance': 'refuted'}
        # Syntactically theta-dependent but numerically ~0 at this sample
        # angle — trust the number (mirrors _part_continuity's own
        # "syntactically theta-dependent but numerically constant" rule),
        # and fall through to the parabola cross-check below to be sure
        # this genuinely holds along every angle, not merely this one.
        lim_polar_s = sp.Integer(0)

    if lim_polar_s.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or not lim_polar_s.is_real:
        raise Abstain('polar ratio limit is not a finite real constant')

    if sp.simplify(lim_polar_s) != 0:
        return {'pass': False, 'kind': 'nonzero-ratio',
                'detail': f'the radial (straight-line) error-ratio limit is {sp.latex(lim_polar_s)}, not 0',
                'provenance': 'refuted'}

    # Radial approach vanishes — NOT sufficient on its own (see docstring).
    # Cross-check several parabolic paths before certifying.
    t = Symbol('t', positive=True)
    for m in (sp.Integer(1), sp.Integer(-1), sp.Integer(2), sp.Rational(1, 2)):
        for sgn in (1, -1):
            h_par = sgn * t
            k_par = m * t ** 2
            dist_par = sp.sqrt(h_par ** 2 + k_par ** 2)
            g_par = f.subs({x: a_val + h_par, y: b_val + k_par}) - fab_val - fx_val * h_par - fy_val * k_par
            try:
                lim_par = sp.limit(g_par / dist_par, t, 0, '+')
            except Exception as e:  # noqa: BLE001
                raise Abstain(f'parabolic cross-check limit failed: {e}') from e
            lim_par_s = sp.simplify(lim_par)
            if lim_par_s.has(sp.Limit):
                raise Abstain('parabolic cross-check did not evaluate in closed form')
            if lim_par_s.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or not lim_par_s.is_real:
                raise Abstain('parabolic cross-check limit is not a finite real constant')
            if sp.simplify(lim_par_s) != 0:
                return {'pass': False, 'kind': 'parabola-nonzero',
                        'detail': f'the radial error-ratio limit vanishes, but along y=b+({sp.latex(m)})(x-a)^2 it is {sp.latex(lim_par_s)} — straight-line sampling alone is not enough',
                        'provenance': 'refuted'}

    return {'pass': True, 'kind': None, 'detail': None, 'provenance': 'proved'}


def analyse_totaldiff(expr_str, a_str, b_str):
    """expr_str: SymPy-syntax string from expr.mjs's parseExpr(text,['x','y']).toSympy().
    a_str/b_str: SymPy-syntax scalar strings from expr.mjs's parseNumber().sympy.

    Topic 2.16: Total differential & tangent plane. Two GENUINELY CHAINED
    clauses at the point (a,b) — see this file's own section header above
    and totaldiffEngine.js's header for the full design rationale.

    Reuses \`_part_parse\`/\`_part_eval\`/\`_part_partial\` VERBATIM from
    analyse_partials — \`defined\` and \`partialX\`/\`partialY\` are EXACTLY the
    same questions partials already answers; the only genuinely new piece
    is \`_td_differentiable_at\` above.

    Overall \`provenance\`: 'unknown' if any ATTEMPTED clause abstained; else
    'numeric' if any input was a float; else 'proved' if defined holds,
    both partials exist, and totallyDifferentiable also holds; else
    'refuted'.

    Confirmed by hand (see totaldiffEngine.js's own header and the module's
    completion report) which presets this decides exactly vs abstains on:
    the paraboloid/sincos 'holds' presets, both corner 'breakPartials'
    presets (Abs is not in CALC_ABSTAIN_FUNCS), the constant and plane
    'degenerate' presets, and the f(a,b)-undefined preset are ALL decided
    exactly. BOTH flagship presets (xy/sqrt(x^2+y^2), sqrt(|xy|)) are
    decided EXACTLY too via \`_td_differentiable_at\`'s theta-dependence
    branch — UNLIKE analyse_partials' own two flagships, which reported
    \`defined: False\` on the raw string exactly at the origin (a genuine
    0/0 in \`x*y/(x^2+y^2)\`), the FIRST totaldiff flagship,
    \`x*y/sqrt(x^2+y^2)\`, has the identical raw-string-undefined-at-the-
    origin issue (confirmed by hand: \`_part_eval\`'s own denominator check
    finds sqrt(x^2+y^2)=0 at the origin) — the SAME documented,
    intentional asymmetry with the numeric tier's own \`override\` input
    (expr.mjs has no piecewise syntax to write f(0,0):=0). The SECOND
    flagship, \`sqrt(|xy|)\`, is genuinely DIFFERENT here: sqrt(Abs(x*y))
    has no denominator at all, so it evaluates to a perfectly well-defined
    0 at the origin with NO override needed — decided exactly as
    \`refuted\` (not totally differentiable) directly on the raw string,
    with no asymmetry to document for this one.
    """
    try:
        f = _part_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        a_val, ea = calc_parse_scalar(a_str)
        b_val, eb = calc_parse_scalar(b_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    exact = ea and eb

    fab_val, defined_ok = _part_eval(f, a_val, b_val)
    defined = {
        'pass': defined_ok,
        'value': _calc_num(fab_val) if defined_ok else None,
        'valueTex': sp.latex(sp.nsimplify(fab_val)) if defined_ok else None,
        'issues': [] if defined_ok else [{'kind': 'undefined', 'detail': 'f(a,b) is not a finite real number'}],
        'provenance': calc_cap('proved' if defined_ok else 'refuted', exact),
    }

    if not defined_ok:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f(a,b) is undefined'}], 'provenance': defined['provenance']}
        return {
            'id': 'totaldiff', 'provenance': defined['provenance'], 'exactInputs': exact,
            'defined': defined, 'partialX': skipped, 'partialY': skipped, 'totallyDifferentiable': skipped,
            'partialsExist': False, 'counterexample': False, 'blockedAct': 'defined',
        }

    def _safe(fn):
        try:
            return fn()
        except Abstain as e:
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
        except Exception as e:  # noqa: BLE001
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': f'unexpected error: {e}'}

    partialX = _safe(lambda: _part_partial(f, 'x', a_val, b_val))
    partialY = _safe(lambda: _part_partial(f, 'y', a_val, b_val))
    for c in (partialX, partialY):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)

    partials_exist = partialX.get('pass') is True and partialY.get('pass') is True

    if not partials_exist:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'a partial derivative does not exist here'}], 'provenance': 'numeric'}
        provs = [defined['provenance'], partialX['provenance'], partialY['provenance']]
        overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('refuted', exact))
        return {
            'id': 'totaldiff', 'provenance': overall, 'exactInputs': exact,
            'defined': defined, 'partialX': partialX, 'partialY': partialY, 'totallyDifferentiable': skipped,
            'partialsExist': False, 'counterexample': False, 'blockedAct': 'partialsExist',
        }

    # \`exactValue\` (see _part_partial's own docstring) is the EXACT SymPy
    # partial derivative — using the float-rounded \`value\` here would
    # reintroduce exactly the tolerance-blind trap this file's own module
    # docstring warns against, on a quantity that IS available exactly.
    fx_val = partialX['exactValue']
    fy_val = partialY['exactValue']
    totallyDifferentiable = _safe(lambda: _td_differentiable_at(f, fx_val, fy_val, a_val, b_val, fab_val))
    if totallyDifferentiable.get('pass') is not None:
        totallyDifferentiable['provenance'] = calc_cap(totallyDifferentiable['provenance'], exact)

    counterexample = bool(partials_exist and totallyDifferentiable.get('pass') is False)

    provs = [defined['provenance'], partialX['provenance'], partialY['provenance'], totallyDifferentiable['provenance']]
    if 'unknown' in provs:
        overall = 'unknown'
    elif 'numeric' in provs:
        overall = 'numeric'
    elif totallyDifferentiable.get('pass') is True:
        overall = calc_cap('proved', exact)
    else:
        overall = calc_cap('refuted', exact)

    return {
        'id': 'totaldiff', 'provenance': overall, 'exactInputs': exact,
        'defined': defined, 'partialX': partialX, 'partialY': partialY, 'totallyDifferentiable': totallyDifferentiable,
        'partialsExist': partials_exist, 'counterexample': counterexample, 'blockedAct': None,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — chainrule: chain rule for composites
#    (topic 2.17), the direct sequel to totaldiff (2.16) ──
# =============================================================================
# THEOREM module. Given z=f(x,y), x=x(t), y=y(t), TWO INDEPENDENT hypotheses
# (unlike totaldiff's own single-object chained pair — see
# chainruleEngine.js's own header for the full design rationale: these are
# facts about DIFFERENT objects, the same "independent facts" shape
# analyse_partials already uses for its own three clauses):
#   (1) pathDifferentiable — x'(t0) AND y'(t0) both exist. \`_cr_path_derivative\`
#       reuses calc_safe_diff_at via the SAME symbol-renaming trick
#       \`_part_partial\` already established for its own y-slice (t -> x,
#       since calc_safe_diff_at hardcodes the module-global \`x\` symbol).
#   (2) f totally differentiable at (a,b) = (x(t0), y(t0)) — reuses
#       \`_part_eval\`/\`_part_partial\`/\`_td_differentiable_at\` VERBATIM from
#       analyse_totaldiff, called directly on the ALREADY-EVALUATED sympy
#       point (a_val, b_val) rather than re-parsing from strings (those
#       three helpers already accept plain sympy values, not raw strings —
#       analyse_totaldiff itself calls them the identical way).
#
# The conclusion is cross-checked two independent ways, mirroring
# chainruleEngine.js's own numeric cross-check: the FORMULA
# f_x(a,b)x'(t0)+f_y(a,b)y'(t0), and DIRECT differentiation of the composite
# g(t)=f(x(t),y(t)) at t0 (built by substituting x(t), y(t) into f, then
# applying the SAME t->x renaming trick before calc_safe_diff_at). When both
# hypotheses hold, these must agree EXACTLY (a symbolic sp.simplify check,
# not a numeric tolerance). When hypothesis (2) fails, they may genuinely
# disagree, or \`direct\` may fail to exist at all (a corner in the
# composite) even though \`formula\` still computes a definite number — the
# SAME asymmetry the numeric tier's own flagship preset demonstrates.
#
# The numeric tier's own \`override\` input is NOT sent here — expr.mjs has
# no piecewise syntax, the SAME documented asymmetry analyse_totaldiff
# already establishes for its own flagship presets. Confirmed by hand: the
# module's own flagship preset (x*y/sqrt(x^2+y^2) along x(t)=y(t)=t)
# reports \`defined: False\` at (0,0) on the raw formula, exactly as
# analyse_totaldiff's identical flagship does — an honest, documented
# asymmetry with the numeric tier's own override, not a contradiction.
_CR_T = Symbol('t', real=True)
_CR_PATH_NAMESPACE = dict(CALC_NAMESPACE)
_CR_PATH_NAMESPACE['t'] = _CR_T


def _cr_parse_path(expr_str):
    """Parse x(t) or y(t) from expr.mjs's parseExpr(text, ['t']).toSympy().

    TRAP, confirmed by hand while building this module: a genuinely
    malformed string like 'not-a-number' can sympify WITHOUT raising at
    all — Python's own \`not\` keyword combines with the bare symbols into a
    boolean expression that sympify happily evaluates to a plain Python
    \`bool\`, which has no \`.free_symbols\` attribute at all. Guarded here
    (raising ValueError, caught by analyse_chainrule's own caller) rather
    than letting an AttributeError escape uncaught."""
    try:
        f = sp.sympify(expr_str, locals=_CR_PATH_NAMESPACE)
        if not isinstance(f, sp.Basic):
            raise ValueError(f'not a valid expression: {expr_str!r}')
    except Exception as e:  # noqa: BLE001
        raise ValueError(f'cannot parse: {e}') from e
    bad = f.free_symbols - {_CR_T}
    if bad:
        raise ValueError(f'unexpected symbols {sorted(map(str, bad))}')
    return f


def _cr_eval1(expr, t0_val):
    """Evaluate a path component at t0_val; returns (value, ok). Mirrors
    \`_part_eval\`'s own denominator-first trap check (see its docstring) —
    the identical .subs() eagerly-simplifies-through-a-0/0 risk applies to
    a single-variable path expression exactly as it does to f(x,y)."""
    try:
        _, den = sp.together(expr).as_numer_denom()
        den_at = sp.simplify(den.subs(_CR_T, t0_val))
        if den_at == 0:
            return None, False
    except Exception:  # noqa: BLE001
        pass
    try:
        v = sp.simplify(expr.subs(_CR_T, t0_val))
    except Exception:  # noqa: BLE001
        return None, False
    if v.has(sp.zoo, sp.oo, -sp.oo, sp.nan) or v.free_symbols:
        return None, False
    try:
        vn = sp.N(v, 30)
    except Exception:  # noqa: BLE001
        return None, False
    if not vn.is_real:
        return None, False
    return v, True


def _cr_path_derivative(expr, t0_val):
    """x'(t0) or y'(t0), decided EXACTLY via a one-sided-limit check at the
    pivot — the SAME renaming trick \`_part_partial\` uses for its own
    y-slice (calc_safe_diff_at hardcodes the module-global \`x\` symbol, so a
    path expression purely in \`t\` is renamed t->x before calling it)."""
    if calc_has_abstain_funcs(expr):
        raise Abstain('path derivative via one-sided limits is unreliable with floor/ceiling/sign/Piecewise')
    renamed = expr.subs(_CR_T, x)
    l, r = calc_safe_diff_at(renamed, t0_val)
    finite = lambda v: v is not None and v.is_finite and v.is_real  # noqa: E731
    if finite(l) and finite(r) and sp.simplify(l - r) == 0:
        return {'pass': True, 'value': _calc_num(l), 'exactValue': l, 'valueTex': sp.latex(sp.nsimplify(l)), 'issues': [], 'provenance': 'proved'}
    issue = {
        'kind': 'corner' if finite(l) and finite(r) else 'cusp',
        'left': _calc_num(l) if l is not None else None, 'right': _calc_num(r) if r is not None else None,
        'detail': 'one-sided slopes disagree' if finite(l) and finite(r) else 'one-sided slope is infinite',
    }
    return {'pass': False, 'value': None, 'issues': [issue], 'provenance': 'refuted'}


def analyse_chainrule(expr_str, xt_str, yt_str, t0_str):
    """expr_str: SymPy-syntax string for f(x,y) from expr.mjs's
    parseExpr(text,['x','y']).toSympy(). xt_str/yt_str: SymPy-syntax
    strings for x(t)/y(t) from expr.mjs's parseExpr(text,['t']).toSympy().
    t0_str: SymPy-syntax scalar string from expr.mjs's parseNumber().sympy.

    Topic 2.17: Chain rule for composites. See this file's own section
    header above and chainruleEngine.js's header for the full design
    rationale.

    \`blockedAct\` walks the SAME four-way halt sequence chainruleEngine.js's
    own analyzeChainRule does: 'pathPoint' (x(t0)/y(t0) undefined) ->
    'pathDiff' (x'(t0) or y'(t0) fails) -> 'fUndefined' (f(a,b) undefined,
    the raw-formula/override asymmetry documented above) ->
    'partialsExist' (f_x(a,b) or f_y(a,b) fails) -> None (formula AND
    direct both computed, decided against each other exactly).

    Overall \`provenance\`: 'unknown' if any ATTEMPTED clause abstained; else
    'numeric' if any input was a float; else 'proved' if every clause holds
    and the two conclusion routes agree EXACTLY; else 'refuted'.

    Confirmed by hand (see chainruleEngine.js's own header and the module's
    completion report) which presets this decides exactly vs abstains on:
    every preset except the two flagship (xy/sqrt(x^2+y^2)) presets is
    decided exactly (the theorem-holds, both breakPathDiff, breakPartials,
    and degenerate presets). The two flagship presets (2.17.6/2.17.7) both
    report \`defined: False\` for f at (0,0) on the raw formula, the SAME
    documented asymmetry with the numeric tier's own override input that
    analyse_totaldiff already establishes for its identical flagship — see
    this file's own section docstring above for the reasoning; a student
    clicking "check with symbolic engine" on either preset sees this exact,
    correct, honestly-reported disagreement, not a defect.
    """
    try:
        f = _part_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        xexpr = _cr_parse_path(xt_str)
        yexpr = _cr_parse_path(yt_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        t0_val, et = calc_parse_scalar(t0_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    exact = et

    def _safe(fn):
        try:
            return fn()
        except Abstain as e:
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
        except Exception as e:  # noqa: BLE001
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': f'unexpected error: {e}'}

    a_val, a_ok = _cr_eval1(xexpr, t0_val)
    b_val, b_ok = _cr_eval1(yexpr, t0_val)
    path_ok = a_ok and b_ok
    pathPoint = {
        'pass': path_ok,
        'a': _calc_num(a_val) if a_ok else None,
        'b': _calc_num(b_val) if b_ok else None,
        'issues': [] if path_ok else [{'kind': 'undefined', 'detail': 'x(t0) or y(t0) is not a finite real number'}],
        'provenance': calc_cap('proved' if path_ok else 'refuted', exact),
    }
    if not path_ok:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'x(t0) or y(t0) is undefined'}], 'provenance': pathPoint['provenance']}
        return {
            'id': 'chainrule', 'provenance': pathPoint['provenance'], 'exactInputs': exact,
            'pathPoint': pathPoint, 'xPrime': skipped, 'yPrime': skipped, 'pathDifferentiable': False,
            'totalDiff': None, 'formula': None, 'direct': None, 'conclusion': None,
            'counterexample': False, 'blockedAct': 'pathPoint',
        }

    xPrime = _safe(lambda: _cr_path_derivative(xexpr, t0_val))
    yPrime = _safe(lambda: _cr_path_derivative(yexpr, t0_val))
    for c in (xPrime, yPrime):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)
    path_diff = xPrime.get('pass') is True and yPrime.get('pass') is True

    if not path_diff:
        provs = [pathPoint['provenance'], xPrime['provenance'], yPrime['provenance']]
        overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('refuted', exact))
        return {
            'id': 'chainrule', 'provenance': overall, 'exactInputs': exact,
            'pathPoint': pathPoint, 'xPrime': xPrime, 'yPrime': yPrime, 'pathDifferentiable': False,
            'totalDiff': None, 'formula': None, 'direct': None, 'conclusion': None,
            'counterexample': False, 'blockedAct': 'pathDiff',
        }

    fab_val, defined_ok = _part_eval(f, a_val, b_val)
    defined = {
        'pass': defined_ok,
        'value': _calc_num(fab_val) if defined_ok else None,
        'issues': [] if defined_ok else [{'kind': 'undefined', 'detail': 'f(a,b) is not a finite real number'}],
        'provenance': calc_cap('proved' if defined_ok else 'refuted', exact),
    }
    if not defined_ok:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f(a,b) is undefined'}], 'provenance': defined['provenance']}
        totalDiff = {'defined': defined, 'partialX': skipped, 'partialY': skipped, 'totallyDifferentiable': skipped, 'partialsExist': False}
        return {
            'id': 'chainrule', 'provenance': defined['provenance'], 'exactInputs': exact,
            'pathPoint': pathPoint, 'xPrime': xPrime, 'yPrime': yPrime, 'pathDifferentiable': True,
            'totalDiff': totalDiff, 'formula': None, 'direct': None, 'conclusion': None,
            'counterexample': False, 'blockedAct': 'fUndefined',
        }

    partialX = _safe(lambda: _part_partial(f, 'x', a_val, b_val))
    partialY = _safe(lambda: _part_partial(f, 'y', a_val, b_val))
    for c in (partialX, partialY):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)
    partials_exist = partialX.get('pass') is True and partialY.get('pass') is True

    if not partials_exist:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'a partial derivative of f does not exist here'}], 'provenance': 'numeric'}
        totalDiff = {'defined': defined, 'partialX': partialX, 'partialY': partialY, 'totallyDifferentiable': skipped, 'partialsExist': False}
        provs = [pathPoint['provenance'], xPrime['provenance'], yPrime['provenance'], defined['provenance'], partialX['provenance'], partialY['provenance']]
        overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('refuted', exact))
        return {
            'id': 'chainrule', 'provenance': overall, 'exactInputs': exact,
            'pathPoint': pathPoint, 'xPrime': xPrime, 'yPrime': yPrime, 'pathDifferentiable': True,
            'totalDiff': totalDiff, 'formula': None, 'direct': None, 'conclusion': None,
            'counterexample': False, 'blockedAct': 'partialsExist',
        }

    fx_val = partialX['exactValue']
    fy_val = partialY['exactValue']
    totallyDifferentiable = _safe(lambda: _td_differentiable_at(f, fx_val, fy_val, a_val, b_val, fab_val))
    if totallyDifferentiable.get('pass') is not None:
        totallyDifferentiable['provenance'] = calc_cap(totallyDifferentiable['provenance'], exact)
    totalDiff = {'defined': defined, 'partialX': partialX, 'partialY': partialY, 'totallyDifferentiable': totallyDifferentiable, 'partialsExist': True}

    xp_val = xPrime['exactValue']
    yp_val = yPrime['exactValue']
    formula_exact = fx_val * xp_val + fy_val * yp_val
    formula = {'value': _calc_num(formula_exact), 'valueTex': sp.latex(sp.nsimplify(formula_exact)), 'exactValue': formula_exact}

    def _direct():
        if calc_has_abstain_funcs(f) or calc_has_abstain_funcs(xexpr) or calc_has_abstain_funcs(yexpr):
            raise Abstain('direct differentiation of the composite is unreliable with floor/ceiling/sign/Piecewise')
        # CONFIRMED BY HAND (see the module's own completion report): the
        # composite substitution can produce an expression sp.limit's own
        # generic (Gruntz) algorithm struggles to simplify DIRECTLY inside
        # a one-sided-limit quotient — e.g. cos(t)**2+sin(t)**2 composed
        # into a difference quotient took over two minutes unsimplified,
        # vs. under a millisecond once sp.simplify() collapses it to the
        # constant 1 FIRST. g_expr is a concrete, fully-substituted
        # expression purely in t at this point (no free parameters left to
        # simplify away prematurely), so simplifying before differentiating
        # is safe and does not discard any generality calc_safe_diff_at
        # itself would otherwise need to preserve.
        g_expr = sp.simplify(f.subs({x: xexpr, y: yexpr}))
        g_renamed = g_expr.subs(_CR_T, x)
        l, r = calc_safe_diff_at(g_renamed, t0_val)
        finite = lambda v: v is not None and v.is_finite and v.is_real  # noqa: E731
        if finite(l) and finite(r) and sp.simplify(l - r) == 0:
            return {'pass': True, 'value': _calc_num(l), 'exactValue': l, 'issues': [], 'provenance': 'proved'}
        issue = {
            'kind': 'corner' if finite(l) and finite(r) else 'cusp',
            'detail': 'one-sided slopes disagree' if finite(l) and finite(r) else 'one-sided slope is infinite',
        }
        return {'pass': False, 'value': None, 'issues': [issue], 'provenance': 'refuted'}

    direct = _safe(_direct)
    if direct.get('pass') is not None:
        direct['provenance'] = calc_cap(direct['provenance'], exact)

    conclusion_match = bool(
        direct.get('pass') is True
        and direct.get('exactValue') is not None
        and sp.simplify(direct['exactValue'] - formula['exactValue']) == 0
    )
    counterexample = totallyDifferentiable.get('pass') is False

    provs = [
        pathPoint['provenance'], xPrime['provenance'], yPrime['provenance'],
        defined['provenance'], partialX['provenance'], partialY['provenance'],
        totallyDifferentiable['provenance'], direct['provenance'],
    ]
    if 'unknown' in provs:
        overall = 'unknown'
    elif 'numeric' in provs:
        overall = 'numeric'
    elif conclusion_match:
        overall = calc_cap('proved', exact)
    else:
        overall = calc_cap('refuted', exact)

    return {
        'id': 'chainrule', 'provenance': overall, 'exactInputs': exact,
        'pathPoint': pathPoint, 'xPrime': xPrime, 'yPrime': yPrime, 'pathDifferentiable': True,
        'totalDiff': totalDiff, 'formula': formula, 'direct': direct,
        'conclusion': {'pass': conclusion_match},
        'counterexample': counterexample, 'blockedAct': None,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — extrema: unconstrained extrema, critical
#    points & the Hessian (topic 2.18), the direct sequel to chainrule
#    (2.17) — now that partial derivatives, total differentiability and the
#    tangent plane are established, this asks WHERE the tangent plane is
#    horizontal and what the second partials say about the shape there ──
# =============================================================================
# THEOREM module. (a,b) is a CRITICAL POINT of f if f_x(a,b)=f_y(a,b)=0.
# Given a critical point where the second partials agree (Clairaut/Schwarz:
# f_xy=f_yx, so the Hessian H=[[f_xx,f_xy],[f_xy,f_yy]] is symmetric), the
# discriminant D=det(H)=f_xx f_yy - f_xy^2 classifies the point: D>0 &
# f_xx>0 -> min; D>0 & f_xx<0 -> max; D<0 -> saddle; D=0 -> genuinely
# INCONCLUSIVE (see extremaPresets.mjs's own flagship x^4+y^4/x^4-y^4 pair —
# the SAME D=0, opposite truths). \`verdict\` is reported as a field ORTHOGONAL
# to \`provenance\`, mirroring analyse_series' own provenance/verdict split
# (topic 2.11) — D=0 is a mathematically exact "the test is silent" fact,
# never an abstention.
#
# Reuses \`_part_parse\`/\`_part_eval\`/\`_part_partial\` VERBATIM from
# analyse_partials/analyse_totaldiff/analyse_chainrule for \`defined\` and the
# FIRST partials (clause 1's own gating prerequisite). The genuinely new
# piece is calling \`_part_partial\` A SECOND TIME on a DERIVED expression —
# \`sp.diff(f, x)\` or \`sp.diff(f, y)\`, still a function of BOTH x and y, so
# \`_part_partial\`'s own x-slice/y-slice-with-renaming logic applies
# UNCHANGED to it — to get each second partial via the SAME one-sided-limit
# discipline used for the first, never a bare \`sp.diff(f, x, y).subs(...)\`
# evaluated directly at the point (the same "never trust diff() itself at a
# suspect point" rule this file's own module docstring states, extended one
# derivative deeper). This gives the module's own signature "show, don't
# assert" version of Clairaut: \`fxyOrder1\` (x differentiated first, via
# \`_part_partial(sp.diff(f,x), 'y', ...)\`) and \`fyxOrder2\` (y first, via
# \`_part_partial(sp.diff(f,y), 'x', ...)\`) are computed by two INDEPENDENT
# calls through two DIFFERENT derived expressions, then compared exactly —
# confirmed by hand on the classic counterexample
# \`x*y*(x**2-y**2)/(x**2+y**2)\` (f(0,0):=0): \`_part_partial\` on
# \`sp.diff(f,x)\` sliced at x=0 (renamed y->x) is EXACTLY \`-x\`, giving
# fxyOrder1 = -1; \`_part_partial\` on \`sp.diff(f,y)\` sliced at y=0 is EXACTLY
# \`x\`, giving fyxOrder2 = +1 — an EXACT, decisive mismatch, unlike the
# numeric (JS) tier's own float-sampled approximation of the same fact.
#
# Unlike analyse_partials' \`defined\`-only gate, this module's blockedAct
# walks a FIVE-way halt sequence mirroring extremaEngine.js's own:
# 'defined' -> 'partialsExist' -> 'criticalPoint' -> 'secondPartialsExist'
# -> 'secondPartialsAgree' -> None (fully classified). No override input is
# sent here — expr.mjs has no piecewise syntax, the SAME documented
# asymmetry analyse_totaldiff/analyse_chainrule already establish for their
# own flagships; the Clairaut preset's raw formula is genuinely undefined
# at the origin (0/0), so \`analyse_extrema\` honestly reports
# \`defined: False\` there — see extremaPresets.mjs's own header, which
# documents this exact, expected asymmetry for its 2.18.5 preset.
def analyse_extrema(expr_str, a_str, b_str):
    """expr_str: SymPy-syntax string for f(x,y) from expr.mjs's
    parseExpr(text,['x','y']).toSympy(). a_str/b_str: SymPy-syntax scalar
    strings from expr.mjs's parseNumber().sympy.

    Topic 2.18: Unconstrained extrema — critical points & the Hessian. See
    this file's own section header above and extremaEngine.js's header for
    the full design rationale.

    Overall \`provenance\`: 'unknown' if any ATTEMPTED clause abstained; else
    'numeric' if any input was a float; else 'proved' if defined holds,
    both first partials exist, (a,b) is exactly critical, and the second
    partials exist and exactly agree; else 'refuted' — note \`provenance\`
    never encodes WHICH classification was reached (min/max/saddle/
    inconclusive all count as a successfully decided, 'proved' analysis
    when every clause holds and D != 0 exactly) — that is \`verdict\`'s job,
    kept deliberately orthogonal (see this section's own header).

    Confirmed by hand (see extremaEngine.js's header and the module's
    completion report) which presets this decides exactly: every "holds"
    preset (2.18.1-2.18.3), the not-critical break (2.18.4), the flagship
    D=0 pair (2.18.6-2.18.7, BOTH decided exactly as 'inconclusive' via
    exact D=0, not a float near-zero reading), and the degenerate
    whole-line case (2.18.8) are ALL decided exactly, \`provenance: 'proved'\`
    on exact input. The Clairaut counterexample (2.18.5) reports
    \`defined: False\` on the raw formula at the origin, honestly, per this
    section's own header — the numeric tier's own override input is what
    makes its counterexample demonstrable end-to-end; the symbolic tier
    still exactly confirms the underlying fact (fxyOrder1=-1 !=
    fyxOrder2=+1) via the module's own selftest, which calls \`_part_partial\`
    directly on the derived expressions rather than through
    \`analyse_extrema\` for that specific fact.
    """
    try:
        f = _part_parse(expr_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        a_val, ea = calc_parse_scalar(a_str)
        b_val, eb = calc_parse_scalar(b_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    exact = ea and eb

    def _safe(fn):
        try:
            return fn()
        except Abstain as e:
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
        except Exception as e:  # noqa: BLE001
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': f'unexpected error: {e}'}

    fab_val, defined_ok = _part_eval(f, a_val, b_val)
    defined = {
        'pass': defined_ok,
        'value': _calc_num(fab_val) if defined_ok else None,
        'issues': [] if defined_ok else [{'kind': 'undefined', 'detail': 'f(a,b) is not a finite real number'}],
        'provenance': calc_cap('proved' if defined_ok else 'refuted', exact),
    }
    if not defined_ok:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f(a,b) is undefined'}], 'provenance': defined['provenance']}
        return {
            'id': 'extrema', 'provenance': defined['provenance'], 'exactInputs': exact,
            'defined': defined, 'partialX': skipped, 'partialY': skipped, 'partialsExist': False,
            'criticalPoint': False, 'fxx': skipped, 'fyy': skipped, 'fxyOrder1': skipped, 'fyxOrder2': skipped,
            'secondPartialsExist': False, 'secondPartialsAgree': None, 'D': None, 'verdict': None,
            'counterexample': False, 'blockedAct': 'defined',
        }

    partialX = _safe(lambda: _part_partial(f, 'x', a_val, b_val))
    partialY = _safe(lambda: _part_partial(f, 'y', a_val, b_val))
    for c in (partialX, partialY):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)
    partials_exist = partialX.get('pass') is True and partialY.get('pass') is True

    if not partials_exist:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'a first partial derivative does not exist here'}], 'provenance': 'numeric'}
        provs = [defined['provenance'], partialX['provenance'], partialY['provenance']]
        overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('refuted', exact))
        return {
            'id': 'extrema', 'provenance': overall, 'exactInputs': exact,
            'defined': defined, 'partialX': partialX, 'partialY': partialY, 'partialsExist': False,
            'criticalPoint': False, 'fxx': skipped, 'fyy': skipped, 'fxyOrder1': skipped, 'fyxOrder2': skipped,
            'secondPartialsExist': False, 'secondPartialsAgree': None, 'D': None, 'verdict': None,
            'counterexample': False, 'blockedAct': 'partialsExist',
        }

    critical = bool(sp.simplify(partialX['exactValue']) == 0 and sp.simplify(partialY['exactValue']) == 0)

    if not critical:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': '(a,b) is not a critical point'}], 'provenance': 'numeric'}
        return {
            'id': 'extrema', 'provenance': calc_cap('refuted', exact), 'exactInputs': exact,
            'defined': defined, 'partialX': partialX, 'partialY': partialY, 'partialsExist': True,
            'criticalPoint': False, 'fxx': skipped, 'fyy': skipped, 'fxyOrder1': skipped, 'fyxOrder2': skipped,
            'secondPartialsExist': False, 'secondPartialsAgree': None, 'D': None, 'verdict': None,
            'counterexample': False, 'blockedAct': 'criticalPoint',
        }

    # The genuinely new piece: _part_partial applied to a DERIVED expression
    # (still a function of x AND y) — see this section's own header.
    fx_expr = sp.diff(f, x)
    fy_expr = sp.diff(f, y)
    fxx = _safe(lambda: _part_partial(fx_expr, 'x', a_val, b_val))
    fyy = _safe(lambda: _part_partial(fy_expr, 'y', a_val, b_val))
    fxyOrder1 = _safe(lambda: _part_partial(fx_expr, 'y', a_val, b_val))  # d/dy[f_x]
    fyxOrder2 = _safe(lambda: _part_partial(fy_expr, 'x', a_val, b_val))  # d/dx[f_y]
    for c in (fxx, fyy, fxyOrder1, fyxOrder2):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)
    second_exist = all(c.get('pass') is True for c in (fxx, fyy, fxyOrder1, fyxOrder2))

    if not second_exist:
        provs = [defined['provenance'], partialX['provenance'], partialY['provenance'],
                 fxx['provenance'], fyy['provenance'], fxyOrder1['provenance'], fyxOrder2['provenance']]
        overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('refuted', exact))
        return {
            'id': 'extrema', 'provenance': overall, 'exactInputs': exact,
            'defined': defined, 'partialX': partialX, 'partialY': partialY, 'partialsExist': True,
            'criticalPoint': True, 'fxx': fxx, 'fyy': fyy, 'fxyOrder1': fxyOrder1, 'fyxOrder2': fyxOrder2,
            'secondPartialsExist': False, 'secondPartialsAgree': None, 'D': None, 'verdict': None,
            'counterexample': False, 'blockedAct': 'secondPartialsExist',
        }

    agree = bool(sp.simplify(fxyOrder1['exactValue'] - fyxOrder2['exactValue']) == 0)

    if not agree:
        provs = [defined['provenance'], partialX['provenance'], partialY['provenance'],
                 fxx['provenance'], fyy['provenance'], fxyOrder1['provenance'], fyxOrder2['provenance']]
        overall = 'unknown' if 'unknown' in provs else calc_cap('refuted', exact)
        return {
            'id': 'extrema', 'provenance': overall, 'exactInputs': exact,
            'defined': defined, 'partialX': partialX, 'partialY': partialY, 'partialsExist': True,
            'criticalPoint': True, 'fxx': fxx, 'fyy': fyy, 'fxyOrder1': fxyOrder1, 'fyxOrder2': fyxOrder2,
            'secondPartialsExist': True, 'secondPartialsAgree': False, 'D': None, 'verdict': None,
            'counterexample': True, 'blockedAct': 'secondPartialsAgree',
        }

    fxy_val = fxyOrder1['exactValue']
    fxx_val = fxx['exactValue']
    fyy_val = fyy['exactValue']
    D_exact = sp.simplify(fxx_val * fyy_val - fxy_val ** 2)
    D_num = _calc_num(D_exact)
    if D_exact == 0:
        verdict = 'inconclusive'
    elif D_exact > 0:
        verdict = 'min' if fxx_val > 0 else 'max'
    else:
        verdict = 'saddle'

    provs = [defined['provenance'], partialX['provenance'], partialY['provenance'],
             fxx['provenance'], fyy['provenance'], fxyOrder1['provenance'], fyxOrder2['provenance']]
    overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('proved', exact))

    return {
        'id': 'extrema', 'provenance': overall, 'exactInputs': exact,
        'defined': defined, 'partialX': partialX, 'partialY': partialY, 'partialsExist': True,
        'criticalPoint': True, 'fxx': fxx, 'fyy': fyy, 'fxyOrder1': fxyOrder1, 'fyxOrder2': fyxOrder2,
        'secondPartialsExist': True, 'secondPartialsAgree': True,
        'D': D_num, 'DTex': sp.latex(sp.nsimplify(D_exact)), 'verdict': verdict,
        'counterexample': False, 'blockedAct': None,
    }


# =============================================================================
# ── Calculus domain (DOMAINS[1]) — Lagrange multipliers: constrained
#    extrema (topic 2.19) ──
# =============================================================================
# THEOREM module, the direct sequel to extrema (2.18) — see
# lagrangeEngine.js's own header for the full design rationale (hypothesis
# structure, the ONE breakable hypothesis being regularity grad g != (0,0),
# and why grad f=(0,0) and grad g=(0,0) mattering TOGETHER is what makes a
# genuine "hypothesis fails yet conclusion holds" instance possible here).
#
# Reuses _PART_NAMESPACE/_part_parse/_part_eval/_part_partial VERBATIM — f
# and g are both ordinary two-variable expressions in the SAME x,y
# namespace partials/totaldiff/chainrule/extrema already established;
# nothing new is needed for a SECOND two-variable expression alongside f —
# analyse_cauchymvt/analyse_ftc already established the "more than one
# expression string in one payload" shape for the single-variable Calculus
# modules; this is its first two-variable instance.
def analyse_lagrange(f_str, g_str, a_str, b_str):
    """f_str/g_str: SymPy-syntax strings from expr.mjs's
    parseExpr(text,['x','y']).toSympy(). a_str/b_str: SymPy-syntax scalar
    strings from expr.mjs's parseNumber().sympy.

    Topic 2.19: Constrained extrema — Lagrange multipliers. See this file's
    own section header above and lagrangeEngine.js's header for the full
    design rationale. Checks a candidate point (a,b): f(a,b)/g(a,b) defined
    -> (a,b) on the constraint g=0 -> both gradients exist -> THE ONE
    hypothesis (regular: grad g != (0,0)) -> the conclusion (does there
    exist lambda with grad f = lambda*grad g).

    \`provenance\`: 'unknown' if any ATTEMPTED clause abstained; else
    'numeric' if any input was a float; else 'proved'/'refuted' depending on
    whether the chain DECIDES the conclusion (lagrangeHolds True or False) —
    mirroring analyse_extrema's own discipline: a decisively refuted
    pre-check/hypothesis is still 'refuted', not merely 'unknown'.
    """
    try:
        f = _part_parse(f_str)
        g = _part_parse(g_str)
    except ValueError as e:
        return calc_unknown(f'bad input: {e}')
    try:
        a_val, ea = calc_parse_scalar(a_str)
        b_val, eb = calc_parse_scalar(b_str)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'bad input: {e}')
    exact = ea and eb

    def _safe(fn):
        try:
            return fn()
        except Abstain as e:
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': str(e)}
        except Exception as e:  # noqa: BLE001
            return {'pass': None, 'issues': [], 'provenance': 'unknown', 'reason': f'unexpected error: {e}'}

    fab_val, f_ok = _part_eval(f, a_val, b_val)
    gab_val, g_ok = _part_eval(g, a_val, b_val)
    fDefined = {
        'pass': f_ok, 'value': _calc_num(fab_val) if f_ok else None,
        'issues': [] if f_ok else [{'kind': 'undefined', 'detail': 'f(a,b) is not a finite real number'}],
        'provenance': calc_cap('proved' if f_ok else 'refuted', exact),
    }
    gDefined = {
        'pass': g_ok, 'value': _calc_num(gab_val) if g_ok else None,
        'issues': [] if g_ok else [{'kind': 'undefined', 'detail': 'g(a,b) is not a finite real number'}],
        'provenance': calc_cap('proved' if g_ok else 'refuted', exact),
    }

    def _base(**extra):
        return {'id': 'lagrange', 'exactInputs': exact, 'fDefined': fDefined, 'gDefined': gDefined, **extra}

    if not (f_ok and g_ok):
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': 'f(a,b) or g(a,b) undefined'}], 'provenance': 'numeric'}
        overall = fDefined['provenance'] if not f_ok else gDefined['provenance']
        return _base(
            provenance=overall,
            onConstraint=skipped, gradF=skipped, gradG=skipped, gradientsExist=False,
            regular=None, lagrangeHolds=None, **{'lambda': None},
            licensed=False, counterexample=False, blockedAct='defined',
        )

    on_constraint = bool(sp.simplify(gab_val) == 0)
    onConstraint = {
        'pass': on_constraint,
        'issues': [] if on_constraint else [{'kind': 'off-constraint', 'detail': 'g(a,b) != 0'}],
        'provenance': calc_cap('proved' if on_constraint else 'refuted', exact),
    }
    if not on_constraint:
        skipped = {'pass': False, 'issues': [{'kind': 'skipped', 'detail': '(a,b) is not on the constraint curve'}], 'provenance': 'numeric'}
        return _base(
            provenance=onConstraint['provenance'],
            onConstraint=onConstraint, gradF=skipped, gradG=skipped, gradientsExist=False,
            regular=None, lagrangeHolds=None, **{'lambda': None},
            licensed=False, counterexample=False, blockedAct='onConstraint',
        )

    fx = _safe(lambda: _part_partial(f, 'x', a_val, b_val))
    fy = _safe(lambda: _part_partial(f, 'y', a_val, b_val))
    gx = _safe(lambda: _part_partial(g, 'x', a_val, b_val))
    gy = _safe(lambda: _part_partial(g, 'y', a_val, b_val))
    for c in (fx, fy, gx, gy):
        if c.get('pass') is not None:
            c['provenance'] = calc_cap(c['provenance'], exact)
    gradients_exist = all(c.get('pass') is True for c in (fx, fy, gx, gy))

    if not gradients_exist:
        provs = [fDefined['provenance'], gDefined['provenance'], onConstraint['provenance'],
                 fx['provenance'], fy['provenance'], gx['provenance'], gy['provenance']]
        overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('refuted', exact))
        return _base(
            provenance=overall, onConstraint=onConstraint,
            gradF={'fx': fx, 'fy': fy, 'pass': fx.get('pass') is True and fy.get('pass') is True},
            gradG={'gx': gx, 'gy': gy, 'pass': gx.get('pass') is True and gy.get('pass') is True},
            gradientsExist=False, regular=None, lagrangeHolds=None, **{'lambda': None},
            licensed=False, counterexample=False, blockedAct='gradientsExist',
        )

    fx_val, fy_val, gx_val, gy_val = fx['exactValue'], fy['exactValue'], gx['exactValue'], gy['exactValue']
    grad_f_zero = bool(sp.simplify(fx_val) == 0 and sp.simplify(fy_val) == 0)
    grad_g_zero = bool(sp.simplify(gx_val) == 0 and sp.simplify(gy_val) == 0)
    regular = not grad_g_zero

    # See lagrangeEngine.js's own header for why grad_f_zero is checked
    # FIRST, before regularity: when grad f = (0,0), grad f = lambda*grad g
    # holds trivially for lambda=0 REGARDLESS of grad g — this is the
    # 'failsYetHolds' flagship (irregular) and the 'degenerate' constant-f
    # case (regular) alike; only when grad f is genuinely nonzero does
    # regularity actually gate the conclusion.
    lam_val = None
    if grad_f_zero:
        lagrange_holds = True
        lam_val = sp.Integer(0)
    elif regular:
        cross = sp.simplify(fx_val * gy_val - fy_val * gx_val)
        lagrange_holds = bool(cross == 0)
        if lagrange_holds:
            lam_val = sp.simplify(fx_val / gx_val) if gx_val != 0 else sp.simplify(fy_val / gy_val)
    else:
        lagrange_holds = False

    licensed = regular and lagrange_holds
    counterexample = (not regular) and lagrange_holds
    blocked = 'regular' if (not regular and not lagrange_holds) else None

    provs = [fDefined['provenance'], gDefined['provenance'], onConstraint['provenance'],
             fx['provenance'], fy['provenance'], gx['provenance'], gy['provenance']]
    overall = 'unknown' if 'unknown' in provs else ('numeric' if 'numeric' in provs else calc_cap('proved' if lagrange_holds else 'refuted', exact))

    return _base(
        provenance=overall, onConstraint=onConstraint,
        gradF={'fx': fx, 'fy': fy, 'pass': True}, gradG={'gx': gx, 'gy': gy, 'pass': True},
        gradientsExist=True, regular=regular, lagrangeHolds=lagrange_holds,
        **{'lambda': _calc_num(lam_val) if lam_val is not None else None,
           'lambdaTex': sp.latex(sp.nsimplify(lam_val)) if lam_val is not None else None},
        licensed=licensed, counterexample=counterexample, blockedAct=blocked,
    )


# ============================================================================
# DOMAINS[2] — Probability & Statistics. First module: \`probabilitylaws\`
# (3.4, Sample spaces, events & the axioms of probability). Reuses the
# shared parse_matrix/parse_entry/provenance_for parsing helpers (a bare
# list[str] parses as a column vector — exactly this module's outcome-
# probability table shape) and \`calc_cap\` (introduced for the Calculus
# domain, but its logic — "float input caps proved/refuted at numeric" — is
# domain-agnostic, so it is reused here rather than re-derived a third time).
# symbolicEngine.py/symbolic.worker.js stay ONE file each across all domains
# (CLAUDE.md: check-module.mjs hardcodes both paths).
# ============================================================================

def analyse_probabilitylaws(outcomes, entries, idx_a, idx_b):
    """outcomes: list[str] (outcome labels); entries: list[str] (one
    probability-table entry per outcome, exact fraction/integer or decimal —
    parsed via the shared parse_matrix/parse_entry helpers exactly like a
    matrix's own entries); idx_a/idx_b: list[int], 0-indexed outcome indices
    naming the two declared events A, B. An INDEPENDENT SymPy implementation
    of probabilitylawsEngine.js's analyzeProbabilityLaw (nothing here calls
    into the JS engine) — same axiom/consequence structure, verified
    separately in symbolicEngine.selftest.py against presets mirrored
    verbatim from probabilitylawsPresets.mjs.

    See probabilitylawsEngine.js's own header for the full design note on
    why axiom (c) (additivity) does not gate the derived consequences the
    way (a) nonNegativity/(b) normalization do: for a FINITE sample space,
    once (a)/(b) hold, finite additivity over ANY disjoint pair is a
    mathematical theorem, not a free assumption — what CAN fail is a
    student's own claim that a NON-disjoint pair may still use the
    P(A)+P(B) shortcut. That failure is reported (additivity.pass ==
    additivity.disjoint) but never halts anything; inclusion-exclusion,
    checked independently below, is unaffected and still holds exactly."""
    if not outcomes or len(outcomes) != len(entries):
        return calc_unknown('sample space and probability table must have the same length')
    try:
        M, exact = parse_matrix(list(entries))
    except ValueError as e:
        return calc_unknown(f'bad probability entry: {e}')

    values = [M[i, 0] for i in range(M.rows)]
    n = len(values)
    idx_a = [int(i) for i in idx_a]
    idx_b = [int(i) for i in idx_b]

    violations = [i for i, v in enumerate(values) if v < 0]
    non_neg_pass = len(violations) == 0
    total = sum(values, S.Zero)
    norm_pass = bool(simplify(total - 1) == 0)
    law_valid = non_neg_pass and norm_pass

    set_b = set(idx_b)
    inter_idx = [i for i in idx_a if i in set_b]
    union_idx = sorted(set(idx_a) | set(idx_b))
    disjoint = len(inter_idx) == 0
    p_a = sum((values[i] for i in idx_a), S.Zero)
    p_b = sum((values[i] for i in idx_b), S.Zero)
    p_union = sum((values[i] for i in union_idx), S.Zero)
    p_inter = sum((values[i] for i in inter_idx), S.Zero)
    p_claimed = p_a + p_b
    additivity = {
        'disjoint': disjoint, 'pass': disjoint,
        'pA': str(p_a), 'pB': str(p_b), 'pUnion': str(p_union),
        'pInter': str(p_inter), 'pClaimed': str(p_claimed),
        'matches': bool(simplify(p_claimed - p_union) == 0),
    }

    complement = monotonic = incl_excl = None
    if law_valid:
        set_a = set(idx_a)
        idx_ac = [i for i in range(n) if i not in set_a]
        p_ac = sum((values[i] for i in idx_ac), S.Zero)
        complement = {
            'pA': str(p_a), 'pAc': str(p_ac), 'total': str(p_a + p_ac),
            'pass': bool(simplify(p_a + p_ac - 1) == 0),
        }
        is_subset = all(i in set_b for i in idx_a)
        if is_subset:
            monotonic = {'applicable': True, 'pA': str(p_a), 'pB': str(p_b), 'pass': bool(p_a <= p_b)}
        else:
            monotonic = {'applicable': False}
        formula = p_a + p_b - p_inter
        incl_excl = {
            'pA': str(p_a), 'pB': str(p_b), 'pInter': str(p_inter),
            'pUnionDirect': str(p_union), 'formula': str(formula),
            'pass': bool(simplify(formula - p_union) == 0),
        }

    blocked = None
    if not non_neg_pass:
        blocked = 'nonNegativity'
    elif not norm_pass:
        blocked = 'normalization'

    return {
        'provenance': calc_cap('proved' if law_valid else 'refuted', exact),
        'exactInputs': exact,
        'nonNegativity': {'pass': non_neg_pass, 'violations': violations},
        'normalization': {'pass': norm_pass, 'total': str(total)},
        'lawValid': law_valid,
        'additivity': additivity,
        'complement': complement,
        'monotonic': monotonic,
        'inclusionExclusion': incl_excl,
        'blockedAct': blocked,
    }


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) — counting: permutations &
#    combinations (topic 3.5) ──
#
# Every input here is a small nonnegative integer a student types directly —
# unlike Linear Algebra/Calculus, this domain has NO float-vs-exact split for
# this topic (there is no "decimal n" the way there is a decimal matrix
# entry), so provenance is always 'proved' (the classification/identity
# genuinely holds) or 'refuted' (it genuinely does not) once n and k parse —
# never 'numeric'. Plain Python \`int\` (arbitrary precision) throughout: no
# sympy/factorial call is needed, and none of the float traps documented
# elsewhere in this file (nsimplify, Matrix.rank() on floats, ...) apply to
# integer counting at all. Mirrors countingEngine.js's own BigInt-exact
# design exactly (see that file's header for the full design rationale) —
# the two engines are independent implementations of the SAME arithmetic,
# which is what makes cross-checking them meaningful.
# =============================================================================


def _cnt_perm(n, k):
    """P(n,k) = n(n-1)...(n-k+1); 0 when k>n (standard convention — see
    countingEngine.js's own permP for the identical rule)."""
    if n < 0 or k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r *= (n - i)
    return r


def _cnt_comb(n, k):
    """C(n,k); 0 when k<0, n<0, or k>n."""
    if n < 0 or k < 0 or k > n:
        return 0
    kk = min(k, n - k)
    num = 1
    for i in range(kk):
        num *= (n - i)
    den = 1
    for i in range(2, kk + 1):
        den *= i
    return num // den


def _cnt_pow(n, k):
    """n^k, ordered sampling with replacement. 0^0 := 1 (one way to make
    zero draws), matching countingEngine.js's powBig."""
    if k == 0:
        return 1
    return n ** k


def _cnt_multichoose(n, k):
    """C(n+k-1,k): unordered sampling with replacement."""
    if k == 0:
        return 1
    if n == 0:
        return 0
    return _cnt_comb(n + k - 1, k)


def _cnt_formula_id(order, replacement):
    if order and not replacement:
        return 'P'
    if order and replacement:
        return 'nk'
    if not order and not replacement:
        return 'C'
    return 'multichoose'


def _cnt_count_for(n, k, order, replacement):
    fid = _cnt_formula_id(order, replacement)
    if fid == 'P':
        return _cnt_perm(n, k)
    if fid == 'nk':
        return _cnt_pow(n, k)
    if fid == 'C':
        return _cnt_comb(n, k)
    return _cnt_multichoose(n, k)


def _cnt_unknown(reason):
    return {'id': 'counting', 'provenance': 'unknown', 'reason': reason}


def analyse_counting(mode, n, k, order=None, replacement=None, trueOrder=None, trueReplacement=None):
    """mode: 'rules' | 'identity'. n/k: ints (or numeric strings) — a student's
    typed nonnegative integers, never a decimal for this topic. order/
    replacement/trueOrder/trueReplacement: bools, only meaningful for
    mode='rules' (see countingEngine.js's evaluateScenario for the identical
    JS-side computation this mirrors).

    RULES: reports whether the STUDENT's classification (order, replacement)
    produces the same count as the scenario's TRUE classification
    (trueOrder, trueReplacement) — this module's own breakable clauses.

    IDENTITY: the Team Captain identity n*C(n-1,k-1) = k*C(n,k) — see
    countingEngine.js's header for why this has NO breakable hypothesis
    (both sides collapse to 0 by convention whenever k=0 or k>n, so the
    identity holds for every nonnegative integer pair; 'regular' (1<=k<=n)
    is reported as an informational flag, not a gate).
    """
    try:
        n = int(n)
        k = int(k)
    except (TypeError, ValueError):
        return _cnt_unknown(f'n and k must be integers (got n={n!r}, k={k!r})')
    if n < 0 or k < 0:
        return _cnt_unknown('n and k must be nonnegative integers')

    if mode == 'identity':
        regular = 1 <= k <= n
        lhs = n * _cnt_comb(n - 1, k - 1)
        rhs = k * _cnt_comb(n, k)
        holds = lhs == rhs
        return {
            'id': 'counting', 'mode': 'identity', 'n': n, 'k': k,
            'regular': regular, 'lhs': lhs, 'rhs': rhs, 'holds': holds,
            'provenance': 'proved' if holds else 'refuted',
        }

    # mode == 'rules'
    try:
        order = bool(order)
        replacement = bool(replacement)
        trueOrder = bool(trueOrder)
        trueReplacement = bool(trueReplacement)
    except (TypeError, ValueError):
        return _cnt_unknown('order/replacement/trueOrder/trueReplacement must be booleans')

    orderMatches = order == trueOrder
    replacementMatches = replacement == trueReplacement
    chosenCount = _cnt_count_for(n, k, order, replacement)
    trueCount = _cnt_count_for(n, k, trueOrder, trueReplacement)
    holds = chosenCount == trueCount
    return {
        'id': 'counting', 'mode': 'rules', 'n': n, 'k': k,
        'orderMatches': orderMatches, 'replacementMatches': replacementMatches,
        'chosenCount': chosenCount, 'trueCount': trueCount, 'holds': holds,
        'chosenFormula': _cnt_formula_id(order, replacement),
        'trueFormula': _cnt_formula_id(trueOrder, trueReplacement),
        'provenance': 'proved' if holds else 'refuted',
    }


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) — descriptive statistics:
#    measuring centre, spread & robustness (topic 3.2) ──
#
# An INDEPENDENT SymPy implementation of descriptivestatsEngine.js's
# analyzeDataset (nothing here calls into the JS engine) — same structure
# (mean/median/mode/sample variance/quartiles/IQR, the two grid-search-
# verified minimization clauses, the empirical robustness probe), verified
# separately in symbolicEngine.selftest.py against presets mirrored verbatim
# from descriptivestatsPresets.mjs. See descriptivestatsEngine.js's own
# header for the full design rationale (variance convention: SAMPLE, divide
# by n-1; quartile convention: exclusive median-of-halves/Tukey hinges).
#
# Exact Rational arithmetic throughout via parse_matrix (a bare list of
# entry strings is treated as a column vector — the SAME helper
# probabilitylaws' own table uses) — decimals flip the whole dataset to
# 'numeric' provenance, never 'proved', exactly like every other module.
# =============================================================================


def _ds_mean(values):
    return sum(values, S.Zero) / len(values)


def _ds_median(values):
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[(n - 1) // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _ds_modes(values):
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts.values())
    modes = sorted([v for v, c in counts.items() if c == best])
    no_unique = best <= 1 and len(values) > 1
    return modes, best, no_unique


def _ds_variance(values):
    n = len(values)
    if n < 2:
        return None
    m = _ds_mean(values)
    ssd = sum(((v - m) ** 2 for v in values), S.Zero)
    return ssd / (n - 1)


def _ds_quartiles(values):
    s = sorted(values)
    n = len(s)
    if n == 1:
        return s[0], s[0]
    if n % 2 == 0:
        lower, upper = s[: n // 2], s[n // 2 :]
    else:
        mid = (n - 1) // 2
        lower, upper = s[:mid], s[mid + 1 :]
    return _ds_median(lower), _ds_median(upper)


def _ds_ssd_at(values, c):
    return sum(((v - c) ** 2 for v in values), S.Zero)


def _ds_sad_at(values, c):
    return sum((Abs(v - c) for v in values), S.Zero)


def _ds_grid_minimizes(values, center, is_ssd):
    """Fine grid search confirming \`center\` (mean or median) beats every
    nearby candidate on its own loss (SSD or SAD) — mirrors
    descriptivestatsEngine.js's checkMeanMinimizesSSD/checkMedianMinimizesSAD
    exactly (same step-choosing rule, same +/-12-step grid), an independent
    SymPy re-verification rather than trusting the textbook fact."""
    s = sorted(values)
    rng = s[-1] - s[0]
    if rng != 0:
        step = rng / 24
    else:
        ref = Abs(center)
        step = ref / 24 if ref != 0 else Q(1, 24)
    loss_at = _ds_ssd_at if is_ssd else _ds_sad_at
    base = loss_at(values, center)
    for k in range(-12, 13):
        if k == 0:
            continue
        cand = center + step * k
        if loss_at(values, cand) < base:
            return False
    return True


def _ds_robustness(values, idx):
    """Replace values[idx] with two independently huge magnitudes (10x/1000x
    the data's own range past the current max) and compare each statistic:
    equal => robust, different => not robust — decided empirically, never
    from a hand-derived threshold, mirroring descriptivestatsEngine.js's
    robustnessProbe exactly (an independent SymPy re-implementation)."""
    n = len(values)
    s = sorted(values)
    max_v = s[-1]
    rng = s[-1] - s[0]
    if rng != 0:
        bump = rng
    else:
        bump = Abs(max_v) if Abs(max_v) != 0 else S.One
    m1 = max_v + bump * 10
    m2 = max_v + bump * 1000

    def with_val(v):
        y = list(values)
        y[idx] = v
        return y

    xs1, xs2 = with_val(m1), with_val(m2)
    mean1, mean2 = _ds_mean(xs1), _ds_mean(xs2)
    med1, med2 = _ds_median(xs1), _ds_median(xs2)
    var1, var2 = _ds_variance(xs1), _ds_variance(xs2)
    q1a, q3a = _ds_quartiles(xs1)
    q1b, q3b = _ds_quartiles(xs2)
    iqr1, iqr2 = q3a - q1a, q3b - q1b
    return {
        'M1': str(m1), 'M2': str(m2),
        'meanRobust': bool(mean1 == mean2),
        'medianRobust': bool(med1 == med2),
        'iqrRobust': bool(iqr1 == iqr2),
        'sdRobust': None if (var1 is None or var2 is None) else bool(var1 == var2),
    }


def analyse_descriptivestats(entries, outlierIndex=None):
    """entries: list[str] — one dataset value per entry (exact integer/
    fraction or decimal, via parse_matrix). outlierIndex: int or None —
    which observation the robustness probe replaces; defaults to the current
    maximum's own index, exactly like descriptivestatsEngine.js's
    analyzeDataset."""
    if not entries:
        return calc_unknown('dataset must have at least one value')
    try:
        M, exact = parse_matrix(list(entries))
    except ValueError as e:
        return calc_unknown(f'bad dataset entry: {e}')

    values = [M[i, 0] for i in range(M.rows)]
    n = len(values)
    s = sorted(values)

    mean = _ds_mean(values)
    median = _ds_median(values)
    modes, mult, no_unique_mode = _ds_modes(values)
    variance = _ds_variance(values)
    q1, q3 = _ds_quartiles(values)
    iqr = q3 - q1
    low_fence = q1 - iqr * Q(3, 2)
    high_fence = q3 + iqr * Q(3, 2)
    outlier_idxs = [i for i, v in enumerate(values) if v < low_fence or v > high_fence]

    mean_pass = _ds_grid_minimizes(values, mean, True)
    median_pass = _ds_grid_minimizes(values, median, False)

    try:
        idx = int(outlierIndex) if outlierIndex is not None else None
    except (TypeError, ValueError):
        idx = None
    if idx is None or idx < 0 or idx >= n:
        idx = values.index(s[-1])
    robustness = _ds_robustness(values, idx)

    definition_valid = mean_pass and median_pass

    return {
        'n': n,
        'mean': str(mean), 'median': str(median),
        'modes': [str(m) for m in modes], 'multiplicity': mult, 'noUniqueMode': no_unique_mode,
        'variance': None if variance is None else str(variance),
        'q1': str(q1), 'q3': str(q3), 'iqr': str(iqr),
        'lowFence': str(low_fence), 'highFence': str(high_fence),
        'outlierIndices': outlier_idxs,
        'meanMinimizesSSD': mean_pass, 'medianMinimizesSAD': median_pass,
        'robustness': robustness,
        'outlierIndex': idx,
        'definitionValid': definition_valid,
        'provenance': calc_cap('proved' if definition_valid else 'refuted', exact),
    }


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) — conditional probability &
#    the multiplication rule (topic 3.6) ──
#
# An INDEPENDENT SymPy implementation of conditionalEngine.js's own
# analyzeConditional (nothing here calls into the JS engine) — same
# structure, verified separately in symbolicEngine.selftest.py against
# presets mirrored verbatim from conditionalPresets.mjs. See
# conditionalEngine.js's own header for the full design note: ONE outcome
# table plus an ORDERED list of >= 2 declared events serves both halves of
# the topic — events[0]=A, events[1]=B for Part 1 ("conditional
# probabilities form a probability law", gated on THE one breakable
# precondition P(B) > 0) and the basic multiplication rule; the FULL
# ordered list is walked as A1..An for the multiplication rule's chain
# generalization, verified two independent ways on the same table (the
# telescoping product of conditional factors vs. the direct joint
# probability summed straight from the outcome indices).
# =============================================================================


def _cond_p_of(values, idx):
    return sum((values[i] for i in idx), S.Zero)


def analyse_conditional(outcomes, entries, events):
    """outcomes: list[str]; entries: list[str] (one probability-table entry
    per outcome, exact fraction/integer or decimal — parsed via the shared
    parse_matrix helper exactly like probabilitylaws' own table); events:
    list[list[int]], length >= 2, ordered 0-indexed outcome index sets.
    events[0]=A, events[1]=B for the axioms/basic multiplication rule; the
    FULL list is walked in order as A1..An for the chain rule."""
    if not outcomes or len(outcomes) != len(entries):
        return calc_unknown('sample space and probability table must have the same length')
    events = [[int(i) for i in ev] for ev in events]
    if len(events) < 2:
        return calc_unknown('need at least two declared events (A and B)')
    try:
        M, exact = parse_matrix(list(entries))
    except ValueError as e:
        return calc_unknown(f'bad probability entry: {e}')

    values = [M[i, 0] for i in range(M.rows)]

    violations = [i for i, v in enumerate(values) if v < 0]
    non_neg_pass = len(violations) == 0
    total = sum(values, S.Zero)
    norm_pass = bool(simplify(total - 1) == 0)
    table_valid = non_neg_pass and norm_pass

    idx_a, idx_b = events[0], events[1]
    set_a, set_b = set(idx_a), set(idx_b)
    p_a = _cond_p_of(values, idx_a)
    p_b = _cond_p_of(values, idx_b)
    conditioning = {'pass': bool(p_b > 0), 'pB': str(p_b)}

    axioms = None
    mult = None
    chain = None
    if table_valid:
        inter_ab = [i for i in idx_a if i in set_b]
        p_and = _cond_p_of(values, inter_ab)
        b_ready = bool(p_b > 0)
        a_ready = bool(p_a > 0)
        p_a_given_b = (p_and / p_b) if b_ready else None
        p_b_given_a = (p_and / p_a) if a_ready else None
        dir1 = None
        if b_ready:
            formula1 = p_a_given_b * p_b
            dir1 = {'formula': str(formula1), 'target': str(p_and), 'pass': bool(simplify(formula1 - p_and) == 0)}
        dir2 = None
        if a_ready:
            formula2 = p_b_given_a * p_a
            dir2 = {'formula': str(formula2), 'target': str(p_and), 'pass': bool(simplify(formula2 - p_and) == 0)}
        mult = {
            'pA': str(p_a), 'pB': str(p_b), 'pAandB': str(p_and),
            'bReady': b_ready, 'aReady': a_ready, 'dir1': dir1, 'dir2': dir2,
        }

        if conditioning['pass']:
            b_minus_a = [i for i in idx_b if i not in set_a]
            p_b_minus_a = _cond_p_of(values, b_minus_a)
            q_a = p_and / p_b
            q_bma = p_b_minus_a / p_b
            q_b = p_b / p_b
            non_neg_pass2 = bool(q_a >= 0) and bool(q_bma >= 0)
            norm_pass2 = bool(simplify(q_b - 1) == 0)
            sum_q = q_a + q_bma
            additivity_pass = bool(simplify(sum_q - q_b) == 0)
            axioms = {
                'pB': str(p_b), 'qAinterB': str(q_a), 'qBminusA': str(q_bma), 'qB': str(q_b),
                'nonNegPass': non_neg_pass2, 'normPass': norm_pass2, 'additivityPass': additivity_pass,
                'pass': non_neg_pass2 and norm_pass2 and additivity_pass,
            }

        prefixes = [events[0]]
        for k in range(1, len(events)):
            prev_set = set(prefixes[k - 1])
            prefixes.append([i for i in events[k] if i in prev_set])
        p_prefixes = [_cond_p_of(values, idx) for idx in prefixes]
        steps = []
        blocked_step = None
        broken = False
        product = p_prefixes[0]
        for k in range(1, len(events)):
            denom = p_prefixes[k - 1]
            numer = p_prefixes[k]
            denom_pos = bool(denom > 0)
            if not denom_pos and blocked_step is None:
                blocked_step = k + 1
            if not denom_pos:
                broken = True
            factor = (numer / denom) if denom_pos else None
            steps.append({
                'step': k + 1, 'numer': str(numer), 'denom': str(denom),
                'factor': (str(factor) if factor is not None else None), 'defined': denom_pos,
            })
            product = None if broken else product * factor
        chain_defined = blocked_step is None
        direct_joint = p_prefixes[-1]
        matches = chain_defined and bool(simplify(product - direct_joint) == 0)
        chain = {
            'n': len(events), 'pPrefixes': [str(x) for x in p_prefixes], 'steps': steps,
            'chainDefined': chain_defined, 'blockedStep': blocked_step,
            'chainProduct': (str(product) if product is not None else None),
            'directJoint': str(direct_joint), 'matches': matches,
        }

    blocked_act = None
    if not table_valid:
        blocked_act = 'nonNegativity' if not non_neg_pass else 'normalization'
    elif not conditioning['pass']:
        blocked_act = 'conditioningUndefined'

    law_valid = table_valid and conditioning['pass']

    return {
        'provenance': calc_cap('proved' if law_valid else 'refuted', exact),
        'exactInputs': exact,
        'tableValid': {'pass': table_valid, 'nonNegPass': non_neg_pass, 'normPass': norm_pass},
        'conditioning': conditioning,
        'axioms': axioms,
        'mult': mult,
        'chain': chain,
        'blockedAct': blocked_act,
        'lawValid': law_valid,
    }


# =============================================================================
# \`bayes\` (3.7, "Total probability & Bayes' rule"). An INDEPENDENT SymPy
# implementation of bayesEngine.js's own analyzeBayes (nothing here calls
# into the JS engine) — same partition/total-probability/Bayes'-rule/
# odds-form structure, verified separately in symbolicEngine.selftest.py
# against presets mirrored verbatim from bayesPresets.mjs.
#
# See bayesEngine.js's own header for the full design note on why the
# partition hypothesis has exactly TWO independently-breakable clauses
# (pairwise disjoint, covers Omega) — a third condition the theorem's own
# statement also lists, P(Bi) > 0 for every i, is reported (\`allPositive\`)
# but never gates anything: the ADDITIVE total-probability sum is
# well-defined regardless (a zero-probability part just contributes 0), only
# the weighted-average FORM needs it to form a likelihood factor at all.
#
# \`total_prob\` (the naive-partition-sum vs. direct-from-table comparison) is
# computed whenever the underlying table is valid, REGARDLESS of whether the
# declared partition itself is genuine — exactly mirroring the JS engine's
# own deliberate choice (see its header): the derivation still HALTS at
# whichever partition clause fails, but the two-routes comparison is what
# lets a "break" preset show a genuine numeric disagreement and a
# "fails-yet-holds" preset show a genuine numeric agreement, instead of
# merely asserting either. Bayes' rule and the odds form are gated on the
# partition genuinely holding.
# =============================================================================

def _bayes_p_of(values, idx):
    return sum((values[i] for i in idx), S.Zero)


def analyse_bayes(outcomes, entries, partition, event_a):
    if not outcomes or len(outcomes) != len(entries):
        return calc_unknown('sample space and probability table must have the same length')
    partition = [[int(i) for i in part] for part in partition]
    if len(partition) < 2:
        return calc_unknown('need a partition of at least two parts')
    event_a = [int(i) for i in event_a]
    try:
        M, exact = parse_matrix(list(entries))
    except ValueError as e:
        return calc_unknown(f'bad probability entry: {e}')

    values = [M[i, 0] for i in range(M.rows)]
    n = len(values)

    violations = [i for i, v in enumerate(values) if v < 0]
    non_neg_pass = len(violations) == 0
    total = sum(values, S.Zero)
    norm_pass = bool(simplify(total - 1) == 0)
    table_valid = non_neg_pass and norm_pass

    overlaps = []
    for i in range(len(partition)):
        set_i = set(partition[i])
        for j in range(i + 1, len(partition)):
            if set_i & set(partition[j]):
                overlaps.append((i, j))
    disjoint_pass = len(overlaps) == 0
    covered = set()
    for part in partition:
        covered |= set(part)
    missing = [i for i in range(n) if i not in covered]
    covers_pass = len(missing) == 0
    p_bi = [_bayes_p_of(values, part) for part in partition]
    zero_indices = [i for i, p in enumerate(p_bi) if not bool(p > 0)]
    partition_pass = disjoint_pass and covers_pass

    total_prob = None
    bayes = None
    odds = None
    if table_valid:
        set_a = set(event_a)
        p_a_and_bi = [_bayes_p_of(values, [i for i in part if i in set_a]) for part in partition]
        p_a_given_bi = [(p_a_and_bi[k] / p_bi[k]) if bool(p_bi[k] > 0) else None for k in range(len(partition))]
        weighted = sum(p_a_and_bi, S.Zero)
        direct = _bayes_p_of(values, event_a)
        matches = bool(simplify(weighted - direct) == 0)
        total_prob = {
            'pBi': [str(x) for x in p_bi], 'pAandBi': [str(x) for x in p_a_and_bi],
            'pAgivenBi': [(str(x) if x is not None else None) for x in p_a_given_bi],
            'totalPWeighted': str(weighted), 'directPA': str(direct), 'matches': matches,
        }

        if partition_pass:
            ready = bool(direct > 0)
            if ready:
                posteriors = []
                post_values = []
                for k in range(len(partition)):
                    raw = p_a_and_bi[k] / direct
                    if bool(p_bi[k] > 0):
                        formula = (p_bi[k] * p_a_given_bi[k]) / weighted
                    else:
                        formula = S.Zero
                    post_values.append(raw)
                    posteriors.append({
                        'index': k, 'raw': str(raw), 'formula': str(formula),
                        'matches': bool(simplify(raw - formula) == 0),
                    })
                sum_posteriors = sum(post_values, S.Zero)
                all_match = all(p['matches'] for p in posteriors)
                bayes = {
                    'ready': True, 'posteriors': posteriors,
                    'sumPosteriors': str(sum_posteriors), 'allMatch': all_match,
                    '_postValues': post_values,
                }
            else:
                bayes = {'ready': False}

            if bayes['ready'] and len(partition) == 2:
                pAgB1, pAgB2 = p_a_given_bi
                post1, post2 = bayes['_postValues']
                if bool(p_bi[0] > 0) and bool(p_bi[1] > 0) and pAgB2 is not None and bool(pAgB2 > 0) and bool(post2 > 0):
                    likelihood_ratio = pAgB1 / pAgB2
                    prior_odds = p_bi[0] / p_bi[1]
                    computed = likelihood_ratio * prior_odds
                    actual = post1 / post2
                    odds = {
                        'applicable': True, 'likelihoodRatio': str(likelihood_ratio),
                        'priorOdds': str(prior_odds), 'computedPosteriorOdds': str(computed),
                        'actualPosteriorOdds': str(actual), 'matches': bool(simplify(computed - actual) == 0),
                    }
                else:
                    odds = {'applicable': False, 'reason': 'a required probability is zero'}
            else:
                odds = {'applicable': False, 'reason': 'odds form needs exactly two partition parts, both positive, and P(A) > 0'}

    blocked_act = None
    if not table_valid:
        blocked_act = 'nonNegativity' if not non_neg_pass else 'normalization'
    elif not disjoint_pass:
        blocked_act = 'notDisjoint'
    elif not covers_pass:
        blocked_act = 'notCovering'

    hypotheses_hold = table_valid and partition_pass
    if bayes is not None:
        bayes.pop('_postValues', None)

    return {
        'provenance': calc_cap('proved' if hypotheses_hold else 'refuted', exact),
        'exactInputs': exact,
        'tableValid': {'pass': table_valid, 'nonNegPass': non_neg_pass, 'normPass': norm_pass},
        'partitionCheck': {
            'pass': partition_pass, 'pairwiseDisjointPass': disjoint_pass, 'coversPass': covers_pass,
            'allPositivePass': len(zero_indices) == 0, 'zeroIndices': zero_indices,
        },
        'totalProb': total_prob,
        'bayes': bayes,
        'odds': odds,
        'blockedAct': blocked_act,
        'hypothesesHold': hypotheses_hold,
    }


# =============================================================================
# \`randomvariables\` (3.9, "Random variables — PMF, PDF & CDF"). An
# INDEPENDENT SymPy implementation of randomvariablesEngine.js's own
# analyzeDiscrete/analyzeContinuous (nothing here calls into the JS engine) —
# see that file's header for the full design note on why nonNegativity/
# normalization are the ONLY two breakable clauses and why the CDF's own
# properties never independently gate anything once they hold.
#
# DISCRETE: xs/ps mirror the JS engine's own (x, p) rows exactly — ps is a
# bare list[str] of probability-table entries, parsed via the SAME shared
# parse_matrix helper probabilitylaws/conditional/descriptivestats already
# use (a decimal ANYWHERE flips the whole table to 'numeric' provenance,
# never 'proved' — non-negotiable #2). xs is a list of plain floats (already
# numeric x-values, not strings needing exact parsing — the x-axis LABELS
# carry no provenance of their own, only the probabilities do).
#
# CONTINUOUS: \`formula\` is a SymPy-syntax string from expr.mjs's toSympy()
# (the module converts its own mathjs-syntax formula before sending — see
# RandomVariablesModule.jsx's checkSymbolic — the same "JS never hands SymPy
# raw user text" discipline every Calculus module already follows).
# \`support_kind\`/\`lo_str\`/\`hi_str\` mirror randomvariablesEngine.js's own
# {kind, lo, hi} shape; a semi-infinite support becomes sp.oo directly
# (no string round-trip needed for +infinity — the JS side never needs to
# print 'oo' itself, unlike improperEngine.js's own parseBound, since a
# random-variable support is only ever right-semi-infinite here, never a
# user-typed bound). Unlike the numeric engine (which is honestly ALWAYS
# 'numeric' provenance for continuous, since Simpson's-rule quadrature is
# inherently approximate — see randomvariablesEngine.js's header),
# \`sp.integrate\` and \`solve_univariate_inequality\` decide both axioms
# EXACTLY on every preset here (confirmed by hand: both calls succeed
# cleanly on the uniform/quadratic/exponential/linear preset family), so the
# continuous branch CAN reach 'proved'/'refuted' symbolically even though its
# numeric twin never does — an intentional, documented asymmetry between the
# two tiers, not a bug. A formula/support this abstains on (e.g. a genuinely
# unintegrable density, or one solve_univariate_inequality cannot decide)
# reports 'unknown' rather than guessing, per non-negotiable #2.
# =============================================================================

def analyse_randomvariables(kind, xs, ps, formula, support_kind, lo_str, hi_str):
    if kind == 'discrete':
        if not xs or not ps or len(xs) != len(ps):
            return calc_unknown('x and p must be non-empty and the same length')
        try:
            xvals = [float(v) for v in xs]
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad x value: {e}')
        if len(set(xvals)) != len(xvals):
            return calc_unknown('duplicate x value')
        try:
            M, exact = parse_matrix(list(ps))
        except ValueError as e:
            return calc_unknown(f'bad probability entry: {e}')
        values = [M[i, 0] for i in range(M.rows)]

        order = sorted(range(len(xvals)), key=lambda i: xvals[i])
        sorted_x = [xvals[i] for i in order]
        sorted_p = [values[i] for i in order]

        violations = [i for i, v in enumerate(sorted_p) if v < 0]
        non_neg_pass = len(violations) == 0
        total = sum(sorted_p, S.Zero)
        norm_pass = bool(simplify(total - 1) == 0)
        law_valid = non_neg_pass and norm_pass

        cdf_total = None
        cdf_all_pass = None
        if law_valid:
            acc = S.Zero
            for xv, p in zip(sorted_x, sorted_p):
                acc = acc + p
            cdf_total = str(acc)
            cdf_all_pass = bool(simplify(acc - 1) == 0)

        blocked = None
        if not non_neg_pass:
            blocked = 'nonNegativity'
        elif not norm_pass:
            blocked = 'normalization'

        return {
            'provenance': calc_cap('proved' if law_valid else 'refuted', exact),
            'kind': 'discrete',
            'exactInputs': exact,
            'nonNegativity': {'pass': non_neg_pass, 'violations': violations},
            'normalization': {'pass': norm_pass, 'total': str(total)},
            'lawValid': law_valid,
            'cdfTotal': cdf_total,
            'cdfPropertiesAllPass': cdf_all_pass,
            'blockedAct': blocked,
        }

    if kind != 'continuous':
        return calc_unknown(f'unknown kind {kind!r}')

    try:
        f = calc_parse(formula)
    except ValueError as e:
        return calc_unknown(f'bad formula: {e}')
    try:
        lo, lo_exact = calc_parse_scalar(lo_str)
    except ValueError as e:
        return calc_unknown(f'bad lower support bound: {e}')

    if support_kind == 'semiInfRight':
        hi, hi_exact = oo, True
    else:
        try:
            hi, hi_exact = calc_parse_scalar(hi_str)
        except ValueError as e:
            return calc_unknown(f'bad upper support bound: {e}')
        if not (hi > lo):
            return calc_unknown('support must have hi > lo')

    exact = lo_exact and hi_exact

    try:
        neg_set = solve_univariate_inequality(f < 0, x, relational=False, domain=Interval(lo, hi))
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot determine sign on the support: {e}')
    non_neg_pass = (neg_set is S.EmptySet)

    try:
        total = sp.integrate(f, (x, lo, hi))
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot integrate exactly: {e}')
    if total.has(sp.Integral) or not total.is_finite:
        return calc_unknown('integral does not evaluate to a finite closed form')
    norm_pass = bool(simplify(total - 1) == 0)

    law_valid = non_neg_pass and norm_pass
    blocked = None
    if not non_neg_pass:
        blocked = 'nonNegativity'
    elif not norm_pass:
        blocked = 'normalization'

    return {
        'provenance': calc_cap('proved' if law_valid else 'refuted', exact),
        'kind': 'continuous',
        'exactInputs': exact,
        'nonNegativity': {'pass': non_neg_pass},
        'normalization': {'pass': norm_pass, 'total': str(total)},
        'lawValid': law_valid,
        'blockedAct': blocked,
    }


# =============================================================================
# \`independence\` (3.8, "Independence of events"). An INDEPENDENT SymPy
# implementation of independenceEngine.js's own analyzeIndependence (nothing
# here calls into the JS engine) — same three-facet structure, verified
# separately in symbolicEngine.selftest.py against presets mirrored verbatim
# from independencePresets.mjs.
#
# See independenceEngine.js's own header for the full design note: THREE
# checkMode facets over one shared outcome-probability table —
#   'pairwise'    — P(A ∩ B) = P(A)P(B), cross-checked against the
#                   conditional characterization P(A|B) = P(A) / P(B|A) =
#                   P(B) wherever the conditioning event has positive
#                   probability.
#   'mutual'      — n >= 3 declared events A1..An are mutually independent
#                   iff the product rule holds for EVERY subset of size
#                   >= 2, checked exhaustively (2^n - n - 1 non-trivial
#                   subsets, safe for n <= 5, the scope this module
#                   supports) via \`itertools.combinations\` — the flagship
#                   lesson (every pair independent, the full triple not) is
#                   this module's signature counterexample.
#   'conditional' — A, B independent GIVEN C means P(A ∩ B|C) = P(A|C)P(B|C)
#                   for a third declared event C with P(C) > 0 (the one
#                   genuinely breakable precondition here); the plain
#                   UNCONDITIONAL product-rule check on the same A, B is
#                   also reported, so the module can show conditional and
#                   unconditional independence are genuinely different
#                   properties (neither implies the other).
#
# Reliability of series/parallel systems (independently-given component
# reliabilities, not events sharing a sample space) has no symbolic tier —
# it is a small pure numeric calculator on the JS side only (see
# independenceEngine.js's own header), exactly like every other
# module-adjacent Applications dataset in this app.
def _indep_p_of(values, idx):
    return sum((values[i] for i in idx), S.Zero)


def analyse_independence(check_mode, outcomes, entries, events):
    """check_mode: 'pairwise' | 'mutual' | 'conditional'. outcomes/entries:
    the shared probability table, parsed via the shared parse_matrix helper
    exactly like probabilitylaws'/conditional's/bayes' own tables. events:
    list[list[int]], ordered 0-indexed outcome index sets — length 2 for
    'pairwise' (A, B), length >= 3 (<= 5) for 'mutual' (A1..An), length 3 for
    'conditional' (A, B, C — C is the conditioning event)."""
    if not outcomes or len(outcomes) != len(entries):
        return calc_unknown('sample space and probability table must have the same length')
    events = [[int(i) for i in ev] for ev in events]
    if len(events) < 2:
        return calc_unknown('need at least two declared events')
    if check_mode == 'mutual' and len(events) < 3:
        return calc_unknown('mutual independence needs at least three declared events')
    if check_mode == 'mutual' and len(events) > 5:
        return calc_unknown('mutual independence here supports at most five declared events (subset enumeration grows as 2^n)')
    if check_mode == 'conditional' and len(events) < 3:
        return calc_unknown('conditional independence needs three declared events: A, B, and the conditioning event C')
    try:
        M, exact = parse_matrix(list(entries))
    except ValueError as e:
        return calc_unknown(f'bad probability entry: {e}')

    values = [M[i, 0] for i in range(M.rows)]

    violations = [i for i, v in enumerate(values) if v < 0]
    non_neg_pass = len(violations) == 0
    total = sum(values, S.Zero)
    norm_pass = bool(simplify(total - 1) == 0)
    table_valid = non_neg_pass and norm_pass

    pairwise = None
    mutual = None
    conditional = None
    blocked_act = None

    if not table_valid:
        blocked_act = 'nonNegativity' if not non_neg_pass else 'normalization'
    elif check_mode == 'pairwise':
        idx_a, idx_b = events[0], events[1]
        set_b = set(idx_b)
        p_a = _indep_p_of(values, idx_a)
        p_b = _indep_p_of(values, idx_b)
        inter_ab = [i for i in idx_a if i in set_b]
        p_ab = _indep_p_of(values, inter_ab)
        product_matches = bool(simplify(p_ab - p_a * p_b) == 0)
        b_ready = bool(p_b > 0)
        a_ready = bool(p_a > 0)
        p_a_given_b = (p_ab / p_b) if b_ready else None
        cond_matches_b = bool(simplify(p_a_given_b - p_a) == 0) if b_ready else None
        p_b_given_a = (p_ab / p_a) if a_ready else None
        cond_matches_a = bool(simplify(p_b_given_a - p_b) == 0) if a_ready else None
        pairwise = {
            'pA': str(p_a), 'pB': str(p_b), 'pAB': str(p_ab), 'productMatches': product_matches,
            'bReady': b_ready, 'pAgivenB': (str(p_a_given_b) if b_ready else None), 'condMatchesB': cond_matches_b,
            'aReady': a_ready, 'pBgivenA': (str(p_b_given_a) if a_ready else None), 'condMatchesA': cond_matches_a,
            'pass': product_matches,
        }
    elif check_mode == 'mutual':
        n = len(events)
        p_single = [_indep_p_of(values, ev) for ev in events]
        subsets = []
        for r in range(2, n + 1):
            for combo in combinations(range(n), r):
                inter = set(range(len(values)))
                for i in combo:
                    inter &= set(events[i])
                p_inter = _indep_p_of(values, sorted(inter))
                prod = S.One
                for i in combo:
                    prod *= p_single[i]
                passed = bool(simplify(p_inter - prod) == 0)
                subsets.append({'bits': list(combo), 'pInter': str(p_inter), 'prod': str(prod), 'pass': passed})
        pairwise_subsets = [s for s in subsets if len(s['bits']) == 2]
        pairwise_all_pass = all(s['pass'] for s in pairwise_subsets)
        mutual_pass = all(s['pass'] for s in subsets)
        pairwise_not_mutual = pairwise_all_pass and not mutual_pass
        mutual = {
            'n': n, 'pSingle': [str(x) for x in p_single], 'subsets': subsets,
            'pairwiseAllPass': pairwise_all_pass, 'mutualPass': mutual_pass, 'pairwiseNotMutual': pairwise_not_mutual,
        }
    elif check_mode == 'conditional':
        idx_a, idx_b, idx_c = events[0], events[1], events[2]
        set_b = set(idx_b)
        p_a = _indep_p_of(values, idx_a)
        p_b = _indep_p_of(values, idx_b)
        inter_ab = [i for i in idx_a if i in set_b]
        p_ab = _indep_p_of(values, inter_ab)
        uncond_indep = bool(simplify(p_ab - p_a * p_b) == 0)
        p_c = _indep_p_of(values, idx_c)
        c_ready = bool(p_c > 0)
        if not c_ready:
            conditional = {
                'pC': str(p_c), 'cReady': False, 'pA': str(p_a), 'pB': str(p_b), 'pAB': str(p_ab),
                'uncondIndep': uncond_indep, 'condIndep': None, 'agrees': None,
            }
            blocked_act = 'conditioningUndefined'
        else:
            set_c = set(idx_c)
            idx_ac = [i for i in idx_a if i in set_c]
            idx_bc = [i for i in idx_b if i in set_c]
            idx_abc = [i for i in inter_ab if i in set_c]
            p_ac = _indep_p_of(values, idx_ac)
            p_bc = _indep_p_of(values, idx_bc)
            p_abc = _indep_p_of(values, idx_abc)
            p_a_given_c = p_ac / p_c
            p_b_given_c = p_bc / p_c
            p_ab_given_c = p_abc / p_c
            cond_indep = bool(simplify(p_ab_given_c - p_a_given_c * p_b_given_c) == 0)
            agrees = (cond_indep == uncond_indep)
            conditional = {
                'pC': str(p_c), 'cReady': True, 'pA': str(p_a), 'pB': str(p_b), 'pAB': str(p_ab), 'uncondIndep': uncond_indep,
                'pAC': str(p_ac), 'pBC': str(p_bc), 'pABC': str(p_abc),
                'pAgivenC': str(p_a_given_c), 'pBgivenC': str(p_b_given_c), 'pABgivenC': str(p_ab_given_c),
                'condIndep': cond_indep, 'agrees': agrees, 'pass': cond_indep,
            }

    holds = None
    if check_mode == 'pairwise' and pairwise is not None:
        holds = pairwise['pass']
    elif check_mode == 'mutual' and mutual is not None:
        holds = mutual['mutualPass']
    elif check_mode == 'conditional' and conditional is not None:
        holds = conditional.get('pass')

    definite = table_valid and blocked_act is None

    return {
        'provenance': calc_cap('proved' if (definite and holds) else 'refuted', exact),
        'exactInputs': exact,
        'checkMode': check_mode,
        'tableValid': {'pass': table_valid, 'nonNegPass': non_neg_pass, 'normPass': norm_pass},
        'pairwise': pairwise,
        'mutual': mutual,
        'conditional': conditional,
        'blockedAct': blocked_act,
        'holds': holds,
    }


# =============================================================================
# \`expectation\` (3.11, "Expectation & variance"). An INDEPENDENT SymPy
# implementation of expectationEngine.js's own two-clause chain (nothing
# here calls into the JS engine) — see expectationEngine.js's own header for
# the full design note. Unlike the JS tier (always 'numeric' provenance,
# since its PMF/PDF is formula-driven and checked by marching/quadrature),
# SymPy's exact Sum/Integrate machinery CAN decide 'proved'/'refuted' on
# most presets here — the same asymmetry analyse_randomvariables's own
# docstring already documents ("the JS engine's own honest quadrature always
# reports 'numeric', the SymPy tier decides more").
#
#   DISCRETE.  x(n), p(n) parsed via _seq_parse (n a positive-integer-valued
#              symbol, reused VERBATIM from analyse_sequences — the same
#              "genuinely reuse an already-tested small helper" precedent
#              series/powerseries already set for this domain). A finite
#              support (n = startN..startN+count-1) sums directly, term by
#              term, over EXACT SymPy values — always decidable. An infinite
#              support hands sp.Sum(term, (n, startN, oo)) to \`.doit()\`:
#                - a genuine SymPy VALUE (finite)  -> convergent, exact.
#                - oo/-oo/zoo/nan                  -> divergent, exact
#                  (a certain, definitive refutation — never an abstention;
#                  the St. Petersburg preset's own sum of the CONSTANT term
#                  1 is exactly this shape: Sum(1, (n,1,oo)).doit() = oo).
#                - AccumBounds                      -> divergent, exact (the
#                  same "AccumBounds is certain, never an abstention"
#                  precedent analyse_improper/analyse_sequences established).
#                - still an unevaluated Sum          -> abstain ('unknown'):
#                  SymPy could not decide this sum's convergence in closed
#                  form (never guessed).
#   CONTINUOUS. f(x) parsed via calc_parse (the shared \`x\` symbol every
#              other Calculus/probability-continuous analyse_* uses).
#              sp.integrate(term, (x, lo, hi)) is called DIRECTLY on the
#              (possibly doubly-infinite) interval — mirroring
#              analyse_improper's own established choice: SymPy's own
#              integration machinery performs the equivalent split-and-limit
#              internally, so no manual piece-marching is needed even for
#              the two-sided-improper Cauchy flagship (lo=-oo, hi=oo).
#              Divergence classification (finite / oo-zoo-nan / AccumBounds /
#              unevaluated-abstain) is the SAME four-way split as the
#              discrete branch, and reuses \`_expect_classify\` for both.
#
# H1 (meanExists) is decided on sum/integral of Abs(x(n))*p(n) (or
# Abs(x)*f(x)); H2 (varianceExists), GATED by H1, on x(n)**2*p(n) (or
# x**2*f(x)) — never signed, so the SAME classification helper applies to
# both without an Abs()-vs-not distinction to track. Once both hold, E[X]
# (signed) and Var(X) = E[X^2]-(E[X])^2 are computed directly and exactly
# where SymPy can; each is independently reported, never asserted from the
# JS tier's own numeric estimate.
#
# Non-negotiable #5 is N/A, the same documented call netchange/ftc/
# gammabeta/chainrule/extrema/expectationEngine.js's own header already
# make: every check here is a direct computation against a GIVEN law, not
# an existence claim with multiple witnesses to search among.
# =============================================================================

def _expect_classify(val):
    """Classify a computed Sum/Integral result. Returns one of:
      ('finite', value)  — a genuine, decided finite SymPy value.
      ('divergent', None) — a certain, definitive divergence (oo/-oo/zoo/nan,
        or a genuinely oscillating AccumBounds — never an abstention).
      ('abstain', reason) — SymPy could not decide this in closed form.
    Shared between the discrete Sum branch and the continuous Integral
    branch — both an unevaluated sp.Sum and an unevaluated sp.Integral are
    caught by the same \`.has(sp.Sum) or .has(sp.Integral)\` guard."""
    if val is None:
        return ('abstain', 'SymPy returned no result')
    if val.has(sp.Sum) or val.has(sp.Integral):
        return ('abstain', 'SymPy could not evaluate this in closed form')
    if isinstance(val, sp.calculus.accumulationbounds.AccumulationBounds):
        return ('divergent', None)
    if val in (oo, -oo, sp.zoo, sp.nan):
        return ('divergent', None)
    if val.is_real is False:
        return ('abstain', 'did not evaluate to a real number')
    return ('finite', val)


def _expect_abs_x_f_integral(f, lo, hi):
    """integral of |x| f(x) dx over [lo, hi] (lo/hi possibly +-oo), computed
    by splitting exactly at x=0 by KNOWN SIGN rather than leaving a literal
    Abs(x) inside sp.integrate. Confirmed by hand while building this
    module, and documented as this module's own addition to the domain's
    running "Abs()-wrapped integrands are frequently where sp.integrate
    gives up" trap family (analyse_improper's own docstring already names
    the general shape): the Cauchy flagship's two-sided
    \`sp.integrate(Abs(x)/(pi*(1+x**2)), (x,-oo,oo))\` returns an UNEVALUATED
    Integral (an honest abstention) even though EACH signed half is
    individually decidable in exact closed form —
    \`sp.integrate(x/(pi*(1+x**2)), (x,0,oo))\` evaluates cleanly to \`oo\`.
    Splitting by sign first turns an abstention into a certain, definitive
    divergence finding."""
    if bool(lo >= 0):
        return sp.integrate(x * f, (x, lo, hi))
    if bool(hi <= 0):
        return sp.integrate(-x * f, (x, lo, hi))
    try:
        left = sp.integrate(-x * f, (x, lo, 0))
        right = sp.integrate(x * f, (x, 0, hi))
    except Exception:  # noqa: BLE001
        return sp.integrate(Abs(x) * f, (x, lo, hi))
    kl, _ = _expect_classify(left)
    kr, _ = _expect_classify(right)
    if kl == 'divergent' or kr == 'divergent':
        return oo
    return left + right


def analyse_expectation(rv_mode, x_formula, p_formula, startN_str, disc_support_kind, count_str,
                         formula, cont_support_kind, lo_str, hi_str):
    """See the module docstring above. \`x_formula\`/\`p_formula\` are SymPy-
    syntax strings from expr.mjs's toSympy(), parsed with n (not x) as the
    free variable — src/modules/expectationEngine.js's own parseExprN
    wrapper calls parseExpr(text, ['n']), mirroring sequencesEngine.js's own
    convention exactly. \`formula\` is the continuous density, parsed with
    the shared \`x\` symbol via calc_parse."""
    if rv_mode == 'discrete':
        try:
            xf = _seq_parse(x_formula)
            pf = _seq_parse(p_formula)
        except ValueError as e:
            return calc_unknown(f'bad formula: {e}')
        try:
            startN_val, startN_exact = calc_parse_scalar(startN_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad startN: {e}')
        if not (startN_val.is_integer and bool(startN_val >= 0)):
            return calc_unknown('startN must be a non-negative integer')
        startN = int(startN_val)

        n = _SEQ_N
        if disc_support_kind == 'finite':
            try:
                count_val, count_exact = calc_parse_scalar(count_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad count: {e}')
            if not (count_val.is_integer and bool(count_val >= 1)):
                return calc_unknown('count must be a positive integer')
            count = int(count_val)
            exact = startN_exact and count_exact
            ns = list(range(startN, startN + count))
            try:
                xs = [xf.subs(n, k) for k in ns]
                ps = [pf.subs(n, k) for k in ns]
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'cannot evaluate formula: {e}')
            mean_val = sp.nsimplify(sum((xv * pv for xv, pv in zip(xs, ps)), S.Zero))
            mean_kind, e2_val = 'finite', sp.nsimplify(sum((xv**2 * pv for xv, pv in zip(xs, ps)), S.Zero))
            var_kind = 'finite'
            mean_exists, variance_exists = True, True
        else:
            exact = startN_exact
            try:
                mean_abs = sp.Sum(Abs(xf) * pf, (n, startN, oo)).doit()
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'cannot evaluate sum: {e}')
            mean_kind, mean_abs_val = _expect_classify(mean_abs)
            if mean_kind == 'abstain':
                return calc_unknown(f'meanExists: {mean_abs_val}')
            mean_exists = (mean_kind == 'finite')
            variance_exists = False
            e2_val = None
            if mean_exists:
                try:
                    mean_val = sp.Sum(xf * pf, (n, startN, oo)).doit()
                except Exception as e:  # noqa: BLE001
                    return calc_unknown(f'cannot evaluate E[X]: {e}')
                try:
                    e2_raw = sp.Sum(xf**2 * pf, (n, startN, oo)).doit()
                except Exception as e:  # noqa: BLE001
                    return calc_unknown(f'cannot evaluate E[X^2]: {e}')
                var_kind, e2_val = _expect_classify(e2_raw)
                if var_kind == 'abstain':
                    return calc_unknown(f'varianceExists: {e2_val}')
                variance_exists = (var_kind == 'finite')
            else:
                mean_val = None

        blocked = None if mean_exists else 'meanExists'
        if mean_exists and not variance_exists:
            blocked = 'varianceExists'
        variance_val = (e2_val - mean_val**2) if (mean_exists and variance_exists) else None
        all_pass = blocked is None
        return {
            'provenance': calc_cap('proved' if all_pass else 'refuted', exact),
            'kind': 'discrete',
            'exactInputs': exact,
            'meanExists': mean_exists,
            'varianceExists': variance_exists if mean_exists else None,
            'blockedAct': blocked,
            'allPass': all_pass,
            'mean': (_calc_num(mean_val) if mean_exists else None),
            'meanTex': (sp.latex(mean_val) if mean_exists else None),
            'variance': (_calc_num(variance_val) if variance_val is not None else None),
            'varianceTex': (sp.latex(variance_val) if variance_val is not None else None),
        }

    if rv_mode != 'continuous':
        return calc_unknown(f'unknown rv_mode {rv_mode!r}')

    try:
        f = calc_parse(formula)
    except ValueError as e:
        return calc_unknown(f'bad formula: {e}')

    if cont_support_kind == 'full':
        lo, hi, exact = -oo, oo, True
    else:
        try:
            lo, lo_exact = calc_parse_scalar(lo_str)
        except ValueError as e:
            return calc_unknown(f'bad lower support bound: {e}')
        if cont_support_kind == 'semiInfRight':
            hi, hi_exact = oo, True
        else:
            try:
                hi, hi_exact = calc_parse_scalar(hi_str)
            except ValueError as e:
                return calc_unknown(f'bad upper support bound: {e}')
            if not (hi > lo):
                return calc_unknown('support must have hi > lo')
        exact = lo_exact and hi_exact

    try:
        mean_abs_raw = _expect_abs_x_f_integral(f, lo, hi)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot integrate |x|f(x): {e}')
    mean_kind, mean_abs_val = _expect_classify(mean_abs_raw)
    if mean_kind == 'abstain':
        return calc_unknown(f'meanExists: {mean_abs_val}')
    mean_exists = (mean_kind == 'finite')

    mean_val, variance_exists, e2_val = None, False, None
    if mean_exists:
        try:
            mean_val = sp.integrate(x * f, (x, lo, hi))
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'cannot integrate x f(x): {e}')
        try:
            e2_raw = sp.integrate(x**2 * f, (x, lo, hi))
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'cannot integrate x^2 f(x): {e}')
        var_kind, e2_val = _expect_classify(e2_raw)
        if var_kind == 'abstain':
            return calc_unknown(f'varianceExists: {e2_val}')
        variance_exists = (var_kind == 'finite')

    blocked = None if mean_exists else 'meanExists'
    if mean_exists and not variance_exists:
        blocked = 'varianceExists'
    variance_val = (e2_val - mean_val**2) if (mean_exists and variance_exists) else None
    all_pass = blocked is None
    return {
        'provenance': calc_cap('proved' if all_pass else 'refuted', exact),
        'kind': 'continuous',
        'exactInputs': exact,
        'meanExists': mean_exists,
        'varianceExists': variance_exists if mean_exists else None,
        'blockedAct': blocked,
        'allPass': all_pass,
        'mean': (_calc_num(mean_val) if mean_exists else None),
        'meanTex': (sp.latex(mean_val) if mean_exists else None),
        'variance': (_calc_num(variance_val) if variance_val is not None else None),
        'varianceTex': (sp.latex(variance_val) if variance_val is not None else None),
    }


# =============================================================================
# \`transformrv\` (3.10, "Transformations of random variables"). An INDEPENDENT
# SymPy implementation of transformrvEngine.js's own two-facet split (nothing
# here calls into the JS engine) — see that file's header for the full design
# note: discrete collecting-preimages (no breakable hypothesis) vs.
# continuous strictly-monotonic-transformation (H1 = monotonic, gating the
# change-of-variables shortcut).
#
# DISCRETE. xs/ps mirror analyse_randomvariables's own (x, p) rows exactly
# (xs plain floats — the x-axis LABELS carry no exactness of their own, only
# the probabilities do, via the shared parse_matrix helper); g_formula is a
# SymPy-syntax string (expr.mjs's toSympy()) evaluated at each x-value.
# Grouping is by FLOAT g(x) value (rounded to 9 decimals, the same tolerance
# transformrvEngine.js's own keyOf uses) — the KEY (which x's collide) is a
# float classification decision exactly as the JS tier documents, but the
# VALUE summed per group is exact SymPy Rational addition, so this facet CAN
# reach 'proved'/'refuted' when the input table is exact.
#
# CONTINUOUS. f_formula/g_formula are BOTH SymPy-syntax strings. Unlike the
# JS numeric tier (which decides monotonicity by SAMPLING — a probe, never a
# proof, for an arbitrary formula), SymPy decides H1 EXACTLY: g' has a
# constant sign on the OPEN interval iff \`solve_univariate_inequality\` finds
# no point where the derivative's sign flips — checked as "g'<=0 nowhere" OR
# "g'>=0 nowhere" on Interval.open(lo, hi) (a decisive, closed-form
# certificate every preset in this module's grammar produces). When
# monotonic, \`sp.solve(Eq(g, y), x)\` is attempted for a closed-form inverse
# h(y); every preset needing an inverse (sqrt(x), 180/x, 2x+1, x) resolves to
# EXACTLY ONE symbolic solution, so f_Y(y) = f_X(h(y))*|dh/dy| is then
# computed and simplified exactly. A genuine scope decision, documented here
# rather than attempted and silently wrong: when H1 is FALSE (g not
# monotonic), this symbolic tier does NOT attempt a general multi-branch
# piecewise inverse — that is exactly the JS numeric tier's own
# region-integral machinery's job (transformrvEngine.js's regionIntegral),
# and forcing a symbolic piecewise solve here would risk exactly the kind of
# "confidently wrong closed form" this app's provenance discipline exists to
# prevent. The module reports H1's own verdict exactly (a certain,
# decisive fact either way) and abstains ONLY on the derived f_Y formula in
# that case — never promoted to 'proved'.
# =============================================================================

def _trv_group_discrete(xvals, pvals, g_formula):
    """Returns (groups, injective) or raises ValueError. groups is a list of
    {'y': float, 'xs': [float,...], 'p': sympy expr} sorted by y."""
    try:
        g = calc_parse(g_formula)
    except ValueError as e:
        raise ValueError(f'bad g(x): {e}') from e
    keyed = {}
    for xv, p in zip(xvals, pvals):
        try:
            yv = float(g.subs(x, xv))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f'g(x) could not be evaluated at x={xv}: {e}') from e
        key = round(yv, 9)
        if key not in keyed:
            keyed[key] = {'y': yv, 'xs': [], 'p': S.Zero}
        keyed[key]['xs'].append(xv)
        keyed[key]['p'] = keyed[key]['p'] + p
    groups = sorted(keyed.values(), key=lambda gg: gg['y'])
    injective = all(len(gg['xs']) == 1 for gg in groups)
    return groups, injective


def analyse_transformrv(rv_mode, xs, ps, g_formula, f_formula, lo_str, hi_str):
    """rv_mode: 'discrete'|'continuous'.
    Discrete:   xs (list[float]), ps (list[str] probability entries), g_formula.
    Continuous: f_formula, lo_str, hi_str (support [lo,hi], finite only —
                mirroring transformrvEngine.js's own scoped-down support
                contract, see that file's header), g_formula.
    """
    if rv_mode == 'discrete':
        if not xs or not ps or len(xs) != len(ps):
            return calc_unknown('x and p must be non-empty and the same length')
        try:
            xvals = [float(v) for v in xs]
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad x value: {e}')
        if len(set(xvals)) != len(xvals):
            return calc_unknown('duplicate x value')
        try:
            M, exact = parse_matrix(list(ps))
        except ValueError as e:
            return calc_unknown(f'bad probability entry: {e}')
        pvals = [M[i, 0] for i in range(M.rows)]

        order = sorted(range(len(xvals)), key=lambda i: xvals[i])
        sorted_x = [xvals[i] for i in order]
        sorted_p = [pvals[i] for i in order]

        violations = [i for i, v in enumerate(sorted_p) if v < 0]
        non_neg_pass = len(violations) == 0
        total = sum(sorted_p, S.Zero)
        norm_pass = bool(simplify(total - 1) == 0)
        p_x_valid = non_neg_pass and norm_pass

        groups_out, injective, p_y_valid = None, None, None
        if p_x_valid:
            try:
                groups, injective = _trv_group_discrete(sorted_x, sorted_p, g_formula)
            except ValueError as e:
                return calc_unknown(str(e))
            p_y_values = [gg['p'] for gg in groups]
            p_y_nonneg = all(bool(v >= 0) for v in p_y_values)
            p_y_total = sum(p_y_values, S.Zero)
            p_y_norm = bool(simplify(p_y_total - 1) == 0)
            p_y_valid = p_y_nonneg and p_y_norm
            groups_out = [{'y': gg['y'], 'xs': gg['xs'], 'p': str(gg['p'])} for gg in groups]

        law_ok = bool(p_x_valid and p_y_valid)
        return {
            'provenance': calc_cap('proved' if law_ok else 'refuted', exact),
            'kind': 'discrete',
            'exactInputs': exact,
            'pXValid': p_x_valid,
            'groups': groups_out,
            'injective': injective,
            'pYValid': p_y_valid,
        }

    if rv_mode != 'continuous':
        return calc_unknown(f'unknown rv_mode {rv_mode!r}')

    try:
        f = calc_parse(f_formula)
    except ValueError as e:
        return calc_unknown(f'bad f(x): {e}')
    try:
        g = calc_parse(g_formula)
    except ValueError as e:
        return calc_unknown(f'bad g(x): {e}')
    try:
        lo, lo_exact = calc_parse_scalar(lo_str)
        hi, hi_exact = calc_parse_scalar(hi_str)
    except ValueError as e:
        return calc_unknown(f'bad support bound: {e}')
    if not bool(hi > lo):
        return calc_unknown('support must have hi > lo')
    exact = lo_exact and hi_exact

    try:
        neg_set = solve_univariate_inequality(f < 0, x, relational=False, domain=Interval(lo, hi))
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot determine sign of f on the support: {e}')
    non_neg_pass = (neg_set is S.EmptySet)

    try:
        total = sp.integrate(f, (x, lo, hi))
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot integrate f exactly: {e}')
    if total.has(sp.Integral) or not total.is_finite:
        return calc_unknown('integral of f does not evaluate to a finite closed form')
    norm_pass = bool(simplify(total - 1) == 0)
    f_x_valid = non_neg_pass and norm_pass

    if not f_x_valid:
        blocked = 'nonNegativity' if not non_neg_pass else 'normalization'
        return {
            'provenance': calc_cap('refuted', exact),
            'kind': 'continuous',
            'exactInputs': exact,
            'nonNegativity': {'pass': non_neg_pass},
            'normalization': {'pass': norm_pass, 'total': str(total)},
            'fXValid': False,
            'blockedAct': blocked,
            'monotonic': None,
            'hExpr': None,
            'fYExpr': None,
        }

    # H1: is g' of constant sign on the OPEN interval? A decisive certificate
    # either way (see the module docstring for why this is exact, unlike the
    # JS tier's own sampling probe).
    try:
        gprime = sp.diff(g, x)
        open_iv = Interval.open(lo, hi)
        non_positive_nowhere = solve_univariate_inequality(gprime <= 0, x, relational=False, domain=open_iv) is S.EmptySet
        non_negative_nowhere = solve_univariate_inequality(gprime >= 0, x, relational=False, domain=open_iv) is S.EmptySet
        monotonic = bool(non_positive_nowhere or non_negative_nowhere)
    except Exception as e:  # noqa: BLE001
        return calc_unknown(f'cannot determine monotonicity of g exactly: {e}')

    h_expr, f_y_expr = None, None
    if monotonic:
        y_sym = sp.Symbol('y', real=True)
        try:
            sols = sp.solve(sp.Eq(g, y_sym), x)
        except Exception:  # noqa: BLE001
            sols = []
        if len(sols) == 1:
            h_expr = sols[0]
            try:
                f_y_expr = simplify(f.subs(x, h_expr) * Abs(sp.diff(h_expr, y_sym)))
            except Exception:  # noqa: BLE001
                f_y_expr = None
        # len(sols) != 1: SymPy could not produce a single unambiguous
        # closed-form inverse — abstain on f_Y specifically (h_expr/f_y_expr
        # stay None) rather than guessing which branch/solution to use.

    provenance = calc_cap('proved' if (monotonic and f_y_expr is not None) else 'refuted', exact) if (monotonic is False or f_y_expr is not None) else 'unknown'
    return {
        'provenance': provenance,
        'kind': 'continuous',
        'exactInputs': exact,
        'nonNegativity': {'pass': non_neg_pass},
        'normalization': {'pass': norm_pass, 'total': str(total)},
        'fXValid': True,
        'blockedAct': None if monotonic else 'monotonic',
        'monotonic': monotonic,
        'hExpr': (str(h_expr) if h_expr is not None else None),
        'fYExpr': (str(f_y_expr) if f_y_expr is not None else None),
        'fYTex': (sp.latex(f_y_expr) if f_y_expr is not None else None),
    }


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) — named distributions:
#    Bernoulli & Binomial distributions (topic 3.12) ──
#
# An INDEPENDENT SymPy implementation of binomialEngine.js's analyzeBinomial
# (nothing here calls into the JS engine). p and every entry of p_arr arrive
# as RAW probability strings exactly as the module's own input fields hold
# them (never pre-converted to a sympy-syntax string via expr.mjs's
# parseNumber) — parsed via parse_entry/parse_matrix, the same exact-vs-
# decimal distinguishing helper every other module in this domain
# (probabilitylaws/conditional/bayes/independence/randomvariables/
# transformrv) already uses for its own probability entries: a decimal
# string like "0.05" is read as an EXACT rational via nsimplify(Float(s))
# (the string itself is already fixed-precision, not a suspect float) but
# still flips the whole result's provenance to 'numeric', mirroring the JS
# engine's own "one decimal taints the table" rule.
#
# Mirrors binomialEngine.js's own structure exactly: build the joint table
# over 2^n_check binary trial-outcome sequences under the declared scenario
# ('iid'/'correlated'/'unequalP'), check identicallyDistributed/jointFactors,
# and compare the enumerated distribution of X=X1+...+Xn_check against the
# Bin(n_check, pUsed) formula DIRECTLY — the one genuinely breakable
# hypothesis, iidBernoulli. Once it holds, normalization (the binomial
# theorem) and the two-route mean/variance theorem (binmv) are verified via
# exact finite sums (n is always concrete here, so "the sum" is a literal
# finite Rational sum, not a symbolic Sum over an unbound n) — never merely
# asserted as n*p / n*p*(1-p).
# =============================================================================


def _bin_joint(n_check, scenario, p_val, p_arr_vals):
    """Build the joint table over 2^n_check binary trial-outcome sequences —
    an independent re-derivation of binomialEngine.js's own buildJoint."""
    seqs = []
    for bits in product((0, 1), repeat=n_check):
        if scenario == 'iid':
            prob = S.One
            for b in bits:
                prob = prob * (p_val if b else (S.One - p_val))
        elif scenario == 'correlated':
            if n_check < 2 or bits[0] == bits[1]:
                prob = p_val if bits[0] else (S.One - p_val)
                for b in bits[2:]:
                    prob = prob * (p_val if b else (S.One - p_val))
            else:
                prob = S.Zero
        else:  # 'unequalP'
            prob = S.One
            for i, b in enumerate(bits):
                pi = p_arr_vals[i]
                prob = prob * (pi if b else (S.One - pi))
        seqs.append((bits, prob))
    marginals = [sum((prob for bits, prob in seqs if bits[i]), S.Zero) for i in range(n_check)]
    return seqs, marginals


def analyse_binomial(n, p_str, n_check, scenario, p_arr=None):
    """n, n_check: ints (or numeric strings). p_str: a RAW probability entry
    string ('0.05', '1/2', '1'). scenario: 'iid'|'correlated'|'unequalP'.
    p_arr: list[str] of n_check raw probability entries, required only when
    scenario='unequalP'."""
    try:
        n = int(n)
        n_check = int(n_check)
    except (TypeError, ValueError):
        return calc_unknown('n and nCheck must be integers')
    if n < 1:
        return calc_unknown('n must be a positive integer')
    if not (1 <= n_check <= 6):
        return calc_unknown('the i.i.d. check needs 1 <= nCheck <= 6 (brute force over 2^n sequences)')

    try:
        p_val, p_exact = parse_entry(p_str)
    except ValueError as e:
        return calc_unknown(f'bad p: {e}')
    if not (0 <= p_val <= 1):
        return calc_unknown('p must satisfy 0 <= p <= 1')

    p_arr_vals, p_arr_exact = None, True
    if scenario == 'unequalP':
        if not p_arr or len(p_arr) != n_check:
            return calc_unknown(f'need exactly {n_check} trial probabilities for the unequal-p scenario')
        p_arr_vals = []
        for s in p_arr:
            try:
                v, ex = parse_entry(s)
            except ValueError as e:
                return calc_unknown(f'bad trial probability: {e}')
            if not (0 <= v <= 1):
                return calc_unknown('every trial probability must satisfy 0 <= p_i <= 1')
            p_arr_vals.append(v)
            p_arr_exact = p_arr_exact and ex

    eff_exact = p_arr_exact if scenario == 'unequalP' else p_exact
    seqs, marginals = _bin_joint(n_check, scenario, p_val, p_arr_vals)

    def eqp(a, b):
        return bool(simplify(a - b) == 0) if eff_exact else abs(float(a) - float(b)) <= 1e-9

    identically_distributed = all(eqp(m, marginals[0]) for m in marginals)

    def joint_factors_check():
        for bits, prob in seqs:
            expected = S.One
            for i in range(n_check):
                mi = marginals[i]
                expected = expected * (mi if bits[i] else (S.One - mi))
            if not eqp(expected, prob):
                return False
        return True
    joint_factors = joint_factors_check()

    p_used = marginals[0] if identically_distributed else sum(marginals, S.Zero) / n_check

    true_dist = [S.Zero] * (n_check + 1)
    for bits, prob in seqs:
        true_dist[sum(bits)] += prob
    formula_dist = [sp.binomial(n_check, k) * p_used**k * (S.One - p_used)**(n_check - k) for k in range(n_check + 1)]

    matches = True
    for k in range(n_check + 1):
        if not eqp(true_dist[k], formula_dist[k]):
            matches = False

    result = {
        'id': 'binomial', 'n': n, 'nCheck': n_check, 'scenario': scenario,
        'iidHolds': matches,
        'identicallyDistributed': identically_distributed,
        'jointFactors': joint_factors,
        'pUsed': _calc_num(p_used),
        'blockedAct': None if matches else 'iidBernoulli',
    }
    if not matches:
        result['provenance'] = calc_cap('refuted', eff_exact)
        return result

    # Normalization (the binomial theorem) — a real, non-gating check via a
    # literal finite sum (n is always concrete here).
    pmf_terms = [sp.binomial(n, k) * p_val**k * (S.One - p_val)**(n - k) for k in range(n + 1)]
    total = sum(pmf_terms, S.Zero)
    norm_pass = bool(simplify(total - 1) == 0) if p_exact else bool(abs(float(total) - 1) <= 1e-6)

    # Theorem binmv, Route A: the direct factorial-moment sums (never merely
    # asserted as np / np(1-p)).
    mean_sum = sum((k * pmf_terms[k] for k in range(n + 1)), S.Zero)
    e2fact_sum = sum((k * (k - 1) * pmf_terms[k] for k in range(n + 1)), S.Zero)
    var_sum = e2fact_sum + mean_sum - mean_sum * mean_sum
    mean_closed = n * p_val
    var_closed = n * p_val * (S.One - p_val)
    if p_exact:
        mean_sum = simplify(mean_sum)
        var_sum = simplify(var_sum)
        mean_agree = bool(simplify(mean_sum - mean_closed) == 0)
        var_agree = bool(simplify(var_sum - var_closed) == 0)
    else:
        mean_agree = bool(abs(float(mean_sum) - float(mean_closed)) <= 1e-6)
        var_agree = bool(abs(float(var_sum) - float(var_closed)) <= 1e-6)

    return {
        **result,
        'provenance': calc_cap('proved', p_exact),
        'allPass': True,
        'normalization': {'pass': norm_pass, 'total': _calc_num(total)},
        'mean': _calc_num(mean_sum), 'meanTex': sp.latex(mean_sum),
        'variance': _calc_num(var_sum), 'varianceTex': sp.latex(var_sum),
        'meanAgree': mean_agree, 'varAgree': var_agree,
        'reducesTo': 'bernoulli' if n == 1 else None,
    }


# ---------------------------------------------------------------------------
# geometric (3.13, "Geometric distribution & memorylessness") — the direct
# sequel to analyse_binomial above. See geometricEngine.js's own header for
# the full mathematical design; this mirrors it independently in SymPy.
# ---------------------------------------------------------------------------
def _geom_joint_iid(n, p_val):
    """Independent re-derivation of geometricEngine.js's own 'iid' joint
    table — identical in structure to _bin_joint(n, 'iid', p_val, None)."""
    seqs = []
    for bits in product((0, 1), repeat=n):
        prob = S.One
        for b in bits:
            prob = prob * (p_val if b else (S.One - p_val))
        seqs.append((bits, prob))
    marginals = [sum((prob for bits, prob in seqs if bits[i]), S.Zero) for i in range(n)]
    return seqs, marginals


def _geom_joint_improving(n, p_start, inc):
    """'improvingP' scenario — mirrors geometricEngine.js's own
    buildImprovingJoint: a trial's own probability, in a GIVEN sequence,
    depends on how many of the PRECEDING trials in THAT sequence were
    failures (regardless of any interspersed success — a documented
    modeling simplification shared with the JS engine, needed so this joint
    table can confirm identicallyDistributed/jointFactors by brute force
    rather than merely computing the schedule's own tail probabilities).
    Kept in exact Rational arithmetic throughout — p_start/inc arrive as
    exact Rationals from parse_entry even when the raw string was a
    decimal, so 'improvingP' with two decimal inputs still computes exactly
    and is only CAPPED to 'numeric'/'refuted' provenance by calc_cap."""
    seqs = []
    for bits in product((0, 1), repeat=n):
        prob = S.One
        fail_count = 0
        for b in bits:
            pi = min(S.One, p_start + inc * fail_count)
            prob = prob * (pi if b else (S.One - pi))
            if not b:
                fail_count += 1
        seqs.append((bits, prob))
    marginals = [sum((prob for bits, prob in seqs if bits[i]), S.Zero) for i in range(n)]
    return seqs, marginals


def _geom_waiting_dist(n, seqs):
    """X = index (1-based) of the first 1-bit, or the tail P(X>n)."""
    dist = [S.Zero] * (n + 1)
    tail = S.Zero
    for bits, prob in seqs:
        idx = None
        for i, b in enumerate(bits):
            if b:
                idx = i
                break
        if idx is None:
            tail += prob
        else:
            dist[idx + 1] += prob
    return dist, tail


def _geom_tail_exact(n_val, p_val, q_val):
    """P(X>n_val) = sum_{k>n_val} q^(k-1)p, via SymPy's OWN infinite Sum —
    a genuinely independent route from simply writing q_val**n_val.
    q_val=0 (p=1) is special-cased: SymPy's infinite-Sum machinery,
    confirmed by hand, silently drops the k=n_val+1 boundary term
    (0**0=1) when the summation index is symbolic, returning 0 instead of
    the correct P(X>0)=1 — a genuinely NEW SymPy trap for this module,
    alongside the existing floor/sign/nsimplify family, worked around by
    deciding the p=1 degenerate case analytically instead (X is
    identically 1, so P(X>n)=1 for n=0 and 0 for n>=1 — exact facts, not a
    limit)."""
    if q_val == 0:
        return S.One if n_val == 0 else S.Zero
    ksym = Symbol('k', positive=True, integer=True)
    return Sum(q_val ** (ksym - 1) * p_val, (ksym, n_val + 1, oo)).doit()


def analyse_geometric(p_str, n_check, scenario, inc_str=None, s=None, t=None):
    """p_str: RAW probability entry string ('0.3', '1/2', '1'). n_check: int
    (or numeric string), 1<=n_check<=6. scenario: 'iid'|'improvingP'.
    inc_str: RAW increment entry string, required only for 'improvingP'.
    s, t: ints (or numeric strings) for the memorylessness check, s>=0,
    t>=1 — checked whenever supplied, REGARDLESS of whether iidBernoulli
    holds (this module's own flagship lesson: memorylessness genuinely
    depends on the i.i.d. hypothesis, so the broken scenario's own
    divergence is a second, honest, always-computed fact, the same shape
    analyse_bayes' own totalProb-computed-regardless-of-validity
    established)."""
    try:
        n_check = int(n_check)
    except (TypeError, ValueError):
        return calc_unknown('nCheck must be an integer')
    if not (1 <= n_check <= 6):
        return calc_unknown('the i.i.d. check needs 1 <= nCheck <= 6 (brute force over 2^n sequences)')

    try:
        p_val, p_exact = parse_entry(p_str)
    except ValueError as e:
        return calc_unknown(f'bad p: {e}')
    if not (0 < p_val <= 1):
        return calc_unknown('p must satisfy 0 < p <= 1')

    inc_val, inc_exact = None, True
    if scenario == 'improvingP':
        try:
            inc_val, inc_exact = parse_entry(inc_str)
        except (ValueError, TypeError) as e:
            return calc_unknown(f'bad increment: {e}')
        if inc_val < 0:
            return calc_unknown('the improving-p increment must be >= 0')
        seqs, marginals = _geom_joint_improving(n_check, p_val, inc_val)
    else:
        seqs, marginals = _geom_joint_iid(n_check, p_val)

    eff_exact = p_exact and inc_exact

    def eqp(a, b):
        return bool(simplify(a - b) == 0)

    identically_distributed = all(eqp(m, marginals[0]) for m in marginals)

    def joint_factors_check():
        for bits, prob in seqs:
            expected = S.One
            for i in range(n_check):
                mi = marginals[i]
                expected = expected * (mi if bits[i] else (S.One - mi))
            if not eqp(expected, prob):
                return False
        return True
    joint_factors = joint_factors_check()

    dist, tail = _geom_waiting_dist(n_check, seqs)
    p_used = marginals[0] if scenario == 'iid' else p_val
    formula_dist = [(S.One - p_used) ** (k - 1) * p_used for k in range(1, n_check + 1)]
    formula_tail = (S.One - p_used) ** n_check

    matches = all(eqp(dist[k], formula_dist[k - 1]) for k in range(1, n_check + 1)) and eqp(tail, formula_tail)

    result = {
        'id': 'geometric', 'nCheck': n_check, 'scenario': scenario,
        'iidHolds': matches,
        'identicallyDistributed': identically_distributed,
        'jointFactors': joint_factors,
        'pUsed': _calc_num(p_used),
        'blockedAct': None if matches else 'iidBernoulli',
    }

    try:
        s_int, t_int = int(s), int(t)
    except (TypeError, ValueError):
        s_int, t_int = None, None
    if s_int is not None and t_int is not None and s_int >= 0 and t_int >= 1:
        if scenario == 'iid':
            q_val = S.One - p_val
            closed_ratio = q_val ** t_int
            tail_s = _geom_tail_exact(s_int, p_val, q_val)
            tail_st = _geom_tail_exact(s_int + t_int, p_val, q_val)
            tail_t = _geom_tail_exact(t_int, p_val, q_val)
            numeric_ratio = tail_st / tail_s if tail_s != 0 else None
            memory_holds = numeric_ratio is not None and eqp(numeric_ratio, closed_ratio) and eqp(numeric_ratio, tail_t)
            result['memoryless'] = {
                'closedRatio': _calc_num(closed_ratio),
                'numericRatio': _calc_num(numeric_ratio) if numeric_ratio is not None else None,
                'holds': bool(memory_holds),
            }
        else:
            def tail_from_schedule(k):
                prod = S.One
                for i in range(k):
                    pi = min(S.One, p_val + inc_val * i)
                    prod = prod * (S.One - pi)
                return prod
            tail_s = tail_from_schedule(s_int)
            tail_st = tail_from_schedule(s_int + t_int)
            naive_t = tail_from_schedule(t_int)
            ratio = tail_st / tail_s if tail_s != 0 else None
            diverges = ratio is None or not eqp(ratio, naive_t)
            result['memoryless'] = {
                'conditionalRatio': _calc_num(ratio) if ratio is not None else None,
                'naiveTailT': _calc_num(naive_t),
                'holds': bool(not diverges),
            }

    if not matches:
        result['provenance'] = calc_cap('refuted', eff_exact)
        return result

    # Normalization (geometric series) — a real, non-gating check that ALWAYS
    # holds for 0 < p <= 1 (an algebraic identity, not something that can
    # independently fail) — the now-established "no independently-breakable
    # third hypothesis" exception shape (riemann/ftc/improper/gammabeta/
    # sequences/extrema/probabilitylaws/bayes' own third condition).
    norm_pass = True

    # Theorem geomv, Route A: literally differentiate the geometric series
    # sum_{k=0}^inf q^k = (1-q)^-1 — sp.diff on the CLOSED FORM itself,
    # mirroring the course notes' own proof, not an asserted derivative.
    qsym = Symbol('q')
    series_closed = 1 / (1 - qsym)
    d1_expr = sp.diff(series_closed, qsym)
    d2_expr = sp.diff(series_closed, qsym, 2)
    q_val = S.One - p_val
    if q_val == 0:
        # See _geom_tail_exact's own docstring for the identical q=0 SymPy
        # trap — the general infinite-Sum/derivative-substitution route
        # mishandles this boundary; p=1 is decided analytically instead.
        mean_sum, var_sum, mean_b = S.One, S.Zero, S.One
        derivative_identities_confirmed = True
    else:
        d1_val = d1_expr.subs(qsym, q_val)
        d2_val = d2_expr.subs(qsym, q_val)
        mean_sum = p_val * d1_val
        e2fact_sum = p_val * q_val * d2_val
        var_sum = e2fact_sum + mean_sum - mean_sum * mean_sum
        # Route B: expectation's (3.11) own Tail-Sum Formula, E[X]=
        # sum_{n>=0}P(X>n) — a genuinely different SymPy computation (a
        # direct Sum(...).doit() over the geometric series in q, not a
        # derivative), citing Topic 3.11 explicitly (see geometricNotes.jsx's
        # connects section).
        nsym = Symbol('n', nonnegative=True, integer=True)
        tail_sum_expr = Sum(qsym ** nsym, (nsym, 0, oo)).doit()
        mean_b = tail_sum_expr.subs(qsym, q_val)
        derivative_identities_confirmed = bool(
            simplify(d1_val - 1 / (1 - q_val) ** 2) == 0
            and simplify(d2_val - 2 / (1 - q_val) ** 3) == 0
        )

    mean_closed = 1 / p_val
    var_closed = q_val / (p_val * p_val)
    mean_agree = bool(simplify(mean_sum - mean_closed) == 0)
    var_agree = bool(simplify(var_sum - var_closed) == 0)
    route_b_agree = bool(simplify(mean_b - mean_closed) == 0)

    return {
        **result,
        'provenance': calc_cap('proved', eff_exact),
        'allPass': True,
        'normalization': {'pass': norm_pass},
        'mean': _calc_num(mean_sum), 'meanTex': sp.latex(mean_sum),
        'variance': _calc_num(var_sum), 'varianceTex': sp.latex(var_sum),
        'meanAgree': mean_agree, 'varAgree': var_agree,
        'meanB': _calc_num(mean_b), 'routeBAgree': route_b_agree,
        'derivativeIdentitiesConfirmed': derivative_identities_confirmed,
    }


# ---------------------------------------------------------------------------
# poisson (3.14, "Poisson distribution") — the direct sequel to
# analyse_binomial/analyse_geometric above. See poissonEngine.js's own
# header for the full mathematical design; this is a genuinely INDEPENDENT
# re-derivation in SymPy, not a port of the JS engine's own marching
# machinery — where the JS tier's provenance is ALWAYS 'numeric' (e^-lambda
# is transcendental, so no rational-arithmetic route exists there even for
# an exact-fraction lambda), THIS function decides every base-layer fact
# EXACTLY for a rational lambda via SymPy's own closed-form infinite-series
# summation (Sum(...).doit() resolves sum_k lambda^k/k! to exp(lambda)
# directly, a known hypergeometric closed form), and decides the
# Approximation Theorem's own hypothesis H1 (poissonScaling) EXACTLY via a
# genuine symbolic LIMIT as n -> infinity — sp.limit(binomial(n,k)*p_n**k*
# (1-p_n)**(n-k), n, oo) resolves cleanly to exp(-lambda)*lambda**k/k! under
# the textbook p_n=lambda/n scaling, and to 0 under a FIXED p_n — both
# confirmed by hand across every shipped preset (lambda up to 20, k up to
# 15) in well under 3 seconds each, comfortably inside the 10s compute
# watchdog. NO SymPy boundary trap analogous to analyse_geometric's own q=0
# finding was found here: normalization/poismv were confirmed by hand at
# lambda=1/1000 and lambda=100 (both extremes) and the H1 limit was
# confirmed at lambda=1/100,k=0 and lambda=20,k=15 — every case decided
# cleanly with no unevaluated Sum/limit and no special-casing needed.
def analyse_poisson(lambda_str, k, scaling_rule, p_fixed_str=None):
    """lambda_str: RAW lambda entry string ('0.7', '1/3', '10'). k: int (or
    numeric string), k>=0. scaling_rule: 'lambdaOverN'|'fixedP'. p_fixed_str:
    RAW probability entry string, required only for scaling_rule='fixedP'."""
    try:
        k = int(k)
    except (TypeError, ValueError):
        return calc_unknown('k must be an integer')
    if k < 0:
        return calc_unknown('k must be a non-negative integer')

    try:
        lambda_val, lambda_exact = parse_entry(lambda_str)
    except ValueError as e:
        return calc_unknown(f'bad lambda: {e}')
    if lambda_val <= 0:
        return calc_unknown('lambda must be > 0 (a Poisson rate is strictly positive)')

    p_fixed_val, p_fixed_exact = None, True
    if scaling_rule == 'fixedP':
        try:
            p_fixed_val, p_fixed_exact = parse_entry(p_fixed_str)
        except (ValueError, TypeError) as e:
            return calc_unknown(f'bad fixed p: {e}')
        if not (0 < p_fixed_val <= 1):
            return calc_unknown('the fixed p must satisfy 0 < p <= 1')
    elif scaling_rule != 'lambdaOverN':
        return calc_unknown("scalingRule must be 'lambdaOverN' or 'fixedP'")

    eff_exact = lambda_exact and p_fixed_exact

    ksym = Symbol('k', integer=True, nonnegative=True)
    target = sp.exp(-lambda_val) * lambda_val ** k / sp.factorial(k)

    # Base layer: normalization (the exponential series) — a real,
    # non-gating, ALWAYS-holding identity for lambda>0, confirmed via
    # SymPy's own closed-form infinite summation, not merely cited.
    total = Sum(lambda_val ** ksym * sp.exp(-lambda_val) / sp.factorial(ksym), (ksym, 0, oo)).doit()
    norm_pass = bool(simplify(total - 1) == 0)

    # Theorem poismv: E[X]=Var(X)=lambda, via the SAME exact closed-form
    # summation (a genuinely independent computation from the JS engine's
    # own marching-sum Route A/B split — SymPy proves the infinite sum
    # directly rather than truncating it).
    mean_sum = Sum(ksym * lambda_val ** ksym * sp.exp(-lambda_val) / sp.factorial(ksym), (ksym, 0, oo)).doit()
    e2_sum = Sum(ksym * (ksym - 1) * lambda_val ** ksym * sp.exp(-lambda_val) / sp.factorial(ksym), (ksym, 0, oo)).doit()
    var_sum = simplify(e2_sum + mean_sum - mean_sum ** 2)
    mean_agree = bool(simplify(mean_sum - lambda_val) == 0)
    var_agree = bool(simplify(var_sum - lambda_val) == 0)

    # The Poisson Approximation Theorem's H1 (poissonScaling): a genuine
    # symbolic LIMIT as n -> infinity of the fixed-k binomial PMF under the
    # declared scaling — decided EXACTLY, never sampled.
    nsym = Symbol('n', positive=True)
    p_n = p_fixed_val if scaling_rule == 'fixedP' else lambda_val / nsym
    expr = sp.binomial(nsym, k) * p_n ** k * (1 - p_n) ** (nsym - k)
    try:
        scaling_limit = sp.limit(expr, nsym, oo)
    except Exception as e:  # noqa: BLE001 — an exotic input SymPy's limit machinery declines
        return calc_unknown(f'could not compute the scaling limit: {e}')
    scaling_holds = bool(simplify(scaling_limit - target) == 0)

    result = {
        'id': 'poisson', 'lambda': _calc_num(lambda_val), 'k': k, 'scalingRule': scaling_rule,
        'normalization': {'pass': norm_pass},
        'mean': _calc_num(mean_sum), 'meanTex': sp.latex(mean_sum),
        'variance': _calc_num(var_sum), 'varianceTex': sp.latex(var_sum),
        'meanAgree': mean_agree, 'varAgree': var_agree,
        'target': _calc_num(target),
        'scalingLimit': _calc_num(scaling_limit), 'scalingLimitTex': sp.latex(scaling_limit),
        'scalingHolds': scaling_holds,
        'blockedAct': None if scaling_holds else 'poissonScaling',
    }
    if not scaling_holds:
        result['provenance'] = calc_cap('refuted', eff_exact)
        return result
    return {
        **result,
        'provenance': calc_cap('proved', eff_exact),
        'allPass': True,
    }


# =============================================================================
# \`uniform\` (3.15, "Uniform distribution — discrete & continuous"). An
# INDEPENDENT SymPy implementation of uniformEngine.js's own analyzeDiscrete
# Uniform/analyzeContinuousUniform (nothing here calls into the JS engine).
# See uniformEngine.js's own header for the full design note: unlike
# binomial/geometric/poisson, this module has NO hypothesis structure at all
# — a definition plus two derived moments, verified as an identity between
# two independently-computed numbers, never an existence claim.
#
# a/b/c/d are parsed via \`parse_entry\` (this domain's own exact-vs-decimal
# helper binomial/geometric/poisson already use for their own probability
# entries) — deliberately NOT expr.mjs's parseNumber/calc_parse_scalar (the
# Calculus-domain helper, which pre-converts even a plain decimal string into
# an exact Rational before it ever reaches SymPy), for exactly the reason
# analyse_binomial's own docstring documents at length: using the wrong
# helper here would silently make every decimal a/b preset read as 'proved'
# instead of 'numeric'.
#
# A genuine asymmetry with the JS numeric engine, the mirror image of
# analyse_ftc's/analyse_randomvariables' own "decides more than expected"
# finding: the JS continuous branch is ALWAYS 'numeric' (Simpson quadrature
# is inherently approximate), but analyse_uniform's continuous branch
# integrates the DEFINING integral directly via \`sp.integrate\` and so CAN
# decide exactly ('proved'/'refuted') whenever a and b are both exact —
# documented here, not merely discovered by accident.
def _unif_p_int(v):
    """v: a SymPy Rational/Integer known to be an integer value. Returns a
    plain Python int."""
    return int(v)


def analyse_uniform(rv_mode, a_str, b_str, c_str=None, d_str=None):
    """rv_mode: 'discrete'|'continuous'. a_str/b_str: RAW entry strings
    ('1', '1/2', '0.5'), parsed via parse_entry — never expr.mjs's
    parseNumber (see module header). c_str/d_str: RAW entry strings for the
    continuous branch's affine-invariance extra fact (Y=cX+d), optional —
    when omitted, the affine check is skipped entirely (not an error)."""
    if rv_mode == 'discrete':
        try:
            a_val, a_exact = parse_entry(a_str)
            b_val, b_exact = parse_entry(b_str)
        except ValueError as e:
            return calc_unknown(f'bad a/b: {e}')
        if not a_val.is_integer or not b_val.is_integer:
            return calc_unknown('a and b must be integers')
        a_int, b_int = _unif_p_int(a_val), _unif_p_int(b_val)
        if a_int > b_int:
            return calc_unknown('need a <= b (a > b is an empty range)')
        n = b_int - a_int + 1
        exact = a_exact and b_exact

        p = Q(1, n)
        total = sum((p for _ in range(n)), S.Zero)
        norm_pass = bool(simplify(total - 1) == 0)

        mean = sum((Q(k) * p for k in range(a_int, b_int + 1)), S.Zero)
        var = sum(((Q(k) - mean) ** 2 * p for k in range(a_int, b_int + 1)), S.Zero)
        mean_closed = Q(a_int + b_int, 2)
        var_closed = Q((b_int - a_int) * (b_int - a_int + 2), 12)
        identity_holds = bool(simplify(mean - mean_closed) == 0) and bool(simplify(var - var_closed) == 0)

        return {
            'provenance': calc_cap('proved' if identity_holds else 'refuted', exact),
            'kind': 'discrete', 'a': a_int, 'b': b_int, 'n': n,
            'degenerate': a_int == b_int,
            'normalizationPass': norm_pass,
            'mean': str(mean), 'variance': str(var),
            'meanClosed': str(mean_closed), 'varianceClosed': str(var_closed),
            'identityHolds': identity_holds,
        }

    if rv_mode != 'continuous':
        return calc_unknown(f'unknown rv_mode {rv_mode!r}')

    try:
        lo, lo_exact = parse_entry(a_str)
        hi, hi_exact = parse_entry(b_str)
    except ValueError as e:
        return calc_unknown(f'bad a/b: {e}')
    if not (hi > lo):
        return calc_unknown('need b > a (a zero-width interval [a,a] has no density to normalize)')
    exact = lo_exact and hi_exact

    width = hi - lo
    f = S.One / width
    try:
        total = sp.integrate(f, (x, lo, hi))
        mean = sp.integrate(f * x, (x, lo, hi))
        e2 = sp.integrate(f * x ** 2, (x, lo, hi))
    except Exception as e:  # noqa: BLE001 — an exotic width SymPy's integrator declines
        return calc_unknown(f'could not integrate exactly: {e}')
    var = simplify(e2 - mean ** 2)
    norm_pass = bool(simplify(total - 1) == 0)
    mean_closed = simplify((lo + hi) / 2)
    var_closed = simplify(width ** 2 / 12)
    identity_holds = bool(simplify(mean - mean_closed) == 0) and bool(simplify(var - var_closed) == 0)

    affine = None
    if c_str is not None and d_str is not None:
        try:
            c_val, c_exact = parse_entry(c_str)
            d_val, d_exact = parse_entry(d_str)
        except ValueError as e:
            affine = {'error': f'bad c/d: {e}'}
        else:
            if c_val == 0:
                affine = {'error': 'c must be nonzero'}
            else:
                new_lo = c_val * lo + d_val if c_val > 0 else c_val * hi + d_val
                new_hi = c_val * hi + d_val if c_val > 0 else c_val * lo + d_val
                expected_density = simplify(S.One / (Abs(c_val) * width))
                # Y=cX+d, X~U[lo,hi]. h(y)=(y-d)/c, |dh/dy|=1/|c| — the
                # transformed density's own genuinely constant value,
                # confirmed by direct substitution into f, not merely
                # asserted from the theorem's own conclusion.
                fY = simplify(f.subs(x, (x - d_val) / c_val) * (S.One / Abs(c_val)))
                density_matches = bool(simplify(fY - expected_density) == 0)
                affine = {
                    'c': str(c_val), 'd': str(d_val),
                    'newLo': str(new_lo), 'newHi': str(new_hi),
                    'expectedDensity': str(expected_density),
                    'densityMatches': density_matches,
                    'stillUniform': density_matches,
                    'provenance': calc_cap('proved' if density_matches else 'refuted', exact and c_exact and d_exact),
                }

    return {
        'provenance': calc_cap('proved' if identity_holds else 'refuted', exact),
        'kind': 'continuous', 'a': str(lo), 'b': str(hi),
        'normalizationPass': norm_pass,
        'mean': str(mean), 'variance': str(var),
        'meanClosed': str(mean_closed), 'varianceClosed': str(var_closed),
        'identityHolds': identity_holds,
        'affine': affine,
    }


# =============================================================================
# \`exponential\` (3.16, "Exponential distribution & memorylessness"). An
# INDEPENDENT SymPy implementation of exponentialEngine.js's own
# analyzeExponential (nothing here calls into the JS engine). See
# exponentialEngine.js's own header for the full design note: the ONE
# genuinely breakable hypothesis, poissonProcessAssumption, asks whether the
# DECLARED rate process is truly homogeneous Poisson (rateMode='constant')
# or a non-homogeneous "aging" hazard (rateMode='aging',
# rate(t)=lambda*(1+k*t)). Unlike the JS numeric tier (always 'numeric' —
# e^{-lambda x} is transcendental to a floating-point evaluator), SymPy CAN
# decide this EXACTLY for a rational lambda/x/k: the hazard integral
# lambda*x + lambda*k*x**2/2 is a polynomial, so both the naive survival
# e^{-lambda x} and the TRUE survival e^{-hazard integral} are exact SymPy
# expressions whose difference \`simplify\` can decide definitively — the same
# "JS always numeric, SymPy decides more" asymmetry randomvariables/ftc/
# poisson already documented, now live for a THIRD time in this domain.
#
# a/b/c/d-style entries here (lambda/x/s/t/k) are parsed via \`parse_entry\`
# (this domain's own exact-vs-decimal helper binomial/geometric/poisson/
# uniform already use) — never expr.mjs's parseNumber, for exactly the
# reason analyse_binomial's own docstring documents at length: using the
# wrong helper would silently make every decimal preset here read as
# 'proved' instead of 'numeric'.
#
# When poissonProcessAssumption is BROKEN, the official proof halts
# ('refuted') and a second, honest fact is still computed and returned: does
# memorylessness happen to survive anyway, under the TRUE (aging) process?
# It does not (mirroring exponentialEngine.js's own analyzeMemorylessness
# Broken) — computed via the SAME exact hazard-integral machinery, never
# merely asserted.
def analyse_exponential(lambda_str, x_str, s_str, t_str, rate_mode, aging_k_str=None):
    """lambda_str/x_str/s_str/t_str: RAW entry strings ('0.001', '1500',
    '1/2'), parsed via parse_entry. rate_mode: 'constant'|'aging'.
    aging_k_str: RAW entry string, required only for rate_mode='aging'."""
    try:
        lambda_val, lambda_exact = parse_entry(lambda_str)
    except ValueError as e:
        return calc_unknown(f'bad lambda: {e}')
    if lambda_val <= 0:
        return calc_unknown('lambda must be > 0 (an Exponential rate is strictly positive)')

    try:
        x_val, x_exact = parse_entry(x_str)
    except ValueError as e:
        return calc_unknown(f'bad x: {e}')
    if x_val <= 0:
        return calc_unknown('x must be > 0 (P(X>0)=1 trivially)')

    try:
        s_val, s_exact = parse_entry(s_str)
    except ValueError as e:
        return calc_unknown(f'bad s: {e}')
    if s_val < 0:
        return calc_unknown('s must be >= 0')

    try:
        t_val, t_exact = parse_entry(t_str)
    except ValueError as e:
        return calc_unknown(f'bad t: {e}')
    if t_val <= 0:
        return calc_unknown('t must be > 0')

    k_val, k_exact = S.Zero, True
    if rate_mode == 'aging':
        try:
            k_val, k_exact = parse_entry(aging_k_str)
        except (ValueError, TypeError) as e:
            return calc_unknown(f'bad aging rate: {e}')
        if k_val <= 0:
            return calc_unknown('the aging rate k must be > 0 (k=0 reduces to the constant-rate case)')
    elif rate_mode != 'constant':
        return calc_unknown("rateMode must be 'constant' or 'aging'")

    eff_exact = lambda_exact and x_exact and s_exact and t_exact and k_exact

    tsym = Symbol('t', nonnegative=True)
    xsym = Symbol('x', positive=True)

    def hazard_expr_at(bound):
        """The exact hazard integral integral_0^bound rate(u)du, and its
        implied survival e^{-that} — the SAME relationship
        checkPoissonProcessAssumption's own hazardSurvival approximates
        numerically by quadrature; SymPy decides it exactly."""
        rate_expr = lambda_val * (1 + k_val * tsym) if rate_mode == 'aging' else lambda_val
        return sp.integrate(rate_expr, (tsym, 0, bound))

    naive_survival = sp.exp(-lambda_val * x_val)
    true_hazard_x = hazard_expr_at(x_val)
    true_survival = sp.exp(-true_hazard_x)
    h1_holds = bool(simplify(naive_survival - true_survival) == 0)

    result = {
        'id': 'exponential', 'lambda': _calc_num(lambda_val), 'x': _calc_num(x_val),
        'rateMode': rate_mode,
        'naiveSurvival': _calc_num(naive_survival), 'naiveSurvivalTex': sp.latex(naive_survival),
        'trueSurvival': _calc_num(true_survival), 'trueSurvivalTex': sp.latex(true_survival),
        'h1Holds': h1_holds,
        'blockedAct': None if h1_holds else 'poissonProcessAssumption',
    }

    if not h1_holds:
        # Honest second fact, computed regardless of official validity (the
        # same shape bayes'/geometric's own broken-scenario checks already
        # established): does memorylessness survive anyway, under the TRUE
        # (aging) process?
        surv_s = sp.exp(-hazard_expr_at(s_val))
        surv_s_plus_t = sp.exp(-hazard_expr_at(s_val + t_val))
        conditional = simplify(surv_s_plus_t / surv_s) if surv_s != 0 else None
        fresh_t = sp.exp(-lambda_val * t_val)
        memoryless_holds = bool(conditional is not None and simplify(conditional - fresh_t) == 0)
        result.update({
            'memoryless': {
                'conditional': _calc_num(conditional) if conditional is not None else None,
                'freshTailT': _calc_num(fresh_t),
                'holds': memoryless_holds,
            },
            'provenance': calc_cap('refuted', eff_exact),
        })
        return result

    # Theorem expmv: E[X]=1/lambda, Var(X)=1/lambda^2, via exact integration
    # over the defining density — decided EXACTLY for a rational lambda.
    try:
        mean_sym = sp.integrate(xsym * lambda_val * sp.exp(-lambda_val * xsym), (xsym, 0, oo))
        e2_sym = sp.integrate(xsym ** 2 * lambda_val * sp.exp(-lambda_val * xsym), (xsym, 0, oo))
    except Exception as e:  # noqa: BLE001 — an exotic lambda SymPy's integrator declines
        return calc_unknown(f'could not integrate exactly: {e}')
    var_sym = simplify(e2_sym - mean_sym ** 2)
    mean_closed = simplify(1 / lambda_val)
    var_closed = simplify(1 / lambda_val ** 2)
    mean_agree = bool(simplify(mean_sym - mean_closed) == 0)
    var_agree = bool(simplify(var_sym - var_closed) == 0)

    # Theorem expmemory: memorylessness, exact algebraic cancellation.
    ratio = simplify(sp.exp(-lambda_val * (s_val + t_val)) / sp.exp(-lambda_val * s_val))
    target = simplify(sp.exp(-lambda_val * t_val))
    memoryless_holds = bool(simplify(ratio - target) == 0)

    identity_holds = mean_agree and var_agree and memoryless_holds

    result.update({
        'mean': _calc_num(mean_sym), 'meanTex': sp.latex(mean_sym),
        'variance': _calc_num(var_sym), 'varianceTex': sp.latex(var_sym),
        'meanAgree': mean_agree, 'varAgree': var_agree,
        'memoryless': {
            'ratio': _calc_num(ratio), 'target': _calc_num(target), 'holds': memoryless_holds,
        },
        'provenance': calc_cap('proved' if identity_holds else 'refuted', eff_exact),
        'allPass': identity_holds,
    })
    return result


# =============================================================================
# \`normal\` (3.17, "Normal distribution"). An INDEPENDENT SymPy implementation
# of normalEngine.js's own analyzeNormal (nothing here calls into the JS
# engine). See normalEngine.js's own header for the full design note: this is
# a CONCEPT module with no hypothesis structure at all (sigma>0 is a domain
# constraint, not a breakable assumption).
#
# INVESTIGATED, CONFIRMED: unlike every prior continuous module in this
# domain (transformrv/expectation/uniform/exponential all had elementary
# antiderivatives that were nonetheless left to quadrature in JS), the
# Gaussian integral int e^{-y^2/2}dy = sqrt(2*pi) is one of the FEW genuinely
# closed-form transcendental integrals SymPy handles NATIVELY —
# sp.integrate(f, (x,-oo,oo)) on the raw defining density decides
# normalization/mean/variance EXACTLY for an exact-rational mu/sigma, a
# genuine asymmetry stronger than randomvariables'/ftc's own "SymPy decides
# more" finding (those relied on solve_univariate_inequality/ordinary
# polynomial integration, not a transcendental special integral). Confirmed
# by hand: analyse_normal('3','1/2') simplifies total/mean/variance to EXACT
# 1/3/(1/4) with zero residual.
#
# Phi ITSELF still has no elementary closed form even in SymPy (Phi(z) =
# (1+erf(z/sqrt(2)))/2 is as far as it reduces) — a decimal Phi(z) value is
# therefore still fundamentally a numeric evaluation (sp.N), never claimed
# 'proved'. But standardized-probability IDENTITIES (Phi(-k)=1-Phi(k); the
# symmetric-interval rule 2*Phi(k)-1; P(X<=x)=Phi((x-mu)/sigma) itself) ARE
# decidable EXACTLY via erf for rational inputs — confirmed by hand (see the
# module's own investigation), with one confirmed TRAP: sp.integrate's own
# raw-density antiderivative reaches the answer via \`erfc\`, not \`erf\`, so a
# direct \`simplify(rawP - stdP)\` can spuriously report \`!= 0\` even when the
# two are identical — FIXED by \`.rewrite(erf)\` on both sides before
# comparing, the same "confirmed by hand, rejected the naive form, fixed"
# discipline every prior module's symbolic-tier retro documents. (This trap
# is not exercised by analyse_normal itself, which never separately computes
# a raw P(X<=x) — see below — but is documented here since it was found
# during this module's own investigation and would bite the next symbolic
# extension of this function.)
#
# For a FLOAT mu/sigma, sp.integrate still returns a genuine SymPy
# expression (e.g. \`-1 + 0.707106781186548*sqrt(2)\`), but algebraic
# simplification does not fold a bare float against an irrational sqrt(2)
# term to a clean symbolic 0 — confirmed by hand — so the float branch is
# graded by a numeric evaluation (\`_calc_num\`) against a tight absolute
# tolerance instead, capped at 'numeric' via calc_cap exactly like every
# other module in this domain.
def analyse_normal(mu_str, sigma_str, a_str=None, b_str=None):
    """mu_str/sigma_str: RAW entry strings ('0', '1/2', '0.005'), parsed via
    parse_entry — never expr.mjs's parseNumber (the domain's own documented
    binomial/geometric/poisson/uniform/exponential trap: using the wrong
    helper would silently make every decimal preset here read as 'proved'
    instead of 'numeric'). a_str/b_str: RAW entry strings for Theorem
    normlin's affine-transform extra fact (Y=aX+b), optional — when omitted,
    that check is skipped entirely (not an error), mirroring
    analyse_uniform's own c_str/d_str contract."""
    try:
        mu_val, mu_exact = parse_entry(mu_str)
        sigma_val, sigma_exact = parse_entry(sigma_str)
    except ValueError as e:
        return calc_unknown(f'bad mu/sigma: {e}')
    if not (sigma_val > 0):
        return calc_unknown('sigma must be > 0 (a Normal distribution requires a strictly positive standard deviation — a domain constraint, not a breakable hypothesis)')
    exact = mu_exact and sigma_exact

    f = 1 / (sigma_val * sp.sqrt(2 * pi)) * sp.exp(-(x - mu_val) ** 2 / (2 * sigma_val ** 2))
    try:
        total = sp.integrate(f, (x, -oo, oo))
        mean_sym = sp.integrate(f * x, (x, -oo, oo))
        e2_sym = sp.integrate(f * x ** 2, (x, -oo, oo))
    except Exception as e:  # noqa: BLE001 — an exotic mu/sigma SymPy's integrator declines
        return calc_unknown(f'could not integrate exactly: {e}')
    var_sym = simplify(e2_sym - mean_sym ** 2)

    total_num, mean_num, var_num = _calc_num(total), _calc_num(mean_sym), _calc_num(var_sym)
    if exact:
        norm_pass = bool(simplify(total - 1) == 0)
        mean_agree = bool(simplify(mean_sym - mu_val) == 0)
        var_agree = bool(simplify(var_sym - sigma_val ** 2) == 0)
    else:
        norm_pass = total_num is not None and abs(total_num - 1) < 1e-9
        mean_agree = mean_num is not None and abs(mean_num - float(mu_val)) < 1e-6 * max(1, abs(float(mu_val)))
        var_agree = var_num is not None and abs(var_num - float(sigma_val) ** 2) < 1e-6 * max(1, float(sigma_val) ** 2)
    identity_holds = norm_pass and mean_agree and var_agree

    # Theorem normlin (optional — see docstring): Y=aX+b ~ N(a*mu+b, a^2*sigma^2),
    # confirmed by direct substitution into the raw density (the change-of-
    # variables formula itself), never merely asserted from the theorem's own
    # conclusion — the same discipline analyse_uniform's own affine check uses.
    linear = None
    if a_str is not None and b_str is not None:
        try:
            a_val, a_exact = parse_entry(a_str)
            b_val, b_exact = parse_entry(b_str)
        except ValueError as e:
            linear = {'error': f'bad a/b: {e}'}
        else:
            if a_val == 0:
                linear = {'error': 'a must be nonzero'}
            else:
                y_sym = Symbol('y', real=True)
                h = (y_sym - b_val) / a_val
                fY = f.subs(x, h) / Abs(a_val)
                new_mu = a_val * mu_val + b_val
                new_sigma2 = a_val ** 2 * sigma_val ** 2
                target = 1 / sp.sqrt(new_sigma2 * 2 * pi) * sp.exp(-(y_sym - new_mu) ** 2 / (2 * new_sigma2))
                lin_exact = exact and a_exact and b_exact
                if lin_exact:
                    density_matches = bool(simplify(fY - target) == 0)
                else:
                    # a bare float against an irrational sqrt(2*pi) term does
                    # not fold to a clean symbolic 0 (see module header) — a
                    # sample-point numeric comparison at several y's is the
                    # honest check instead, confirmed by hand to agree with
                    # the exact branch to ~1e-15 on every tested case.
                    try:
                        new_mu_f, new_sigma_f = float(new_mu), abs(float(a_val)) * float(sigma_val)
                        sample_ys = [new_mu_f + k * new_sigma_f for k in (-3, -1, 0, 1, 3)]
                        density_matches = all(
                            abs(float(fY.subs(y_sym, yv)) - float(target.subs(y_sym, yv))) <= 1e-6
                            for yv in sample_ys
                        )
                    except Exception:  # noqa: BLE001
                        density_matches = False
                linear = {
                    'a': str(a_val), 'b': str(b_val),
                    'newMu': str(new_mu), 'newSigma2': str(new_sigma2),
                    'densityMatches': density_matches,
                    'provenance': calc_cap('proved' if density_matches else 'refuted', lin_exact),
                }

    # Standard normal identities — 2*Phi(k)-1 for k=1,2,3, decided EXACTLY via
    # erf regardless of mu/sigma (the identity is pure in k — see module
    # header) — informational, non-gating, the same shape every prior
    # "extra fact" in this domain uses.
    z = Symbol('z', real=True)
    Phi = sp.Rational(1, 2) * (1 + sp.erf(z / sp.sqrt(2)))
    identities = [{'k': k, 'value': _calc_num(2 * Phi.subs(z, k) - 1)} for k in (1, 2, 3)]

    return {
        'provenance': calc_cap('proved' if identity_holds else 'refuted', exact),
        'mu': str(mu_val), 'sigma': str(sigma_val),
        'total': total_num, 'mean': mean_num, 'variance': var_num,
        'normPass': norm_pass, 'meanAgree': mean_agree, 'varAgree': var_agree,
        'identityHolds': identity_holds,
        'linear': linear,
        'identities': identities,
    }


# ---------------------------------------------------------------------------
# distconnections (3.18, "Origins & connections between distributions") —
# the domain's own synthesis/review topic. See distconnectionsEngine.js's
# own header for the full mathematical design. TWO facets, \`mode\`:
#
#   'negbin'   — the Negative Binomial (beyond the nominal syllabus). Exact
#     mean/variance (E[X]=r/p, Var(X)=r(1-p)/p^2) and, when k is supplied,
#     the exact PMF value C(k-1,r-1)p^r(1-p)^(k-r) — both decided EXACTLY
#     for a rational p via ordinary SymPy arithmetic (never sp.nbinom or any
#     other closed-form special-function shortcut, mirroring
#     analyse_gammabeta's/analyse_improper's own "the raw defining object,
#     not the closed-form special function, is the primary evaluator"
#     choice). Normalization is decided via SymPy's own infinite Sum over
#     the negative-binomial series
#     (p^r * Sum_{j=0}^oo C(j+r-1,r-1) q^j = 1), the SAME "confirm the
#     series identity itself via SymPy's own Sum, never merely cited"
#     discipline analyse_geometric's own normalization check already uses —
#     confirmed by hand for r=1,3,4 to return exactly 1 (never an
#     unevaluated Sum), so no boundary trap analogous to analyse_geometric's
#     own q=0 finding was found here.
#
#   'paradigm' — the Poisson Paradigm. Decides ONLY the exact
#     E[X]=sum(p_j)=lambda identity for a rational p_arr (a genuine, if
#     algebraically direct, fact — lambda IS defined as sum(p_j), so this
#     confirms the fundamental-bridge identity survives exact rational
#     arithmetic, never silently promoted to a claim about the
#     approximation's SHAPE). The approximation QUALITY itself (the total
#     variation distance against Poisson(lambda)) is NOT a symbolic
#     question — no exact closed form exists for a heterogeneous p_arr's
#     own convolution against a transcendental Poisson target — and is
#     explicitly abstained on ('approxQuality': 'unknown') rather than
#     guessed, per non-negotiable #2.
def analyse_distconnections(mode, r=None, p_str=None, k=None, p_arr=None):
    if mode == 'negbin':
        try:
            r = int(r)
        except (TypeError, ValueError):
            return calc_unknown('r must be an integer')
        if r < 1:
            return calc_unknown('r must be a positive integer')
        try:
            p_val, p_exact = parse_entry(p_str)
        except ValueError as e:
            return calc_unknown(f'bad p: {e}')
        if not (0 < p_val <= 1):
            return calc_unknown('p must satisfy 0 < p <= 1')
        q_val = S.One - p_val

        # Exact mean/variance via X = sum of r independent Geometric(p)
        # stretches — the SAME identity distconnectionsEngine.js's own
        # Route B reuses geometricEngine.js's machinery for.
        mean = r / p_val
        variance = r * q_val / p_val**2

        # Normalization: p^r * Sum_{j=0}^oo C(j+r-1,r-1) q^j = 1 — a genuine
        # exact identity decided via SymPy's own infinite Sum, never merely
        # cited (see docstring above).
        j = Symbol('j', nonnegative=True, integer=True)
        norm_sum = (p_val ** r) * Sum(sp.binomial(j + r - 1, r - 1) * q_val ** j, (j, 0, oo)).doit()
        if p_exact:
            norm_holds = bool(simplify(norm_sum - 1) == 0)
        else:
            norm_holds = bool(abs(float(norm_sum) - 1) <= 1e-9)

        result = {
            'id': 'distconnections', 'mode': 'negbin', 'r': r,
            'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
            'variance': _calc_num(variance), 'varianceTex': sp.latex(variance),
            'normalizationHolds': norm_holds,
            'reducesTo': 'geometric' if r == 1 else None,
            'provenance': calc_cap('proved', p_exact),
        }

        if k is not None:
            try:
                k = int(k)
            except (TypeError, ValueError):
                return calc_unknown('k must be an integer')
            if k < r:
                return calc_unknown('k must satisfy k >= r')
            pmf = sp.binomial(k - 1, r - 1) * p_val ** r * q_val ** (k - r)
            result['pmf'] = _calc_num(pmf)
            result['pmfTex'] = sp.latex(pmf)
        return result

    if mode == 'paradigm':
        if not p_arr:
            return calc_unknown('need a list of event probabilities p_1..p_n')
        p_vals = []
        p_exact = True
        for s in p_arr:
            try:
                v, ex = parse_entry(s)
            except ValueError as e:
                return calc_unknown(f'bad p_j: {e}')
            if not (0 < v < 1):
                return calc_unknown('every p_j must satisfy 0 < p_j < 1')
            p_vals.append(v)
            p_exact = p_exact and ex
        lam = sum(p_vals, S.Zero)
        if p_exact:
            lam = simplify(lam)
        return {
            'id': 'distconnections', 'mode': 'paradigm', 'n': len(p_vals),
            'lambda': _calc_num(lam), 'lambdaTex': sp.latex(lam),
            'meanExactlyLambda': True,
            'provenance': calc_cap('proved', p_exact),
            'approxQuality': 'unknown',
            'note': 'E[X]=lambda is an exact algebraic identity (the fundamental bridge) for any p_j; whether the SHAPE genuinely resembles Poisson(lambda) is a numeric total-variation-distance question, not decided symbolically.',
        }

    return calc_unknown(f'unknown mode: {mode!r} (expected "negbin" or "paradigm")')


# =============================================================================
# \`mgf\` (3.19, "Moments & the moment generating function") — the direct
# sequel to \`expectation\` (3.11): M_X(s) = E[e^{sX}], discrete
# Sum(e^{sx}p(x)) or continuous Integral(e^{sx}f(x)dx). Mirrors
# mgfEngine.js's own shape exactly: same rv_mode split, same formula-over-
# an-index/coordinate law (never a finite table), and the SAME S_LADDER
# ([1/5, 1/10, 1/20, 1/40], both signs) used to decide mgfExists — this
# module DECIDES the ladder point-wise via genuine closed-form Sum/
# Integral evaluation AT each rational magnitude, rather than attempting a
# single symbolic-s Piecewise convergence condition once: a symbolic-s
# Sum/Integral's own generic convergence certificate (e.g.
# \`Abs(arg(s)+pi)<=pi/2\`) cannot be safely substituted at s=0 (arg(0) is
# undefined — confirmed by hand, "Invalid NaN comparison"), and evaluating
# AT each concrete rational ladder point sidesteps that entirely while
# staying decisive: a certain divergence at ANY tested magnitude refutes
# mgfExists outright (the same operational "no interval at all vs. one
# narrower than every tested magnitude" reading mgfEngine.js's own
# checkMgfExists already documents, kept identical on purpose here for
# one-to-one selftest parity with the JS ladder).
#
# Once mgfExists holds, Theorem momentsfrommgf is decided by constructing
# the GENERAL closed-form M(s) (a genuine function of the free symbol
# _MGF_S, via Sum/Integral.doit()), differentiating at s=0, and cross-
# checking against an INDEPENDENTLY-evaluated E[X]/E[X^2] (a second,
# separate Sum/Integral call on x*p(x)/x^2*p(x)) — the SAME "two
# independent routes" discipline mgfEngine.js's own analyzeMoments
# already uses, now symbolic. Two confirmed traps, both fixed, documented
# rather than hidden:
#   1. A Piecewise closed form guarded by a convergence condition (e.g.
#      \`Piecewise((1/(1-s/2), Abs(arg(s)+pi)<=pi/2), (Integral(...), True))\`
#      for Exponential(2)) cannot be differentiated-then-substituted at
#      s=0 directly — \`_mgf_pick_branch\` picks the Piecewise's own
#      closed-form branch FIRST (mgfExists is already independently
#      confirmed by the ladder by the time this runs, so a genuine
#      closed-form branch is known to apply near s=0), then differentiates
#      the PLAIN expression, sidestepping the guard condition entirely.
#   2. A genuinely two-sided-improper full-line integral (the Cauchy
#      flagship; the general shape ANY 'full'-support law could take) is
#      frequently left unevaluated by a single sp.integrate call even
#      when each signed half is independently decidable — the SAME
#      "two-sided-improper integrands are where sp.integrate often gives
#      up" trap family analyse_expectation's own _expect_abs_x_f_integral
#      already documents for meanExists; fixed the identical way, by
#      splitting at x=0 and combining two independently-classified
#      halves. On the Cauchy preset specifically, EACH half still
#      abstains for every tested ladder magnitude (neither half resolves
#      to a certain divergence, unlike the discrete/continuous Pareto
#      flagships, which DO resolve to a clean \`oo\`) — so this module
#      honestly reports mgfExists as 'unknown' on Cauchy, a genuine,
#      confirmed-by-hand asymmetry with the discrete tier's own decisive
#      refutation, not a bug to route around.
#
# Theorem mgflinear/mgfsum are NOT re-derived symbolically — a documented
# scope decision, the same "investigated, then declined" precedent
# transformrv's own H1-false branch and gammabeta's declined quadrature
# re-derivation already set: mgfEngine.js's own analyzeLinear already
# cross-checks the closed-form substitution against a full, independently
# recomputed INDUCED LAW (an affine change of variables, the continuous
# case needing exactly transformrv's own change-of-variables machinery a
# second time to redo symbolically for no additional confidence — the JS
# tier already covers it exactly, with a 100-case seeded fuzz pass);
# mgfsum/Uniqueness are built from a small FRESH finite-table construction
# with no typed formula input at all, so there is nothing of the
# student's own input left to decide symbolically beyond what the numeric
# tier already demonstrates exactly (confirmed to machine precision in
# mgfEngine.test.mjs).
# =============================================================================

_MGF_S = sp.symbols('mgf_s', real=True)
_MGF_LADDER = [sp.Rational(1, 5), sp.Rational(1, 10), sp.Rational(1, 20), sp.Rational(1, 40)]


def _mgf_pick_branch(expr):
    """See module docstring trap (1). Picks the first Piecewise branch
    whose own expression is not an unevaluated Sum/Integral; a bare
    (non-Piecewise) expression is returned unchanged."""
    if not isinstance(expr, sp.Piecewise):
        return expr
    for piece_expr, _cond in expr.args:
        if not (piece_expr.has(sp.Sum) or piece_expr.has(sp.Integral)):
            return piece_expr
    return expr


def _mgf_discrete_at(xf, pf, startN, support_kind, count, sval):
    n = _SEQ_N
    if support_kind == 'finite':
        try:
            val = sum((sp.exp(sval * xf.subs(n, k)) * pf.subs(n, k) for k in range(startN, startN + count)), S.Zero)
        except Exception as e:  # noqa: BLE001
            return ('abstain', str(e))
        return _expect_classify(val)
    try:
        raw = sp.Sum(sp.exp(sval * xf) * pf, (n, startN, oo)).doit()
    except Exception as e:  # noqa: BLE001
        return ('abstain', str(e))
    return _expect_classify(raw)


def _mgf_discrete_expr(xf, pf, startN, support_kind, count):
    """The GENERAL closed-form M_X(s) as a function of _MGF_S, used only
    for Theorem momentsfrommgf (never for deciding mgfExists itself —
    see module docstring for why the ladder decides that pointwise)."""
    n = _SEQ_N
    if support_kind == 'finite':
        return sum((sp.exp(_MGF_S * xf.subs(n, k)) * pf.subs(n, k) for k in range(startN, startN + count)), S.Zero)
    return sp.Sum(sp.exp(_MGF_S * xf) * pf, (n, startN, oo)).doit()


def _mgf_continuous_at(f, lo, hi, sval):
    try:
        if lo == -oo and hi == oo:
            left = sp.integrate(sp.exp(sval * x) * f, (x, -oo, 0))
            right = sp.integrate(sp.exp(sval * x) * f, (x, 0, oo))
            kl, lv = _expect_classify(left)
            kr, rv = _expect_classify(right)
            if kl == 'divergent' or kr == 'divergent':
                return ('divergent', None)
            if kl == 'abstain' or kr == 'abstain':
                return ('abstain', 'unevaluated even after splitting the full line at x=0')
            return ('finite', lv + rv)
        raw = sp.integrate(sp.exp(sval * x) * f, (x, lo, hi))
    except Exception as e:  # noqa: BLE001
        return ('abstain', str(e))
    return _expect_classify(raw)


def _mgf_continuous_expr(f, lo, hi):
    if lo == -oo and hi == oo:
        direct = sp.integrate(sp.exp(_MGF_S * x) * f, (x, lo, hi))
        if not (direct.has(sp.Integral) and _mgf_pick_branch(direct).has(sp.Integral)):
            return direct
        left = sp.integrate(sp.exp(_MGF_S * x) * f, (x, -oo, 0))
        right = sp.integrate(sp.exp(_MGF_S * x) * f, (x, 0, oo))
        return left + right
    return sp.integrate(sp.exp(_MGF_S * x) * f, (x, lo, hi))


def _mgf_moments(M_expr, trueEX_expr, trueEX2_expr):
    """Theorem momentsfrommgf. Returns None (an honest abstention, never a
    guessed value) when SymPy cannot produce a differentiable closed form
    for M(s) or the independent E[X]/E[X^2] route itself abstains."""
    if M_expr is None:
        return None
    branch = _mgf_pick_branch(M_expr)
    try:
        M1 = sp.simplify(sp.diff(branch, _MGF_S).subs(_MGF_S, 0).doit())
        M2 = sp.simplify(sp.diff(branch, _MGF_S, 2).subs(_MGF_S, 0).doit())
    except Exception:  # noqa: BLE001
        return None
    if M1.has(sp.Sum) or M1.has(sp.Integral) or M2.has(sp.Sum) or M2.has(sp.Integral):
        return None
    ex_expr = trueEX_expr.doit() if hasattr(trueEX_expr, 'doit') else trueEX_expr
    ex2_expr = trueEX2_expr.doit() if hasattr(trueEX2_expr, 'doit') else trueEX2_expr
    ek, ev = _expect_classify(ex_expr)
    e2k, e2v = _expect_classify(ex2_expr)
    if ek != 'finite' or e2k != 'finite':
        return None
    agree = bool(sp.simplify(M1 - ev) == 0) and bool(sp.simplify(M2 - e2v) == 0)
    variance = sp.simplify(M2 - M1 ** 2) if agree else None
    return {
        'agree': agree,
        'eX': _calc_num(M1), 'eXTex': sp.latex(M1),
        'eX2': _calc_num(M2), 'eX2Tex': sp.latex(M2),
        'variance': (_calc_num(variance) if variance is not None else None),
    }


def analyse_mgf(rv_mode, x_formula, p_formula, startN_str, disc_support_kind, count_str,
                 formula, cont_support_kind, lo_str, hi_str, a_str, b_str):
    """See the module docstring above. a_str/b_str are accepted for
    payload-shape parity with mgfEngine.js's own request (Theorem
    mgflinear's a/b) but are NOT used — see the docstring's own
    documented scope decision for why mgflinear is JS-only."""
    del a_str, b_str
    if rv_mode == 'discrete':
        try:
            xf = _seq_parse(x_formula)
            pf = _seq_parse(p_formula)
        except ValueError as e:
            return calc_unknown(f'bad formula: {e}')
        try:
            startN_val, startN_exact = calc_parse_scalar(startN_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad startN: {e}')
        if not (startN_val.is_integer and bool(startN_val >= 0)):
            return calc_unknown('startN must be a non-negative integer')
        startN = int(startN_val)
        exact = startN_exact
        count = None
        if disc_support_kind == 'finite':
            try:
                count_val, count_exact = calc_parse_scalar(count_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad count: {e}')
            if not (count_val.is_integer and bool(count_val >= 1)):
                return calc_unknown('count must be a positive integer')
            count = int(count_val)
            exact = exact and count_exact

        def at(sval):
            return _mgf_discrete_at(xf, pf, startN, disc_support_kind, count, sval)

        def general_expr():
            return _mgf_discrete_expr(xf, pf, startN, disc_support_kind, count)

        n = _SEQ_N
        if disc_support_kind == 'finite':
            trueEX_expr = sum((xf.subs(n, k) * pf.subs(n, k) for k in range(startN, startN + count)), S.Zero)
            trueEX2_expr = sum((xf.subs(n, k) ** 2 * pf.subs(n, k) for k in range(startN, startN + count)), S.Zero)
        else:
            trueEX_expr = sp.Sum(xf * pf, (n, startN, oo)).doit()
            trueEX2_expr = sp.Sum(xf ** 2 * pf, (n, startN, oo)).doit()
    elif rv_mode == 'continuous':
        try:
            f = calc_parse(formula)
        except ValueError as e:
            return calc_unknown(f'bad formula: {e}')
        if cont_support_kind == 'full':
            lo, hi, exact = -oo, oo, True
        else:
            try:
                lo, lo_exact = calc_parse_scalar(lo_str)
            except ValueError as e:
                return calc_unknown(f'bad lower support bound: {e}')
            if cont_support_kind == 'semiInfRight':
                hi, hi_exact = oo, True
            else:
                try:
                    hi, hi_exact = calc_parse_scalar(hi_str)
                except ValueError as e:
                    return calc_unknown(f'bad upper support bound: {e}')
                if not (hi > lo):
                    return calc_unknown('support must have hi > lo')
            exact = lo_exact and hi_exact

        def at(sval):
            return _mgf_continuous_at(f, lo, hi, sval)

        def general_expr():
            return _mgf_continuous_expr(f, lo, hi)

        trueEX_expr = sp.integrate(x * f, (x, lo, hi))
        trueEX2_expr = sp.integrate(x ** 2 * f, (x, lo, hi))
    else:
        return calc_unknown(f'unknown rv_mode {rv_mode!r}')

    zero_kind, zero_val = at(S.Zero)

    kinds = []
    for mag in _MGF_LADDER:
        k1, _ = at(mag)
        k2, _ = at(-mag)
        kinds.append(k1)
        kinds.append(k2)
    if 'divergent' in kinds:
        mgf_exists = False
    elif 'abstain' in kinds:
        mgf_exists = None
    else:
        mgf_exists = True

    if mgf_exists is not True:
        return {
            'id': 'mgf', 'kind': rv_mode,
            'provenance': calc_cap('refuted', exact) if mgf_exists is False else 'unknown',
            'mgfExists': mgf_exists,
            'atZero': (_calc_num(zero_val) if zero_kind == 'finite' else None),
            'blockedAct': 'mgfExists' if mgf_exists is False else None,
            'allPass': False,
            'moments': None,
        }

    try:
        M_expr = general_expr()
    except Exception:  # noqa: BLE001
        M_expr = None
    moments = _mgf_moments(M_expr, trueEX_expr, trueEX2_expr)
    all_pass = bool(moments and moments.get('agree'))
    return {
        'id': 'mgf', 'kind': rv_mode,
        'provenance': calc_cap('proved', exact),
        'mgfExists': True,
        'atZero': (_calc_num(zero_val) if zero_kind == 'finite' else None),
        'blockedAct': None,
        'allPass': all_pass,
        'moments': moments,
    }


# =============================================================================
# \`inequalities\` (3.20, "Markov & Chebyshev inequalities"). An INDEPENDENT
# SymPy implementation of inequalitiesEngine.js's own two-theorem module
# (theorem_mode 'markov'|'chebyshev') — nothing here calls into the JS
# engine. See inequalitiesEngine.js's own header for the full design note
# on the module's genuinely NEW verification shape (a true value vs. a
# bound, not an identity).
#
# For an exact-rational law, this decides the TRUE tail probability EXACTLY
# (sum the actual PMF over the exact set of support points satisfying the
# threshold; integrate the actual density over the exact sub-interval) and
# compares it SYMBOLICALLY against the stated bound (E[X]/a or sigma^2/c^2)
# — landing on 'proved'/'refuted' for the COMPARISON itself when both sides
# are exact rationals, this domain's usual exact-vs-numeric split
# (calc_cap, applied to \`holds\` rather than to an identity's own agreement).
#
# Markov's own hypothesis (X >= 0) is decided differently by support shape:
# a FINITE discrete support is checked EXHAUSTIVELY (every point, exact);
# an INFINITE discrete support and a continuous support are decided
# structurally — a continuous X is nonnegative iff its own support's lower
# bound is >= 0 (X's "value function" is the identity, so this is exactly
# the lower endpoint), and an infinite discrete X's own x(n) formula is
# checked via \`solve_univariate_inequality(x(n) < 0, ...)\` over the
# declared range — genuinely decided, never assumed, but honestly
# ABSTAINING (never guessing) if solve_univariate_inequality itself cannot
# resolve the inequality to a definite empty/nonempty set.
#
# The TRUE tail probability for an INFINITE discrete support additionally
# needs to know WHICH n satisfy x(n)>=a — solved the same way
# (solve_univariate_inequality), then handed to sp.Sum(...).doit() over the
# resolved tail; if the solved set is not a plain right-unbounded interval
# (the only shape every shipped infinite-support Markov preset needs), this
# abstains rather than guess at the summation range.
#
# Chebyshev's own reduction to Markov on Y=(X-mu)^2 is NOT re-derived
# symbolically as a second Markov call (unlike the JS engine's own
# checkMarkov reuse) — the true two-sided tail P(|X-mu|>=c) is computed
# DIRECTLY as two one-sided pieces (P(X<=mu-c)+P(X>=mu+c)), the same Route B
# the JS engine's own \`tailProbAbsDeviation\` uses, since a symbolic
# quadratic-indicator route would need to re-derive the same case-split
# for no additional confidence.
# =============================================================================

def _ineq_nonneg_discrete_finite(xf, startN, count):
    n = _SEQ_N
    try:
        vals = [xf.subs(n, k) for k in range(startN, startN + count)]
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot evaluate x(n): {e}') from e
    bad = next((v for v in vals if v.is_real is False or bool(v < 0)), None)
    return bad is None


def _ineq_solve_univariate_over_n(xf, n, rel, threshold, domain):
    """Solving a relational built directly from \`n\` (e.g. \`xf >= a\`, with
    \`xf\` in terms of \`_SEQ_N\`) has a confirmed SymPy trap, documented at
    length here: \`_SEQ_N\` carries \`positive=True\` (needed elsewhere so a
    discrete sequence's own n is read as a genuine positive-integer index),
    and \`x(n)>=a\`-style relationals AUTO-EVALUATE to a bare \`True\`/\`False\`
    at CONSTRUCTION time whenever the assumption alone already settles
    them (e.g. plain \`n >= 1\` immediately becomes \`sympy.true\`, since a
    positive INTEGER is always >= 1) — confirmed by hand: even
    substituting a fresh symbol into an ALREADY-collapsed \`True\` cannot
    recover the original comparison, so the substitution must happen
    BEFORE the relational is ever built. \`rel\` is \`operator.lt\`/
    \`operator.ge\` (or any two-argument comparison callable), applied to
    the SUBSTITUTED expression (in terms of a fresh, assumption-free
    integer symbol) and \`threshold\` — never to \`xf\` itself."""
    m = Symbol('_ineq_m', integer=True)
    return sp.solve_univariate_inequality(rel(xf.subs(n, m), threshold), m, domain=domain, relational=False)


def _ineq_nonneg_discrete_infinite(xf, startN):
    n = _SEQ_N
    try:
        sol = _ineq_solve_univariate_over_n(xf, n, operator.lt, S.Zero, Interval(startN, oo))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot decide X>=0: {e}') from e
    return sol == S.EmptySet


def _ineq_tail_ge_discrete_finite(xf, pf, startN, count, a):
    n = _SEQ_N
    try:
        terms = [pf.subs(n, k) for k in range(startN, startN + count) if bool(xf.subs(n, k) >= a)]
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot evaluate the tail: {e}') from e
    return sum(terms, S.Zero)


def _ineq_tail_ge_discrete_infinite(xf, pf, startN, a):
    """The tail P(x(n)>=a) for an infinite discrete support — solves for
    the set of n satisfying the threshold first (see module docstring),
    then sums the PMF over that set. Abstains (never guesses) if the
    solved set is not a plain right-unbounded interval [n0, oo)."""
    n = _SEQ_N
    try:
        sol = _ineq_solve_univariate_over_n(xf, n, operator.ge, a, Interval(startN, oo))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot resolve the tail set: {e}') from e
    if sol == S.EmptySet:
        return S.Zero
    if not (isinstance(sol, Interval) and sol.right == oo):
        raise Abstain(f'tail set is not a plain right-unbounded interval: {sol}')
    # Smallest integer n satisfying the resolved bound: n>left (open) is
    # floor(left)+1 for ANY real left (integer or not); n>=left (closed)
    # is ceiling(left).
    n0 = sp.floor(sol.left) + 1 if sol.left_open else sp.ceiling(sol.left)
    try:
        total = sp.Sum(pf, (n, n0, oo)).doit()
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot sum the resolved tail: {e}') from e
    kind, val = _expect_classify(total)
    if kind != 'finite':
        raise Abstain(f'tail sum did not resolve to a finite value ({kind})')
    return val


def _ineq_mean_discrete(xf, pf, startN, support_kind, count):
    """Returns ('finite', value) / ('divergent', None) / ('abstain', reason)
    — the SAME three-way classification analyse_expectation's own
    _expect_classify uses, so a genuinely divergent mean (St. Petersburg)
    is reported as a certain, definitive 'divergent' — refuted, never
    'unknown' — distinctly from an honest SymPy abstention."""
    n = _SEQ_N
    if support_kind == 'finite':
        try:
            return ('finite', sum((xf.subs(n, k) * pf.subs(n, k) for k in range(startN, startN + count)), S.Zero))
        except Exception as e:  # noqa: BLE001
            return ('abstain', f'cannot evaluate E[X]: {e}')
    try:
        raw = sp.Sum(xf * pf, (n, startN, oo)).doit()
    except Exception as e:  # noqa: BLE001
        return ('abstain', f'cannot sum E[X]: {e}')
    return _expect_classify(raw)


def _ineq_var_discrete(xf, pf, startN, support_kind, count, mean_val):
    """Returns ('finite', value) / ('divergent', None) / ('abstain', reason)
    — see _ineq_mean_discrete's own header for why this is a 3-way
    classification rather than a value-or-raise."""
    n = _SEQ_N
    if support_kind == 'finite':
        try:
            e2 = sum((xf.subs(n, k) ** 2 * pf.subs(n, k) for k in range(startN, startN + count)), S.Zero)
        except Exception as e:  # noqa: BLE001
            return ('abstain', f'cannot evaluate E[X^2]: {e}')
        return ('finite', sp.simplify(e2 - mean_val ** 2))
    try:
        raw = sp.Sum(xf ** 2 * pf, (n, startN, oo)).doit()
    except Exception as e:  # noqa: BLE001
        return ('abstain', f'cannot sum E[X^2]: {e}')
    kind, e2 = _expect_classify(raw)
    if kind != 'finite':
        return (kind, None)
    return ('finite', sp.simplify(e2 - mean_val ** 2))


def _ineq_two_tail_discrete_finite(xf, pf, startN, count, mu, c):
    n = _SEQ_N
    try:
        terms = [pf.subs(n, k) for k in range(startN, startN + count) if bool(Abs(xf.subs(n, k) - mu) >= c)]
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot evaluate the two-sided tail: {e}') from e
    return sum(terms, S.Zero)


def _ineq_integral_ge(f, lo, hi, a):
    lo_eff = sp.Max(a, lo)
    if hi != oo and bool(lo_eff >= hi):
        return S.Zero
    try:
        raw = sp.integrate(f, (x, lo_eff, hi))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot integrate the tail: {e}') from e
    kind, val = _expect_classify(raw)
    if kind != 'finite':
        raise Abstain(f'tail integral did not resolve to a finite value ({kind})')
    return val


def _ineq_integral_le(f, lo, hi, t):
    hi_eff = sp.Min(t, hi)
    if lo != -oo and bool(hi_eff <= lo):
        return S.Zero
    try:
        raw = sp.integrate(f, (x, lo, hi_eff))
    except Exception as e:  # noqa: BLE001
        raise Abstain(f'cannot integrate the lower tail: {e}') from e
    kind, val = _expect_classify(raw)
    if kind != 'finite':
        raise Abstain(f'lower tail integral did not resolve to a finite value ({kind})')
    return val


def _ineq_mean_continuous(f, lo, hi):
    """Returns ('finite', value) / ('divergent', None) / ('abstain',
    reason) — see _ineq_mean_discrete's own header."""
    if lo == -oo and hi == oo:
        # Splitting at x=0 by known sign, the SAME fix analyse_expectation's
        # own _expect_abs_x_f_integral already documents (a two-sided-
        # improper sp.integrate call frequently gives up even when each
        # signed half is individually decidable — the Cauchy flagship).
        try:
            left = sp.integrate(x * f, (x, -oo, 0))
            right = sp.integrate(x * f, (x, 0, oo))
        except Exception as e:  # noqa: BLE001
            return ('abstain', f'cannot integrate E[X]: {e}')
        kl, lv = _expect_classify(left)
        kr, rv = _expect_classify(right)
        if kl == 'divergent' or kr == 'divergent':
            return ('divergent', None)
        if kl == 'abstain' or kr == 'abstain':
            return ('abstain', 'E[X] unevaluated even after splitting the full line at x=0')
        return ('finite', lv + rv)
    try:
        raw = sp.integrate(x * f, (x, lo, hi))
    except Exception as e:  # noqa: BLE001
        return ('abstain', f'cannot integrate E[X]: {e}')
    return _expect_classify(raw)


def _ineq_var_continuous(f, lo, hi, mean_val):
    """Returns ('finite', value) / ('divergent', None) / ('abstain',
    reason) — see _ineq_mean_discrete's own header."""
    try:
        raw = sp.integrate(x ** 2 * f, (x, lo, hi))
    except Exception as e:  # noqa: BLE001
        return ('abstain', f'cannot integrate E[X^2]: {e}')
    kind, e2 = _expect_classify(raw)
    if kind != 'finite':
        return (kind, None)
    return ('finite', sp.simplify(e2 - mean_val ** 2))


def analyse_inequalities(theorem_mode, rv_mode, x_formula, p_formula, startN_str, disc_support_kind, count_str,
                          formula, cont_support_kind, lo_str, hi_str, a_str, c_str):
    """See the module docstring above."""
    try:
        if rv_mode == 'discrete':
            try:
                xf = _seq_parse(x_formula)
                pf = _seq_parse(p_formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            try:
                startN_val, startN_exact = calc_parse_scalar(startN_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad startN: {e}')
            if not (startN_val.is_integer and bool(startN_val >= 0)):
                return calc_unknown('startN must be a non-negative integer')
            startN = int(startN_val)
            exact = startN_exact
            count = None
            if disc_support_kind == 'finite':
                try:
                    count_val, count_exact = calc_parse_scalar(count_str)
                except Exception as e:  # noqa: BLE001
                    return calc_unknown(f'bad count: {e}')
                if not (count_val.is_integer and bool(count_val >= 1)):
                    return calc_unknown('count must be a positive integer')
                count = int(count_val)
                exact = exact and count_exact
        elif rv_mode == 'continuous':
            try:
                f = calc_parse(formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            if cont_support_kind == 'full':
                lo, hi, exact = -oo, oo, True
            else:
                try:
                    lo, lo_exact = calc_parse_scalar(lo_str)
                except ValueError as e:
                    return calc_unknown(f'bad lower support bound: {e}')
                if cont_support_kind == 'semiInfRight':
                    hi, hi_exact = oo, True
                else:
                    try:
                        hi, hi_exact = calc_parse_scalar(hi_str)
                    except ValueError as e:
                        return calc_unknown(f'bad upper support bound: {e}')
                    if not (hi > lo):
                        return calc_unknown('support must have hi > lo')
                exact = lo_exact and hi_exact
        else:
            return calc_unknown(f'unknown rv_mode {rv_mode!r}')

        if theorem_mode == 'markov':
            try:
                a_val, a_exact = calc_parse_scalar(a_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad a: {e}')
            if not bool(a_val > 0):
                return calc_unknown('a must be positive')
            exact = exact and a_exact

            if rv_mode == 'discrete':
                nonneg = _ineq_nonneg_discrete_finite(xf, startN, count) if disc_support_kind == 'finite' \\
                    else _ineq_nonneg_discrete_infinite(xf, startN)
                mean_kind, mean = _ineq_mean_discrete(xf, pf, startN, disc_support_kind, count)
                true_val = _ineq_tail_ge_discrete_finite(xf, pf, startN, count, a_val) if disc_support_kind == 'finite' \\
                    else _ineq_tail_ge_discrete_infinite(xf, pf, startN, a_val)
            else:
                nonneg = bool(lo >= 0)
                mean_kind, mean = _ineq_mean_continuous(f, lo, hi)
                true_val = _ineq_integral_ge(f, lo, hi, a_val)

            if mean_kind != 'finite':
                # E[X] itself does not exist for this law — Markov's own
                # bound (E[X]/a) cannot even be FORMED, a distinct failure
                # from the true tail simply exceeding a known bound.
                # Scope decision (documented in the module docstring): this
                # module presupposes a finite mean, the same way the
                # informal theorem statement itself does by writing E[X]/a
                # at all; every shipped preset has a finite mean, so this
                # branch is honesty, not a gap being routed around.
                return calc_unknown(f'E[X] does not exist for this law ({mean_kind})') if mean_kind == 'abstain' \\
                    else {'theoremMode': 'markov', 'kind': rv_mode, 'provenance': calc_cap('refuted', exact),
                          'nonneg': nonneg, 'mean': None, 'trueValue': None, 'bound': None, 'holds': None,
                          'blockedAct': 'meanExists'}

            bound = sp.simplify(mean / a_val)
            holds = bool(sp.simplify(true_val - bound) <= 0)
            return {
                'theoremMode': 'markov', 'kind': rv_mode,
                'provenance': calc_cap('proved' if holds else 'refuted', exact),
                'nonneg': nonneg,
                'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
                'trueValue': _calc_num(true_val), 'trueValueTex': sp.latex(true_val),
                'bound': _calc_num(bound), 'boundTex': sp.latex(bound),
                'holds': holds,
                'blockedAct': None if nonneg else 'nonnegative',
            }

        if theorem_mode != 'chebyshev':
            return calc_unknown(f'unknown theoremMode {theorem_mode!r}')

        try:
            c_val, c_exact = calc_parse_scalar(c_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad c: {e}')
        if not bool(c_val > 0):
            return calc_unknown('c must be positive')
        exact = exact and c_exact

        # meanExists / varianceExists — the SAME chained-clause cascade
        # analyse_expectation's own header documents: a certain divergence
        # is 'refuted' (never an abstention), an honest SymPy failure to
        # decide is 'unknown', and varianceExists is only even ATTEMPTED
        # once meanExists holds.
        if rv_mode == 'discrete':
            mean_kind, mean = _ineq_mean_discrete(xf, pf, startN, disc_support_kind, count)
        else:
            mean_kind, mean = _ineq_mean_continuous(f, lo, hi)
        if mean_kind == 'abstain':
            return calc_unknown(f'meanExists: {mean}')
        if mean_kind != 'finite':
            return {
                'theoremMode': 'chebyshev', 'kind': rv_mode,
                'provenance': calc_cap('refuted', exact),
                'meanExists': False, 'mean': None,
                'varianceExists': None, 'variance': None,
                'blockedAct': 'meanExists',
                'cheb': None,
            }

        if rv_mode == 'discrete':
            var_kind, variance = _ineq_var_discrete(xf, pf, startN, disc_support_kind, count, mean)
        else:
            var_kind, variance = _ineq_var_continuous(f, lo, hi, mean)
        if var_kind == 'abstain':
            return calc_unknown(f'varianceExists: {variance}')
        if var_kind != 'finite':
            return {
                'theoremMode': 'chebyshev', 'kind': rv_mode,
                'provenance': calc_cap('refuted', exact),
                'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
                'varianceExists': False, 'variance': None,
                'blockedAct': 'varianceExists',
                'cheb': None,
            }

        if rv_mode == 'discrete' and disc_support_kind == 'finite':
            true_val = _ineq_two_tail_discrete_finite(xf, pf, startN, count, mean, c_val)
        elif rv_mode == 'continuous':
            true_val = _ineq_integral_le(f, lo, hi, mean - c_val) + _ineq_integral_ge(f, lo, hi, mean + c_val)
        else:
            raise Abstain('the two-sided tail is only decided here for a finite discrete or continuous support')

        bound = sp.simplify(variance / (c_val ** 2))
        holds = bool(sp.simplify(true_val - bound) <= 0)
        return {
            'theoremMode': 'chebyshev', 'kind': rv_mode,
            'provenance': calc_cap('proved' if holds else 'refuted', exact),
            'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
            'varianceExists': True, 'variance': _calc_num(variance), 'varianceTex': sp.latex(variance),
            'blockedAct': None,
            'cheb': {
                'trueValue': _calc_num(true_val), 'trueValueTex': sp.latex(true_val),
                'bound': _calc_num(bound), 'boundTex': sp.latex(bound),
                'holds': holds,
            },
        }
    except Abstain as e:
        return calc_unknown(str(e))


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) — modes of convergence (topic 3.21) ──
# =============================================================================
# CONCEPT module: FOUR independent checks (never chained/gated -- none is a
# precondition for another) on ONE of five NAMED, PARAMETERIZED
# constructions -- see src/modules/convergenceEngine.js's own header for the
# full design rationale. Unlike every other module in this file, this is
# NOT a free-form-formula module: there is no expr.mjs parsing of a general
# f(x)/table here at all, only a construction id plus, for two of the five
# constructions, a small number of scalar parameters (p, q for the spike
# family; c for the degenerate constant).
#
# n is \`_SEQ_N\` (the SAME positive-integer symbol analyse_sequences already
# defines), reusing its own \`limit_seq\`/\`sp.limit\` machinery via a direct
# \`sp.limit\` call on each construction's own closed form -- confirmed by
# hand that SymPy decides every closed form this module needs EXACTLY,
# including the typewriter construction's floor-based P(Y_n=1) =
# 2^-floor(log2 n): \`sp.limit(2**(-sp.floor(sp.log(n, 2))), n, oo)\` resolves
# cleanly to 0 with no abstention needed -- a genuine departure from this
# domain's usual floor/ceiling/sign trap family (see this file's own module
# docstring), because here floor sits INSIDE a monotonically-diverging
# argument rather than creating a jump discontinuity \`continuous_domain\`
# must classify.
def _conv_mode(pass_, detail):
    """A DECIDED (proved/refuted) mode result — used for the three
    constructions (minUniform, typewriter's inProbability/meanSquare/
    inDistribution facets, symmetric) with no free scalar inputs at all, so
    there is never a float to cap against."""
    return {'pass': bool(pass_), 'provenance': 'proved' if pass_ else 'refuted', 'detail': detail}


def _conv_overall(fields_attempted):
    """Weakest-tier reduction across every ATTEMPTED (non-None) mode field
    -- the SAME reduction analyse_partials's own docstring documents,
    applied here to four independent facts rather than partials' own four
    clauses: 'unknown' if any attempted field abstained; else 'numeric' if
    any attempted field is 'numeric'; else 'refuted' if any attempted field
    is 'refuted'; else 'proved'."""
    provs = [f['provenance'] for f in fields_attempted if f is not None]
    if not provs:
        return 'unknown'
    if 'unknown' in provs:
        return 'unknown'
    if 'numeric' in provs:
        return 'numeric'
    if 'refuted' in provs:
        return 'refuted'
    return 'proved'


def analyse_convergence(construction, p_str, q_str, c_str):
    """construction: 'minUniform'|'spike'|'typewriter'|'symmetric'|
    'degenerate'. p_str/q_str: SymPy-syntax scalar strings (expr.mjs's
    parseNumber().sympy) for the 'spike' family only, else None/null.
    c_str: same, for the 'degenerate' constant only, else None/null.

    Returns a FLAT dict with the usual top-level \`provenance\` PLUS four
    per-mode sub-dicts (\`inProbability\`, \`almostSure\`, \`meanSquare\`,
    \`inDistribution\`), each \`{pass, provenance, detail}\` OR \`None\` when the
    construction genuinely does not evaluate that mode — the spike
    family's own documented scope decision (see convergenceEngine.js's
    header: an arbitrary (p,q) does not admit a clean closed-form
    in-distribution test point or an exact almost-sure argument, so both
    stay \`None\`/"not evaluated" rather than guessed at, on BOTH tiers).

    minUniform's \`almostSure\` is an ENTAILED consequence, not independently
    re-derived: Y_n = min(Y_{n-1}, X_n) <= Y_{n-1} is a structural fact of
    the min operator needing no computation, and Y_n >= 0 always — a
    monotone bounded sequence converges surely to some limit, which must
    then equal the ALREADY-DECIDED in-probability limit (0) by uniqueness
    of a probability limit — the same "report a derived, not re-probed,
    consequence" choice \`analyse_gammabeta\`'s own identity fields already
    made. typewriter's \`almostSure\` is a genuine EXACT combinatorial
    argument needing no limit at all (every omega in [0,1) lies in exactly
    one length-2^-k subinterval per block k, so Y_n(omega)=1 infinitely
    often for literally every omega) -- decided 'refuted', never an
    abstention. spike's \`almostSure\` depends on an independence MODELING
    ASSUMPTION this function does not encode (see convergenceEngine.js's
    own header) and stays \`None\` on this tier too, exactly mirroring the
    numeric tier's own choice to report only Monte Carlo EVIDENCE there,
    never a symbolic verdict.
    """
    n = _SEQ_N

    if construction == 'minUniform':
        eps_list = [sp.Rational(1, 2), sp.Rational(1, 5), sp.Rational(1, 10)]
        ip_pass = all(sp.limit((1 - e) ** n, n, oo) == 0 for e in eps_list)
        inProbability = _conv_mode(ip_pass, 'lim_{n->oo} (1-eps)^n = 0 for eps in (0,1)')

        ms_lim = sp.limit(sp.Rational(2, 1) / ((n + 1) * (n + 2)), n, oo)
        meanSquare = _conv_mode(ms_lim == 0, 'lim_{n->oo} 2/((n+1)(n+2)) = 0')

        y_list = [sp.Rational(1, 10), sp.Rational(3, 10), sp.Rational(6, 10), sp.Rational(9, 10)]
        id_pass = all(sp.limit(1 - (1 - y) ** n, n, oo) == 1 for y in y_list)
        inDistribution = _conv_mode(id_pass, 'lim_{n->oo} [1-(1-y)^n] = 1 for y in (0,1)')

        almostSure = {
            'pass': inProbability['pass'], 'provenance': inProbability['provenance'],
            'detail': 'Y_n = min(Y_{n-1}, X_n) <= Y_{n-1} always (a structural fact of the min operator) and Y_n >= 0 always -- a monotone bounded sequence converges surely to some limit; since Y_n -> 0 in probability too, and a probability limit is a.s. unique, the a.s. limit is 0 too.',
        }
        fields = {'inProbability': inProbability, 'almostSure': almostSure, 'meanSquare': meanSquare, 'inDistribution': inDistribution}

    elif construction == 'spike':
        try:
            p_val, p_exact = calc_parse_scalar(p_str)
            q_val, q_exact = calc_parse_scalar(q_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad p/q: {e}')
        if not bool(p_val > 0):
            return calc_unknown('p must be positive')
        if not bool(q_val > 0):
            return calc_unknown('q must be positive')
        exact = p_exact and q_exact

        ip_lim = sp.limit(n ** (-q_val), n, oo)
        ip_pass = bool(ip_lim == 0)
        inProbability = {'pass': ip_pass, 'provenance': calc_cap('proved' if ip_pass else 'refuted', exact),
                          'detail': 'lim_{n->oo} n^-q = 0 for q > 0'}

        exponent = sp.nsimplify(2 * p_val - q_val)
        ms_lim = sp.limit(n ** exponent, n, oo)
        ms_pass = bool(ms_lim == 0)
        meanSquare = {'pass': ms_pass, 'provenance': calc_cap('proved' if ms_pass else 'refuted', exact),
                      'detail': f'E[Y_n^2] = n^(2p-q) = n^{exponent}; converges to 0 iff 2p < q'}

        fields = {'inProbability': inProbability, 'almostSure': None, 'meanSquare': meanSquare, 'inDistribution': None}

    elif construction == 'typewriter':
        base = 2 ** (-sp.floor(sp.log(n, 2)))
        ip_lim = sp.limit(base, n, oo)
        ip_pass = bool(ip_lim == 0)
        inProbability = _conv_mode(ip_pass, 'P(Y_n=1) = 2^-floor(log2 n) -> 0')
        meanSquare = _conv_mode(ip_pass, 'E[Y_n^2] = P(Y_n=1) (indicator identity: Y_n in {0,1}) -> 0')

        id_lim = sp.limit(1 - base, n, oo)
        inDistribution = _conv_mode(id_lim == 1, 'F_{Y_n}(y) = 1 - 2^-floor(log2 n) -> 1 for y in (0,1)')

        almostSure = {
            'pass': False, 'provenance': 'refuted',
            'detail': 'Every omega in [0,1) lies in EXACTLY ONE length-2^-k subinterval per block k -- Y_n(omega)=1 infinitely often, for every omega. The sequence converges NOWHERE, a strictly stronger failure than merely almost-sure non-convergence.',
        }
        fields = {'inProbability': inProbability, 'almostSure': almostSure, 'meanSquare': meanSquare, 'inDistribution': inDistribution}

    elif construction == 'symmetric':
        inProbability = _conv_mode(False, 'P(|Y_n-Y|>=eps) = 1 for eps<=2 (Y is never 0) -- constant, never -> 0')
        meanSquare = _conv_mode(False, 'E[(Y_n-Y)^2] = E[4Y^2] = 4 -- constant, never -> 0')
        inDistribution = _conv_mode(True, 'Y_n = -Y has the identical law as Y by symmetry -- F_{Y_n} = F_Y exactly for every n')
        almostSure = _conv_mode(False, 'Y_n(omega) = Y(omega) requires Y(omega) = 0, never true for a +/-1 coin -- fails at every outcome')
        fields = {'inProbability': inProbability, 'almostSure': almostSure, 'meanSquare': meanSquare, 'inDistribution': inDistribution}

    elif construction == 'degenerate':
        try:
            c_val, c_exact = calc_parse_scalar(c_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad c: {e}')
        exact = c_exact
        prov = calc_cap('proved', exact)
        inProbability = {'pass': True, 'provenance': prov, 'detail': f'|Y_n-Y| = 0 always (c = {c_val})'}
        meanSquare = {'pass': True, 'provenance': prov, 'detail': f'E[(Y_n-Y)^2] = 0 always (c = {c_val})'}
        inDistribution = {'pass': True, 'provenance': prov, 'detail': f'F_{{Y_n}} = F_Y exactly (c = {c_val})'}
        almostSure = {'pass': True, 'provenance': prov, 'detail': f'Y_n(omega) = Y(omega) = {c_val} exactly, for every n and every omega'}
        fields = {'inProbability': inProbability, 'almostSure': almostSure, 'meanSquare': meanSquare, 'inDistribution': inDistribution}

    else:
        return calc_unknown(f'unknown construction {construction!r}')

    overall = _conv_overall(list(fields.values()))
    return {'id': 'convergence', 'construction': construction, 'provenance': overall, **fields}


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) -- the weak law of large
#    numbers (topic 3.22) --
# =============================================================================
# THEOREM module, the direct sequel to inequalities (3.20)/convergence
# (3.21) -- see wllnEngine.js's own header for the full design. H1
# (meanExists) gates everything; H2 (varianceExists) is the PROOF-SPECIFIC
# (Chebyshev) hypothesis, not the theorem's own true (weaker) requirement.
# Reuses inequalitiesEngine's own \`_ineq_mean_discrete\`/\`_ineq_var_discrete\`/
# \`_ineq_mean_continuous\`/\`_ineq_var_continuous\` module-level functions
# DIRECTLY -- the SAME three-way ('finite'/'divergent'/'abstain')
# classification \`analyse_inequalities\` already uses for its own
# meanExists/varianceExists chain, now on this module's per-trial law.
#
# The Chebyshev-BOUND limit (sigma^2/(n*eps^2) -> 0 as n->infty) is decided
# EXACTLY via \`sp.limit\` on \`_SEQ_N\` -- trivial once sigma^2 is a known
# finite constant, but demonstrated via a genuine symbolic limit rather than
# merely cited, per non-negotiable #1.
#
# The TRUE P(|M_n-mu|>=eps) is decided EXACTLY, for a SMALL finite discrete
# per-trial law and a small n_check, via genuine convolution/enumeration
# over the joint distribution of X_1+...+X_n_check (mirroring
# \`analyse_binomial\`'s own exact small-n brute-force cross-check) --
# abstaining honestly (returning \`exactSmallN: None\`, never guessing) for a
# larger n, an infinite discrete support, or a continuous law, where only
# the numeric (JS) tier can march.
def analyse_wlln(rv_mode, x_formula, p_formula, startN_str, disc_support_kind, count_str,
                  formula, cont_support_kind, lo_str, hi_str, eps_str, n_check_str):
    """See the module docstring above."""
    n = _SEQ_N
    try:
        if rv_mode == 'discrete':
            try:
                xf = _seq_parse(x_formula)
                pf = _seq_parse(p_formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            try:
                startN_val, startN_exact = calc_parse_scalar(startN_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad startN: {e}')
            if not (startN_val.is_integer and bool(startN_val >= 0)):
                return calc_unknown('startN must be a non-negative integer')
            startN = int(startN_val)
            exact = startN_exact
            count = None
            if disc_support_kind == 'finite':
                try:
                    count_val, count_exact = calc_parse_scalar(count_str)
                except Exception as e:  # noqa: BLE001
                    return calc_unknown(f'bad count: {e}')
                if not (count_val.is_integer and bool(count_val >= 1)):
                    return calc_unknown('count must be a positive integer')
                count = int(count_val)
                exact = exact and count_exact
        elif rv_mode == 'continuous':
            try:
                f = calc_parse(formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            if cont_support_kind == 'full':
                lo, hi, exact = -oo, oo, True
            else:
                try:
                    lo, lo_exact = calc_parse_scalar(lo_str)
                except ValueError as e:
                    return calc_unknown(f'bad lower support bound: {e}')
                if cont_support_kind == 'semiInfRight':
                    hi, hi_exact = oo, True
                else:
                    try:
                        hi, hi_exact = calc_parse_scalar(hi_str)
                    except ValueError as e:
                        return calc_unknown(f'bad upper support bound: {e}')
                    if not (hi > lo):
                        return calc_unknown('support must have hi > lo')
                exact = lo_exact and hi_exact
        else:
            return calc_unknown(f'unknown rv_mode {rv_mode!r}')

        try:
            eps_val, eps_exact = calc_parse_scalar(eps_str)
        except Exception as e:  # noqa: BLE001
            return calc_unknown(f'bad eps: {e}')
        if not bool(eps_val > 0):
            return calc_unknown('eps must be positive')
        exact = exact and eps_exact

        n_check = 4
        if n_check_str is not None:
            try:
                nc_val, _ = calc_parse_scalar(n_check_str)
                if nc_val.is_integer and bool(nc_val >= 1):
                    n_check = int(nc_val)
            except Exception:  # noqa: BLE001
                pass

        # H1 -- meanExists -- reusing analyse_inequalities' own module-level
        # helpers DIRECTLY.
        if rv_mode == 'discrete':
            mean_kind, mean = _ineq_mean_discrete(xf, pf, startN, disc_support_kind, count)
        else:
            mean_kind, mean = _ineq_mean_continuous(f, lo, hi)
        if mean_kind == 'abstain':
            return calc_unknown(f'meanExists: {mean}')
        if mean_kind != 'finite':
            return {
                'kind': rv_mode, 'provenance': calc_cap('refuted', exact),
                'meanExists': False, 'mean': None,
                'varianceExists': None, 'variance': None,
                'blockedAct': 'meanExists',
                'exactSmallN': None,
            }

        # H2 -- varianceExists (proof-specific) -- same reuse.
        if rv_mode == 'discrete':
            var_kind, variance = _ineq_var_discrete(xf, pf, startN, disc_support_kind, count, mean)
        else:
            var_kind, variance = _ineq_var_continuous(f, lo, hi, mean)
        if var_kind == 'abstain':
            return calc_unknown(f'varianceExists: {variance}')
        if var_kind != 'finite':
            return {
                'kind': rv_mode, 'provenance': calc_cap('refuted', exact),
                'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
                'varianceExists': False, 'variance': None,
                'blockedAct': 'varianceExists',
                'exactSmallN': None,
            }

        # The Chebyshev BOUND's own limit, decided EXACTLY (trivial once
        # sigma^2 is a known finite constant, but demonstrated via a genuine
        # sp.limit rather than merely cited).
        bound_seq = variance / (n * eps_val ** 2)
        bound_limit = sp.limit(bound_seq, n, oo)
        bound_limit_zero = bool(bound_limit == 0)

        # exactSmallN -- EXACT convolution/enumeration, finite discrete only,
        # abstaining (None, never guessed) otherwise.
        exact_small_n = None
        if rv_mode == 'discrete' and disc_support_kind == 'finite' and count ** n_check <= 5000:
            xs = [xf.subs(n, startN + i) for i in range(count)]
            ps = [pf.subs(n, startN + i) for i in range(count)]
            from itertools import product as _product
            true_val = S.Zero
            for combo in _product(range(count), repeat=n_check):
                s = sum((xs[i] for i in combo), S.Zero)
                w = S.One
                for i in combo:
                    w *= ps[i]
                m = sp.Rational(1, n_check) * s
                if bool(Abs(m - mean) >= eps_val):
                    true_val += w
            bound_at_ncheck = variance / (n_check * eps_val ** 2)
            holds = bool(sp.simplify(true_val - bound_at_ncheck) <= 0)
            exact_small_n = {
                'nCheck': n_check,
                'trueValue': _calc_num(true_val), 'trueValueTex': sp.latex(true_val),
                'bound': _calc_num(bound_at_ncheck), 'boundTex': sp.latex(bound_at_ncheck),
                'holds': holds,
            }

        return {
            'kind': rv_mode, 'provenance': calc_cap('proved', exact),
            'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
            'varianceExists': True, 'variance': _calc_num(variance), 'varianceTex': sp.latex(variance),
            'blockedAct': None,
            'boundLimitZero': bound_limit_zero,
            'exactSmallN': exact_small_n,
        }
    except Abstain as e:
        return calc_unknown(str(e))


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) -- the strong law of large
#    numbers (topic 3.23) --
# =============================================================================
# THEOREM module, the direct sequel to wlln (3.22)/convergence (3.21) --
# see sllnEngine.js's own header for the full design. ONE hypothesis only
# (meanExists) -- reuses inequalitiesEngine's own \`_ineq_mean_discrete\`/
# \`_ineq_mean_continuous\` module-level functions DIRECTLY, the SAME reuse
# \`analyse_wlln\` already established for its own H1. Unlike \`analyse_wlln\`,
# there is NO H2 (varianceExists) here at all: SLLN's true scope needs only
# a finite mean, a genuinely striking asymmetry with WLLN's own textbook
# proof (which needs finite variance for Chebyshev to run).
#
# The almost-sure convergence CLAIM ITSELF is NOT further decided here --
# genuinely, not a stylistic omission: the real proof of the Strong Law is
# hard and beyond this course (the source material states this plainly),
# and no SymPy computation could certify "only finitely many exceedances"
# for an arbitrary declared law any more than the numeric tier's own
# finite-horizon Monte Carlo evidence can PROVE it. This is a genuinely new
# instance of this domain's "how much does the symbolic tier add"
# asymmetry family, landing at "nothing more, for once" -- contrast
# \`analyse_randomvariables\`/\`analyse_ftc\`/\`analyse_uniform\`/\`analyse_normal\`,
# which all decide MORE than their own numeric tiers. Once \`meanExists\` is
# decided (exactly, for a rational law), the result reports
# \`asDecidable: False\` with a \`note\` explaining why, rather than inventing
# a verdict on the core claim.
def analyse_slln(rv_mode, x_formula, p_formula, startN_str, disc_support_kind, count_str,
                  formula, cont_support_kind, lo_str, hi_str):
    """See the module docstring above."""
    try:
        if rv_mode == 'discrete':
            try:
                xf = _seq_parse(x_formula)
                pf = _seq_parse(p_formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            try:
                startN_val, startN_exact = calc_parse_scalar(startN_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad startN: {e}')
            if not (startN_val.is_integer and bool(startN_val >= 0)):
                return calc_unknown('startN must be a non-negative integer')
            startN = int(startN_val)
            exact = startN_exact
            count = None
            if disc_support_kind == 'finite':
                try:
                    count_val, count_exact = calc_parse_scalar(count_str)
                except Exception as e:  # noqa: BLE001
                    return calc_unknown(f'bad count: {e}')
                if not (count_val.is_integer and bool(count_val >= 1)):
                    return calc_unknown('count must be a positive integer')
                count = int(count_val)
                exact = exact and count_exact
        elif rv_mode == 'continuous':
            try:
                f = calc_parse(formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            if cont_support_kind == 'full':
                lo, hi, exact = -oo, oo, True
            else:
                try:
                    lo, lo_exact = calc_parse_scalar(lo_str)
                except ValueError as e:
                    return calc_unknown(f'bad lower support bound: {e}')
                if cont_support_kind == 'semiInfRight':
                    hi, hi_exact = oo, True
                else:
                    try:
                        hi, hi_exact = calc_parse_scalar(hi_str)
                    except ValueError as e:
                        return calc_unknown(f'bad upper support bound: {e}')
                    if not (hi > lo):
                        return calc_unknown('support must have hi > lo')
                exact = lo_exact and hi_exact
        else:
            return calc_unknown(f'unknown rv_mode {rv_mode!r}')

        # H1 -- meanExists -- reusing analyse_inequalities' own module-level
        # helpers DIRECTLY (the same reuse analyse_wlln already established).
        # This is the ONLY hypothesis this module checks.
        if rv_mode == 'discrete':
            mean_kind, mean = _ineq_mean_discrete(xf, pf, startN, disc_support_kind, count)
        else:
            mean_kind, mean = _ineq_mean_continuous(f, lo, hi)
        if mean_kind == 'abstain':
            return calc_unknown(f'meanExists: {mean}')
        if mean_kind != 'finite':
            return {
                'kind': rv_mode, 'provenance': calc_cap('refuted', exact),
                'meanExists': False, 'mean': None,
                'blockedAct': 'meanExists',
                'asDecidable': False,
                'note': 'meanExists fails -- M_n has nothing to converge to. The almost-sure convergence claim itself is not decidable by this symbolic tier in any case (see this function\\'s own docstring); the numeric tier\\'s Monte Carlo evidence is the only route once mean/no-mean is settled.',
            }

        return {
            'kind': rv_mode, 'provenance': calc_cap('proved', exact),
            'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
            'blockedAct': None,
            'asDecidable': False,
            'note': 'The mean exists (decided exactly above) -- the theorem\\'s ONLY hypothesis holds. The almost-sure convergence conclusion itself is NOT further decidable by this symbolic tier: the real proof of the Strong Law is beyond this course, and no SymPy computation here certifies "only finitely many exceedances" for an arbitrary law any more than finite-horizon simulation can. See the numeric tier for Monte Carlo evidence (the exceedance-plateau demonstration).',
        }
    except Abstain as e:
        return calc_unknown(str(e))


# =============================================================================
# ── Probability & Statistics domain (DOMAINS[2]) -- the Central Limit
#    Theorem (topic 3.24) -- the LAST topic in the 24-topic backlog --
# =============================================================================
# THEOREM module -- see cltEngine.js's own header for the full design. H1
# (meanExists) gates H2 (varianceExists) -- reuses \`_ineq_mean_discrete\`/
# \`_ineq_mean_continuous\`/\`_ineq_var_discrete\`/\`_ineq_var_continuous\`
# DIRECTLY, the SAME reuse \`analyse_wlln\`/\`analyse_slln\` already established
# for their own H1 (and, for wlln, H2).
#
# UNLIKE \`analyse_wlln\`, a broken H2 here reports a \`provenance: 'refuted'\`
# verdict with \`variance: None\` and NOTHING further attempted -- CLT's own
# Z_n divides by sigma*sqrt(n), so a missing (infinite) variance makes the
# STATEMENT itself undefined, not merely one particular proof of it (\`mean\`
# is still reported, since H1 independently holds -- the same "report what
# is known as soon as it is known" precedent \`analyse_wlln\` already sets).
#
# \`exactSmallN\`, when computable (a FINITE discrete per-trial law, small
# enough that \`size**n_check\` stays tractable), builds Z_n's own EXACT law
# by brute-force convolution over every length-n_check trial sequence (the
# SAME odometer-style enumeration \`analyse_wlln\`'s own \`exact_small_n\`
# already uses, adapted to Z_n's affine transform of the raw sum) and
# reports the EXACT rational P(Z_n<=z) alongside a DECIMAL Phi(z) via
# \`sp.erf\` -- Phi ITSELF has no elementary closed form (see \`analyse_normal\`'s
# own header), so even an exact-rational per-trial law can only ever produce
# a \`numeric\` comparison against Phi, never a \`proved\` one; this asymmetry
# (mean/variance decide exactly, but the actual CLT comparison itself never
# fully closes) is documented here, not hidden, the same "SymPy decides
# less" instance \`analyse_slln\`'s own header already records for its own
# almost-sure conclusion.
def analyse_clt(rv_mode, x_formula, p_formula, startN_str, disc_support_kind, count_str,
                 formula, cont_support_kind, lo_str, hi_str, z_str, n_check_str):
    """See the module docstring above."""
    n = _SEQ_N
    try:
        if rv_mode == 'discrete':
            try:
                xf = _seq_parse(x_formula)
                pf = _seq_parse(p_formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            try:
                startN_val, startN_exact = calc_parse_scalar(startN_str)
            except Exception as e:  # noqa: BLE001
                return calc_unknown(f'bad startN: {e}')
            if not (startN_val.is_integer and bool(startN_val >= 0)):
                return calc_unknown('startN must be a non-negative integer')
            startN = int(startN_val)
            exact = startN_exact
            count = None
            if disc_support_kind == 'finite':
                try:
                    count_val, count_exact = calc_parse_scalar(count_str)
                except Exception as e:  # noqa: BLE001
                    return calc_unknown(f'bad count: {e}')
                if not (count_val.is_integer and bool(count_val >= 1)):
                    return calc_unknown('count must be a positive integer')
                count = int(count_val)
                exact = exact and count_exact
        elif rv_mode == 'continuous':
            try:
                f = calc_parse(formula)
            except ValueError as e:
                return calc_unknown(f'bad formula: {e}')
            if cont_support_kind == 'full':
                lo, hi, exact = -oo, oo, True
            else:
                try:
                    lo, lo_exact = calc_parse_scalar(lo_str)
                except ValueError as e:
                    return calc_unknown(f'bad lower support bound: {e}')
                if cont_support_kind == 'semiInfRight':
                    hi, hi_exact = oo, True
                else:
                    try:
                        hi, hi_exact = calc_parse_scalar(hi_str)
                    except ValueError as e:
                        return calc_unknown(f'bad upper support bound: {e}')
                    if not (hi > lo):
                        return calc_unknown('support must have hi > lo')
                exact = lo_exact and hi_exact
        else:
            return calc_unknown(f'unknown rv_mode {rv_mode!r}')

        z_val = S.Zero
        if z_str is not None:
            try:
                z_val, z_exact = calc_parse_scalar(z_str)
                exact = exact and z_exact
            except Exception:  # noqa: BLE001
                z_val = S.Zero

        n_check = 4
        if n_check_str is not None:
            try:
                nc_val, _ = calc_parse_scalar(n_check_str)
                if nc_val.is_integer and bool(nc_val >= 1):
                    n_check = int(nc_val)
            except Exception:  # noqa: BLE001
                pass

        # H1 -- meanExists -- reusing analyse_inequalities' own module-level
        # helpers DIRECTLY (the same reuse analyse_wlln/analyse_slln already
        # established).
        if rv_mode == 'discrete':
            mean_kind, mean = _ineq_mean_discrete(xf, pf, startN, disc_support_kind, count)
        else:
            mean_kind, mean = _ineq_mean_continuous(f, lo, hi)
        if mean_kind == 'abstain':
            return calc_unknown(f'meanExists: {mean}')
        if mean_kind != 'finite':
            return {
                'kind': rv_mode, 'provenance': calc_cap('refuted', exact),
                'meanExists': False, 'mean': None,
                'varianceExists': None, 'variance': None,
                'blockedAct': 'meanExists',
                'exactSmallN': None,
            }

        # H2 -- varianceExists -- same reuse. UNLIKE analyse_wlln, a broken
        # H2 here is NOT merely "this proof cannot run" -- Z_n's own
        # STATEMENT (its denominator) is undefined, so nothing further is
        # even attempted, exactly mirroring cltEngine.js's own numeric tier.
        if rv_mode == 'discrete':
            var_kind, variance = _ineq_var_discrete(xf, pf, startN, disc_support_kind, count, mean)
        else:
            var_kind, variance = _ineq_var_continuous(f, lo, hi, mean)
        if var_kind == 'abstain':
            return calc_unknown(f'varianceExists: {variance}')
        if var_kind != 'finite':
            return {
                'kind': rv_mode, 'provenance': calc_cap('refuted', exact),
                'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
                'varianceExists': False, 'variance': None,
                'blockedAct': 'varianceExists',
                'exactSmallN': None,
            }

        # exactSmallN -- EXACT convolution/enumeration for Z_n itself (finite
        # discrete only), comparing against a DECIMAL Phi(z) via erf -- Phi
        # has no elementary closed form (see this function's own docstring),
        # so this comparison can never fully close even for an exact-rational
        # law, unlike analyse_wlln's own exact_small_n (which compares two
        # exact rationals throughout).
        exact_small_n = None
        if rv_mode == 'discrete' and disc_support_kind == 'finite' and count ** n_check <= 5000:
            xs = [xf.subs(n, startN + i) for i in range(count)]
            ps = [pf.subs(n, startN + i) for i in range(count)]
            sigma_exact = sp.sqrt(variance)
            true_val = S.Zero
            for combo in product(range(count), repeat=n_check):
                s = sum((xs[i] for i in combo), S.Zero)
                w = S.One
                for i in combo:
                    w *= ps[i]
                zn = (s - n_check * mean) / (sigma_exact * sp.sqrt(n_check))
                if bool(zn <= z_val):
                    true_val += w
            Phi = sp.Rational(1, 2) * (1 + sp.erf(z_val / sp.sqrt(2)))
            phi_num = _calc_num(Phi)
            exact_small_n = {
                'nCheck': n_check, 'z': _calc_num(z_val),
                'trueValue': _calc_num(true_val), 'trueValueTex': sp.latex(true_val),
                'phi': phi_num,
                'diff': abs(_calc_num(true_val) - phi_num),
            }

        return {
            'kind': rv_mode, 'provenance': calc_cap('proved', exact),
            'meanExists': True, 'mean': _calc_num(mean), 'meanTex': sp.latex(mean),
            'varianceExists': True, 'variance': _calc_num(variance), 'varianceTex': sp.latex(variance),
            'blockedAct': None,
            'exactSmallN': exact_small_n,
        }
    except Abstain as e:
        return calc_unknown(str(e))
`),self.postMessage({type:`ready`}),t}function n(){return e||=t(),e}n().catch(e=>{self.postMessage({type:`init-error`,error:String(e&&e.message||e)})});let r={rowreduce:(e,t)=>e.globals.get(`analyse_rowreduce`)(e.toPy(t.matrix)),linsystems:(e,t)=>e.globals.get(`analyse_linsystems`)(e.toPy(t.matrix)),vectorspaces:(e,t)=>t.target?e.globals.get(`analyse_vectorspaces`)(e.toPy(t.vectors),e.toPy(t.target)):e.globals.get(`analyse_vectorspaces`)(e.toPy(t.vectors)),orthogonality:(e,t)=>e.globals.get(`analyse_orthogonality`)(e.toPy(t.vectors)),fundspaces:(e,t)=>e.globals.get(`analyse_fundspaces`)(e.toPy(t.matrix)),lineartransform:(e,t)=>e.globals.get(`analyse_lineartransform`)(e.toPy(t.matrix)),eigen:(e,t)=>e.globals.get(`analyse_eigen`)(e.toPy(t.matrix)),determinants:(e,t)=>e.globals.get(`analyse_determinants`)(e.toPy(t.matrix)),inverses:(e,t)=>t.b?e.globals.get(`analyse_inverses`)(e.toPy(t.matrix),e.toPy(t.b)):e.globals.get(`analyse_inverses`)(e.toPy(t.matrix)),leastsquares:(e,t)=>e.globals.get(`analyse_leastsquares`)(e.toPy(t.matrix),e.toPy(t.b)),spectral:(e,t)=>e.globals.get(`analyse_spectral`)(e.toPy(t.matrix)),svd:(e,t)=>e.globals.get(`analyse_svd`)(e.toPy(t.matrix)),changeofbasis:(e,t)=>t.otherBasis?e.globals.get(`analyse_changeofbasis`)(e.toPy(t.basis),e.toPy(t.vector),e.toPy(t.otherBasis)):e.globals.get(`analyse_changeofbasis`)(e.toPy(t.basis),e.toPy(t.vector)),quadraticforms:(e,t)=>e.globals.get(`analyse_quadraticforms`)(e.toPy(t.matrix)),factorizations:(e,t)=>e.globals.get(`analyse_factorizations`)(e.toPy(t.matrix),t.mode),rolle:(e,t)=>e.globals.get(`analyse_rolle`)(t.expr,t.a,t.b),mvt:(e,t)=>e.globals.get(`analyse_mvt`)(t.expr,t.a,t.b),limits:(e,t)=>e.globals.get(`analyse_limits`)(t.expr,t.c,t.override??null),ivt:(e,t)=>e.globals.get(`analyse_ivt`)(t.expr,t.a,t.b,t.k),riemann:(e,t)=>e.globals.get(`analyse_riemann`)(t.expr,t.a,t.b),netchange:(e,t)=>e.globals.get(`analyse_netchange`)(t.expr,t.a,t.b),ftc:(e,t)=>e.globals.get(`analyse_ftc`)(t.f,t.F,t.a,t.b),improper:(e,t)=>e.globals.get(`analyse_improper`)(t.expr,t.a,t.b),gammabeta:(e,t)=>e.globals.get(`analyse_gammabeta`)(t.kind,t.p,t.q??null),sequences:(e,t)=>e.globals.get(`analyse_sequences`)(t.expr,t.startN),series:(e,t)=>e.globals.get(`analyse_series`)(t.expr,t.startN,t.testMode),powerseries:(e,t)=>e.globals.get(`analyse_powerseries`)(t.expr,t.startN,t.testMode),cauchymvt:(e,t)=>e.globals.get(`analyse_cauchymvt`)(t.f,t.g,t.a,t.b),taylor:(e,t)=>e.globals.get(`analyse_taylor`)(t.f,t.a,t.x,t.n),partials:(e,t)=>e.globals.get(`analyse_partials`)(t.expr,t.a,t.b),totaldiff:(e,t)=>e.globals.get(`analyse_totaldiff`)(t.expr,t.a,t.b),chainrule:(e,t)=>e.globals.get(`analyse_chainrule`)(t.expr,t.xt,t.yt,t.t0),extrema:(e,t)=>e.globals.get(`analyse_extrema`)(t.expr,t.a,t.b),lagrange:(e,t)=>e.globals.get(`analyse_lagrange`)(t.f,t.g,t.a,t.b),probabilitylaws:(e,t)=>e.globals.get(`analyse_probabilitylaws`)(e.toPy(t.outcomes),e.toPy(t.entries),e.toPy(t.idxA),e.toPy(t.idxB)),counting:(e,t)=>e.globals.get(`analyse_counting`)(t.mode,t.n,t.k,t.order??null,t.replacement??null,t.trueOrder??null,t.trueReplacement??null),descriptivestats:(e,t)=>e.globals.get(`analyse_descriptivestats`)(e.toPy(t.entries),t.outlierIndex??null),conditional:(e,t)=>e.globals.get(`analyse_conditional`)(e.toPy(t.outcomes),e.toPy(t.entries),e.toPy(t.events)),bayes:(e,t)=>e.globals.get(`analyse_bayes`)(e.toPy(t.outcomes),e.toPy(t.entries),e.toPy(t.partition),e.toPy(t.eventA)),randomvariables:(e,t)=>e.globals.get(`analyse_randomvariables`)(t.kind,e.toPy(t.xs??[]),e.toPy(t.ps??[]),t.formula??null,t.supportKind??null,t.lo??null,t.hi??null),independence:(e,t)=>e.globals.get(`analyse_independence`)(t.checkMode,e.toPy(t.outcomes),e.toPy(t.entries),e.toPy(t.events)),expectation:(e,t)=>e.globals.get(`analyse_expectation`)(t.rvMode,t.xFormula??null,t.pFormula??null,t.startN??null,t.discSupportKind??null,t.count??null,t.formula??null,t.contSupportKind??null,t.lo??null,t.hi??null),transformrv:(e,t)=>e.globals.get(`analyse_transformrv`)(t.rvMode,e.toPy(t.xs??[]),e.toPy(t.ps??[]),t.gFormula??null,t.fFormula??null,t.lo??null,t.hi??null),binomial:(e,t)=>e.globals.get(`analyse_binomial`)(t.n,t.p,t.nCheck,t.scenario,e.toPy(t.pArr??[])),geometric:(e,t)=>e.globals.get(`analyse_geometric`)(t.p,t.nCheck,t.scenario,t.inc??null,t.s,t.t),poisson:(e,t)=>e.globals.get(`analyse_poisson`)(t.lambda,t.k,t.scalingRule,t.pFixed??null),uniform:(e,t)=>e.globals.get(`analyse_uniform`)(t.rvMode,t.a,t.b,t.c??null,t.d??null),exponential:(e,t)=>e.globals.get(`analyse_exponential`)(t.lambda,t.x,t.s,t.t,t.rateMode,t.agingK??null),normal:(e,t)=>e.globals.get(`analyse_normal`)(t.mu,t.sigma,t.a??null,t.b??null),distconnections:(e,t)=>e.globals.get(`analyse_distconnections`)(t.mode,t.r??null,t.p??null,t.k??null,e.toPy(t.pArr??[])),mgf:(e,t)=>e.globals.get(`analyse_mgf`)(t.rvMode,t.xFormula??null,t.pFormula??null,t.startN??null,t.discSupportKind??null,t.count??null,t.formula??null,t.contSupportKind??null,t.lo??null,t.hi??null,t.a??null,t.b??null),inequalities:(e,t)=>e.globals.get(`analyse_inequalities`)(t.theoremMode,t.rvMode,t.xFormula??null,t.pFormula??null,t.startN??null,t.discSupportKind??null,t.count??null,t.formula??null,t.contSupportKind??null,t.lo??null,t.hi??null,t.a??null,t.c??null),convergence:(e,t)=>e.globals.get(`analyse_convergence`)(t.construction,t.p??null,t.q??null,t.c??null),wlln:(e,t)=>e.globals.get(`analyse_wlln`)(t.rvMode,t.xFormula??null,t.pFormula??null,t.startN??null,t.discSupportKind??null,t.count??null,t.formula??null,t.contSupportKind??null,t.lo??null,t.hi??null,t.eps??null,t.nCheck??null),slln:(e,t)=>e.globals.get(`analyse_slln`)(t.rvMode,t.xFormula??null,t.pFormula??null,t.startN??null,t.discSupportKind??null,t.count??null,t.formula??null,t.contSupportKind??null,t.lo??null,t.hi??null),clt:(e,t)=>e.globals.get(`analyse_clt`)(t.rvMode,t.xFormula??null,t.pFormula??null,t.startN??null,t.discSupportKind??null,t.count??null,t.formula??null,t.contSupportKind??null,t.lo??null,t.hi??null,t.z??null,t.nCheck??null)};self.onmessage=async e=>{let{requestId:t,kind:i,payload:a}=e.data,o;try{let e=await n(),s=r[i];if(!s)throw Error(`unknown analysis kind: ${i}`);o=s(e,a);let c=o.toJs({dict_converter:Object.fromEntries});self.postMessage({requestId:t,result:c})}catch(e){self.postMessage({requestId:t,error:String(e&&e.message||e)})}finally{o?.destroy?.()}}})();