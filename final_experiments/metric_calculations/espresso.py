from typing import Tuple, Set, Dict, List
from final_experiments.types import ClassRules, Clause, Rules
from final_experiments.util import global_bounds
from itertools import product
from pyeda.boolalg.espresso import espresso, set_config
from pyeda.boolalg import espresso as _e
assert _e.RTYPE == 4, "RTYPE value check"

# Remove RTYPE from our
delattr(_e, "RTYPE")
set_config(skip_make_sparse=1) # stupidly the binaries are not compiled with OFF flag

Interval   = Tuple[float, float]

def _class_atoms(
        class_rules: ClassRules,
        g_bounds: Dict[int, Interval],
) -> Dict[int, List[Interval]]:
    """
    For every dimension that appears in *this* class:
        cut-points = {all lo/hi in the class} ∪ {global min, global max}
        atoms      = consecutive intervals between cut-points
    """
    cuts: Dict[int, Set[float]] = {}
    for clause in class_rules:
        for dim, (lo, hi) in clause:
            cuts.setdefault(dim, set()).update([lo, hi])
    for dim in cuts:                               # add global min/max
        glo, ghi = g_bounds[dim]
        cuts[dim].update([glo, ghi])

    atoms: Dict[int, List[Interval]] = {}
    for dim, pts in cuts.items():
        pts = sorted(pts)
        atoms[dim] = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return atoms                                   # dim -> [intervals...]

def _mv_vocab(atoms: Dict[int, List[Interval]]
              ) -> Dict[int, Dict[Interval, int]]:
    return {d: {iv: i for i, iv in enumerate(iv_list)}
            for d, iv_list in atoms.items()}

def _rules_to_espresso_text(
        class_rules: ClassRules,
        atoms: Dict[int, List[Interval]],
) -> Tuple[List[Tuple[str, str]], Dict[int, Dict[Interval, int]]]:
    """
    Build a *finite* truth table for this class:
        - variables = only the dims that appear in the class
        - ON rows   = atomic tuples covered by any clause
        - OFF rows  = remaining atomic tuples inside [global min,max]
    """
    symbol_id = _mv_vocab(atoms)                     # dim ->  id map
    dims      = list(atoms.keys())                   # stable order

    # ON tuples
    ON: Set[Tuple[int, ...]] = set()
    for clause in class_rules:
        # per-dim token list (id list or full range)
        token_ranges: List[List[int]] = []
        for dim in dims:
            hit = next((iv for d, iv in clause if d == dim), None)
            if hit is None:                          # unconstrained
                token_ranges.append(list(range(len(atoms[dim]))))
            else:                                    # constrained
                lo, hi = hit
                token_ranges.append(
                    [i for i, iv in enumerate(atoms[dim])
                     if iv[0] >= lo and iv[1] <= hi]
                )
        ON.update(product(*token_ranges))

    # OFF tuples
    TOTAL = product(*[range(len(atoms[d])) for d in dims])
    OFF   = set(TOTAL).difference(ON)

    def tup_row(t: Tuple[int, ...], bit: str) -> Tuple[str, str]:
        return (" ".join(map(str, t)), bit)

    pla_text = [tup_row(t, "1") for t in ON]# + [tup_row(t, "2") for t in OFF]
    return pla_text, symbol_id

def _espresso_text_to_rules(
        pla_text: List[Tuple[str, str]],
        symbol_id: Dict[int, Dict[Interval, int]],
        dims: List[int],
) -> ClassRules:
    inverse = {d: {sid: iv for iv, sid in mp.items()}
               for d, mp in symbol_id.items()}
    new_rules: ClassRules = []
    for cube, bit in pla_text:
        if bit.strip() != "1":
            continue                      # keep ON rows only
        literals: Clause = []
        for dim_local, token in enumerate(cube.split()):
            if token == "-":                 # don’t-care -> skip
                continue
            token_int   = int(token)
            dim_global  = dims[dim_local]
            iv          = inverse[dim_global][token_int]
            literals.append((dim_global, iv))
        new_rules.append(literals)
    return new_rules


def reformulate_class_with_espresso(
        class_rules: ClassRules,
        g_bounds: Dict[int, Interval],
        max_tuples: int = 1_000_000,
) -> ClassRules:
    """
    Return a logically equivalent ClassRules with
    fewer (or equal) literals, using per-class Espresso
    minimisation that keeps coverage EXACT.

    If the atomic grid would exceed `max_tuples`
    the original class is returned unchanged.
    """
    if not class_rules:                                # empty class
        return []

    atoms = _class_atoms(class_rules, g_bounds)
    grid_size = 1
    for iv_list in atoms.values():
        grid_size *= len(iv_list)
    if grid_size > max_tuples:                         # safeguard
        return class_rules

    pla_text, sym_id = _rules_to_espresso_text(class_rules, atoms)
    dims = list(atoms.keys())                          # order again
    minimised    = _run_espresso_mv(pla_text, n_vars=len(dims))
    return _espresso_text_to_rules(minimised, sym_id, dims)


def _run_espresso_mv(pla_text: List[Tuple[str, str]], n_vars: int
                     ) -> List[Tuple[str, str]]:
    """
    Parameters
    ----------
    pla_text : [('0 1 2', '1'), ...]  # tokens separated by blanks
    n_vars   : number of input columns (= len(dims))

    Returns the minimised PLA rows (same string format)
    containing ONLY the ON-set cubes (output '1').
    """

    WITHIN_F__TYPE = 1     # ON-set
    DONT_CARE_TYPE = 2     # Dont care-set , stupidly the binaries are not compiled with OFF flag
    def encode_cube(cube_str: str) -> Tuple[int, ...]:

        tok2int = lambda tok: 2 if tok == '-' else int(tok)
        return tuple(tok2int(tok) for tok in cube_str.split())

    cover = []
    bits_in_rows = {bit for _, bit in pla_text}
    print("OUTPUT BITS PRESENT:", bits_in_rows)
    assert bits_in_rows <= {"1", "2"}, "Found a '0' row!"

    for cube, bit in pla_text:
        invec  = encode_cube(cube)              # positional form
        outvec = (int(bit),)                    # single output
        cover.append((invec, outvec))

    # give BOTH F-type (1-rows) and R-type (0-rows)

    print("INTYPE BEING SENT:", WITHIN_F__TYPE | DONT_CARE_TYPE)
    minimized = espresso(n_vars, 1, cover, WITHIN_F__TYPE) # | DONT_CARE_TYPE)

    # Convert back to  string PLA format,
    # but keep only cubes with output bit  1
    rows: List[Tuple[str, str]] = []
    for invec, outvec in minimized:
        if outvec[0] != 1:        # ignore 0 (OFF) rows
            continue
        tok = lambda v: "-" if v == 2 else str(v)
        cube_str = " ".join(tok(v) for v in invec)
        rows.append((cube_str, "1"))
    return rows



def reformulate_rules_with_espresso(rules: Rules,
                                    max_tuples: int = 1_000_000) -> Rules:
    g_bounds = global_bounds(rules)
    new_rules: Rules = []
    for cls in rules:
        new_rules.append(
            reformulate_class_with_espresso(cls, g_bounds, max_tuples)
        )
    return new_rules