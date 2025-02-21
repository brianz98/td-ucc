import forte, forte.utils
import itertools
import numpy as np
import functools
import math


def sym_dir_prod(sym_list):
    if len(sym_list) == 0:
        return 0
    return functools.reduce(lambda x, y: x ^ y, sym_list)


def make_tn_operator(exc, eps, orbs, orbsym, root_sym, pair=None, general=False):
    if general:
        occ = orbs[0] + orbs[1]
        vir = orbs[0] + orbs[1]
    else:
        occ = orbs[0]
        vir = orbs[1]
    op = forte.SparseOperatorList()
    denom = []

    # loop over beta excitation level
    for nb in range(exc + 1):
        na = exc - nb
        # loop over alpha occupied
        for ao in itertools.combinations(occ, na):
            ao_sym = sym_dir_prod(orbsym[list(ao)])
            # loop over alpha virtual
            for av in itertools.combinations(vir, na):
                av_sym = sym_dir_prod(orbsym[list(av)])
                # loop over beta occupied
                for bo in itertools.combinations(occ, nb):
                    bo_sym = sym_dir_prod(orbsym[list(bo)])
                    # loop over beta virtual
                    for bv in itertools.combinations(vir, nb):
                        bv_sym = sym_dir_prod(orbsym[list(bv)])
                        if ao_sym ^ av_sym ^ bo_sym ^ bv_sym != root_sym:
                            continue
                        if pair is not None and exc >= pair:
                            # definition of 'pair excitation':
                            # iiaa for doubles (1 unique)
                            # iijaab for triples (2 unique)
                            # iijjaabb for quadruples (2 unique)
                            # etc..
                            if len(set(ao + bo)) != math.ceil(exc / 2) or len(
                                set(av + bv)
                            ) != math.ceil(exc / 2):
                                continue
                        # compute the denominators
                        e_aocc = functools.reduce(lambda x, y: x + eps[y], ao, 0.0)
                        e_avir = functools.reduce(lambda x, y: x + eps[y], av, 0.0)
                        e_bocc = functools.reduce(lambda x, y: x + eps[y], bo, 0.0)
                        e_bvir = functools.reduce(lambda x, y: x + eps[y], bv, 0.0)
                        denom.append(e_aocc + e_bocc - e_bvir - e_avir)
                        op_str = []
                        for i in av:
                            op_str.append(f"{i}a+")
                        for i in bv:
                            op_str.append(f"{i}b+")
                        for i in reversed(bo):
                            op_str.append(f"{i}b-")
                        for i in reversed(ao):
                            op_str.append(f"{i}a-")
                        op.add(f"{' '.join(op_str)}", 0.0)
    return op, denom
