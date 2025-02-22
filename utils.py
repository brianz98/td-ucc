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


def enumerate_determinants(
    nael, nbel, orbs, orbsym, nhole, npart, root_sym, ms2, core=None
):
    occ, vir = orbs
    states = []

    for ah in range(nhole + 1):
        bh = nhole - ah
        for ap in range(npart + 1):
            bp = npart - ap
            if (ah - ap) - (bh - bp) != ms2:
                continue
            if any([ah > nael, bh > nbel]):
                continue
            for ao in itertools.combinations(occ, nael - ah):
                ao_sym = sym_dir_prod(orbsym[list(ao)])
                for av in itertools.combinations(vir, ap):
                    av_sym = sym_dir_prod(orbsym[list(av)])
                    for bo in itertools.combinations(occ, nbel - bh):
                        bo_sym = sym_dir_prod(orbsym[list(bo)])
                        for bv in itertools.combinations(vir, bp):
                            bv_sym = sym_dir_prod(orbsym[list(bv)])
                            if ao_sym ^ av_sym ^ bo_sym ^ bv_sym != root_sym and root_sym != -1:
                                continue
                            if core is not None:
                                # no core orbital can be doubly occupied
                                if any([(ao+bo).count(i) == 2 for i in core]):
                                    continue
                            d = forte.Determinant()
                            for i in ao:
                                d.set_alfa_bit(i, True)
                            for i in av:
                                d.set_alfa_bit(i, True)
                            for i in bo:
                                d.set_beta_bit(i, True)
                            for i in bv:
                                d.set_beta_bit(i, True)
                            states.append(d)
    return states
