import os
import sys

import pyscf.mcscf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_n2_casci66_davidson():
    mol = pyscf.gto.M(
        atom="""
    N
    N 1 1.1""",
        basis="6-31g",
        symmetry="d2h",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    ci = SparseCI(mf, 6, 6, verbose=2)
    e, _ = ci.kernel(solver="davidson", nroots=1)
    assert np.isclose(e[0] - ci.e_ref, -0.07939231694, atol=1e-8)


if __name__ == "__main__":
    test_n2_casci66_davidson()
