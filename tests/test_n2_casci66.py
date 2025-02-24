import os
import sys

import pyscf.mcscf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_n2_casci66():
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
    e, _ = ci.kernel(solver="eigh", nroots=5)
    assert np.isclose(e[0] - ci.e_ref, -0.07939231694, atol=1e-8)
    assert np.isclose(e[1] - e[0], 0.580854586247, atol=1e-8)
    assert np.isclose(e[2] - e[0], 0.650080375482, atol=1e-8)
    assert np.isclose(e[3] - e[0], 0.653724853903, atol=1e-8)
    assert np.isclose(e[4] - e[0], 0.719683975872, atol=1e-8)



if __name__ == "__main__":
    test_n2_casci66()
