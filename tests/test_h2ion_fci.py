import os
import sys

import pyscf.mcscf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2ion_fci():
    mol = pyscf.gto.M(
        atom="""
    H
    H 1 1.1""",
        basis="sto-6g",
        symmetry="d2h",
        charge=1,
        spin=1,
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    ci = SparseCI(mf, 2, (1, 0), verbose=2)
    e, _ = ci.kernel(solver="eigh")
    assert np.isclose(e[0], -0.5859588012891606, atol=1e-8)


if __name__ == "__main__":
    test_h2ion_fci()
