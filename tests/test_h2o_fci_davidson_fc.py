import os
import sys
import pytest
import pyscf.mcscf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *

def test_h2o_fci_davidson_fc():
    mol = pyscf.gto.M(
        atom="""
        O
        H 1 1.1
        H 1 1.1 2 104""",
        basis="6-31g",
        symmetry="c2v",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    ci = SparseCI(mf, 12, 8, verbose=5)
    e, _ = ci.kernel(
        nroots=1,
        solver="davidson",
        guess_method="pspace",
        pspace_dim=500,
        verbose=5,
        max_cycle=100,
        tol=1e-8,
    )
    assert np.isclose(e, -76.10335034184608, atol=1e-8)


if __name__ == "__main__":
    test_h2o_fci_davidson_fc()
