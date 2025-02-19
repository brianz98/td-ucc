import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_ccsd():
    mol = pyscf.gto.M(
        atom="""
    O
    H 1 1.1
    H 1 1.1 2 104""",
        basis="sto-6g",
        symmetry="c2v",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cc = SparseCC(mf, verbose=2, cc_type="cc")

    cc.make_cluster_operator(max_exc=2)
    cc.kernel()
    assert np.isclose(cc.e_corr, -0.071272110420, atol=1e-8)


if __name__ == "__main__":
    test_h2o_ccsd()
