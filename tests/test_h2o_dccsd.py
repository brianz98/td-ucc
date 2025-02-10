import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_duccsd():
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
    sparse_cc = SparseCC(mf, verbose=5, cc_type="ducc")

    sparse_cc.make_cluster_operator(max_exc=2)
    sparse_cc.kernel()
    assert np.isclose(sparse_cc.energy, -75.728060006019, atol=1e-8)


if __name__ == "__main__":
    test_h2o_duccsd()
