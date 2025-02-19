import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2_duccsd_nosym():
    mol = pyscf.gto.M(
        atom="""
    H 0 0 0
    H 0 0 0.7354""",
        basis="6-31++g(d,p)",
        symmetry="c1",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cc = SparseCC(mf, verbose=2, cc_type="ducc")

    cc.make_cluster_operator(max_exc=2)
    cc.kernel()
    assert np.isclose(cc.e_corr, -0.033875238656, atol=1e-8)


if __name__ == "__main__":
    test_h2_duccsd_nosym()
