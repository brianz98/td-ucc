import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_beh2_ccsdtq_pp():
    mol = pyscf.gto.M(
        atom="""
    Be 0.0     0.0     0.0
    H  0   1.310011  0.0
    H  0   -1.310011  0.0""",
        basis="beh.nw",
        symmetry="c2v",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cc = SparseCC(mf, verbose=2, cc_type="cc")

    cc.make_cluster_operator(max_exc=4, pp=True)
    cc.kernel()
    assert np.isclose(cc.e_corr, -0.057165169573, atol=1e-8)


if __name__ == "__main__":
    test_beh2_ccsdtq_pp()
