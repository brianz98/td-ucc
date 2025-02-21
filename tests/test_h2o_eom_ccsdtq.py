import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_eom_ccsdtq():
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

    cc.make_cluster_operator(max_exc=4)
    cc.kernel()
    e, _ = cc.run_eom(nhole=4, npart=4, ms2=0, print_eigvals=True)
    assert np.isclose(e[0] - cc.e_cc, 0.273489391393, atol=1e-8)
    assert np.isclose(e[1] - cc.e_cc, 0.320716367276, atol=1e-8)
    assert np.isclose(e[2] - cc.e_cc, 0.356073574550, atol=1e-8)
    assert np.isclose(e[3] - cc.e_cc, 0.363868969265, atol=1e-8)
    assert np.isclose(e[4] - cc.e_cc, 0.390243577551, atol=1e-8)


if __name__ == "__main__":
    test_h2o_eom_ccsdtq()
