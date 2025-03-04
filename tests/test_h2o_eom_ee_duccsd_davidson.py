import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_eom_ee_duccsd_davidson():
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
    cc = SparseCC(mf, verbose=2, cc_type="ducc")

    cc.make_cluster_operator(max_exc=2)
    cc.kernel()
    e, c = cc.run_eom(
        nhole=2,
        npart=2,
        ms2=0,
        print_eigvals=True,
        solver="davidson",
        guess_method="eye",
        max_cycle=100,
        nroots=6,
    )
    assert np.isclose(e[0] - cc.e_cc, 0.274357896910, atol=1e-8)
    assert np.isclose(e[1] - cc.e_cc, 0.321080833292, atol=1e-8)
    assert np.isclose(e[2] - cc.e_cc, 0.357318814370, atol=1e-8)
    assert np.isclose(e[3] - cc.e_cc, 0.364014821771, atol=1e-8)
    assert np.isclose(e[4] - cc.e_cc, 0.390220287852, atol=1e-8)
    assert np.isclose(e[5] - cc.e_cc, 0.422972981800, atol=1e-8)


if __name__ == "__main__":
    test_h2o_eom_ee_duccsd_davidson()
