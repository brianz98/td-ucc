import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_eom_dip_4h2p_ccsd_davidson():
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
    cc = SparseCC(mf, verbose=5, cc_type="cc")

    cc.make_cluster_operator(max_exc=2)
    cc.kernel()
    e, _ = cc.run_eom(
        nhole=4,
        npart=2,
        ms2=0,
        print_eigvals=True,
        solver="davidson",
        nroots=4,
        max_cycle=500,
        guess_method="eye",
        verbose=100,
    )
    assert np.isclose(e[0] - cc.e_cc, 1.218152825496, atol=1e-8)
    assert np.isclose(e[1] - cc.e_cc, 1.281144406455, atol=1e-8)
    assert np.isclose(e[2] - cc.e_cc, 1.327918131752, atol=1e-8)
    assert np.isclose(e[3] - cc.e_cc, 1.358893337784, atol=1e-8)


if __name__ == "__main__":
    test_h2o_eom_dip_4h2p_ccsd_davidson()
