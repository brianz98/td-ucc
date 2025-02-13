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
    sparse_cc = SparseCC(mf, verbose=5, cc_type="ducc")

    sparse_cc.make_cluster_operator(max_exc=2)
    sparse_cc.kernel()
    e,c = sparse_cc.run_eom(nhole=2, npart=2, ms2=0, print_eigvals=True, solver="davidson", guess_method="eye", max_cycle=100, nroots=6)
    assert np.isclose(e[0], -75.453758387522, atol=1e-8)
    assert np.isclose(e[1], -75.407022619187, atol=1e-8)
    assert np.isclose(e[2], -75.370821647008, atol=1e-8)
    assert np.isclose(e[3], -75.364139164375, atol=1e-8)
    assert np.isclose(e[4], -75.337921422111, atol=1e-8)
    assert np.isclose(e[5], -75.305162475278, atol=1e-8)

if __name__ == "__main__":
    test_h2o_eom_ee_duccsd_davidson()
