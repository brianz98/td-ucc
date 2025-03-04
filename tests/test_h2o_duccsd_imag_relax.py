import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_duccsd_imag_relax():
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
    cc.imag_time_relaxation(dt=0.02, maxiter=5000, e_conv=1e-12)
    assert np.isclose(cc.e_corr, -0.071355412429, atol=1e-8)


if __name__ == "__main__":
    test_h2o_duccsd_imag_relax()
