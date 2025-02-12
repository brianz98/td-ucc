import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_h2o_eom_ee_duccsd():
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
    sparse_cc.run_eom(nhole=2, npart=2, ms2=0, print_eigvals=True)
    assert np.isclose(sparse_cc.eom_eigval[0], 0.274374987352, atol=1e-8)
    assert np.isclose(sparse_cc.eom_eigval[1], 0.321110755687, atol=1e-8)
    assert np.isclose(sparse_cc.eom_eigval[2], 0.357311727862, atol=1e-8)
    assert np.isclose(sparse_cc.eom_eigval[3], 0.363994210503, atol=1e-8)
    assert np.isclose(sparse_cc.eom_eigval[4], 0.390211952757, atol=1e-8)


if __name__ == "__main__":
    test_h2o_eom_ee_duccsd()
