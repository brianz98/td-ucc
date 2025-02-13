import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_beh2_eom_dip_duccsdt_pp():
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
    sparse_cc = SparseCC(mf, verbose=5, cc_type="ducc")

    sparse_cc.make_cluster_operator(max_exc=3, pp=True)
    sparse_cc.kernel()
    e, _ = sparse_cc.run_eom(nhole=3, npart=1, ms2=0, print_eigvals=True)
    assert np.isclose(e[0] - sparse_cc.energy, 1.163166080181, atol=1e-8)
    assert np.isclose(e[1] - sparse_cc.energy, 1.163841778103, atol=1e-8)
    assert np.isclose(e[2] - sparse_cc.energy, 1.476338030622, atol=1e-8)
    assert np.isclose(e[3] - sparse_cc.energy, 1.478893425901, atol=1e-8)
    assert np.isclose(e[4] - sparse_cc.energy, 1.631801692044, atol=1e-8)


if __name__ == "__main__":
    test_beh2_eom_dip_duccsdt_pp()
