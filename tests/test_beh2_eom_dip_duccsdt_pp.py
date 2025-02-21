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
    cc = SparseCC(mf, verbose=2, cc_type="ducc")

    cc.make_cluster_operator(max_exc=3, pair=3)
    cc.kernel()
    e, _ = cc.run_eom(nhole=3, npart=1, ms2=0, print_eigvals=True)
    assert np.isclose(e[0] - cc.e_cc, 1.163161418271, atol=1e-8)
    assert np.isclose(e[1] - cc.e_cc, 1.163836737955, atol=1e-8)
    assert np.isclose(e[2] - cc.e_cc, 1.476350236104, atol=1e-8)
    assert np.isclose(e[3] - cc.e_cc, 1.478840007796, atol=1e-8)
    assert np.isclose(e[4] - cc.e_cc, 1.631808665124, atol=1e-8)


if __name__ == "__main__":
    test_beh2_eom_dip_duccsdt_pp()
