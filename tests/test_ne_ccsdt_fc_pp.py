import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_ne_duccsdt_fc_pp():
    mol = pyscf.gto.M(atom="Ne", basis="6-31g", symmetry="c1")
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cc = SparseCC(mf, verbose=2, cc_type="cc", frozen_core=1)

    cc.make_cluster_operator(max_exc=3, pair=3)
    cc.kernel()
    # assert np.isclose(cc.e_corr, -0.189729548769, atol=1e-8)


if __name__ == "__main__":
    test_ne_duccsdt_fc_pp()
