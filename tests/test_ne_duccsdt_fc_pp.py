import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_ne_duccsdt_fc_pp():
    mol = pyscf.gto.M(atom="Ne", basis="cc-pvdz", symmetry="d2h")
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cc = SparseCC(mf, verbose=5, cc_type="ducc", frozen_core=1)

    cc.make_cluster_operator(max_exc=3, pp=True)
    cc.kernel()
    assert np.isclose(cc.e_cc, -128.678505307688, atol=1e-8)


if __name__ == "__main__":
    test_ne_duccsdt_fc_pp()
