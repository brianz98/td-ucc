import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_ne_duccsd_fc():
    mol = pyscf.gto.M(atom="Ne", basis="cc-pvdz", symmetry="d2h")
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    sparse_cc = SparseCC(mf, verbose=5, cc_type="ducc", frozen_core=1)

    sparse_cc.make_cluster_operator(max_exc=2)
    sparse_cc.kernel()
    assert np.isclose(sparse_cc.energy, -128.677997135198, atol=1e-8)


if __name__ == "__main__":
    test_ne_duccsd_fc()
