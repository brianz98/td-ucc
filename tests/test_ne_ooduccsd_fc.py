import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def test_ne_ooduccsd_fc():
    mol = pyscf.gto.M(atom="Ne", basis="cc-pvdz", symmetry="d2h")
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    cc = SparseCC(mf, verbose=2, cc_type="ducc", frozen_core=1)

    cc.make_cluster_operator(max_exc=2)
    cc.kernel(do_oo=True)
    assert np.isclose(cc.e_corr, -0.189228115951, atol=1e-8)


if __name__ == "__main__":
    test_ne_ooduccsd_fc()
