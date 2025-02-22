import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from sparse_cc import *


def td_pert(t):
    emax = 0.07
    omega = 0.1
    et = np.array([0, 0, emax])
    if t < 0 or t > 6 * np.pi / omega:
        return np.array([0, 0, 0])
    elif t < 2 * np.pi / omega:
        return omega * t / (2 * np.pi) * np.sin(omega * t) * et
    elif t < 4 * np.pi / omega:
        return et * np.sin(omega * t)
    elif t < 6 * np.pi / omega:
        return (3 - omega * t / (2 * np.pi)) * et * np.sin(omega * t)


def test_h2_td_eom_duccsd_props():
    mol = pyscf.gto.M(
        atom="""
    H 0 0 0
    H 0 0 0.7354""",
        basis="6-311++g(d,p)",
        symmetry="c1",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    cc = SparseCC(mf, verbose=2, cc_type="ducc")
    cc.make_dipole_operator()
    cc.make_cluster_operator(max_exc=2)
    cc.kernel()
    dets = cc.make_eom_basis(2, 2, ms2=0, first=True)
    propfuncs = [
        lambda state: cc.get_occ(state, 0),
        lambda state: cc.get_dipole_z(state),
    ]
    c0 = np.zeros(len(dets), dtype=np.complex128)
    c0[0] = 1
    prop = cc.kernel_td_eom(c0, dets, td_pert, 10, propfuncs, max_step=0.01)

    assert np.isclose(np.real(prop[-1, 1]), 1.966349110, atol=1e-8)
    assert np.isclose(np.imag(prop[-1, 1]), 0.0, atol=1e-12)
    assert np.isclose(np.real(prop[-1, 2]), 0.06028912, atol=1e-8)
    assert np.isclose(np.imag(prop[-1, 2]), 0.0, atol=1e-12)


if __name__ == "__main__":
    test_h2_td_eom_duccsd_props()
