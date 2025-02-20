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


def test_h2_td_fci_props():
    mol = pyscf.gto.M(
        atom="""
    H 0 0 0
    H 0 0 0.7354""",
        basis="6-311++g(d,p)",
        symmetry="c1",
    )
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    ci = SparseCI(mf, 14, 2, verbose=10)
    ci.make_dipole_operator()
    e, c = ci.kernel(solver="eigh")
    dets = ci.enumerate_determinants(-1)
    propfuncs = [
        lambda state: ci.get_occ(state, 0),
        lambda state: ci.get_dipole_z(state),
    ]
    prop = ci.kernel_td(c[:, 0], dets, td_pert, 5, propfuncs, max_step=0.01)
    assert np.isclose(np.real(prop[-1, 1]), 1.96690835, atol=1e-8)
    assert np.isclose(np.imag(prop[-1, 1]), 0.0, atol=1e-12)
    assert np.isclose(np.real(prop[-1, 2]), 0.00826807, atol=1e-8)
    assert np.isclose(np.imag(prop[-1, 2]), 0.0, atol=1e-12)


if __name__ == "__main__":
    test_h2_td_fci_props()
