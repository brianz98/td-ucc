import itertools
import functools
import time
import numpy as np
import forte, forte.utils
import wicked as w
import pyscf, pyscf.cc, pyscf.mp, pyscf.fci
from collections import deque
import scipy, scipy.linalg

MINIMAL_PRINT_LEVEL = 1
NORMAL_PRINT_LEVEL = 2
DEBUG_PRINT_LEVEL = 3


def sym_dir_prod(sym_list):
    if len(sym_list) == 0:
        return 0
    return functools.reduce(lambda x, y: x ^ y, sym_list)


def power_method(ham_op, psi_0, iter):
    psi_old = psi_0
    for i in range(iter):
        psi_new = forte.apply_op(ham_op, psi_old)
        energy = forte.overlap(psi_old, psi_new)
        print(f"Iteration {i+1}: Energy = {energy}")
        psi_old = forte.normalize(psi_new)


class SparseCC:
    def __init__(
        self,
        mf,
        verbose=False,
        cc_type="cc",
        root_sym=0,
        exp_max_k=19,
        exp_screen_thresh=1e-12,
        ham_screen_thresh=1e-12,
        frozen_core=0,
    ):
        # get the number of MOs and alpha/beta electrons per irrep
        if cc_type not in ["cc", "ucc", "dcc", "ducc"]:
            raise RuntimeError("Invalid CC type")
        if not isinstance(frozen_core, int):
            raise RuntimeError("frozen_core must be an integer")
        self.mf = mf
        self.verbose = verbose
        self.mol = mf.mol
        self.factorized = "d" in cc_type
        self.unitary = "u" in cc_type
        self.root_sym = root_sym

        self.nmo = self.mol.nao - frozen_core
        self.nael = self.mol.nelec[0] - frozen_core
        self.nbel = self.mol.nelec[1] - frozen_core
        self.eps = self.mf.mo_energy[frozen_core:]

        self.point_group = self.mol.groupname
        if self.point_group != "C1":
            self.orbsym = self.mf.mo_coeff.orbsym[frozen_core:]
        else:
            self.orbsym = [0] * self.nmo
        self.occ = list(range(self.nael))
        self.vir = list(range(self.nael, self.nmo))

        if self.factorized:
            self.exp_op = forte.SparseFactExp(screen_thresh=exp_screen_thresh)
        else:
            self.exp_op = forte.SparseExp(
                maxk=exp_max_k, screen_thresh=exp_screen_thresh
            )
        if self.unitary:
            self.exp_apply_op = self.exp_op.apply_antiherm
        else:
            self.exp_apply_op = self.exp_op.apply_op
        self.ham_screen_thresh = ham_screen_thresh

        if self.nael != self.nbel:
            raise RuntimeError(
                "The number of alpha and beta electrons must be the same"
            )

        # Specify the occupation of the the Hartree–Fock determinant
        hfref = forte.Determinant()
        # set the occupation numbers of the determinant
        for i in range(self.nael):
            hfref.set_alfa_bit(i, True)
        for i in range(self.nbel):
            hfref.set_beta_bit(i, True)

        if self.verbose >= DEBUG_PRINT_LEVEL:
            print(f"Number of orbitals:        {self.nmo}")
            print(f"Number of alpha electrons: {self.nael}")
            print(f"Number of beta electrons:  {self.nbel}")
            print(f"Reference determinant:     {hfref.str(self.nmo)}")

        self.make_hamiltonian(frozen_core=frozen_core)

        self.ref = forte.SparseState({hfref: 1.0})
        eref = forte.overlap(self.ref, forte.apply_op(self.ham_op, self.ref))
        assert np.isclose(
            eref, mf.e_tot
        ), f"Reference energy mismatch: {eref} != {mf.e_tot}"

    def make_hamiltonian(self, frozen_core=None):
        scalar = self.mol.energy_nuc()
        hcore = np.einsum(
            "pi,qj,pq->ij", self.mf.mo_coeff, self.mf.mo_coeff, self.mf.get_hcore()
        )
        eri = pyscf.ao2mo.full(self.mol.intor("int2e"), self.mf.mo_coeff)

        if frozen_core > 0:
            fc = slice(0, frozen_core)
            av = slice(frozen_core, None)
            scalar += 2 * np.einsum("ii", hcore[fc, fc])
            scalar += 2 * np.einsum("iijj->", eri[fc, fc, fc, fc])
            scalar -= np.einsum("ijji->", eri[fc, fc, fc, fc])
            hcore[av, av] += 2 * np.einsum("pqjj->pq", eri[av, av, fc, fc])
            hcore[av, av] -= np.einsum("pjjq->pq", eri[av, fc, fc, av])
            hcore = hcore[av, av].copy()
            eri = eri[av, av, av, av].copy()

        oei_a = forte.ndarray_from_numpy(hcore)

        tei = eri.swapaxes(1, 2).copy()
        tei_aa = forte.ndarray_copy_from_numpy(tei - tei.swapaxes(2, 3))
        tei_ab = forte.ndarray_from_numpy(tei)

        self.ham_op = forte.sparse_operator_hamiltonian(
            scalar, oei_a, oei_a, tei_aa, tei_ab, tei_aa
        )

    def make_cluster_operator(self, max_exc, pp=True):
        # Prepare the cluster operator (closed-shell case)
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print(f"Occupied orbitals: {self.occ}")
            print(f"Virtual orbitals:  {self.vir}")

        self.op = forte.SparseOperatorList()
        self.denominators = []

        # loop over total excitation level
        for n in range(1, max_exc + 1):
            # loop over beta excitation level
            for nb in range(n + 1):
                na = n - nb
                # loop over alpha occupied
                for ao in itertools.combinations(self.occ, na):
                    ao_sym = sym_dir_prod(self.orbsym[list(ao)])
                    # loop over alpha virtual
                    for av in itertools.combinations(self.vir, na):
                        av_sym = sym_dir_prod(self.orbsym[list(av)])
                        # loop over beta occupied
                        for bo in itertools.combinations(self.occ, nb):
                            bo_sym = sym_dir_prod(self.orbsym[list(bo)])
                            # loop over beta virtual
                            for bv in itertools.combinations(self.vir, nb):
                                bv_sym = sym_dir_prod(self.orbsym[list(bv)])
                                if ao_sym ^ av_sym ^ bo_sym ^ bv_sym != self.root_sym:
                                    continue
                                if n >= 3 and pp:
                                    if len(set(ao + bo)) != 2 or len(set(av + bv)) != 2:
                                        continue
                                # compute the denominators
                                e_aocc = functools.reduce(
                                    lambda x, y: x + self.eps[y], ao, 0.0
                                )
                                e_avir = functools.reduce(
                                    lambda x, y: x + self.eps[y], av, 0.0
                                )
                                e_bocc = functools.reduce(
                                    lambda x, y: x + self.eps[y], bo, 0.0
                                )
                                e_bvir = functools.reduce(
                                    lambda x, y: x + self.eps[y], bv, 0.0
                                )
                                self.denominators.append(
                                    e_aocc + e_bocc - e_bvir - e_avir
                                )
                                op_str = []  # a list to hold the operator triplets
                                for i in av:
                                    op_str.append(f"{i}a+")
                                for i in bv:
                                    op_str.append(f"{i}b+")
                                for i in reversed(bo):
                                    op_str.append(f"{i}b-")
                                for i in reversed(ao):
                                    op_str.append(f"{i}a-")
                                self.op.add(f"{' '.join(op_str)}", 0.0)

        if self.verbose >= DEBUG_PRINT_LEVEL:
            print(f"\n==> Cluster operator <==")
            print(f"Number of amplitudes: {len(self.op)}")
            print(f"Operator components:")
            for sqop, c in self.op:
                print(f"{sqop}")

    def cc_residual_equations(self):
        # Step 1. Compute exp(T)|Phi>
        if self.factorized:
            wfn = self.exp_apply_op(self.op, self.ref, inverse=False)
        else:
            wfn = self.exp_apply_op(self.op, self.ref, scaling_factor=1.0)

        # Step 2. Compute H exp(T)|Phi>
        Hwfn = forte.apply_op(self.ham_op, wfn, screen_thresh=self.ham_screen_thresh)

        # Step 3. Compute exp(-T) H exp(T)|Phi>
        if self.factorized:
            R = self.exp_apply_op(self.op, Hwfn, inverse=True)
        else:
            R = self.exp_apply_op(self.op, Hwfn, scaling_factor=-1.0)

        # Step 4. Project residual onto excited determinants
        self.residual = forte.get_projection(self.op, self.ref, R)

        self.energy = 0.0
        for d, c in self.ref.items():
            self.energy += c * R[d]
        self.energy = self.energy.real

    def update_amps(self, diis):
        t = self.op.coefficients()
        if diis.use_diis:
            t_old = np.array(t).copy()
        # update the amplitudes
        for i in range(len(self.op)):
            t[i] += self.residual[i] / self.denominators[i]
        # push new amplitudes to the T operator
        if diis.use_diis:
            t_new = np.array(t).copy()
            diis.add_vector(t_new, t_new - t_old)
            self.op.set_coefficients(diis.compute())
        else:
            self.op.set_coefficients(t)

    def kernel(self, **kwargs):
        e_conv = kwargs.get("e_conv", 1e-12)
        maxiter = kwargs.get("maxiter", 100)
        diis_nvecs = kwargs.get("diis_nvecs", 6)
        diis_start = kwargs.get("diis_start", 2)

        diis = DIIS(diis_nvecs, diis_start)

        start = time.time()

        # initialize T = 0
        t = [0.0] * len(self.op)
        self.op.set_coefficients(t)

        # initalize E = 0
        old_e = 0.0

        if self.verbose >= NORMAL_PRINT_LEVEL:
            print("=" * 65)
            print(f"{'Iter':<9} {'Energy':<20} {'Delta Energy':<20} {'Time':<11}")
            print("-" * 65)

        for iter in range(maxiter):
            iterstart = time.time()
            # 1. evaluate the CC residual equations
            self.cc_residual_equations()

            # 2. update the CC equations
            self.update_amps(diis)

            iterend = time.time()

            # 3. print information
            if self.verbose >= NORMAL_PRINT_LEVEL:
                print(
                    f"{iter:<9d} {self.energy:<20.12f} {self.energy - old_e:<20.12f} {iterend-iterstart:<11.3f}"
                )

            # 4. check for convergence of the energy
            if abs(self.energy - old_e) < e_conv:
                break
            old_e = self.energy

        if iter == maxiter - 1:
            print("Warning: CC iterations did not converge!")
        self.e_corr = self.energy - self.mf.e_tot
        if self.verbose >= NORMAL_PRINT_LEVEL:
            print("=" * 65)
            print(f" Total time: {time.time()-start:11.3f} [s]")

        if self.verbose >= MINIMAL_PRINT_LEVEL:
            print(f" CC energy:             {self.energy:20.12f} [Eh]")
            print(f" CC correlation energy: {self.e_corr:20.12f} [Eh]")

    def evaluate_grad_ovlp(self):
        # Evaluates <Psi_u|exp(-S) d/dt exp(S)|Psi_0> = <Phi_u|U_v+ s_v U_v|Phi_0>
        # U_v = Prod_{i=v}^{N} exp(s_i * t_i(t))
        grad = np.zeros((len(self.op), len(self.op)), dtype=np.complex128)
        for nu in reversed(range(len(self.op))):
            op_nu = self.op(nu)
            kappa_nu = forte.SparseOperator()
            kappa_nu.add(op_nu[0], 1.0)
            if self.unitary:
                kappa_nu.add(op_nu[0].adjoint(), -1.0)

            u_nu = forte.SparseOperatorList()
            u_nu.add(op_nu[0], op_nu[1])

            if nu == len(self.op) - 1:
                u_psi = self.exp_apply_op(u_nu, self.ref, inverse=False)
            else:
                u_psi = self.exp_apply_op(u_nu, u_psi, inverse=False)
            k_u_psi = forte.apply_op(kappa_nu, u_psi)
            u_k_u_psi = self.exp_apply_op(u_nu, k_u_psi, inverse=True)
            grad[:, nu] = forte.get_projection(self.op, self.ref, u_k_u_psi)
        return grad

    def update_amps_imag_time(self, dt):
        t = self.op.coefficients()
        grad = self.evaluate_grad_ovlp()
        delta = np.linalg.solve(grad, self.residual)
        # update the amplitudes
        for i in range(len(self.op)):
            t[i] -= delta[i] * dt
        # push new amplitudes to the T operator
        self.op.set_coefficients(t)

    def imag_time_relaxation(self, dt=0.005, **kwargs):
        e_conv = kwargs.get("e_conv", 1e-8)
        maxiter = kwargs.get("maxiter", 1000)
        start = time.time()

        # initialize T = 0
        t = [0.0] * len(self.op)
        self.op.set_coefficients(t)

        # initalize E = 0
        old_e = 0.0

        if self.verbose >= NORMAL_PRINT_LEVEL:
            print("=" * 80)
            print(
                f"{'Iter':^5} {'Energy':^20} {'Delta Energy':^20} {'Imag Time':^15} {'Time':^15}"
            )
            print("-" * 80)

        for iter in range(maxiter):
            iterstart = time.time()
            # 1. evaluate the CC residual equations
            self.cc_residual_equations()

            # 2. update the CC equations
            self.update_amps_imag_time(dt)

            iterend = time.time()

            # 3. print information
            if self.verbose >= NORMAL_PRINT_LEVEL:
                print(
                    f"{iter:<5d} {self.energy:<20.12f} {self.energy - old_e:<20.12f} {iter*dt:<15.3f} {iterend-iterstart:<15.3f}"
                )

            # 4. check for convergence of the energy
            if abs(self.energy - old_e) < e_conv:
                break
            old_e = self.energy

        if iter == maxiter - 1:
            print("Warning: CC iterations did not converge!")
        self.e_corr = self.energy - self.mf.e_tot
        if self.verbose >= NORMAL_PRINT_LEVEL:
            print("=" * 80)
            print(f" Total time: {time.time()-start:11.3f} [s]")

        if self.verbose >= MINIMAL_PRINT_LEVEL:
            print(f" CC energy:             {self.energy:20.12f} [Eh]")
            print(f" CC correlation energy: {self.e_corr:20.12f} [Eh]")

    def make_ee_eom_basis(self, max_exc):
        # EE-EOMCC
        # R = [1, t_i^a, t_ij^ab, ...]
        _ee_eom_basis = [self.ref]  # Reference determinant

        for k in range(1, max_exc + 1):  # k is the excitation level
            for ak in range(k + 1):  # alpha excitation level
                bk = k - ak
                for ao in itertools.combinations(self.occ, self.nael - ak):
                    ao_sym = sym_dir_prod(self.orbsym[list(ao)])
                    for av in itertools.combinations(self.vir, ak):
                        av_sym = sym_dir_prod(self.orbsym[list(av)])
                        for bo in itertools.combinations(self.occ, self.nbel - bk):
                            bo_sym = sym_dir_prod(self.orbsym[list(bo)])
                            for bv in itertools.combinations(self.vir, bk):
                                bv_sym = sym_dir_prod(self.orbsym[list(bv)])
                                if ao_sym ^ av_sym ^ bo_sym ^ bv_sym != self.root_sym:
                                    continue
                                d = forte.Determinant()
                                for i in ao:
                                    d.set_alfa_bit(i, True)
                                for i in av:
                                    d.set_alfa_bit(i, True)
                                for i in bo:
                                    d.set_beta_bit(i, True)
                                for i in bv:
                                    d.set_beta_bit(i, True)
                                _ee_eom_basis.append(forte.SparseState({d: 1.0}))

        print(f"Number of EE-EOM basis states: {len(_ee_eom_basis)}")

        return _ee_eom_basis

    def make_ip_eom_basis(self, max_exc):
        # IP-EOMCC
        # R = [t_{i}^{}, t_{ij}^{a}, ...]

        _ip_eom_basis = []

        for k in range(1, max_exc + 1):  # k is the excitation level
            j = k - 1  # number of creation operators
            for ak in range(k + 1):  # alpha excitation level
                bk = k - ak  # beta excitation level
                for aj in range(j + 1):
                    bj = j - aj
                    for ao in itertools.combinations(self.occ, self.nael - ak):
                        ao_sym = sym_dir_prod(self.orbsym[list(ao)])
                        for av in itertools.combinations(self.vir, aj):
                            av_sym = sym_dir_prod(self.orbsym[list(av)])
                            for bo in itertools.combinations(self.occ, self.nbel - bk):
                                bo_sym = sym_dir_prod(self.orbsym[list(bo)])
                                for bv in itertools.combinations(self.vir, bj):
                                    bv_sym = sym_dir_prod(self.orbsym[list(bv)])
                                    if (
                                        ao_sym ^ av_sym ^ bo_sym ^ bv_sym
                                        != self.root_sym
                                    ):
                                        continue
                                    d = forte.Determinant()
                                    for i in ao:
                                        d.set_alfa_bit(i, True)
                                    for i in av:
                                        d.set_alfa_bit(i, True)
                                    for i in bo:
                                        d.set_beta_bit(i, True)
                                    for i in bv:
                                        d.set_beta_bit(i, True)
                                    _ip_eom_basis.append(forte.SparseState({d: 1.0}))

        return _ip_eom_basis

    def make_ea_eom_basis(self, max_exc):
        # EA-EOM-CC
        # R = [t_{}^{a}, t_{i}^{ab}, ...]

        _ea_eom_basis = []

        for k in range(
            1, max_exc + 1
        ):  # k is the creation level (number of creation operators)
            j = k - 1  # number of annihilation operators
            for ak in range(k + 1):  # alpha creation level
                bk = k - ak  # beta creation level
                for aj in range(j + 1):  # alpha annihilation level
                    bj = j - aj  # beta annihilation level
                    for ao in itertools.combinations(self.occ, self.nael - aj):
                        ao_sym = sym_dir_prod(self.orbsym[list(ao)])
                        for av in itertools.combinations(self.vir, ak):
                            av_sym = sym_dir_prod(self.orbsym[list(av)])
                            for bo in itertools.combinations(self.occ, self.nbel - bj):
                                bo_sym = sym_dir_prod(self.orbsym[list(bo)])
                                for bv in itertools.combinations(self.vir, bk):
                                    bv_sym = sym_dir_prod(self.orbsym[list(bv)])
                                    if (
                                        ao_sym ^ av_sym ^ bo_sym ^ bv_sym
                                        != self.root_sym
                                    ):
                                        continue
                                    d = forte.Determinant()
                                    for i in ao:
                                        d.set_alfa_bit(i, True)
                                    for i in av:
                                        d.set_alfa_bit(i, True)
                                    for i in bo:
                                        d.set_beta_bit(i, True)
                                    for i in bv:
                                        d.set_beta_bit(i, True)
                                    _ea_eom_basis.append(forte.SparseState({d: 1.0}))

        return _ea_eom_basis

    def make_hbar(self, dets):
        H = np.zeros((len(dets), len(dets)), dtype=np.complex128)
        algo = "oprod" if self.unitary else "naive"

        if algo == "naive":
            for j in range(len(dets)):
                # exp(S)|j>
                wfn = self.exp_apply_op(
                    self.op,
                    dets[j],
                    scaling_factor=1.0,
                )
                # H exp(S)|j>
                Hwfn = forte.apply_op(self.ham_op, wfn, self.ham_screen_thresh)
                # exp(-S) H exp(S)|j>
                R = self.exp_apply_op(
                    self.op,
                    Hwfn,
                    scaling_factor=-1.0,
                )
                for i in range(j, len(dets)):
                    # <i|exp(-S) H exp(S)|j>
                    H[i, j] = forte.overlap(dets[i], R)
                    H[j, i] = H[i, j]
        elif algo == "oprod":
            _wfn_list = []
            _Hwfn_list = []

            for i in range(len(dets)):
                wfn = self.exp_apply_op(
                    self.op,
                    dets[i],
                    scaling_factor=1.0,
                )
                Hwfn = forte.apply_op(self.ham_op, wfn)
                _wfn_list.append(wfn)
                _Hwfn_list.append(Hwfn)

            for i in range(len(dets)):
                for j in range(len(dets)):
                    H[i, j] = forte.overlap(_wfn_list[i], _Hwfn_list[j])
                    H[j, i] = H[i, j]

        return H

    def run_eom(self, max_exc, mode, print_eigvals=True):
        if self.unitary:

            def _eig(H):
                eigval, eigvec = np.linalg.eigh(H)
                eigval -= eigval[0]
                eigval_argsort = np.array(range(len(eigval)))
                return eigval, eigvec, eigval_argsort

        else:

            def _eig(H):
                eigval, eigvec = scipy.linalg.eig(H)
                eigval_argsort = np.argsort(np.real(eigval))
                eigval -= eigval[eigval_argsort[0]]
                eigval = np.real(eigval)
                return eigval, eigvec, eigval_argsort

        if mode == "ip":
            self.eom_basis = self.make_ip_eom_basis(max_exc)
        elif mode == "ea":
            self.eom_basis = self.make_ea_eom_basis(max_exc)
        elif mode == "ee":
            self.eom_basis = self.make_ee_eom_basis(max_exc)

        self.s2 = np.zeros((len(self.eom_basis),) * 2)
        for i, ibasis in enumerate(self.eom_basis):
            for j, jbasis in enumerate(self.eom_basis):
                self.s2[i, j] = forte.spin2(
                    next(ibasis.items())[0], next(jbasis.items())[0]
                )

        self.eom_hbar = self.make_hbar(self.eom_basis)
        self.eom_eigval, self.eom_eigvec, self.eom_eigval_argsort = _eig(self.eom_hbar)

        if print_eigvals:
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
            print(f"{'#':^4} {'E_exc / Eh':^25} {'<S^2>':^10}  {'S':^5}")
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")
            for i in range(1, len(self.eom_eigval)):
                s2_val = (
                    self.eom_eigvec[:, self.eom_eigval_argsort[i]].T
                    @ self.s2
                    @ self.eom_eigvec[:, self.eom_eigval_argsort[i]]
                )
                s = round(2 * (-1 + np.sqrt(1 + 4 * s2_val)))
                s /= 4
                print(
                    f"{i:^4d} {self.eom_eigval[self.eom_eigval_argsort[i]]:^25.12f} {abs(s2_val):^10.3f} {abs(s):^5.1f}"
                )
            print(f"{'='*4:^4}={'='*25:^25}={'='*10:^10}={'='*5:^5}")


class DIIS:
    def __init__(self, nvecs=6, start=2):
        if nvecs == None or start == None:
            self.use_diis = False
        else:
            self.use_diis = True
            self.nvecs = nvecs
            self.start = start
            self.error = deque(maxlen=nvecs)
            self.vector = deque(maxlen=nvecs)

    def add_vector(self, vector, error):
        if self.use_diis == False:
            return
        self.vector.append(vector)
        self.error.append(error)

    def compute(self):
        if self.use_diis == False:
            return
        B = np.zeros((len(self.vector), len(self.vector)))
        for i in range(len(self.vector)):
            for j in range(len(self.vector)):
                B[i, j] = np.dot(self.error[i], self.error[j]).real
        A = np.zeros((len(self.vector) + 1, len(self.vector) + 1))
        A[:-1, :-1] = B
        A[-1, :] = -1
        A[:, -1] = -1
        b = np.zeros(len(self.vector) + 1)
        b[-1] = -1
        x = np.linalg.solve(A, b)

        new_vec = np.zeros_like(self.vector[0])
        for i in range(len(x) - 1):
            new_vec += x[i] * self.vector[i]

        return new_vec
