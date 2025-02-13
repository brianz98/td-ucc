import itertools
import functools
import time
import numpy as np
import forte, forte.utils
import pyscf, pyscf.cc, pyscf.mp, pyscf.fci
from pyscf.lib.linalg_helper import davidson1, davidson_nosym1
from collections import deque
import scipy, scipy.linalg, scipy.constants

EH_TO_EV = scipy.constants.value("Hartree energy in eV")
MINIMAL_PRINT_LEVEL = 1
NORMAL_PRINT_LEVEL = 2
DEBUG_PRINT_LEVEL = 3
NIRREP = {"c1": 1, "cs": 2, "ci": 2, "c2": 2, "c2v": 4, "c2h": 4, "d2": 4, "d2h": 8}


def sym_dir_prod(sym_list):
    if len(sym_list) == 0:
        return 0
    return functools.reduce(lambda x, y: x ^ y, sym_list)


class SparseBase:
    def __init__(
        self, mf, verbose=False, root_sym=0, mo_space=None, ham_screen_thresh=1e-12
    ):
        self.mf = mf
        self.mol = mf.mol
        self.root_sym = root_sym
        self.verbose = verbose

        if mo_space is None:
            mo_space = {}
            mo_space["frzn"] = slice(0, 0)
            mo_space["corr"] = slice(0, self.mol.nao)
        self.frzn = mo_space["frzn"]
        self.corr = mo_space["corr"]
        self.nfrzn = self.frzn.stop
        self.ncorr = self.corr.stop - self.nfrzn

        self.nmo = self.ncorr
        self.nael = self.mol.nelec[0] - self.nfrzn
        self.nbel = self.mol.nelec[1] - self.nfrzn
        self.eps = self.mf.mo_energy[self.corr]

        self.point_group = self.mol.groupname.lower()
        if self.point_group != "c1":
            self.orbsym = self.mf.mo_coeff.orbsym[self.corr]
        else:
            self.orbsym = [0] * self.nmo

        self.nirrep = NIRREP[self.point_group]
        self.occ = list(range(self.nael))
        self.vir = list(range(self.nael, self.nmo))
        self.ham_screen_thresh = ham_screen_thresh

        # Specify the occupation of the the Hartree–Fock determinant
        self.hfref = forte.Determinant()
        # set the occupation numbers of the determinant
        for i in range(self.nael):
            self.hfref.set_alfa_bit(i, True)
        for i in range(self.nbel):
            self.hfref.set_beta_bit(i, True)

        if self.verbose >= DEBUG_PRINT_LEVEL:
            print(f"Number of orbitals:        {self.nmo}")
            print(f"Number of alpha electrons: {self.nael}")
            print(f"Number of beta electrons:  {self.nbel}")
            print(f"Reference determinant:     {self.hfref.str(self.nmo)}")

        self.make_hamiltonian()
        self.make_s2_operator()

        self.ref = forte.SparseState({self.hfref: 1.0})
        eref = forte.overlap(self.ref, forte.apply_op(self.ham_op, self.ref))
        assert np.isclose(
            eref, mf.e_tot
        ), f"Reference energy mismatch: {eref} != {mf.e_tot}"

    def make_hamiltonian(self):
        scalar = self.mol.energy_nuc()
        hcore = np.einsum(
            "pi,qj,pq->ij", self.mf.mo_coeff, self.mf.mo_coeff, self.mf.get_hcore()
        )
        eri = pyscf.ao2mo.full(self.mol.intor("int2e"), self.mf.mo_coeff)

        if self.nfrzn > 0:
            scalar += 2 * np.einsum("ii", hcore[self.frzn, self.frzn])
            scalar += 2 * np.einsum(
                "iijj->", eri[self.frzn, self.frzn, self.frzn, self.frzn]
            )
            scalar -= np.einsum(
                "ijji->", eri[self.frzn, self.frzn, self.frzn, self.frzn]
            )
            hcore[self.corr, self.corr] += 2 * np.einsum(
                "pqjj->pq", eri[self.corr, self.corr, self.frzn, self.frzn]
            )
            hcore[self.corr, self.corr] -= np.einsum(
                "pjjq->pq", eri[self.corr, self.frzn, self.frzn, self.corr]
            )
            hcore = hcore[self.corr, self.corr].copy()
            eri = eri[self.corr, self.corr, self.corr, self.corr].copy()

        oei_a = forte.ndarray_from_numpy(hcore)

        tei = eri.swapaxes(1, 2).copy()
        tei_aa = forte.ndarray_copy_from_numpy(tei - tei.swapaxes(2, 3))
        tei_ab = forte.ndarray_from_numpy(tei)

        self.ham_op = forte.sparse_operator_hamiltonian(
            scalar, oei_a, oei_a, tei_aa, tei_ab, tei_aa
        )

    def make_s2_operator(self):
        sup = forte.SparseOperator()
        sdn = forte.SparseOperator()
        sz = forte.SparseOperator()
        for i in range(self.nmo):
            sup.add(f"{i}a+ {i}b-", 1.0)
            sdn.add(f"{i}b+ {i}a-", 1.0)
            sz.add(f"{i}a+ {i}a-", 0.5)
            sz.add(f"{i}b+ {i}b-", -0.5)
        self.s2_op = sup @ sdn + sz @ sz - sz

    get_mo_space = NotImplemented


class SparseCI(SparseBase):
    def __init__(
        self,
        mf,
        ncas,
        nelecas,
        verbose=False,
        root_sym=0,
        ham_screen_thresh=1e-12,
    ):
        mo_space = self.get_mo_space(mf, ncas, nelecas)
        super().__init__(
            mf,
            verbose=verbose,
            root_sym=root_sym,
            mo_space=mo_space,
            ham_screen_thresh=ham_screen_thresh,
        )

    def get_mo_space(self, mf, ncas, nelecas):
        assert isinstance(nelecas, int), "nelecas must be an integer"
        mo_space = {}
        nelec = mf.mol.nelectron
        nfrzn = (nelec - nelecas) // 2
        mo_space["frzn"] = slice(0, nfrzn)
        mo_space["corr"] = slice(nfrzn, nfrzn + ncas)
        mo_space["virt"] = slice(nfrzn + ncas, mf.mol.nao)
        return mo_space

    def make_hamiltonian_matrix(self, dets):
        ndets = len(dets)
        hmat = np.zeros((ndets, ndets), dtype=np.complex128)
        for i in range(ndets):
            idet = forte.SparseState({dets[i]: 1.0})
            h_idet = forte.apply_op(self.ham_op, idet)
            for j in range(i, ndets):
                jdet = forte.SparseState({dets[j]: 1.0})
                hmat[i, j] = forte.overlap(jdet, h_idet)
                hmat[j, i] = hmat[i, j]
        return hmat

    def davidson(self, psi_0, maxiter):
        psi_old = psi_0
        for i in range(iter):
            psi_new = forte.apply_op(self.ham_op, psi_old)
            energy = forte.overlap(psi_old, psi_new)
            print(f"Iteration {i+1}: Energy = {energy}")
            psi_old = forte.normalize(psi_new)

    def compute_preconditioner(self, dets, maxspace=100):
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print("Computing preconditioner")
        ndets = len(dets)
        precond = np.ones(ndets, dtype=np.complex128)
        for i in range(maxspace):
            idet = forte.SparseState({dets[i]: 1.0})
            h_idet = forte.apply_op(self.ham_op, idet)
            precond[i] = forte.overlap(idet, h_idet)
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print("Preconditioner computed")
        return precond

    def guess_x0(self, dim, nroots, method="cisd"):
        assert method in ["eye", "cis", "cisd"], "Invalid guess method"
        x0 = np.zeros((nroots, dim))
        if method == "eye":
            x0[:nroots, :nroots] = np.eye(nroots)
        elif method == "cis":
            _, c = self.kernel_cin(truncation=1)
            x0[:nroots, : c.shape[0]] = c[:, :nroots].T
        elif method == "cisd":
            _, c = self.kernel_cin(truncation=2)
            x0[:nroots, : c.shape[0]] = c[:, :nroots].T
        return x0

    def kernel_cin(self, truncation=2):
        cin_dets = sorted(
            forte.hilbert_space(
                self.nmo,
                self.nael,
                self.nbel,
                self.hfref,
                truncation,
                self.nirrep,
                self.orbsym,
                self.root_sym,
            )
        )
        if self.verbose >= NORMAL_PRINT_LEVEL:
            print(f"Number of determinants: {len(cin_dets)}")
        hmat = self.make_hamiltonian_matrix(cin_dets)
        evals, evecs = np.linalg.eigh(hmat)
        return evals, evecs

    def kernel(self, solver, **davidson_kwargs):
        if solver not in ["eigh", "davidson"]:
            raise RuntimeError("Invalid solver")

        ci_dets = sorted(
            forte.hilbert_space(
                self.nmo, self.nael, self.nbel, self.nirrep, self.orbsym, self.root_sym
            )
        )
        if self.verbose >= NORMAL_PRINT_LEVEL:
            print(f"Number of determinants: {len(ci_dets)}")

        if solver == "eigh":
            hmat = self.make_hamiltonian_matrix(ci_dets)
            eigvals, eigvecs = np.linalg.eigh(hmat)
        elif solver == "davidson":
            ndets = len(ci_dets)
            precond_dim = min(davidson_kwargs.get("precond_dim", 100), ndets)
            davidson_kwargs.pop("precond_dim", None)
            precond = self.compute_preconditioner(ci_dets, maxspace=precond_dim).real
            # prec = lambda dx, e, x0: dx/(precond-e)
            guess_method = davidson_kwargs.get("guess_method", "cisd")
            davidson_kwargs.pop("guess_method", None)
            x0 = self.guess_x0(ndets, davidson_kwargs.get("nroots", 1), guess_method)

            def aop(xs):
                xc = np.zeros_like(xs, dtype=np.complex128)
                for idx, x in enumerate(xs):
                    xstate = forte.SparseState({d: c for d, c in zip(ci_dets, x)})
                    x1 = forte.apply_op(self.ham_op, xstate)
                    for i in range(ndets):
                        xc[idx, i] = x1[ci_dets[i]]
                return xc

            conv, eigvals, eigvecs = davidson1(aop, x0, precond, **davidson_kwargs)
            if not all(conv):
                raise RuntimeError("Davidson iterations did not converge")
        return eigvals, eigvecs


class SparseCC(SparseBase):
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
        mo_space = self.get_mo_space(mf, frozen_core)
        super().__init__(
            mf,
            verbose=verbose,
            root_sym=root_sym,
            mo_space=mo_space,
            ham_screen_thresh=ham_screen_thresh,
        )
        # get the number of MOs and alpha/beta electrons per irrep
        if cc_type not in ["cc", "ucc", "dcc", "ducc"]:
            raise RuntimeError("Invalid CC type")

        self.verbose = verbose
        self.factorized = "d" in cc_type
        self.unitary = "u" in cc_type

        if self.factorized:
            self.exp_op = forte.SparseFactExp(screen_thresh=exp_screen_thresh)
        else:
            self.exp_op = forte.SparseExp(
                maxk=exp_max_k, screen_thresh=exp_screen_thresh
            )
        if self.unitary:
            if self.factorized:
                self.apply_exp_op = functools.partial(
                    self.exp_op.apply_antiherm, inverse=False
                )
                self.apply_exp_op_inv = functools.partial(
                    self.exp_op.apply_antiherm, inverse=True
                )
            else:
                self.apply_exp_op = functools.partial(
                    self.exp_op.apply_antiherm, scaling_factor=1.0
                )
                self.apply_exp_op_inv = functools.partial(
                    self.exp_op.apply_antiherm, scaling_factor=-1.0
                )
        else:
            if self.factorized:
                self.apply_exp_op = functools.partial(
                    self.exp_op.apply_op, inverse=False
                )
                self.apply_exp_op_inv = functools.partial(
                    self.exp_op.apply_op, inverse=True
                )
            else:
                self.apply_exp_op = functools.partial(
                    self.exp_op.apply_op, scaling_factor=1.0
                )
                self.apply_exp_op_inv = functools.partial(
                    self.exp_op.apply_op, scaling_factor=-1.0
                )

        if self.nael != self.nbel:
            raise RuntimeError(
                "The number of alpha and beta electrons must be the same"
            )

    def get_mo_space(self, mf, frozen_core):
        mo_space = {}
        mo_space["frzn"] = slice(0, frozen_core)
        mo_space["corr"] = slice(frozen_core, mf.mol.nao)
        return mo_space

    def make_cluster_operator(self, max_exc, pp=True):
        # Prepare the cluster operator (closed-shell case)
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print(f"Occupied orbitals: {self.occ}")
            print(f"Virtual orbitals:  {self.vir}")

        self.t1 = forte.SparseOperatorList()
        self.tn = forte.SparseOperatorList()
        self.t1_denom = []
        self.tn_denom = []

        # do singles first
        for i in self.occ:
            for a in self.vir:
                if self.orbsym[i] ^ self.orbsym[a] != self.root_sym:
                    continue
                self.t1.add(f"{a}a+ {i}a-", 0.0)
                self.t1_denom.append(self.eps[i] - self.eps[a])
                self.t1.add(f"{a}b+ {i}b-", 0.0)
                self.t1_denom.append(self.eps[i] - self.eps[a])

        # loop over total excitation level
        for n in range(2, max_exc + 1):
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
                                self.tn_denom.append(e_aocc + e_bocc - e_bvir - e_avir)
                                op_str = []  # a list to hold the operator triplets
                                for i in av:
                                    op_str.append(f"{i}a+")
                                for i in bv:
                                    op_str.append(f"{i}b+")
                                for i in reversed(bo):
                                    op_str.append(f"{i}b-")
                                for i in reversed(ao):
                                    op_str.append(f"{i}a-")
                                self.tn.add(f"{' '.join(op_str)}", 0.0)
        self.op = self.t1 + self.tn
        self.denominators = self.t1_denom + self.tn_denom
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print(f"\n==> Cluster operator <==")
            print(f"Number of amplitudes: {len(self.op)}")
            print(f"Operator components:")
            for sqop, c in self.op:
                print(f"{sqop}")

    def cc_residual_equations(self):
        # Step 1. Compute exp(T)|Phi>
        wfn = self.apply_exp_op(self.op, self.ref)

        # Step 2. Compute H exp(T)|Phi>
        Hwfn = forte.apply_op(self.ham_op, wfn, screen_thresh=self.ham_screen_thresh)

        # Step 3. Compute exp(-T) H exp(T)|Phi>
        R = self.apply_exp_op_inv(self.op, Hwfn)

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
                u_psi = self.apply_exp_op(u_nu, self.ref, inverse=False)
            else:
                u_psi = self.apply_exp_op(u_nu, u_psi, inverse=False)
            k_u_psi = forte.apply_op(kappa_nu, u_psi)
            u_k_u_psi = self.apply_exp_op_inv(u_nu, k_u_psi, inverse=True)
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

    def make_eom_basis(self, nhole, npart, ms2=0, first=False):
        states = []

        hp = []
        if nhole >= npart:
            for i in range(npart + 1):
                hp.append((i + nhole - npart, i))
        else:
            for i in range(nhole + 1):
                hp.append((i, i + npart - nhole))

        if not first and nhole == npart:
            hp.remove((0, 0))

        for h, p in hp:
            for ah in range(h + 1):
                bh = h - ah
                for ap in range(p + 1):
                    bp = p - ap
                    if (ah - ap) - (bh - bp) != ms2:
                        continue
                    for ao in itertools.combinations(self.occ, self.nael - ah):
                        ao_sym = sym_dir_prod(self.orbsym[list(ao)])
                        for av in itertools.combinations(self.vir, ap):
                            av_sym = sym_dir_prod(self.orbsym[list(av)])
                            for bo in itertools.combinations(self.occ, self.nbel - bh):
                                bo_sym = sym_dir_prod(self.orbsym[list(bo)])
                                for bv in itertools.combinations(self.vir, bp):
                                    bv_sym = sym_dir_prod(self.orbsym[list(bv)])
                                    # if (
                                    #     ao_sym ^ av_sym ^ bo_sym ^ bv_sym
                                    #     != self.root_sym
                                    # ):
                                    # continue
                                    d = forte.Determinant()
                                    for i in ao:
                                        d.set_alfa_bit(i, True)
                                    for i in av:
                                        d.set_alfa_bit(i, True)
                                    for i in bo:
                                        d.set_beta_bit(i, True)
                                    for i in bv:
                                        d.set_beta_bit(i, True)
                                    states.append(d)
        if self.verbose >= NORMAL_PRINT_LEVEL:
            print(f"Number of {nhole}h{npart}p EOM states: {len(states)}")
        return sorted(states)

    def make_hbar_eom(self, basis):
        H = np.zeros((len(basis), len(basis)), dtype=np.complex128)
        basis_states = [forte.SparseState({d: 1.0}) for d in basis]

        for j in range(len(basis)):
            # exp(S)|j>
            wfn = self.apply_exp_op(self.op, basis_states[j])
            # H exp(S)|j>
            Hwfn = forte.apply_op(self.ham_op, wfn, self.ham_screen_thresh)
            # exp(-S) H exp(S)|j>
            R = self.apply_exp_op_inv(self.op, Hwfn)

            # <i|exp(-S) H exp(S)|j>
            if self.unitary:
                for i in range(j, len(basis_states)):
                    H[i, j] = forte.overlap(basis_states[i], R)
                    H[j, i] = H[i, j]
            else:
                for i in range(len(basis_states)):
                    H[i, j] = forte.overlap(basis_states[i], R)

        return H

    def eom_eig(self, basis):
        hbar = self.make_hbar_eom(basis)

        if self.unitary:
            eigval, eigvec = np.linalg.eigh(hbar)
        else:
            eigval, eigvec = scipy.linalg.eig(hbar)
            eigval_argsort = np.argsort(np.real(eigval))
            eigval = eigval[eigval_argsort]
            eigvec = eigvec[:, eigval_argsort]
            eigval = np.real(eigval)
        return eigval, eigvec

    def guess_x0(self, dim, nroots, method="cisd"):
        assert method in ["eye"], "Invalid guess method"
        x0 = np.zeros((nroots, dim))
        if method == "eye":
            x0[:nroots, :nroots] = np.eye(nroots)
        # elif method == "cis":
        #     _, c = self.kernel_cin(truncation=1)
        #     x0[:nroots, : c.shape[0]] = c[:, :nroots].T
        # elif method == "cisd":
        #     _, c = self.kernel_cin(truncation=2)
        #     x0[:nroots, : c.shape[0]] = c[:, :nroots].T
        return x0

    def compute_preconditioner(self, basis, maxspace=100):
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print("Computing preconditioner")
        nbasis = len(basis)
        precond = np.ones(nbasis, dtype=np.complex128)
        for i in range(maxspace):
            idet = forte.SparseState({basis[i]: 1.0})
            ui = self.apply_exp_op(self.op, idet)
            hui = forte.apply_op(self.ham_op, ui)
            uhui = self.apply_exp_op_inv(self.op, hui)
            precond[i] = forte.overlap(idet, uhui)
        if self.verbose >= DEBUG_PRINT_LEVEL:
            print("Preconditioner computed")
        return precond

    def eom_davidson(self, basis, **davidson_kwargs):
        nbasis = len(basis)
        precond_dim = min(davidson_kwargs.get("precond_dim", 100), nbasis)
        davidson_kwargs.pop("precond_dim", None)
        precond = self.compute_preconditioner(basis, maxspace=precond_dim).real
        guess_method = davidson_kwargs.get("guess_method", "cisd")
        davidson_kwargs.pop("guess_method", None)
        x0 = self.guess_x0(nbasis, davidson_kwargs.get("nroots", 5), guess_method)

        def aop(xs):
            xc = np.zeros_like(xs, dtype=np.complex128)
            for idx, x in enumerate(xs):
                xstate = forte.SparseState({d: c for d, c in zip(basis, x)})
                ux = self.apply_exp_op(self.op, xstate)
                hux = forte.apply_op(self.ham_op, ux)
                uhux = self.apply_exp_op_inv(self.op, hux)
                for i in range(nbasis):
                    xc[idx, i] = uhux[basis[i]]
            return xc

        conv, eigvals, eigvecs = davidson_nosym1(aop, x0, precond, **davidson_kwargs)
        if not all(conv):
            raise RuntimeError("Davidson iterations did not converge")
        return eigvals, eigvecs

    def get_s2(self, state):
        return forte.overlap(state, forte.apply_op(self.s2_op, state)).real

    def run_eom(
        self,
        nhole,
        npart,
        ms2,
        print_eigvals=True,
        first=False,
        solver="eig",
        **davidson_kwargs,
    ):
        nroots = davidson_kwargs.get("nroots", 5)
        assert solver in ["eig", "davidson"], "Invalid EOM solver"
        eom_basis = self.make_eom_basis(nhole, npart, ms2, first=first)
        if solver == "eig":
            eom_eigval, eom_eigvec = self.eom_eig(eom_basis)
            eom_eigval = eom_eigval[:nroots]
            eom_eigvec = eom_eigvec[:, :nroots].T
        elif solver == "davidson":
            eom_eigval, eom_eigvec = self.eom_davidson(eom_basis, **davidson_kwargs)
            nroots = len(eom_eigval)
        if print_eigvals:
            print("=" * 78)
            print(
                f"{'Root':^4} {'E / Eh':^20} {'ω / Eh':^20} {'ω / eV':^20} {'<S^2>':^10}"
            )
            print("-" * 78)
            for i in range(nroots):
                s2 = self.get_s2(
                    forte.SparseState(
                        {d: c for d, c in zip(eom_basis, eom_eigvec[i])}
                    )
                )
                print(
                    f"{i:^4d} {eom_eigval[i]:^20.12f} {eom_eigval[i] - self.energy:^20.12f} {(eom_eigval[i] - self.energy) * EH_TO_EV:^20.12f} {s2:^10.3f}"
                )
            print("=" * 78)
        return eom_eigval, eom_eigvec


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
