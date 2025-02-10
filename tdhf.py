import pyscf, pyscf.dft
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg
import subprocess


class TDHF:
    def __init__(
        self,
        mf,
        td_pert,
        dt,
        max_t,
        restart_freq=500,
        expm_algo="scipy",
        expm_algo_params=None,
    ):
        self.mf = mf
        self.mol = mf.mol
        self.td_pert = td_pert
        if isinstance(mf, pyscf.scf.rohf.ROHF):
            self.dm_ao_t = self.mf.make_rdm1()[0]
            print("This is only available for 1e systems!")
            self.get_fock = lambda dm=None: self.mf.get_hcore()
        elif isinstance(mf, pyscf.scf.rhf.RHF):
            self.dm_ao_t = self.mf.make_rdm1()
            self.get_fock = self.mf.get_fock
        else:
            raise ValueError(
                "mf must be an instance of pyscf.scf.rhf.RHF or pyscf.scf.rohf.ROHF"
            )
        self.dm_ao_tm1 = self.dm_ao_t.copy()
        self.fock_ao = self.get_fock()
        self.restart_freq = restart_freq
        self.atom_positions = self.mol.atom_coords()

        if expm_algo == "scipy":
            self.expm = self.expm_scipy
        elif expm_algo == "bch":
            try:
                assert (
                    expm_algo_params is not None
                    and "bch_order" in expm_algo_params.keys()
                )
            except AssertionError:
                raise ValueError("bch_order must be specified in expm_algo_params")
            self.expm_algo_params = expm_algo_params
            self.expm = self.expm_bch

        self.atom_ao_slices = [
            slice(*_) for _ in self.mf.mol.aoslice_nr_by_atom()[:, 2:]
        ]
        self.atom_charges = self.mol.atom_charges()

        self.dipole_ao = self.mol.intor("int1e_r")

        self.dt = dt
        self.max_t = max_t
        self.maxiter = int(max_t / dt)
        self.iter = 0
        self.t = 0

        self.ovlp = self.mol.intor("int1e_ovlp")
        self.s_evals, self.s_evecs = np.linalg.eigh(self.ovlp)
        self.s_sqrt = self.s_evecs @ np.diag(self.s_evals**0.5) @ self.s_evecs.T
        self.X = self.s_evecs @ np.diag(self.s_evals ** (-0.5))
        self.Xinv = np.diag(self.s_evals ** (0.5)) @ self.s_evecs.T
        self.mo_coeff_0 = self.Xinv @ self.mf.mo_coeff.copy()

    def expm_scipy(self, A, B):
        _expA = scipy.linalg.expm(A)
        return _expA @ B @ _expA.conj().T

    def expm_bch(self, A, B):
        _order = self.expm_algo_params["bch_order"]
        _comm = B.copy()
        for _i in range(_order):
            _comm = np.dot(A, _comm) - np.dot(_comm, A)
            B = B + _comm / (_i + 1)
        return B

    def get_hf_wfn(self):
        self.mf = pyscf.scf.HF(self.mol)
        self.mf.kernel()

    def get_ao_fock(self, t, dm_ao):
        return self.get_fock(dm=dm_ao) + np.einsum(
            "xij,x->ij", self.dipole_ao, self.td_pert(t)
        )

    def get_ao_dm(self, dm_on):
        return self.X @ dm_on @ self.X.T

    def get_on_fock(self, t, dm_ao):
        return self.X.T @ self.get_ao_fock(t, dm_ao) @ self.X

    def get_on_dm(self, dm_ao):
        return self.Xinv @ dm_ao @ self.Xinv.T

    def mmut_propagator(self, dm_on, fock_on):
        return self.expm(-2j * self.dt * fock_on, dm_on)

    def em2_propagator(self, dm_on, fock_on):
        p_fe = self.expm(-1j * self.dt * fock_on, dm_on)
        fm = 0.5 * (
            dm_on + self.get_on_fock(self.dt * (self.iter + 1), self.get_ao_dm(p_fe))
        )
        return self.expm(-1j * self.dt * fm, dm_on)

    def lowdin_pop_analysis(self, dm_ao):
        _sps = self.s_sqrt @ dm_ao @ self.s_sqrt
        for i in range(self.mol.natm):
            self.lowdin_pop[self.iter, i] = self.atom_charges[i] - np.trace(
                _sps[self.atom_ao_slices[i], self.atom_ao_slices[i]]
            )

    def mulliken_pop_analysis(self, dm_ao):
        _ps = dm_ao @ self.ovlp
        for i in range(self.mol.natm):
            self.mulliken_pop[self.iter, i] = self.atom_charges[i] - np.trace(
                _ps[self.atom_ao_slices[i], self.atom_ao_slices[i]]
            )

    def run_tdhf(self, save_traj=False, save_traj_freq=1, traj_name=""):
        self.energies = np.zeros((self.maxiter, self.mf.mo_energy.shape[0]))
        self.occ = np.zeros(
            (self.maxiter, self.mf.mo_energy.shape[0]), dtype="complex128"
        )
        self.lowdin_pop = np.zeros((self.maxiter, self.mf.mol.natm))
        self.mulliken_pop = np.zeros((self.maxiter, self.mf.mol.natm))
        self.dipole_inst = np.zeros((self.maxiter, 3))
        if save_traj:
            dm_traj_file = open(traj_name + "dm_traj.npy", "wb")
            mo_traj_file = open(traj_name + "mo_traj.npy", "wb")
        while self.iter < self.maxiter:
            self.dipole_inst[self.iter, :] = np.einsum(
                "a,ai->i", self.atom_charges, self.atom_positions
            ) - np.einsum("xij,ij->x", self.dipole_ao, self.dm_ao_t)

            fock_on = self.get_on_fock(self.t, self.dm_ao_t)
            self.energies[self.iter, :], _coeff = np.linalg.eigh(fock_on)
            if save_traj and self.iter % save_traj_freq == 0:
                np.save(mo_traj_file, self.X @ _coeff)
            self.dm_on_tm1 = self.get_on_dm(self.dm_ao_tm1)

            self.occ[self.iter, :] = np.einsum(
                "ik,ij,jk->k", self.mo_coeff_0, self.dm_on_tm1, self.mo_coeff_0
            )

            self.mulliken_pop_analysis(self.dm_ao_t)
            self.lowdin_pop_analysis(self.dm_ao_t)
            if self.iter % self.restart_freq == 0:
                self.dm_on_t = self.em2_propagator(self.dm_on_tm1, fock_on)
            else:
                self.dm_on_t = self.mmut_propagator(self.dm_on_tm1, fock_on)
                self.dm_ao_tm1 = self.dm_ao_t.copy()

            self.dm_ao_t = self.get_ao_dm(self.dm_on_t)
            if save_traj and self.iter % save_traj_freq == 0:
                np.save(dm_traj_file, self.dm_ao_t)

            self.iter += 1
            self.t += self.dt
        if save_traj:
            dm_traj_file.close()
            mo_traj_file.close()
