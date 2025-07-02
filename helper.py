import forte
import forte.utils
import itertools
import functools
import time
import numpy as np
import psi4
import scipy.linalg
import scipy.optimize

def cc_variational_functional(t, op, ref, ham_op, exp_op, screen_thresh_H, screen_thresh_exp, maxk):
    '''
    E[A] = <Psi|exp(-A) H exp(A)|Psi>
    '''
    op.set_coefficients(t)
    # Step 1. Compute exp(S)|Phi>
    wfn = exp_op.compute(op, ref, scaling_factor=1.0, screen_thresh=screen_thresh_exp, maxk=maxk)
    
    # Step 2. Compute H exp(S)|Phi>
    Hwfn = ham_op.compute(wfn, screen_thresh_H)
    
    # Step 3. Compute exp(-S) H exp(S)|Phi>
    R = exp_op.compute(op.adjoint(), Hwfn, scaling_factor=1.0, screen_thresh=screen_thresh_exp, maxk=maxk)
    
    # Step 4. Get the energy: <Phi|R>
    # E = <ref|R>, R is a StateVector, which can be looked up by the determinant
    energy = 0.0
    for det,coeff in ref.items():
        energy += coeff * R[det].real
    
    norm = forte.overlap(wfn, wfn).real
    return energy/norm

def sym_dir_prod(occ_list, sym_list):
    if (len(occ_list) == 0): 
        return 0
    elif (len(occ_list) == 1):
        return sym_list[occ_list[0]]
    else:
        return functools.reduce(lambda i, j:  i ^ j, [sym_list[x] for x in occ_list])

def process_psi4_wfn(psi4_wfn):
    frzc = np.array(psi4_wfn.frzcpi().to_tuple())
    docc = np.array(psi4_wfn.doccpi().to_tuple())
    gas1 = list(docc - frzc)
    gas3 = np.array(psi4_wfn.nmopi().to_tuple()) - docc
    forte_objs = forte.utils.prepare_forte_objects(psi4_wfn, mo_spaces={'FROZEN_DOCC': list(frzc),'GAS1':gas1, 'GAS3':list(gas3)})
    # ints = forte_objs['ints']
    as_ints = forte_objs['as_ints']
    # scf_info = forte_objs['scf_info']
    mo_space_info = forte_objs['mo_space_info']

    # nmo = mo_space_info.size('CORRELATED')
    nael = len(forte_objs['mo_space_info'].corr_absolute_mo('GAS1'))
    nbel = len(forte_objs['mo_space_info'].corr_absolute_mo('GAS1'))

    # if nael != nbel:
        # raise RuntimeError('The number of alpha and beta electrons must be the same')

    occ = mo_space_info.corr_absolute_mo('GAS1')
    vir = mo_space_info.corr_absolute_mo('GAS3')

    nmopi = list(np.array(psi4_wfn.nmopi().to_tuple()) - frzc)
    # point_group = psi4_wfn.molecule().point_group().symbol()
    
    nirrep = mo_space_info.nirrep()

    naelpi = np.array(psi4_wfn.nalphapi().to_tuple()) - frzc
    nbelpi = np.array(psi4_wfn.nbetapi().to_tuple()) - frzc

    # occ_sym = mo_space_info.symmetry('GAS1')
    # vir_sym = mo_space_info.symmetry('GAS3')
    all_sym = mo_space_info.symmetry('CORRELATED') 

    hfref = forte.Determinant()
    irrep_start = [sum(nmopi[:h]) for h in range(nirrep)]
    for h in range(nirrep):
        for i in range(naelpi[h]): hfref.set_alfa_bit(irrep_start[h] + i, True)
        for i in range(nbelpi[h]): hfref.set_beta_bit(irrep_start[h] + i, True)
    
    ref = forte.StateVector({hfref :1.0})
    ham_op = forte.SparseHamiltonian(as_ints)

    return ham_op, psi4_wfn.epsilon_a_subset('MO','ACTIVE'), occ, vir, all_sym, ref

class ArbOrderSRCC:
    def __init__(self, psi4_wfn, type, max_exc, sym=0, verbose=True, screen_thresh_H=1e-14, screen_thresh_exp=1e-14, maxk=19):
        if type == 'TCC':
            self.var = self.unitary = False
        if type == 'VCC':
            self.var = True
            self.unitary = False
        elif type == 'UCC':
            self.var = False
            self.unitary = True
        self.max_exc = max_exc
        self.verbose = verbose
        self.screen_thresh_H = screen_thresh_H
        self.screen_thresh_exp = screen_thresh_exp
        self.maxk = maxk
        self.sym = sym
        self.exp_op = forte.SparseExp()
        self.ham_op, self.mo_energies, self.occ, self.vir, self.all_sym, self.ref = process_psi4_wfn(psi4_wfn)
        self.make_T(max_exc)

    def make_T(self, max_exc):
        self.op = forte.SparseOperator(antihermitian=True) if self.unitary else forte.SparseOperator(antihermitian=False)
        self.ea = self.mo_energies
        self.eb = self.mo_energies
        self.denominators = []
        for n in range(1,max_exc + 1):
            for nb in range(n + 1):
                na = n - nb
                for ao in itertools.combinations(self.occ, na):
                    ao_sym = sym_dir_prod(ao, self.all_sym)
                    for av in itertools.combinations(self.vir, na):
                        av_sym = sym_dir_prod(av, self.all_sym)
                        for bo in itertools.combinations(self.occ, nb):
                            bo_sym = sym_dir_prod(bo, self.all_sym)
                            for bv in itertools.combinations(self.vir, nb):
                                bv_sym = sym_dir_prod(bv, self.all_sym)
                                if (ao_sym ^ av_sym ^ bo_sym ^ bv_sym == self.sym):
                                    e_aocc = functools.reduce(lambda x, y: x + self.ea.get(y),ao,0.0)
                                    e_avir = functools.reduce(lambda x, y: x + self.ea.get(y),av,0.0)
                                    e_bocc = functools.reduce(lambda x, y: x + self.eb.get(y),bo,0.0)
                                    e_bvir = functools.reduce(lambda x, y: x + self.eb.get(y),bv,0.0)
                                    self.denominators.append(e_aocc + e_bocc - e_bvir - e_avir)
                                    l = []
                                    for i in ao: l.append((False,True,i))
                                    for i in bo: l.append((False,False,i))
                                    for a in reversed(bv): l.append((True,False,a))
                                    for a in reversed(av): l.append((True,True,a))
                                    self.op.add_term(l,0.0)

    def run_ccn_variational(self):
        self.t = [0.0] * self.op.size()
        res = scipy.optimize.minimize(fun=cc_variational_functional, x0=self.t, \
                                      args=(self.op,self.ref,self.ham_op,self.exp_op,\
                                            self.screen_thresh_H,self.screen_thresh_exp,self.maxk),\
                                      method='BFGS')
        print(res)
