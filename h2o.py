from sparse_cc import *

mol = pyscf.gto.M(atom="""
O
H 1 1.1
H 1 1.1 2 104""", basis='6-31g', symmetry='c2v')
mf = pyscf.scf.RHF(mol)
mf.kernel()
sparse_cc = SparseCC(mf,verbose=5,unitary=True)

sparse_cc.make_cluster_operator(max_exc=2)
sparse_cc.kernel()

cc = pyscf.cc.CCSD(mf)
cc.kernel()