from sparse_cc import *

mol = pyscf.gto.M(atom="""
Be 0.0     0.0     0.0
H  0   1.310011  0.0
H  0   -1.310011  0.0""", basis='beh.nw', symmetry='c2v')
mf = pyscf.scf.RHF(mol)
mf.kernel()
sparse_cc = SparseCC(mf,verbose=5,unitary=True)

sparse_cc.make_cluster_operator(max_exc=2)
sparse_cc.kernel()

sparse_cc.run_eom(2, "ee")