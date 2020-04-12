import numpy as np
from scipy.sparse import block_diag as sparse_block_diag
from scipy.linalg import block_diag as dense_block_diag
from scipy.linalg import cho_factor, cho_solve
import theano
import theano.tensor as tt

# sys.path.append("../../volcano")
# from optimizers import Adam, NAdam
# from time_dependent_solver import TimeDependentMapSolver

from volcano.optimizers import Adam, NAdam
from volcano.time_dependent_solver import TimeDependentMapSolver

np.random.seed(42)

# Create some mock data
L = 45
N = 81
K = 4
npts = np.random.randint(500, size=L)
f_list = [np.random.rand(n) for n in npts]
ferr_list = [1e-2 * np.ones_like(f_list[i]) for i in range(L)]
A_list = [np.random.rand(n, N) for n in npts]
solver = TimeDependentMapSolver(f_list, ferr_list, A_list, K)


def test_bilinear_form():

    P = np.random.rand(solver.N, solver.K)
    Q = np.random.rand(solver.K, solver.L)
    Y = P @ Q

    Qp = solver.get_Qp(Q)
    Pp = solver.get_Pp(P)

    f1 = solver.A @ Y.T.flatten()
    f2 = solver.A @ Qp @ P.T.flatten()
    f3 = solver.A @ Pp @ Q.T.flatten()

    assert np.allclose(f1, f2)
    assert np.allclose(f2, f3)
