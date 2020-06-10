import numpy as np
from scipy.linalg import cho_factor, cho_solve
from volcano.time_variable_map import TimeDependentMapSolver

np.random.seed(42)

# Create some mock data
L = 20
N = 81
K = 5
npts = np.random.randint(500, size=L)
f_list = [np.random.rand(n) for n in npts]
ferr_list = [1e-2 * np.ones_like(f_list[i]) for i in range(L)]
A_list = [np.random.rand(n, N) for n in npts]

tgrid = np.linspace(0, 50, L)

# P matrix
P0_mu = np.ones(K)
P1_mu = np.zeros((N - 1, K))
P_mu = np.vstack([P0_mu, P1_mu])

P0_sig = 0.5 * np.ones(K)
P1_sig = 0.01 * np.ones((N - 1, K))
P_sig = np.vstack([P0_sig, P1_sig])

# Q Matrix
Q_mu = np.zeros((K, L))
Q_sig = 2 * np.ones((K, L))
Q_sig_gp = 2 * np.ones(K)
Q_rho_gp = 25.0 * np.ones(K)


def test_bilinear_form():
    solver = TimeDependentMapSolver(
        f_list,
        ferr_list,
        A_list,
        K,
        P_mu=P_mu,
        P_sig=P_sig,
        Q_mu=Q_mu,
        Q_sig=Q_sig,
    )

    solver.P = np.random.rand(solver.N, solver.K)
    solver.Q = np.random.rand(solver.K, solver.L)
    Y = solver.Y

    Qp = solver.get_Qp(solver.Q)
    Pp = solver.get_Pp(solver.P)

    f1 = solver.A @ Y.T.flatten()
    f2 = solver.A @ Qp @ solver.P.T.flatten()
    f3 = solver.A @ Pp @ solver.Q.T.flatten()

    assert np.allclose(f1, f2)
    assert np.allclose(f2, f3)


def test_loss():
    solver = TimeDependentMapSolver(
        f_list,
        ferr_list,
        A_list,
        K,
        P_mu=P_mu,
        P_sig=P_sig,
        Q_mu=Q_mu,
        Q_sig=Q_sig,
    )

    solver.P = np.random.rand(solver.N, solver.K)
    solver.Q = np.random.rand(solver.K, solver.L)

    p_mu = solver.P_mu.T.flatten()
    q_mu = solver.Q_mu.T.flatten()
    p_sig = solver.P_sig.T.flatten()
    q_sig = solver.Q_sig.T.flatten()

    solver._compute_cov()

    lnlike = -0.5 * np.sum(
        (solver.f - solver.model()).reshape(-1) ** 2
        * solver._F_CInv.reshape(-1)
    )
    lnprior = -0.5 * np.sum(
        (solver._p - p_mu) ** 2 / p_sig ** 2
    ) - 0.5 * np.sum((solver._p - p_mu) ** 2 / p_sig ** 2)

    loss = -(lnlike + lnprior).item()

    print("loss1:", loss)
    print("loss2:", solver.loss())

    assert np.allclose(loss, solver.loss())
