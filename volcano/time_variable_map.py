import numpy as np
from scipy.sparse import block_diag as sparse_block_diag
from scipy.linalg import block_diag as dense_block_diag
from scipy.linalg import cho_factor, cho_solve
import celerite
import theano
import theano.tensor as tt
import theano.sparse as ts
from tqdm import tqdm
from volcano.optimizers import Adam, NAdam
from starry_process import SP

__all__ = ["TimeDependentMapSolver"]

# Adapted from https://github.com/rodluger/paparazzi/blob/master/paparazzi/doppler.py


class TimeDependentMapSolver(object):
    """
    Time dependent map solver class. Solves for matrices P and Q where
    Y = PQ is a matrix with L columns where each column represent the model
    for a given light curve in a time ordered collection of L observed light
    curves. The columns of P are the SH coefficients of the K basis maps and
    the columns of Q specify the linear combination of these basis maps which
    best models each light curve.
    
    Args:
        f_list (ndarray): List of observed light curves.
        ferr_list (ndarray): Corresponding uncertainties for each light curve.
        A_list: List of design matrices for each light curve, each of shape
            (npts, N) where N is the nr. of SH coefficients of the map.
        K (int, optional): The number of basis maps used to represent the
            map for any given light curve. By definition K << L where L
            is the total number of light curves. Defaults to 3.
        t_grid (ndarray, optional): Grid of size L on which the GP is to be
            evaluated.
        P_mu (float or ndarray, optional): Prior mean on the spherical
            harmonic coefficients of the basis maps of shape (N, K)  where N
            is the nr. of coefficients and K is the number of basis maps.
        P_sig (float or ndarray, optional): Prior standard deviation on
            P. 
        Q_mu (float or ndarray, optional): Prior mean on the coefficient
            matrix of shape (K, L) where K is the number of basis maps and L
            is the number of light curves.
        Q_sig (ndarray, optional): Prior standard deviation for the GP on the
            coefficient matrix. This is needs to be an array of size K.
        Q_rho_gp (float, optional): Prior lenghtscale for the GP on the
            coefficient matrix. This is needs to be an array of size K. 
    """

    def __init__(
        self,
        f_list,
        ferr_list,
        A_list,
        K=4,
        tgrid=None,
        P_mu=None,
        P_sig=None,
        Q_mu=None,
        Q_sig=None,
        Q_sig_gp=None,
        Q_rho_gp=None,
    ):

        # Data
        self.f = np.concatenate(f_list)
        self.ferr = np.concatenate(ferr_list)
        self._F_CInv = np.ones_like(self.f) / self.ferr ** 2
        self.t_grid = tgrid

        # Design matrix
        self.A = sparse_block_diag(A_list).tocsc()

        # Shape params
        self.N = np.shape(A_list[0])[1]
        self.L = len(f_list)
        self.K = K

        # Inference params
        self.P_mu = P_mu
        self.P_sig = P_sig
        self.Q_mu = Q_mu
        self.Q_sig = Q_sig
        self.Q_sig_gp = Q_sig_gp
        self.Q_rho_gp = Q_rho_gp
        self._q_CInv = None
        self._p_CInv = None

        # Initialize
        self._p = None
        self._q = None

    @property
    def P_sig(self):
        return self._P_sig

    @P_sig.setter
    def P_sig(self, value):
        if (value is not None) and (np.shape(value) != (self.N, self.K)):
            raise ValueError("Input needs have shape (N, K).")
        self._P_sig = value

    @property
    def Q_sig(self):
        return self._Q_sig

    @Q_sig.setter
    def Q_sig(self, value):
        if (value is not None) and (np.shape(value) != (self.K, self.L)):
            raise ValueError("Input needs have shape (K, L).")
        self._Q_sig = value

    @property
    def Q_rho_gp(self):
        return self._Q_rho_gp

    @Q_rho_gp.setter
    def Q_rho_gp(self, value):
        self._Q_rho_gp = value

    @property
    def Q(self):
        return self._q.reshape(self.K, self.L, order="F")

    @Q.setter
    def Q(self, value):
        if np.shape(value) != (self.K, self.L):
            raise ValueError("Input needs have shape (K, L).")
        self._q = value.T.reshape(-1)

    @property
    def P(self):
        return self._p.reshape(self.N, self.K, order="F")

    @P.setter
    def P(self, value):
        if np.shape(value) != (self.N, self.K):
            raise ValueError("Input needs have shape (N, K).")
        self._p = value.T.reshape(-1)

    @property
    def Y(self):
        return np.dot(self.P, self.Q)

    def get_Qp(self, Q):
        """
        Compute the Q' matrix given Q.
        """
        I_stacked = np.tile(np.eye(self.N), (self.L, self.K))
        Q_repeat = np.repeat(np.repeat(Q.T, self.N, axis=0), self.N, axis=1)

        return Q_repeat * I_stacked

    def get_Pp(self, P):
        """
        Compute the P' matrix given P.
        """
        I = np.eye(self.L, dtype=int)

        return np.kron(I, P)  # Repeat P on diagonal

    def _compute_cov(self):
        # Compute the covariance matrix for vec(Q)
        inv_covs = []
        if self._Q_rho_gp is not None:
            # Each row of Q has a GP with different hyperparameters
            for k in range(self.K):
                kernel = celerite.terms.Matern32Term(
                    np.log(self.Q_sig_gp[k]), np.log(self.Q_rho_gp[k])
                )
                gp = celerite.GP(kernel)
                q_C = gp.get_matrix(self.t_grid)

                q_cho_C = cho_factor(q_C)
                q_CInv = cho_solve(q_cho_C, np.eye(int(self.L)))
                inv_covs.append(q_CInv)

            # The complete cov matrix is block diagonal with K identical blocks
            q_CInv = dense_block_diag(*(inv_covs))

        else:
            for i in range(self.K):
                inv_covs.append(np.diag(1.0 / self.Q_sig[i, :] ** 2))
            q_CInv = dense_block_diag(*(inv_covs))

        # The above CINv was constructed for the Q matrix unrolled row wise,
        # to compute the covariance matrix for q=vec(Q), we need to permute
        # the inverse covariance matrix
        perm = np.concatenate(
            [[k * self.L + i for k in range(self.K)] for i in range(self.L)]
        )
        tmp = q_CInv[perm, :]
        self._q_CInv = tmp[:, perm]

        # Compute the covariance matrix for vec(P)
        inv_covs = []
        for k in range(self.K):
            inv_covs.append(np.diag(1 / self.P_sig[:, k] ** 2))
        p_CInv = dense_block_diag(*(inv_covs))

        self._p_CInv = p_CInv

    def model(self):
        """
        Model for the flux.
        """
        # Unroll p into P
        P = self._p.reshape(self.N, self.K, order="F")

        # P' matrix
        Pp = self.get_Pp(P)

        return self.A.dot(Pp.dot(self._q))

    def loss(self):
        """
        Return the loss function for the current parameters.

        """
        # Likelihood and prior
        lnlike = -0.5 * np.sum(
            (self.f - self.model()).reshape(-1) ** 2 * self._F_CInv.reshape(-1)
        )
        lnprior = -0.5 * np.dot(
            np.dot(
                (self._p - self.P_mu.T.reshape(-1)).reshape(1, -1),
                self._p_CInv,
            ),
            (self._p - self.P_mu.T.reshape(-1)).reshape(-1, 1),
        )

        -0.5 * np.dot(
            np.dot(
                (self._q - self.Q_mu.T.reshape(-1)).reshape(1, -1),
                self._q_CInv,
            ),
            (self._q - self.Q_mu.T.reshape(-1)).reshape(-1, 1),
        )

        return -(lnlike + lnprior).item()

    def _compute_p(self, T=1.0):
        """
        Linear solve for ``p`` given ``Q'`` given an optional temperature.

        Returns the Cholesky decomposition of the covariance of ``p``.
        """
        # Q' matrix
        Qp = self.get_Qp(self.Q)

        # Design matrix
        A = self.A.dot(Qp)

        ATCInv = np.multiply(A.T, (self._F_CInv / T).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.f).reshape(-1))[:, None]

        CInv = self._p_CInv
        p_mu = self.P_mu.T.reshape(-1, 1)
        CInvmu = np.dot(CInv, p_mu)
        cho_p = cho_factor(ATCInvA + CInv)
        self._p = cho_solve(cho_p, (ATCInvf + CInvmu).reshape(-1)).reshape(-1)

        return cho_p

    def _compute_q(self, T=1.0):
        """
        Linear solve for ``q`` given ``P'`` given an optional temperature. 

        Returns the Cholesky decomposition of the covariance of ``q``.

        """
        # P' matrix
        Pp = self.get_Pp(self.P)

        # Design matrix
        A = self.A.dot(Pp)

        ATCInv = np.multiply(A.T, (self._F_CInv / T).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.f).reshape(-1))[:, None]

        CInv = self._q_CInv
        q_mu = self.Q_mu.T.reshape(-1, 1)
        CInvmu = np.dot(CInv, q_mu)
        cho_q = cho_factor(ATCInvA + CInv)
        self._q = cho_solve(cho_q, (ATCInvf + CInvmu).reshape(-1)).reshape(-1)

        return cho_q

    def solve(
        self,
        P=None,
        Q=None,
        P_guess=None,
        Q_guess=None,
        niter=100,
        T=1.25,
        dlogT=-0.25,
        optimizer="NAdam",
        dcf=10.0,
        quiet=False,
        **kwargs
    ):
        """
        Solve the bilinear problem.
        
        Returns:
            ``(loss, cho_p, cho_q)``, a tuple containing the array of
            loss values during the optimization and the Cholesky factorization
            of the covariance matrices of ``p`` and ``q``, if available 
            (otherwise the latter two are set to ``None``.)
        """
        # Check the optimizer is valid
        if optimizer.lower() == "nadam":
            optimizer = NAdam
        elif optimizer.lower() == "adam":
            optimizer = Adam
        else:
            raise ValueError("Invalid optimizer.")

        # Compute GP
        self._compute_cov()

        # Figure out what to solve for
        known = []
        if Q is not None:
            known += ["Q"]
        if P is not None:
            known += ["P"]

        if ("P" in known) and ("Q" in known):
            # Nothing to do here but ingest the values!
            self.P = P
            self.Q = Q
            return self.loss(), None, None

        elif "P" in known:
            # Easy: it's a linear problem
            self.P = P
            cho_q = self._compute_q()
            return self.loss(), None, cho_q

        elif "Q" in known:
            # Still a linear problem!
            self.Q = Q
            cho_p = self._compute_p()
            return self.loss(), cho_p, None

        else:
            # Non-linear. Let's use (N)Adam.
            # Initialize the variables
            self.P = P_guess
            self.Q = Q_guess

            # Tempering params
            if T > 1.0:
                T_arr = 10 ** np.arange(np.log10(T), 0, dlogT)
                T_arr = np.append(T_arr, [1.0])
                niter_bilin = len(T_arr)
            else:
                T_arr = [1.0]
                niter_bilin = 1

            # Loss array
            loss_val = np.zeros(niter_bilin + niter + 1)
            loss_val[0] = self.loss()

            # Iterative bi-linear solve
            if niter_bilin > 0:
                if not quiet:
                    print("Running bi-linear solver...")

                best_loss = loss_val[0]
                best_p = self.P
                best_q = self.Q

                for n in tqdm(range(niter_bilin), disable=quiet):

                    # Compute `p` using the previous `q`
                    self._compute_p(T=T_arr[n])

                    # Compute `q` using the current `p`
                    self._compute_q(T=T_arr[n])

                    loss_val[n + 1] = self.loss()

                    if loss_val[n + 1] < best_loss:
                        best_loss = loss_val[n + 1]
                        best_p = self.P
                        best_q = self.Q

                self.P = best_p
                self.Q = best_q

            # Non-linear solve
            if niter > 0:

                # Theano nonlienar solve. Variables:
                p = theano.shared(self._p)
                q = theano.shared(self._q)

                theano_vars = [p, q]

                # Compute the model
                A = ts.as_sparse_variable(self.A)
                P = p.reshape((self.K, self.N)).T
                Q = q.reshape((self.L, self.K)).T
                Y = tt.dot(P, Q)
                f_pred = ts.dot(A, Y.T.flatten())

                # Compute the loss
                r = tt.reshape(self.f - f_pred, (-1,))
                cov = tt.reshape(self._F_CInv, (-1,))
                lnlike = -0.5 * tt.sum(r ** 2 * cov)
                lnprior = (
                    -0.5
                    * tt.dot(
                        tt.dot(
                            tt.reshape((p - self.P_mu.T.reshape(-1)), (1, -1)),
                            self._p_CInv,
                        ),
                        tt.reshape((p - self.P_mu.T.reshape(-1)), (-1, 1)),
                    )[0, 0]
                    - 0.5
                    * tt.dot(
                        tt.dot(
                            tt.reshape((q - self.Q_mu.T.reshape(-1)), (1, -1)),
                            self._q_CInv,
                        ),
                        tt.reshape((q - self.Q_mu.T.reshape(-1)), (-1, 1)),
                    )[0, 0]
                )

                loss = -(lnlike + lnprior)
                best_loss = loss.eval()
                best_p = p.eval()
                best_q = q.eval()

                if not quiet:
                    print("Running non-linear solver...")

                upd = optimizer(loss, theano_vars, **kwargs)
                train = theano.function([], [p, q, loss], updates=upd)
                for n in tqdm(
                    1 + niter_bilin + np.arange(niter), disable=quiet
                ):
                    p_val, s_val, loss_val[n] = train()
                    if loss_val[n] < best_loss:
                        best_loss = loss_val[n]
                        best_p = p_val
                        best_q = s_val

                # We are done!
                self._p = best_p
                self._q = best_q

            # Estimate the covariance of `p` conditioned on `q`
            # and the covariance of `q` conditioned on `p`.
            p_curr = np.array(self._p)
            cho_p = self._compute_p()
            self._p = p_curr

            q_curr = np.array(self._q)
            cho_q = self._compute_q()
            self._q = q_curr

            return loss_val, cho_p, cho_q

    def sample_with_gibbs(
        self, P_guess=None, Q_guess=None, nsamples=1000, quiet=False,
    ):
        """
        Sample the joint posterior p(P, Q) using Gibbs sampling.
        """
        # Store the samples
        Q_samples = []
        P_samples = []

        # Initialize
        self.P = P_guess
        self.Q = Q_guess

        # Compute GP
        self._compute_cov()

        # Sampling loop
        for n in tqdm(range(nsamples), disable=quiet):
            param_order = np.random.choice(np.arange(2), 2, replace=False)

            # Randomly switch order between conditional samples
            for k in param_order:
                # Sample P|Q
                if k == 0:
                    Qp = self.get_Qp(self.Q)
                    # Design matrix
                    A = self.A.dot(Qp)

                    ATCInv = np.multiply(A.T, (self._F_CInv).reshape(-1))
                    ATCInvA = ATCInv.dot(A)
                    ATCInvf = np.dot(ATCInv, (self.f).reshape(-1))[:, None]

                    CInv = self._p_CInv
                    p_mu = self.P_mu.T.reshape(-1, 1)
                    CInvmu = np.dot(CInv, p_mu)
                    cho_p = cho_factor(ATCInvA + CInv)
                    Cp = cho_solve(cho_p, np.eye(int(self.N * self.K)))

                    #                    cho_Cp, lower = cho_factor(Cp, lower=True)
                    #
                    #                    # Get the cholesky decomposition of the covariance matrix
                    #                    L_p = np.tril(cho_Cp)
                    #
                    #                    # Sample X from standard normal
                    #                    X = np.random.normal(size=(int(self.N * self.K), 1))
                    #
                    #                    # Apply the transformation to get sample from p
                    #                    p_sample = L_p.dot(X).reshape(-1) + self.P_mu.T.reshape(-1)

                    p_sample = np.random.multivariate_normal(
                        self.P_mu.T.reshape(-1), Cp, size=1
                    ).reshape(-1)
                    self._p = p_sample

                    P_samples.append(self.P)

                # Sample Q|P
                else:
                    Pp = self.get_Pp(self.P)

                    # Design matrix
                    A = self.A.dot(Pp)

                    ATCInv = np.multiply(A.T, (self._F_CInv).reshape(-1))
                    ATCInvA = ATCInv.dot(A)
                    ATCInvf = np.dot(ATCInv, (self.f).reshape(-1))[:, None]

                    CInv = self._q_CInv
                    q_mu = self.Q_mu.T.reshape(-1, 1)
                    CInvmu = np.dot(CInv, q_mu)
                    cho_q = cho_factor(ATCInvA + CInv)
                    Cq = cho_solve(cho_q, np.eye(int(self.K * self.L)))
                    #                    cho_Cq, lower = cho_factor(Cq, lower=True)
                    #
                    #                    # Get the cholesky decomposition of the covariance matrix
                    #                    L_q = np.tril(cho_Cq)
                    #
                    #                    # Sample X from standard normal
                    #                    X = np.random.normal(size=(int(self.K * self.L), 1))
                    #
                    #                    # Apply the transformation to get sample from q
                    #                    q_sample = L_q.dot(X).reshape(-1) + self.Q_mu.T.reshape(-1)

                    q_sample = np.random.multivariate_normal(
                        self.Q_mu.T.reshape(-1), Cq, size=1
                    ).reshape(-1)

                    self.q = q_sample

                    Q_samples.append(self.Q)

        return P_samples, Q_samples
