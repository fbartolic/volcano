import numpy as np
from scipy.sparse import block_diag as sparse_block_diag
from scipy.linalg import block_diag as dense_block_diag
from scipy.linalg import cho_factor, cho_solve
import theano
import theano.tensor as tt
from tqdm import tqdm
from volcano.optimizers import Adam, NAdam

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
        p_mu (float or ndarray, optional): Prior mean on  the spherical
            harmonic coefficients of the basis maps p=vec(P).
            Defaults to 0.
        p_sig (float or ndarray, optional): Prior standard deviation on
            p Defaults to 0.01.
        q_mu (float or ndarray, optional): Prior mean on the coefficient
            vector q = vec(Q). Defaults to zero.
        q_sig (float, optional): Prior standard deviation on q. 
            Defaults to 1.
    """

    def __init__(
        self,
        f_list,
        ferr_list,
        A_list,
        K=3,
        p_mu=0.0,
        p_sig=0.01,
        q_mu=0.0,
        q_sig=1.0,
    ):

        # Data
        self.f = np.concatenate(f_list)
        self.ferr = np.concatenate(ferr_list)
        self._F_CInv = np.ones_like(self.f) / self.ferr ** 2

        # Design matrix
        self.A = dense_block_diag(*A_list)

        # Shape params
        self.N = np.shape(A_list[0])[1]
        self.L = len(f_list)
        self.K = K

        # Inference params
        self.p_mu = p_mu
        self.p_sig = p_sig
        self.q_mu = q_mu
        self.q_sig = q_sig

        # Initialize
        self.p = p_mu * np.ones(int(self.N * self.K))
        self.q = q_mu * np.ones(int(self.K * self.L))

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

    def model(self):
        """
        Model for the flux.
        """
        # Unroll p into P
        P = self.p.reshape(self.N, self.K)

        # P' matrix
        Pp = self.get_Pp(P)

        return np.dot(np.dot(self.A, Pp), self.q)

    def loss(self):
        """
        Return the loss function for the current parameters.

        """
        # Likelihood and prior
        lnlike = -0.5 * np.sum(
            (self.f - self.model()).reshape(-1) ** 2 * self._F_CInv.reshape(-1)
        )
        lnprior = -0.5 * np.sum(
            (self.p - self.p_mu) ** 2 / self.p_sig ** 2
        ) - 0.5 * np.sum((self.q - self.q_mu) ** 2 / self.q_sig ** 2)
        return -(lnlike + lnprior).item()

    def compute_p(self, T=1.0):
        """
        Linear solve for ``p`` given ``Q'`` given an optional temperature.

        Returns the Cholesky decomposition of the covariance of ``p``.
        """
        # Reshape q into Q
        Q = self.q.reshape(self.K, self.L)
        Qp = self.get_Qp(Q)

        # Design matrix
        A = np.dot(self.A, Qp)

        ATCInv = np.multiply(A.T, (self._F_CInv / T).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.f).reshape(-1))  # - A0)

        cinv = np.ones(len(self.p)) / self.p_sig ** 2
        mu = np.ones(len(self.p)) * self.p_mu
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + cinv)
        cho_C = cho_factor(ATCInvA)
        self.p = cho_solve(cho_C, ATCInvf + cinv * mu)

        return cho_C

    def compute_q(self, T=1.0):
        """
        Linear solve for ``q`` given ``P'`` given an optional temperature. 

        Returns the Cholesky decomposition of the covariance of ``q``.

        """
        # Reshape p into P
        P = self.p.reshape(self.N, self.K)

        # P' matrix
        Pp = self.get_Pp(P)

        # Design matrix
        A = np.dot(self.A, Pp)

        ATCInv = np.multiply(A.T, (self._F_CInv / T).reshape(-1))
        ATCInvA = ATCInv.dot(A)
        ATCInvf = np.dot(ATCInv, (self.f).reshape(-1))

        cinv = np.ones(len(self.q)) / self.q_sig ** 2
        mu = np.ones(len(self.q)) * self.q_mu
        np.fill_diagonal(ATCInvA, ATCInvA.diagonal() + cinv)
        cho_C = cho_factor(ATCInvA)

        cho_q = cho_solve(cho_C, ATCInvf + cinv * mu)

        return cho_q

    def solve(
        self,
        p=None,
        q=None,
        p_guess=None,
        q_guess=None,
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

        # Figure out what to solve for
        known = []
        if q is not None:
            known += ["q"]
        if p is not None:
            known += ["p"]

        if ("p" in known) and ("q" in known):

            # Nothing to do here but ingest the values!
            self.p = p
            self.q = q
            return self.loss(), None, None

        elif "p" in known:

            # Easy: it's a linear problem
            self.p = p
            cho_q = self.compute_q()
            return self.loss(), None, cho_q

        elif "q" in known:

            # Still a linear problem!
            self.q = q
            cho_p = self.compute_p()
            return self.loss(), cho_p, None

        else:

            # Non-linear. Let's use (N)Adam.
            # Initialize the variables
            self.p = p_guess
            self.q = q_guess

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
                best_p = self.p
                best_q = self.q

                for n in tqdm(range(niter_bilin), disable=quiet):

                    # Compute `p` using the previous `q`
                    self.compute_p(T=T_arr[n])

                    # Compute `q` using the current `p`
                    self.compute_q(T=T_arr[n])

                    loss_val[n + 1] = self.loss()

                    if loss_val[n + 1] < best_loss:
                        best_loss = loss_val[n + 1]
                        best_p = self.p
                        best_q = self.q

                self.p = best_p
                self.q = best_q

            # Non-linear solve
            if niter > 0:

                # Theano nonlienar solve. Variables:
                p = theano.shared(self.p)
                q = theano.shared(self.q)

                theano_vars = [p, q]

                # Compute the model
                A = tt.as_tensor_variable(self.A)
                P = p.reshape((self.N, self.K))
                Q = q.reshape((self.K, self.L))
                Y = tt.dot(P, Q)
                f_pred = tt.dot(A, Y.T.flatten())

                # Compute the loss
                r = tt.reshape(self.f - f_pred, (-1,))
                cov = tt.reshape(self._F_CInv, (-1,))
                lnlike = -0.5 * tt.sum(r ** 2 * cov)
                lnprior = -0.5 * tt.sum(
                    (p - self.p_mu) ** 2 / self.p_sig ** 2
                ) - 0.5 * tt.sum((q - self.q_mu) ** 2 / self.q_sig ** 2)
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
                self.p = best_p
                self.q = best_q

            # Estimate the covariance of `p` conditioned on `q`
            # and the covariance of `q` conditioned on `p`.
            p_curr = np.array(self.p)
            cho_p = self.compute_p()
            self.p = p_curr

            q_curr = np.array(self.q)
            cho_q = self.compute_q()
            self.q = q_curr

            return loss_val, cho_p, cho_q
