import torch
import numpy as np


class DDPM_SDE:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = config.sde.N
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Calculate drift coeff. and diffusion coeff. in forward SDE
        """
        beta_t = self.beta_0 + t*(self.beta_1 - self.beta_0)
        beta_t = beta_t[:, None, None, None]
        beta_t = beta_t.to(x.device)

        drift = -beta_t/2 * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x_0, t):
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        alpha_t = torch.exp(-1/2 * (t*self.beta_0 + t**2/2 * (self.beta_1 - self.beta_0)))
        alpha_t = alpha_t[:, None, None, None]
        alpha_t = alpha_t.to(x_0.device)

        mean = torch.sqrt(alpha_t)*x_0
        std = torch.sqrt(1 - alpha_t)

        return mean, std
    
    def marginal_std(self, t):
        """
        Calculate marginal q(x_t|x_0)'s std
        """
        alpha_t = torch.exp(-1 / 2 * (t * self.beta_0 + t ** 2 / 2 * (self.beta_1 - self.beta_0)))
        alpha_t = alpha_t[:, None, None, None]
        alpha_t = alpha_t.to(t.device)

        std = torch.sqrt(1 - alpha_t)

        return std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def reverse(self, score_fn, ode_sampling=False):
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          ode_sampling: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde

        # Build the class for reverse-time SDE.
        class RSDE:
            def __init__(self):
                self.N = N
                self.ode_sampling = ode_sampling

            @property
            def T(self):
                return T

            def sde(self, x, t, y=None):
                """
                Create the drift and diffusion functions for the reverse SDE/ODE.
                
                
                y is here for class-conditional generation through score SDE/ODE
                """
                
                """
                Calculate drift and diffusion for reverse SDE/ODE
                
                
                ode_sampling - True -> reverse SDE
                ode_sampling - False -> reverse ODE
                """
                drift, diffusion = sde_fn(x, t)
                score = score_fn(x, t, y=y)

                if self.ode_sampling:
                    drift = drift - diffusion ** 2 / 2 * score
                    diffusion = 0
                else:
                    drift = drift - diffusion ** 2 * score

                return drift, diffusion

        return RSDE()


class EulerDiffEqSolver:
    def __init__(self, sde, score_fn, ode_sampling = False):
        self.sde = sde
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling
        self.rsde = sde.reverse(score_fn, ode_sampling)

    def step(self, x, t, y=None):
        """
        Implement reverse SDE/ODE Euler solver
        """
        
        
        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        drift, diffusion = self.rsde.sde(x, t, y)
        dt = - 1/self.rsde.N

        x_mean = x + drift * dt
        x = x_mean + diffusion * np.sqrt(np.abs(dt)) * self.sde.prior_sampling(x.shape).to(x.device)

        return x, x_mean
