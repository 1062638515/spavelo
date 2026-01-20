import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import numpy as np
from sklearn.mixture import GaussianMixture
from torch.distributions import Distribution, Gamma, constraints
from torch.distributions.constraints import Constraint
from torch.distributions import Poisson as PoissonTorch
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
import jax.numpy as jnp
import warnings
from numbers import Number


class DenseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, activation="relu", dropout=0, dtype=torch.float64, norm="batchnorm"):
        super(DenseEncoder, self).__init__()
        self.layers = buildNetwork([input_dim]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.enc_mu = nn.Linear(hidden_dims[-1], latent_dim, dtype=dtype)
        self.enc_var = nn.Linear(hidden_dims[-1], latent_dim, dtype=dtype)

    def forward(self, x):
        h = self.layers(x)
        mu = self.enc_mu(h)
        var = torch.exp(self.enc_var(h).clamp(-15, 15))
        return mu, var


class VeloDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, linear_decoder=False, activation="relu", dropout=0, dtype=torch.float64, norm="batchnorm", 
                 use_state_decoder=False):
        super(VeloDecoder, self).__init__()
        self.linear_decoder = linear_decoder
        self.output_dim = output_dim
        self.use_state_decoder = use_state_decoder
        
        if not linear_decoder:
            self.rho_first_decoder = buildNetwork([latent_dim]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        else:
            self.rho_first_decoder = buildNetwork([latent_dim]+hidden_dims+[output_dim], network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.pi_first_decoder = buildNetwork([latent_dim]+hidden_dims, network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)

        # categorical pi
        # 4 states
        self.px_pi_decoder = nn.Linear(hidden_dims[-1], 4 * output_dim)

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid())

        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(hidden_dims[-1], output_dim), nn.Sigmoid())

        self.linear_scaling_tau = nn.Parameter(torch.zeros(output_dim))
        self.linear_scaling_tau_intercept = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x, latent_dim, y_spliced=None, y_unspliced=None):
        x_in = x
        if latent_dim is not None:
            mask = torch.zeros_like(x)
            mask[..., latent_dim] = 1
            x_in = x * mask
        rho_first = self.rho_first_decoder(x_in)

        if not self.linear_decoder:
            px_rho = self.px_rho_decoder(rho_first)
            px_tau = self.px_tau_decoder(rho_first)
        else:
            px_rho = nn.Sigmoid()(rho_first)
            px_tau = 1 - nn.Sigmoid()(
                rho_first * self.linear_scaling_tau.exp()
                + self.linear_scaling_tau_intercept
            )

        # cells by genes by 4
        if self.use_state_decoder:
            if y_spliced is None or y_unspliced is None:
                raise ValueError("y_spliced and y_unspliced must be provided when use_state_decoder=True")
            px_pi = self.px_pi_decoder(y_spliced, y_unspliced)
        else:
            pi_first = self.pi_first_decoder(x)
            px_pi = nn.Softplus()(
                torch.reshape(self.px_pi_decoder(pi_first), (x.shape[0], self.output_dim, -1))
            )

        return px_pi, px_rho, px_tau


class SpatialVeloDecoder(VeloDecoder):
    def __init__(self, latent_dim, hidden_dims, output_dim, linear_decoder=False, activation="relu", dropout=0, dtype=torch.float64, norm="batchnorm", use_state_decoder=False):
        super(SpatialVeloDecoder, self).__init__(latent_dim, hidden_dims, output_dim, linear_decoder=linear_decoder, activation=activation, dropout=dropout, dtype=dtype, norm=norm, use_state_decoder=use_state_decoder)
        self.linear_decoder = linear_decoder
        self.output_dim = output_dim

        self.px_alpha_rho_decoder = buildNetwork([latent_dim]+hidden_dims+[output_dim], network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
        self.px_alpha_rho_decoder.append(nn.Sigmoid())

    def forward(self, x, latent_dim, y_spliced=None, y_unspliced=None):
        px_alpha_rho = self.px_alpha_rho_decoder(x)

        px_pi, px_rho, px_tau = super(SpatialVeloDecoder, self).forward(x, latent_dim, y_spliced=y_spliced, y_unspliced=y_unspliced)
        return px_pi, px_rho, px_tau, px_alpha_rho


class MultiSpatialVeloDecoder(SpatialVeloDecoder):
    def __init__(self, latent_dim, hidden_dims, output_dim, linear_decoder=False, activation="relu", dropout=0, dtype=torch.float64, norm="batchnorm", n_states=10, n_t_decoder=6, use_state_decoder=False, alpha=None, beta=None, gamma=None):
        super(MultiSpatialVeloDecoder, self).__init__(latent_dim, hidden_dims, output_dim, linear_decoder=linear_decoder, activation=activation, dropout=dropout, dtype=dtype, norm=norm, use_state_decoder=use_state_decoder, alpha=alpha, beta=beta, gamma=gamma)

        # Override px_pi_decoder for MultiSpatialVeloDecoder (it doesn't use StateDecoder)
        self.px_pi_decoder = nn.Linear(hidden_dims[-1], n_states * output_dim)
        self.n_t_decoder = n_t_decoder

        for i in range(max(0, n_t_decoder - 2)):
            decoder = buildNetwork([latent_dim]+hidden_dims+[output_dim], network="decoder", activation=activation, dropout=dropout, dtype=dtype, norm=norm)
            self.__setattr__(f"rho_{i}", decoder)

    def forward(self, x, latent_dim, y_spliced=None, y_unspliced=None):
        tau = []
        for i in range(max(0, self.n_t_decoder - 2)):
            decoder = self.__getattr__(f"rho_{i}")
            px_rho = decoder(x)
            tau.append(px_rho)

        px_pi, px_rho, px_tau, px_alpha_rho = super(MultiSpatialVeloDecoder, self).forward(x, latent_dim, y_spliced=y_spliced, y_unspliced=y_unspliced)
        tau = [px_rho, px_tau] + tau
        return px_pi, tau, px_alpha_rho


def buildNetwork(layers, network="decoder", activation="relu", dropout=0., dtype=torch.float64, norm="both"):
    net = []
    if network == "encoder" and dropout > 0:
        net.append(nn.Dropout(p=dropout))
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i], dtype=dtype))
        if norm=="both":
            net.append(nn.BatchNorm1d(layers[i], dtype=dtype))
            net.append(nn.LayerNorm(layers[i], elementwise_affine=False, dtype=dtype))
        elif norm == "batchnorm":
            net.append(nn.BatchNorm1d(layers[i], dtype=dtype))
        elif norm == "layernorm":
            net.append(nn.LayerNorm(layers[i], elementwise_affine=False, dtype=dtype))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        elif activation=="elu":
            net.append(nn.ELU())
        if dropout > 0:
            net.append(nn.Dropout(p=dropout))
    return nn.Sequential(*net)


def gauss_cross_entropy(mu1, var1, mu2, var2):
    """
    Computes the element-wise cross entropy
    Given q(z) ~ N(z| mu1, var1)
    returns E_q[ log N(z| mu2, var2) ]
    args:
        mu1:  mean of expectation (batch, tmax, 2) tf variable
        var1: var  of expectation (batch, tmax, 2) tf variable
        mu2:  mean of integrand (batch, tmax, 2) tf variable
        var2: var of integrand (batch, tmax, 2) tf variable
    returns:
        cross_entropy: (batch, tmax, 2) tf variable
    """

    term0 = 1.8378770664093453  # log(2*pi)
    term1 = torch.log(var2)
    term2 = (var1 + mu1 ** 2 - 2 * mu1 * mu2 + mu2 ** 2) / var2

    cross_entropy = -0.5 * (term0 + term1 + term2)

    return cross_entropy


def _convert_counts_logits_to_mean_disp(
    total_count: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NB parameterizations conversion.

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.

    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.

    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta


class _Optional(Constraint):
    def __init__(self, constraint: Constraint):
        self.constraint = constraint

    def check(self, value: torch.Tensor) -> torch.Tensor:
        if value is None:
            return torch.ones(1, dtype=torch.bool)
        return self.constraint.check(value)

    def __repr__(self) -> str:
        return f"Optional({self.constraint})"


def optional_constraint(constraint: Constraint) -> Constraint:
    """Returns a wrapped constraint that allows optional values."""
    return _Optional(constraint)


def torch_lgamma_mps(x: torch.Tensor) -> torch.Tensor:
    """Used in mac Mx devices while broadcasting a tensor

    Parameters
    ----------
    x
        Data

    Returns
    -------
    lgamma tensor that perform on a copied version of the tensor
    """
    return torch.lgamma(x.contiguous())


def log_nb_positive(
    x: torch.Tensor | jnp.ndarray,
    mu: torch.Tensor | jnp.ndarray,
    theta: torch.Tensor | jnp.ndarray,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
) -> torch.Tensor | jnp.ndarray:
    """Log likelihood (scalar) of a minibatch according to a nb model.

    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    log_fn
        log function
    lgamma_fn
        log gamma function
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    return res


def _gamma(theta: torch.Tensor, mu: torch.Tensor, on_mps: bool = False) -> Gamma:
    concentration = theta
    rate = theta / mu
    # Important remark: Gamma is parametrized by the rate = 1/scale!
    gamma_d = (
        Gamma(concentration=concentration.to("cpu"), rate=rate.to("cpu"))
        if on_mps  # TODO: NEED TORCH MPS FIX for 'aten::_standard_gamma'
        else Gamma(concentration=concentration, rate=rate)
    )
    return gamma_d


class NegativeBinomial(Distribution):
    r"""Negative binomial distribution.

    One of the following parameterizations must be provided:

    (1), (`total_count`, `probs`) where `total_count` is the number of failures until
    the experiment is stopped and `probs` the success probability. (2), (`mu`, `theta`)
    parameterization, which is the one used by scvi-tools. These parameters respectively
    control the mean and inverse dispersion of the distribution.

    In the (`mu`, `theta`) parameterization, samples from the negative binomial are generated as
    follows:

    1. :math:`w \sim \textrm{Gamma}(\underbrace{\theta}_{\text{shape}},
       \underbrace{\theta/\mu}_{\text{rate}})`
    2. :math:`x \sim \textrm{Poisson}(w)`

    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    probs
        The success probability.
    mu
        Mean of the distribution.
    theta
        Inverse dispersion.
    scale
        Normalized mean expression of the distribution.
    validate_args
        Raise ValueError if arguments do not match constraints
    """

    arg_constraints = {
        "mu": optional_constraint(constraints.greater_than_eq(0)),
        "theta": optional_constraint(constraints.greater_than_eq(0)),
        "scale": optional_constraint(constraints.greater_than_eq(0)),
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        mu: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        validate_args: bool = False,
    ):
        self.on_mps = (
            mu.device.type == "mps" if total_count is None else total_count.device.type == "mps"
        )  # TODO: This is used until torch will solve the MPS issues
        self._eps = 1e-8
        if (mu is None) == (total_count is None):
            raise ValueError(
                "Please use one of the two possible parameterizations. Refer to the documentation "
                "for more information."
            )

        using_param_1 = total_count is not None and (logits is not None or probs is not None)
        if using_param_1:
            logits = logits if logits is not None else probs_to_logits(probs)
            total_count = total_count.type_as(logits)
            total_count, logits = broadcast_all(total_count, logits)
            if self.on_mps:  # TODO: This is used until torch will solve the MPS issues
                total_count, logits = total_count.contiguous(), logits.contiguous()
            mu, theta = _convert_counts_logits_to_mean_disp(total_count, logits)
            scale = mu / torch.sum(mu, dim=-1, keepdim=True)
        else:
            mu, theta = broadcast_all(mu, theta)
            if self.on_mps:
                mu, theta = mu.contiguous(), theta.contiguous()
        self.mu = mu
        self.theta = theta
        self.scale = scale
        if isinstance(mu, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = mu.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> torch.Tensor:
        return self.mu

    def get_normalized(self, key) -> torch.Tensor:
        if key == "mu":
            return self.mu
        elif key == "scale":
            return self.scale
        else:
            raise ValueError(f"normalized key {key} not recognized")

    @property
    def variance(self) -> torch.Tensor:
        return self.mean + (self.mean**2) / self.theta

    @torch.inference_mode()
    def sample(
        self,
        sample_shape: torch.Size | tuple | None = None,
    ) -> torch.Tensor:
        """Sample from the distribution."""
        sample_shape = sample_shape or torch.Size()
        gamma_d = self._gamma()  # TODO: TORCH MPS FIX - DONE ON CPU CURRENTLY
        p_means = gamma_d.sample(sample_shape)

        # Clamping as distributions objects can have buggy behaviors when
        # their parameters are too high
        l_train = torch.clamp(p_means, max=1e8)
        counts = (
            PoissonTorch(l_train).sample().to("mps")
            if self.on_mps  # TODO: NEED TORCH MPS FIX for 'aten::poisson'
            else PoissonTorch(l_train).sample()
        )  # Shape : (n_samples, n_cells_batch, n_vars)
        return counts

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            try:
                self._validate_sample(value)
            except ValueError:
                warnings.warn(
                    "The value argument must be within the support of the distribution",
                    UserWarning,
                    stacklevel=2,
                )

        lgamma_fn = torch_lgamma_mps if self.on_mps else torch.lgamma  # TODO: TORCH MPS FIX
        return log_nb_positive(
            value, mu=self.mu, theta=self.theta, eps=self._eps, lgamma_fn=lgamma_fn
        )

    def _gamma(self) -> Gamma:
        return _gamma(self.theta, self.mu, self.on_mps)

    def __repr__(self) -> str:
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]
        args_string = ", ".join(
            [
                f"{p}: "
                f"{self.__dict__[p] if self.__dict__[p].numel() == 1 else self.__dict__[p].size()}"
                for p in param_names
                if self.__dict__[p] is not None
            ]
        )
        return self.__class__.__name__ + "(" + args_string + ")"

