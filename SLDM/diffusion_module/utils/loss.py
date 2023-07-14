"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def variance_KL_loss(latents, noisy_latents, timesteps, model_pred_mean, model_pred_var, noise_scheduler,posterior_mean_coef1, posterior_mean_coef2, posterior_log_variance_clipped):
    model_pred_mean = model_pred_mean.detach()
    true_mean = (
            posterior_mean_coef1.to(device=timesteps.device)[timesteps].float()[..., None, None, None] * latents
            + posterior_mean_coef2.to(device=timesteps.device)[timesteps].float()[..., None, None, None] * noisy_latents
    )

    true_log_variance_clipped = posterior_log_variance_clipped.to(device=timesteps.device)[timesteps].float()[
        ..., None, None, None]

    if noise_scheduler.variance_type == "learned":
        model_log_variance = model_pred_var
        #model_pred_var = th.exp(model_log_variance)
    else:
        min_log = true_log_variance_clipped
        max_log = th.log(noise_scheduler.betas.to(device=timesteps.device)[timesteps].float()[..., None, None, None])
        frac = (model_pred_var + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        #model_pred_var = th.exp(model_log_variance)

    sqrt_recip_alphas_cumprod = th.sqrt(1.0 / noise_scheduler.alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / noise_scheduler.alphas_cumprod - 1)

    pred_xstart = (sqrt_recip_alphas_cumprod.to(device=timesteps.device)[timesteps].float()[
                       ..., None, None, None] * noisy_latents
                   - sqrt_recipm1_alphas_cumprod.to(device=timesteps.device)[timesteps].float()[
                       ..., None, None, None] * model_pred_mean)

    model_mean = (
            posterior_mean_coef1.to(device=timesteps.device)[timesteps].float()[..., None, None, None] * pred_xstart
            + posterior_mean_coef2.to(device=timesteps.device)[timesteps].float()[..., None, None, None] * noisy_latents
    )

    # model_mean = out["mean"] model_log_variance = out["log_variance"]
    kl = normal_kl(
        true_mean, true_log_variance_clipped, model_mean, model_log_variance
    )
    kl = kl.mean() / np.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        latents, means=model_mean, log_scales=0.5 * model_log_variance
    )
    assert decoder_nll.shape == latents.shape
    decoder_nll = decoder_nll.mean() / np.log(2.0)

    # At the first timestep return the decoder NLL,
    # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    kl_loss = th.where((timesteps == 0), decoder_nll, kl).mean()
    return kl_loss

def get_variance(noise_scheduler):
    alphas_cumprod_prev = th.cat([th.tensor([1.0]), noise_scheduler.alphas_cumprod[:-1]])

    posterior_mean_coef1 = (
            noise_scheduler.betas * th.sqrt(alphas_cumprod_prev) / (1.0 - noise_scheduler.alphas_cumprod)
    )

    posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * th.sqrt(noise_scheduler.alphas)
            / (1.0 - noise_scheduler.alphas_cumprod)
    )

    posterior_variance = (
            noise_scheduler.betas * (1.0 - alphas_cumprod_prev) / (1.0 - noise_scheduler.alphas_cumprod)
    )
    posterior_log_variance_clipped = th.log(
        th.cat([posterior_variance[1][..., None], posterior_variance[1:]])
    )
    #res = posterior_log_variance_clipped.to(device=timesteps.device)[timesteps].float()
    return posterior_mean_coef1, posterior_mean_coef2, posterior_log_variance_clipped #res[..., None, None, None]
