# Name(s): Justice Mason
# Project: RBNN + Gravity
# Date: 07/22

# Adpated from "Topological Obstructions and How to Avoid Them" by Esmaeili and Walters et al., 2023
# Link: https://proceedings.neurips.cc/paper_files/paper/2023/file/1c12ccfc7720f6b680edea17300bfc2b-Paper-Conference.pdf

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.func import vmap, jacfwd


from math_utils import logm_taylor, safe_atanh
from distributions_utils import UniformSphere

from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline, rational_quadratic_spline
from normflows.utils.splines import unconstrained_rational_quadratic_spline as mrqs

# code adapted from from pyro
# https://github.com/pyro-ppl/pyro/blob/dev/pyro/distributions/transforms/spline.py

def _searchsorted(sorted_sequence, values):
    return torch.sum(values[..., None] >= sorted_sequence, dim=-1) - 1

def _select_bins(x, idx):
    idx = idx.clamp(min=0, max=x.size(-1) - 1)
    if len(idx.shape) >= len(x.shape):
        x = x.reshape((1,) * (len(idx.shape) - len(x.shape)) + x.shape)
        x = x.expand(idx.shape[:-2] + (-1,) * 2)

    return x.gather(-1, idx).squeeze(-1)

def expand_for_sampling(x, r):
    if len(r.shape[:-1]) >= len(x.shape[:-2]):
        x = x.reshape((1,) * (len(r.shape) - len(x.shape)) + x.shape)
        x = x.expand(r.shape[:-1] + (-1,) * 2)
    return x

def _calculate_knots(lengths, lower, upper):
    # Cumulative widths gives x (y for inverse) position of knots
    knots = torch.cumsum(lengths, dim=-1)

    # Pad left of last dimension with 1 zero to compensate for dim lost to cumsum
    knots = F.pad(knots, pad=(1, 0), mode="constant", value=0.0)

    # Translate [0,1] knot points to [-B, B]
    knots = (upper - lower) * knots + lower

    # Convert the knot points back to lengths
    # NOTE: Are following two lines a necessary fix for accumulation (round-off) error?
    knots[..., 0] = lower
    knots[..., -1] = upper
    lengths = knots[..., 1:] - knots[..., :-1]

    return lengths, knots

def circular_spline(
    inputs,
    widths,
    heights,
    derivatives,
    lambdas,
    inverse=False,
    bound=3.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    min_lambda=0.025,
    eps=1e-6,
):
    # Ensure bound is positive
    # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B] rather than allowing arbitrary
    # corners to the bounding box. If you want a different bounding box you can apply an affine transform before and
    # after the input
    assert bound > 0.0

    widths = F.softmax(widths, dim=-1)
    heights = F.softmax(heights, dim=-1)
    derivatives = F.softplus(derivatives)
    lambdas = None if lambdas is None else torch.sigmoid(lambdas)


    num_bins = widths.shape[-1]
    # import pdb; pdb.set_trace()
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    left, right = -bound, bound
    bottom, top = -bound, bound
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask

    # outputs, logabsdet ~ (batch_dim, input_dim)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
    # then add width fraction of remaining length
    # NOTE: Do this here rather than higher up because we want everything to ensure numerical
    # stability within this function
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives

    # Cumulative widths are x (y for inverse) position of knots
    # Similarly, cumulative heights are y (x for inverse) position of knots
    widths, cumwidths = _calculate_knots(widths, left, right)
    heights, cumheights = _calculate_knots(heights, bottom, top)

    # Pad left and right derivatives with fixed values at first and last knots
    # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
    # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
    derivatives = F.pad(
        derivatives, pad=(1, 1), mode="constant", value=1.0 - min_derivative
    )

    # Get the index of the bin that each input is in
    # bin_idx ~ (batch_dim, input_dim, 1)
    bin_idx = _searchsorted(
        cumheights + eps if inverse else cumwidths + eps, inputs
    ).unsqueeze(-1)

    # Select the value for the relevant bin for the variables used in the main calculation
    input_widths = _select_bins(widths, bin_idx)
    input_cumwidths = _select_bins(cumwidths, bin_idx)
    input_cumheights = _select_bins(cumheights, bin_idx)
    input_delta = _select_bins(heights / widths, bin_idx)
    input_derivatives = _select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _select_bins(derivatives[..., 1:], bin_idx)
    input_heights = _select_bins(heights, bin_idx)

    # Calculate monotonic *linear* rational spline
    if lambdas is not None:
        lambdas = (1 - 2 * min_lambda) * lambdas + min_lambda
        input_lambdas = _select_bins(lambdas, bin_idx)

        # The weight, w_a, at the left-hand-side of each bin
        # We are free to choose w_a, so set it to 1
        wa = 1.0

        # The weight, w_b, at the right-hand-side of each bin
        # This turns out to be a multiple of the w_a
        # TODO: Should this be done in log space for numerical stability?
        wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

        # The weight, w_c, at the division point of each bin
        # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
        wc = (
            input_lambdas * wa * input_derivatives
            + (1 - input_lambdas) * wb * input_derivatives_plus_one
        ) / input_delta

        # Calculate y coords of bins
        ya = input_cumheights
        yb = input_heights + input_cumheights
        yc = ((1.0 - input_lambdas) * wa * ya + input_lambdas * wb * yb) / (
            (1.0 - input_lambdas) * wa + input_lambdas * wb
        )

        if inverse:
            numerator = (input_lambdas * wa * (ya - inputs)) * (
                inputs <= yc
            ).float() + (
                (wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc
            ) * (
                inputs > yc
            ).float()

            denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
                inputs <= yc
            ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

            theta = numerator / denominator

            outputs = theta * input_widths + input_cumwidths

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (inputs <= yc).float()
                + wb * wc * (1 - input_lambdas) * (yb - yc) * (inputs > yc).float()
            ) * input_widths

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

        else:
            theta = (inputs - input_cumwidths) / input_widths

            numerator = (wa * ya * (input_lambdas - theta) + wc * yc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            denominator = (wa * (input_lambdas - theta) + wc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * (1 - theta) + wb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            outputs = numerator / denominator

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (theta <= input_lambdas).float()
                + wb
                * wc
                * (1 - input_lambdas)
                * (yb - yc)
                * (theta > input_lambdas).float()
            ) / input_widths

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

    # Calculate monotonic *quadratic* rational spline
    else:
        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            # Make sure outside_interval input can be reversed as identity.
            discriminant = discriminant.masked_fill(outside_interval_mask, 0)
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = -(torch.log(derivative_numerator.clamp(min=1e-6)) - 2 * torch.log(denominator.clamp(min=1e-6)))

        else:
            theta = (inputs - input_cumwidths) / input_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator.clamp(min=1e-6)

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator.clamp(min=1e-6)) - 2 * torch.log(denominator.clamp(min=1e-6))

    # Apply the identity function outside the bounding box
    # import pdb; pdb.set_trace()
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0
    return outputs, logabsdet.sum(-1)

def circular_rq_spline(inputs,
                    unnormalized_widths,
                    unnormalized_heights,
                    unnormalized_derivatives,
                    inverse=False,
                    tails="circular",
                    tail_bound=3.0,
                    min_bin_width=1E-3,
                    min_bin_height=1E-3,
                    min_derivative=1E-3):
    """
    """
    if len(inputs.shape) > 2:
            unnormalized_widths = expand_for_sampling(x=unnormalized_widths, r=inputs)
            unnormalized_heights = expand_for_sampling(x=unnormalized_heights, r=inputs)
            unnormalized_derivatives = expand_for_sampling(x=unnormalized_derivatives, r=inputs)

    if inverse:
        output_int, logabsdet_cs = circular_shift_transform(x=inputs, inverse=inverse)
        outputs, logabsdet_mrqs = mrqs(inputs=output_int,
                                        unnormalized_widths=unnormalized_widths,
                                        unnormalized_heights=unnormalized_heights,
                                        unnormalized_derivatives=unnormalized_derivatives,
                                        inverse=inverse,
                                        tails="circular",
                                        tail_bound=tail_bound,
                                        min_bin_width=min_bin_width,
                                        min_bin_height=min_bin_height,
                                        min_derivative=min_derivative)

        logabsdet = logabsdet_cs + logabsdet_mrqs
    else:
        # import pdb; pdb.set_trace()
        outputs_int, logabsdet_mrqs = mrqs(inputs=inputs,
                                        unnormalized_widths=unnormalized_widths,
                                        unnormalized_heights=unnormalized_heights,
                                        unnormalized_derivatives=unnormalized_derivatives,
                                        inverse=inverse,
                                        tails="circular",
                                        tail_bound=tail_bound,
                                        min_bin_width=min_bin_width,
                                        min_bin_height=min_bin_height,
                                        min_derivative=min_derivative)
        
        outputs, logabsdet_cs = circular_shift_transform(x=outputs_int, inverse=inverse)
        logabsdet = logabsdet_mrqs + logabsdet_cs
    
    return outputs, logabsdet.sum(-1)

def monotonic_rational_spline(
    inputs,
    widths,
    heights,
    derivatives,
    lambdas=None,
    inverse=False,
    bound=3.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    min_lambda=0.025,
    eps=1e-6,
):
    """
    Calculating a monotonic rational spline (linear or quadratic) or its inverse,
    plus the log(abs(detJ)) required for normalizing flows.
    NOTE: I omit the docstring with parameter descriptions for this method since it
    is not considered "public" yet!
    """

    # Ensure bound is positive
    # NOTE: For simplicity, we apply the identity function outside [-B, B] X [-B, B] rather than allowing arbitrary
    # corners to the bounding box. If you want a different bounding box you can apply an affine transform before and
    # after the input
    assert bound > 0.0

    # added by Justice
    widths = F.softmax(widths, dim=-1)
    heights = F.softmax(heights, dim=-1)
    derivatives = F.softplus(derivatives)
    lambdas = None if lambdas is None else torch.sigmoid(lambdas)

    num_bins = widths.shape[-1]
    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    # inputs, inside_interval_mask, outside_interval_mask ~ (batch_dim, input_dim)
    left, right = -bound, bound
    bottom, top = -bound, bound
    inside_interval_mask = (inputs >= left) & (inputs <= right)
    outside_interval_mask = ~inside_interval_mask

    # outputs, logabsdet ~ (batch_dim, input_dim)
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # For numerical stability, put lower/upper limits on parameters. E.g .give every bin min_bin_width,
    # then add width fraction of remaining length
    # NOTE: Do this here rather than higher up because we want everything to ensure numerical
    # stability within this function
    widths = min_bin_width + (1.0 - min_bin_width * num_bins) * widths
    heights = min_bin_height + (1.0 - min_bin_height * num_bins) * heights
    derivatives = min_derivative + derivatives

    # Cumulative widths are x (y for inverse) position of knots
    # Similarly, cumulative heights are y (x for inverse) position of knots
    widths, cumwidths = _calculate_knots(widths, left, right)
    heights, cumheights = _calculate_knots(heights, bottom, top)

    # Pad left and right derivatives with fixed values at first and last knots
    # These are 1 since the function is the identity outside the bounding box and the derivative is continuous
    # NOTE: Not sure why this is 1.0 - min_derivative rather than 1.0. I've copied this from original implementation
    derivatives = F.pad(
        derivatives, pad=(1, 1), mode="constant", value=1.0 - min_derivative
    )

    # Get the index of the bin that each input is in
    # bin_idx ~ (batch_dim, input_dim, 1)
    bin_idx = _searchsorted(
        cumheights + eps if inverse else cumwidths + eps, inputs
    ).unsqueeze(-1)

    # Select the value for the relevant bin for the variables used in the main calculation
    input_widths = _select_bins(widths, bin_idx)
    input_cumwidths = _select_bins(cumwidths, bin_idx)
    input_cumheights = _select_bins(cumheights, bin_idx)
    input_delta = _select_bins(heights / widths, bin_idx)
    input_derivatives = _select_bins(derivatives, bin_idx)
    input_derivatives_plus_one = _select_bins(derivatives[..., 1:], bin_idx)
    input_heights = _select_bins(heights, bin_idx)

    # Calculate monotonic *linear* rational spline
    if lambdas is not None:
        lambdas = (1 - 2 * min_lambda) * lambdas + min_lambda
        input_lambdas = _select_bins(lambdas, bin_idx)

        # The weight, w_a, at the left-hand-side of each bin
        # We are free to choose w_a, so set it to 1
        wa = 1.0

        # The weight, w_b, at the right-hand-side of each bin
        # This turns out to be a multiple of the w_a
        # TODO: Should this be done in log space for numerical stability?
        wb = torch.sqrt(input_derivatives / input_derivatives_plus_one) * wa

        # The weight, w_c, at the division point of each bin
        # Recall that each bin is divided into two parts so we have enough d.o.f. to fit spline
        wc = (
            input_lambdas * wa * input_derivatives
            + (1 - input_lambdas) * wb * input_derivatives_plus_one
        ) / input_delta

        # Calculate y coords of bins
        ya = input_cumheights
        yb = input_heights + input_cumheights
        yc = ((1.0 - input_lambdas) * wa * ya + input_lambdas * wb * yb) / (
            (1.0 - input_lambdas) * wa + input_lambdas * wb
        )

        if inverse:
            numerator = (input_lambdas * wa * (ya - inputs)) * (
                inputs <= yc
            ).float() + (
                (wc - input_lambdas * wb) * inputs + input_lambdas * wb * yb - wc * yc
            ) * (
                inputs > yc
            ).float()

            denominator = ((wc - wa) * inputs + wa * ya - wc * yc) * (
                inputs <= yc
            ).float() + ((wc - wb) * inputs + wb * yb - wc * yc) * (inputs > yc).float()

            theta = numerator / denominator

            outputs = theta * input_widths + input_cumwidths

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (inputs <= yc).float()
                + wb * wc * (1 - input_lambdas) * (yb - yc) * (inputs > yc).float()
            ) * input_widths

            if torch.any(derivative_numerator < 0.0):
                import pdb; pdb.set_trace()

            if torch.any(torch.abs(denominator) < 0.0):
                import pdb; pdb.set_trace()

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

        else:
            theta = (inputs - input_cumwidths) / input_widths

            numerator = (wa * ya * (input_lambdas - theta) + wc * yc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * yc * (1 - theta) + wb * yb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            denominator = (wa * (input_lambdas - theta) + wc * theta) * (
                theta <= input_lambdas
            ).float() + (wc * (1 - theta) + wb * (theta - input_lambdas)) * (
                theta > input_lambdas
            ).float()

            outputs = numerator / denominator

            derivative_numerator = (
                wa * wc * input_lambdas * (yc - ya) * (theta <= input_lambdas).float()
                + wb
                * wc
                * (1 - input_lambdas)
                * (yb - yc)
                * (theta > input_lambdas).float()
            ) / input_widths

            if torch.any(derivative_numerator < 0.0):
                import pdb; pdb.set_trace()

            if torch.any(torch.abs(denominator) < 0.0):
                import pdb; pdb.set_trace()

            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(
                torch.abs(denominator)
            )

    # Calculate monotonic *quadratic* rational spline
    else:
        if inverse:
            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c
            # Make sure outside_interval input can be reversed as identity.
            discriminant = discriminant.masked_fill(outside_interval_mask, 0)
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            if (denominator < 0.0).any():
                import pdb; pdb.set_trace()

            if (derivative_numerator < 0.0).any():
                import pdb; pdb.set_trace()

            logabsdet = -(torch.log(derivative_numerator) - 2 * torch.log(denominator))

        else:
            theta = (inputs - input_cumwidths) / input_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            if (denominator < 0.0).any():
                import pdb; pdb.set_trace()

            if (derivative_numerator < 0.0).any():
                import pdb; pdb.set_trace()
            
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    assert torch.all(numerator.isnan() == False), f"The value(s) that broke the loop were: {numerator[numerator.isnan()]}."
    assert torch.all(numerator.isinf() == False), f"The value(s) that broke the loop were: {numerator[numerator.isinf()]}."
    assert torch.all(denominator.isnan() == False), f"The value(s) that broke the loop were: {denominator[denominator.isnan()]}."
    assert torch.all(denominator.isinf() == False), f"The value(s) that broke the loop were: {denominator[denominator.isinf()]}."
    # assert (torch.abs(denominator) > 0.0).all(), f"The value(s) that broke the loop were: {denominator[torch.abs(denominator) < 0.0]}."
    # assert (torch.abs(denominator) > 1e-16).all(), f"The value(s) that broke the loop were: {denominator[torch.abs(denominator) < 1e-16]}."
    
    # Apply the identity function outside the bounding box
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0.0
    return outputs, logabsdet.sum(-1)

def cylindrical_spline(
    inputs,
    cyn_nn,
    widths_g,
    heights_g,
    derivatives_g,
    lambdas_g=None,
    inverse=False,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
    min_lambda=0.025,
    eps=1e-6,
):
    # assume we've transformed from spherical to cylindrical coordinates
    if inverse:
        # initialize
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs).sum(-1)
        
        # split input (z', r') = (f(z), g(r)) where :math: `r' \in \R`
        z_prime, r_prime = inputs[..., :-1], inputs[..., -1].unsqueeze(-1)
        
        # convert from R^2 to theta
        if not torch.allclose(torch.linalg.norm(z_prime, dim=-1), torch.ones_like(torch.linalg.norm(z_prime, dim=-1)), rtol=1e-1, atol=1e-1):
            import pdb; pdb.set_trace()

        theta_prime, log_r2_theta = theta_to_r2_transform(x=z_prime, inverse=True)
        assert log_r2_theta.isnan().any() == False and log_r2_theta.isinf().any() == False
        logabsdet -= log_r2_theta
        
        # calculate g(r) for a monotonic rational-quadratic or linear spline
        if len(r_prime.shape) > 2:
            widths_g = expand_for_sampling(x=widths_g, r=r_prime)
            heights_g = expand_for_sampling(x=heights_g, r=r_prime)
            derivatives_g = expand_for_sampling(x=derivatives_g, r=r_prime)

        r_, logabsdet_g = rational_quadratic_spline(inputs=r_prime,
                                        unnormalized_widths=widths_g,
                                        unnormalized_heights= heights_g,
                                        unnormalized_derivatives=derivatives_g,
                                        inverse=inverse,
                                        left=-1.0,
                                        right=1.0,
                                        bottom=-1.0,
                                        top=1.0,
                                        min_bin_width=min_bin_width,
                                        min_bin_height=min_bin_height,
                                        min_derivative=min_derivative)
        
        assert logabsdet_g.isnan().any() == False and logabsdet_g.isinf().any() == False
        r, logabsdet_r_ = tanh_norm_transform(x=r_, inverse=False, radius=1.)

        # print(f"\n Max of r_prime: {r_prime.max()}; Min of r_prime: {r_prime.min()} ... \n")
        # print(f"\n Max of g^{-1}(r): {r_.max()}; Min of g(r): {r_.min()} ... \n")
        # print(f"\n Max of r: {r.max()}; Min of r: {r.min()} ... \n")
        assert torch.all(r <= 1.) and torch.all(r >=-1.), AssertionError(f"Inverse: The output of tanh isn't less than or equal to one: {r[(r > 1. and r < -1.)]}")
        
        '''
        if len(logabsdet_g.shape) > 1:
            import pdb; pdb.set_trace()
        '''
        logabsdet -= logabsdet_g.squeeze(-1)
        logabsdet -= logabsdet_r_.squeeze(-1)

        # compute parameters for circular spline using r_prime = g_r
        D = 1 # dimension of theta
        C = widths_g.size()[-1] # same number of bins as other spline

        if lambdas_g:
            y_g = cyn_nn(r_prime)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
                
            uw, uh, ud, ul = torch.split(y_g, [D*C, D*C, D*(C-1), D*C], dim=-1)
            uw, uh, ud, ul = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1), ul.reshape(-1,D,C)
        else:
            y_g = cyn_nn(r_prime)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)

            uw, uh, ud = torch.split(y_g, [D*C, D*C, D*(C-1)], dim=-1)
            uw, uh, ud = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1)
            ul = None

        # use tanh
        # import pdb; pdb.set_trace()
        z_prime_tanh, logabsdet_tanh = tanh_norm_transform(x=theta_prime, radius=math.pi, inverse=False)
        assert logabsdet_tanh.isnan().any() == False and logabsdet_tanh.isinf().any() == False
        assert torch.all(z_prime_tanh <= math.pi) and torch.all(z_prime_tanh >= -math.pi), AssertionError(f"Inverse: The output of tanh isn't less than or equal to pi: {z_prime_tanh[(z_prime_tanh > math.pi and z_prime_tanh < -math.pi)]}")
        logabsdet -= logabsdet_tanh

        # calculate :math: `f^{-1}(z; g(r))` using the inverse of circular spline parameterized by a function of the output of r' = g(r)
        z_theta, logabsdet_f = circular_spline(inputs=z_prime_tanh.unsqueeze(-1),
                                           widths=uw,
                                           heights=uh,
                                           derivatives=ud,
                                           lambdas=ul,
                                           inverse=True,
                                           bound=math.pi,
                                           min_bin_height=min_bin_height,
                                           min_bin_width=min_bin_width,
                                           min_derivative=min_derivative,
                                           min_lambda=min_lambda,
                                           eps=eps)

        assert logabsdet_f.isnan().any() == False and logabsdet_f.isinf().any() == False
        z, log_theta_r2 = r2_to_theta_transform(x=z_theta, inverse=True)
        
        assert log_theta_r2.isnan().any() == False and log_theta_r2.isinf().any() == False
        logabsdet -= log_theta_r2.sum(-1)

        '''
        if len(logabsdet_g.shape) > 1:
            import pdb; pdb.set_trace()
        '''
        
        if len(z.shape)!= len(r.squeeze(-1).shape):
            outputs = torch.cat((z, r), dim=-1)
        else:
            outputs = torch.cat((z, r.squeeze(-1)), dim=-1)

        logabsdet -= logabsdet_f
        assert logabsdet.isnan().any() == False and logabsdet.isinf().any() == False
        return outputs, logabsdet
    
    else:
        # initialize
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs).sum(-1)
        
        # split input x = (z, r) where :math: `r \in \R`
        z = inputs[..., :-1]
        r = inputs[..., -1].unsqueeze(-1)

        # convert z from R^2 to polar
        theta, log_r2_theta = r2_to_theta_transform(x=z, inverse=False)
        assert torch.all(log_r2_theta.isnan() == False) and torch.all(log_r2_theta.isinf() == False)
        logabsdet += log_r2_theta
        
        # calculate g(r) for a monotonic rational-quadratic or linear spline
        if len(r.shape) > 2:
            widths_g = expand_for_sampling(x=widths_g, r=r)
            heights_g = expand_for_sampling(x=heights_g, r=r)
            derivatives_g = expand_for_sampling(x=derivatives_g, r=r)

            #import pdb; pdb.set_trace()
            
        g_r_, logabsdet_g = rational_quadratic_spline(inputs=r,
                                        unnormalized_widths=widths_g,
                                        unnormalized_heights= heights_g,
                                        unnormalized_derivatives=derivatives_g,
                                        inverse=inverse,
                                        left=-1.0,
                                        right=1.0,
                                        bottom=-1.0,
                                        top=1.0,
                                        min_bin_width=min_bin_width,
                                        min_bin_height=min_bin_height,
                                        min_derivative=min_derivative)


        if torch.any(g_r_.isnan()) == True:
            import pdb; pdb.set_trace()
            
        if torch.any(g_r_.isinf()) == True:
            import pdb; pdb.set_trace()
            
        assert torch.all(g_r_.isinf() == False)

        # confirmation prints
        # print(f"\n Max of r: {r.max()}; Min of r: {r.min()} ... \n")
        # print(f"\n Max of g(r): {g_r_.max()}; Min of g(r): {g_r_.min()} ... \n")
        # print(f"\n Max of $\theta$: {theta.max()}; Min of $\theta$: {theta.min()} ... \n")

        assert torch.all(logabsdet_g.isnan() == False)
        assert torch.all(logabsdet_g.isinf() == False)
        
        g_r, logabsdet_g_tanh = tanh_norm_transform(x=g_r_, inverse=False, radius=1.)
        # print(f"\n Max of tanh(g(r)): {g_r.max()}; Min of tanh(g(r)): {g_r.min()} ... \n")
        assert torch.all(g_r <= 1.) and torch.all(g_r >=-1.), AssertionError(f"Forward: The output of tanh isn't less than or equal to one: {g_r[(g_r > 1. and g_r < -1.)]}")
        
        logabsdet += logabsdet_g.squeeze(-1) # nans here
        logabsdet += logabsdet_g_tanh

        # compute parameters for circular spline using g_r
        D = 1
        C = widths_g.size()[-1] # same number of bins as other spline

        if lambdas_g is not None:
            y_g = cyn_nn(g_r)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
            
            # import pdb; pdb.set_trace()
            uw, uh, ud, ul = torch.split(y_g, [D*C, D*C, D*(C-1), D*C], dim=-1)
            uw, uh, ud, ul = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1), ul.reshape(-1,D,C)
        else:
            y_g = cyn_nn(g_r)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
            
            uw, uh, ud = torch.split(y_g, [D*C, D*C, D*(C-1)], dim=-1)
            uw, uh, ud = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1)
            ul = None
        
        # calculate f(z; g(r)) using the circular spline parameterized by a function of the output of g(r)
        f_z_, logabsdet_f = circular_spline(inputs=theta.unsqueeze(-1),
                                           widths=uw,
                                           heights=uh,
                                           derivatives=ud,
                                           lambdas=ul,
                                           inverse=False,
                                           bound=math.pi,
                                           min_bin_height=min_bin_height,
                                           min_bin_width=min_bin_width,
                                           min_derivative=min_derivative,
                                           min_lambda=min_lambda,
                                           eps=eps)

        assert logabsdet_f.isnan().any() == False and logabsdet_f.isinf().any() == False

        # apply tanh to the output z
        f_z_tanh, logabsdet_tanh = tanh_norm_transform(x=f_z_, inverse=False)
        assert logabsdet_tanh.isnan().any() == False and logabsdet_tanh.isinf().any() == False
        assert torch.all(f_z_tanh <= math.pi) and torch.all(f_z_tanh >= -math.pi), AssertionError(f"Forward: The output of tanh isn't less than or equal to pi: {f_z_tanh[(f_z_tanh > math.pi and f_z_tanh < -math.pi)]}")
        logabsdet += logabsdet_tanh

        # apply tanh to the output r
        # print(f"\n Max of f(z; g(r)): {f_z_.max()}; Min of f(z; g(r)): {f_z_.min()} ... \n")
        # print(f"\n Max of tanh(f(z; g(r))): {f_z_tanh.max()}; Min of tanh(f(z; g(r))): {f_z_tanh.min()} ... \n")

        # convert from theta to R^2
        f_z, log_theta_r2 = theta_to_r2_transform(x=f_z_tanh, inverse=False)
        assert log_theta_r2.isnan().any() == False and log_theta_r2.isinf().any() == False
        logabsdet += log_theta_r2.sum(-1)
        
        outputs = torch.cat((f_z, g_r), dim=-1)
        logabsdet += logabsdet_f
        return outputs, logabsdet

def cylindrical_spline_v2(inputs,
                        cyn_nn,
                        affine_weights,
                        widths_g,
                        heights_g,
                        derivatives_g,
                        lambdas_g=None,
                        inverse=False,
                        min_bin_width=1e-3,
                        min_bin_height=1e-3,
                        min_derivative=1e-3,
                        min_lambda=0.025,
                        eps=1e-6,
):
    if inverse:
        # initialize
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs).sum(-1)
        r_mu, r_sigma, z_mu, z_sigma = torch.split(affine_weights, [1, 1, 1, 1], dim=-1)

        # split input (z', r') = (f(z), g(r)) where :math: `r' \in \R`
        z_prime, r_prime = inputs[..., :-1], inputs[..., -1]

        theta_z_prime, log_r2_theta = theta_to_r2_transform(x=z_prime, inverse=inverse)
        assert log_r2_theta.isnan().any() == False and log_r2_theta.isinf().any() == False
        logabsdet -= log_r2_theta

        # r' -> inverse tanh -> inverse US -> inverse affine
        g_r_prime, logabsdet_r_tanh_inv = tanh_norm_transform(x=r_prime, inverse=False, radius=1.)
        logabsdet -= logabsdet_r_tanh_inv.squeeze(-1)

        # calculate g(r) for a monotonic rational-quadratic or linear spline
        if len(r_prime.shape) > 2:
            widths_g = expand_for_sampling(x=widths_g, r=g_r_prime)
            heights_g = expand_for_sampling(x=heights_g, r=g_r_prime)
            derivatives_g = expand_for_sampling(x=derivatives_g, r=g_r_prime)

        # import pdb; pdb.set_trace()
        # print(g_r_prime.shape)
        r_us_inv, logabsdet_r_us_inv = unconstrained_rational_quadratic_spline(inputs=g_r_prime.unsqueeze(-1),
                                                                            unnormalized_widths=widths_g,
                                                                            unnormalized_heights=heights_g,
                                                                            unnormalized_derivatives=derivatives_g,
                                                                            inverse=inverse,
                                                                            tails="linear",
                                                                            tail_bound=1.0-eps,
                                                                            min_bin_width=min_bin_width,
                                                                            min_bin_height=min_bin_height,
                                                                            min_derivative=min_derivative)
        
        assert logabsdet_r_us_inv.isnan().any() == False and logabsdet_r_us_inv.isinf().any() == False
        logabsdet -= logabsdet_r_us_inv.squeeze(-1)

        r, logabsdet_r = affine_transform(x=r_us_inv, mu=r_mu, logsigma=r_sigma)
        logabsdet -= logabsdet_r

        # theta_z -> inverse tanh -> CS -> inverse affine
        theta_z_tanh_inv, logabsdet_tanh_inv = tanh_norm_transform(x=theta_z_prime, inverse=False)
        logabsdet -= logabsdet_tanh_inv

        # compute parameters for circular spline using r_prime = g_r
        D = 1 # dimension of theta
        C = widths_g.size()[-1] # same number of bins as other spline

        if lambdas_g:
            y_g = cyn_nn(g_r_prime)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
                
            uw, uh, ud, ul = torch.split(y_g, [D*C, D*C, D*(C-1), D*C], dim=-1)
            uw, uh, ud, ul = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1), ul.reshape(-1,D,C)
        else:
            y_g = cyn_nn(g_r_prime.unsqueeze(-1))
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)

            uw, uh, ud = torch.split(y_g, [D*C, D*C, D*(C-1)], dim=-1)
            uw, uh, ud = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1)
            ul = None
        
        # calculate :math: `f^{-1}(z; g(r))` using the inverse of circular spline parameterized by a function of the output of r' = g(r)
        #import pdb; pdb.set_trace()
        
        z_theta_cs, logabsdet_f = circular_spline(inputs=theta_z_tanh_inv.unsqueeze(-1),
                                           widths=uw,
                                           heights=uh,
                                           derivatives=ud,
                                           lambdas=ul,
                                           inverse=inverse,
                                           bound=math.pi-eps,
                                           min_bin_height=min_bin_height,
                                           min_bin_width=min_bin_width,
                                           min_derivative=min_derivative,
                                           min_lambda=min_lambda,
                                           eps=eps)

        assert logabsdet_f.isnan().any() == False and logabsdet_f.isinf().any() == False
        logabsdet -= logabsdet_f

        z_theta, logabsdet_aff_inv = affine_transform(x=z_theta_cs, mu=z_mu, logsigma=z_sigma, inverse=inverse)
        logabsdet -= logabsdet_aff_inv

        z, log_theta_r2 = r2_to_theta_transform(x=z_theta, inverse=inverse)
        
        assert log_theta_r2.isnan().any() == False and log_theta_r2.isinf().any() == False
        logabsdet -= log_theta_r2.sum(-1)

        # concatenate (z, r)
        if len(z.shape)!= len(r.squeeze(-1).shape):
            outputs = torch.cat((z, r), dim=-1)
        else:
            outputs = torch.cat((z, r.squeeze(-1)), dim=-1)

        # import pdb; pdb.set_trace()
        return outputs, logabsdet
    else:
        # initialize
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs).sum(-1)
        r_mu, r_sigma, z_mu, z_sigma = torch.split(affine_weights, [1, 1, 1, 1], dim=-1)

        # split input x = (z, r) where :math: `r \in \R`
        z = inputs[..., :-1]
        r = inputs[..., -1].unsqueeze(-1)

        # convert z from R^2 to polar
        theta_z, log_r2_theta = r2_to_theta_transform(x=z, inverse=False)
        assert torch.all(log_r2_theta.isnan() == False) and torch.all(log_r2_theta.isinf() == False)
        #if len(log_r2_theta.shape) > 1:
            # import pdb; pdb.set_trace()

        # print(log_r2_theta)
        logabsdet += log_r2_theta

        # map r through affine -> US -> tanh
        r_aff, sum_log_abs_det_aff_r =  affine_transform(r, mu=r_mu, logsigma=r_sigma, inverse=inverse)
        logabsdet += sum_log_abs_det_aff_r
        
        # print(r_aff.shape)
        # calculate g(r) for a monotonic rational-quadratic or linear spline
        if len(r.shape) > 2:
            widths_g = expand_for_sampling(x=widths_g, r=r)
            heights_g = expand_for_sampling(x=heights_g, r=r)
            derivatives_g = expand_for_sampling(x=derivatives_g, r=r)

            #import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()

        # push r throught univariate rational spline
        '''
        g_r_us, logabsdet_g = rational_quadratic_spline(inputs=r_aff,
                                        unnormalized_widths=widths_g,
                                        unnormalized_heights= heights_g,
                                        unnormalized_derivatives=derivatives_g,
                                        inverse=inverse,
                                        left=-1.0,
                                        right=1.0,
                                        bottom=-1.0,
                                        top=1.0,
                                        min_bin_width=min_bin_width,
                                        min_bin_height=min_bin_height,
                                        min_derivative=min_derivative)
        '''
        g_r_us, logabsdet_g = unconstrained_rational_quadratic_spline(inputs=r_aff,
                                                                        unnormalized_widths=widths_g,
                                                                        unnormalized_heights=heights_g,
                                                                        unnormalized_derivatives=derivatives_g,
                                                                        inverse=inverse,
                                                                        tails="linear",
                                                                        tail_bound=1.0-eps,
                                                                        min_bin_width=min_bin_width,
                                                                        min_bin_height=min_bin_height,
                                                                        min_derivative=min_derivative)
        logabsdet += logabsdet_g.squeeze(-1)

        # push it through tanh
        g_r, logabsdet_g_tanh = tanh_norm_transform(x=g_r_us, inverse=inverse, radius=1.)
        logabsdet += logabsdet_g_tanh

        # compute parameters for circular spline using g_r
        D = 1
        C = widths_g.size()[-1] # same number of bins as other spline

        if lambdas_g is not None:
            y_g = cyn_nn(g_r_us)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
            
            # import pdb; pdb.set_trace()
            uw, uh, ud, ul = torch.split(y_g, [D*C, D*C, D*(C-1), D*C], dim=-1)
            uw, uh, ud, ul = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1), ul.reshape(-1,D,C)
        else:
            y_g = cyn_nn(g_r_us)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
            
            #import pdb; pdb.set_trace()
            uw, uh, ud = torch.split(y_g, [D*C, D*C, D*(C-1)], dim=-1)
            uw, uh, ud = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1)
            ul = None
        
        # import pdb; pdb.set_trace()

        theta_z_aff, logabsdet_theta_z_aff = affine_transform(x=theta_z.unsqueeze(-1), mu=z_mu, logsigma=z_sigma, inverse=inverse)

        # import pdb; pdb.set_trace()
        logabsdet += logabsdet_theta_z_aff
        
        # calculate f(z; g(r)) using the circular spline parameterized by a function of the output of g(r)
        f_z_, logabsdet_f = circular_spline(inputs=theta_z_aff,
                                           widths=uw,
                                           heights=uh,
                                           derivatives=ud,
                                           lambdas=ul,
                                           inverse=False,
                                           bound=math.pi-eps,
                                           min_bin_height=min_bin_height,
                                           min_bin_width=min_bin_width,
                                           min_derivative=min_derivative,
                                           min_lambda=min_lambda,
                                           eps=eps)

        assert logabsdet_f.isnan().any() == False and logabsdet_f.isinf().any() == False

        # import pdb; pdb.set_trace()
        # apply tanh to the output z
        f_z_tanh, logabsdet_tanh = tanh_norm_transform(x=f_z_, inverse=inverse)

        # import pdb; pdb.set_trace()
        logabsdet += logabsdet_tanh

        # convert from theta to R^2
        f_z, log_theta_r2 = theta_to_r2_transform(x=f_z_tanh, inverse=False)
        assert log_theta_r2.isnan().any() == False and log_theta_r2.isinf().any() == False
        logabsdet += log_theta_r2.sum(-1)
        
        outputs = torch.cat((f_z, g_r), dim=-1)
        logabsdet += logabsdet_f
        # import pdb; pdb.set_trace()
        return outputs, logabsdet

def cylindrical_spline_v3(inputs,
                        cyn_nn,
                        affine_weights,
                        widths_g,
                        heights_g,
                        derivatives_g,
                        lambdas_g=None,
                        inverse=False,
                        min_bin_width=1e-3,
                        min_bin_height=1e-3,
                        min_derivative=1e-3,
                        eps=1e-3
):
    if inverse:
        # initialize
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs).sum(-1)
        r_mu, r_sigma, z_mu, z_sigma = torch.split(affine_weights, [1, 1, 1, 1], dim=-1)

        # split input (z', r') = (f(z), g(r)) where :math: `r' \in \R`
        z_prime, r_prime = inputs[..., :-1], inputs[..., -1]

        theta_z_prime, log_r2_theta = theta_to_r2_transform(x=z_prime, inverse=inverse)
        assert log_r2_theta.isnan().any() == False and log_r2_theta.isinf().any() == False
        logabsdet -= log_r2_theta

        # calculate g(r) for a monotonic rational-quadratic or linear spline
        if len(r_prime.shape) > 2:
            widths_g = expand_for_sampling(x=widths_g, r=r_prime)
            heights_g = expand_for_sampling(x=heights_g, r=r_prime)
            derivatives_g = expand_for_sampling(x=derivatives_g, r=r_prime)

        r_us_inv, logabsdet_r_us_inv = unconstrained_rational_quadratic_spline(inputs=r_prime.unsqueeze(-1),
                                                                            unnormalized_widths=widths_g,
                                                                            unnormalized_heights=heights_g,
                                                                            unnormalized_derivatives=derivatives_g,
                                                                            inverse=inverse,
                                                                            tails="linear",
                                                                            tail_bound=0.9,
                                                                            min_bin_width=min_bin_width,
                                                                            min_bin_height=min_bin_height,
                                                                            min_derivative=min_derivative)
        
        assert logabsdet_r_us_inv.isnan().any() == False and logabsdet_r_us_inv.isinf().any() == False
        logabsdet -= logabsdet_r_us_inv.squeeze(-1)
        r = r_us_inv

        # theta_z -> CS -> inverse affine
        # compute parameters for circular spline using r_prime = g_r
        D = 1 # dimension of theta
        C = widths_g.size()[-1] # same number of bins as other spline

        if lambdas_g:
            y_g = cyn_nn(r_prime)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
                
            uw, uh, ud, ul = torch.split(y_g, [D*C, D*C, D*(C-1), D*C], dim=-1)
            uw, uh, ud, ul = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1), ul.reshape(-1,D,C)
        else:
            y_g = cyn_nn(r_prime.unsqueeze(-1))
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)

            uw, uh, ud = torch.split(y_g, [D*C, D*C, D*(C-1)], dim=-1)
            uw, uh, ud = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1)
            ul = None
        
        # calculate :math: `f^{-1}(z; g(r))` using the inverse of circular spline parameterized by a function of the output of r' = g(r)
        z_theta_cs, logabsdet_f = circular_rq_spline(inputs=theta_z_prime.unsqueeze(-1),
                                                        unnormalized_widths=uw,
                                                        unnormalized_heights=uh,
                                                        unnormalized_derivatives=ud,
                                                        inverse=inverse,
                                                        tails="circular",
                                                        tail_bound=3.,
                                                        min_bin_height=min_bin_height,
                                                        min_bin_width=min_bin_width,
                                                        min_derivative=min_derivative,)

        assert torch.allclose(torch.linalg.norm(z_theta_cs, dim=-1), torch.ones_like(torch.linalg.norm(z_theta_cs, dim=-1)))
        assert logabsdet_f.isnan().any() == False and logabsdet_f.isinf().any() == False
        logabsdet -= logabsdet_f

        z_theta, logabsdet_aff_inv = affine_transform(x=z_theta_cs, mu=z_mu, logsigma=z_sigma, inverse=inverse)
        logabsdet -= logabsdet_aff_inv

        z, log_theta_r2 = r2_to_theta_transform(x=z_theta, inverse=inverse)
        
        assert log_theta_r2.isnan().any() == False and log_theta_r2.isinf().any() == False
        logabsdet -= log_theta_r2.sum(-1)

        # concatenate (z, r)
        if len(z.shape)!= len(r.squeeze(-1).shape):
            outputs = torch.cat((z, r), dim=-1)
        else:
            outputs = torch.cat((z, r.squeeze(-1)), dim=-1)

        return outputs, logabsdet
    else:
        # initialize
        outputs = torch.zeros_like(inputs)
        logabsdet = torch.zeros_like(inputs).sum(-1)
        r_mu, r_sigma, z_mu, z_sigma = torch.split(affine_weights, [1, 1, 1, 1], dim=-1)

        # split input x = (z, r) where :math: `r \in \R`
        z = inputs[..., :-1]
        r = inputs[..., -1].unsqueeze(-1)

        # convert z from R^2 to polar
        theta_z, log_r2_theta = r2_to_theta_transform(x=z, inverse=False)
        assert torch.all(log_r2_theta.isnan() == False) and torch.all(log_r2_theta.isinf() == False)

        logabsdet += log_r2_theta

        # calculate g(r) for a monotonic rational-quadratic or linear spline
        if len(r.shape) > 2:
            widths_g = expand_for_sampling(x=widths_g, r=r)
            heights_g = expand_for_sampling(x=heights_g, r=r)
            derivatives_g = expand_for_sampling(x=derivatives_g, r=r)


        # push r throught univariate rational spline
        g_r_us, logabsdet_g = unconstrained_rational_quadratic_spline(inputs=r,
                                                                        unnormalized_widths=widths_g,
                                                                        unnormalized_heights=heights_g,
                                                                        unnormalized_derivatives=derivatives_g,
                                                                        inverse=inverse,
                                                                        tails="linear",
                                                                        tail_bound=0.9,
                                                                        min_bin_width=min_bin_width,
                                                                        min_bin_height=min_bin_height,
                                                                        min_derivative=min_derivative)
        logabsdet += logabsdet_g.squeeze(-1)

        # push it through tanh
        # compute parameters for circular spline using g_r
        D = 1
        C = widths_g.size()[-1] # same number of bins as other spline

        if lambdas_g is not None:
            y_g = cyn_nn(g_r_us)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
            
            # import pdb; pdb.set_trace()
            uw, uh, ud, ul = torch.split(y_g, [D*C, D*C, D*(C-1), D*C], dim=-1)
            uw, uh, ud, ul = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1), ul.reshape(-1,D,C)
        else:
            y_g = cyn_nn(g_r_us)
            if len(y_g.size()[:-1]) > 1:
                y_g = y_g.mean(0)
            
            uw, uh, ud = torch.split(y_g, [D*C, D*C, D*(C-1)], dim=-1)
            uw, uh, ud = uw.reshape(-1,D,C), uh.reshape(-1,D,C), ud.reshape(-1,D,C-1)
            ul = None
        
        # import pdb; pdb.set_trace()

        theta_z_aff, logabsdet_theta_z_aff = affine_transform(x=theta_z.unsqueeze(-1), mu=z_mu, logsigma=z_sigma, inverse=inverse)

        # import pdb; pdb.set_trace()
        logabsdet += logabsdet_theta_z_aff
        
        # calculate f(z; g(r)) using the circular spline parameterized by a function of the output of g(r)
        f_z_, logabsdet_f = circular_rq_spline(inputs=theta_z_aff,
                                           unnormalized_widths=uw,
                                           unnormalized_heights=uh,
                                           unnormalized_derivatives=ud,
                                           inverse=False,
                                           tails="circular",
                                           tail_bound=3.0,
                                           min_bin_height=min_bin_height,
                                           min_bin_width=min_bin_width,
                                           min_derivative=min_derivative)

        assert logabsdet_f.isnan().any() == False and logabsdet_f.isinf().any() == False

        # import pdb; pdb.set
        # import pdb; pdb.set_trace()
        # logabsdet += logabsdet_tanh
        f_z_tanh = f_z_
        g_r = g_r_us

        assert torch.allclose(torch.linalg.norm(g_r, dim=-1), torch.ones_like(torch.linalg.norm(g_r, dim=-1)))
        # convert from theta to R^2
        f_z, log_theta_r2 = theta_to_r2_transform(x=f_z_tanh, inverse=False)
        assert log_theta_r2.isnan().any() == False and log_theta_r2.isinf().any() == False
        logabsdet += log_theta_r2.sum(-1)
        
        outputs = torch.cat((f_z, g_r), dim=-1)
        logabsdet += logabsdet_f
        # import pdb; pdb.set_trace()
        return outputs, logabsdet
    
def clamp_preserve_gradients(x, min, max):
    # This helper function clamps gradients but still passes through the gradient in clamped regions
    return x + (x.clamp(min, max) - x).detach()


def affine_transform(x, mu, logsigma, clip=False, presevre_gradients_after_clip=False, clip_val=5., inverse=False):
    if clip:
        if presevre_gradients_after_clip:
            logsigma = clamp_preserve_gradients(logsigma, -clip_val, clip_val)
            mu = clamp_preserve_gradients(mu, -clip_val, clip_val)
        else:
            logsigma.clip_(-clip_val, clip_val)
            mu.clip_(-clip_val, clip_val)
    sum_log_abs_det_jacobians = (F.softplus(logsigma) + 1e-7).log().sum(-1)
    if inverse:
        z = (x - mu) / (F.softplus(logsigma) + 1e-7)
        sum_log_abs_det_jacobians = -sum_log_abs_det_jacobians
    else:
        z = mu + (F.softplus(logsigma) + 1e-7) * x
    
    # import pdb; pdb.set_trace()
    return z, sum_log_abs_det_jacobians.expand(*x.size()[:-1])

def permutation_transform(x, permutation, inverse=False):
    """"""
    if inverse:
        inverse_perm = torch.sort(permutation)[1]
        z = x[..., inverse_perm]
        sum_log_abs_det_jacobians = torch.zeros(*x.size()[:-1], device=x.device)
    else:
        # import pdb; pdb.set_trace()
        z = x[..., permutation]
        sum_log_abs_det_jacobians = torch.zeros(*x.size()[:-1], device=x.device)
    return z, sum_log_abs_det_jacobians.expand(*x.size()[:-1])

def r2_to_theta_transform(x, epsilon: float = 1e-6, inverse=False):
    """"""
    if inverse:
        # import pdb; pdb.set_trace()
        z1 = torch.cos(x)
        z2 = torch.sin(x)
        z = torch.cat((z1, z2), dim=-1)

        # import pdb; pdb.set_trace()
        sum_log_abs_det_jac = torch.log(torch.abs(torch.cos(x)).clamp(min=1e-8)) # torch.log(torch.sin(x).abs().clamp(min=1e-8)) + torch.log(torch.cos(x).abs().clamp(min=1e-8))

    else:
        assert x.size()[-1] == 2
        # assert torch.all(torch.linalg.norm(x, dim=-1) - 1. < 1e-5)
        # x = (x,y)
        z = torch.atan2(x[..., -1], x[..., -2])
        sum_log_abs_det_jac = -torch.log(torch.abs(x[..., -2]).clamp(min=1e-8)) # .sum(-1)# (torch.abs(x).clamp(min=1e-8)).log().sum(-1)
    
    return z, sum_log_abs_det_jac

def theta_to_r2_transform(x, epsilon: float = 1e-6, inverse=False):
    """"""
    if inverse:
        assert x.size()[-1] == 2
        if not torch.allclose(torch.linalg.norm(x, dim=-1), torch.ones_like(torch.linalg.norm(x, dim=-1)), rtol=1e-1, atol=1e-1):
            import pdb; pdb.set_trace()

        z = torch.atan2(x[..., -1], x[..., -2])
        sum_log_abs_det_jac = -torch.log(torch.abs(x[..., -2]).clamp(min=1e-8)).sum(-1)
    else:
        z1 = torch.cos(x)
        z2 = torch.sin(x)
        z = torch.cat((z1, z2), dim=-1)

        sum_log_abs_det_jac =  torch.log(torch.abs(torch.cos(x)).clamp(min=1e-8))
    
    return z, sum_log_abs_det_jac

def tanh_norm_transform(x, params=None, inverse=False, radius=math.pi):
# Code from https://github.com/pimdh/relie/blob/master/relie/flow/radial_tanh_transform.py
    epsilon = 1e-6
    radius = radius#math.pi

    if inverse:
        org_dtype = x.dtype

        x = x.unsqueeze(-1)
        x_norm = x.norm(dim=-1, keepdim=True)
        mask = x_norm > 1e-8
        x_norm = torch.where(mask, x_norm, torch.ones_like(x_norm))

        z_arg = x_norm / radius

        z = torch.where(
            mask, torch.atanh(z_arg) * x / x_norm, x / radius
        ).to(org_dtype)

        '''
        z = torch.where(
            mask, safe_atanh(z_arg) * x / x_norm, x / radius
        ).to(org_dtype)
        '''
        # stop here
        assert z.isnan().any() == False and z.isinf().any() == False

        x_norm = x.norm(dim=-1)
        d = x.shape[-1]
        tanh = x_norm / radius
        log_dtanh = torch.log1p(torch.clamp(-tanh ** 2, min=-1.+epsilon, max=1.-epsilon))
        log_radius = torch.full_like(log_dtanh, math.log(radius))
        sum_log_abs_det_jacobians = -d * torch.where(x_norm > 1e-8, log_dtanh + log_radius, log_radius)
    else:
        org_dtype = x.dtype
        x = x.double()
        # import pdb; pdb.set_trace()
        x_norm = x.norm(dim=-1, keepdim=True)
        mask = x_norm > 1e-8
        x_norm = torch.where(mask, x_norm, torch.ones_like(x_norm))

        # import pdb; pdb.set_trace()

        z = torch.where(
            mask, torch.tanh(x_norm) * x / x_norm * radius, x * radius
        ).to(org_dtype)

        z_norm = z.norm(dim=-1)
        d = x.shape[-1]
        tanh = z_norm / radius
        log_dtanh = torch.log1p(torch.clamp(-tanh ** 2, min=-1.+epsilon, max=1.-epsilon))
        log_radius = torch.full_like(log_dtanh, math.log(radius))
        sum_log_abs_det_jacobians = d * torch.where(z_norm > 1e-8, log_dtanh + log_radius, log_radius)
    return z, sum_log_abs_det_jacobians

def circular_affine_transform(x, weights, inverse=False):
    """"""
    W = weights.reshape(-1, 2, 2)
    if inverse:
        W_inv = torch.linalg.inv(W)
        W_inv_x = torch.bmm(W_inv, x)
        outputs = W_inv_x /torch.linalg.norm(W_inv_x)

        detJ = torch.linalg.det(W_inv)/(torch.linalg.norm(W_inv_x, dim=(-2, -1))**2)
        logabsdet = torch.log(torch.abs(detJ))
    else:
        W_x = torch.bmm(W, x)
        outputs = W_x/torch.linalg.norm(W_x)

        detJ = torch.linalg.det(W)/(torch.linalg.norm(W_x, dim=(-2, -1))**2)

    return outputs
 
def cylinder_to_sphere_transform(x, params=None, inverse=False):
    """
    Transform input x from cylindrical coordinates to spherical coordinates or its inverse and the log(abs(det(J))) for normalizing flows.
    """
    eps = 1e-6
    if inverse:
        assert torch.all((torch.linalg.norm(x, dim=-1) - 1)**2 < 1e-5)
        x_1D, x_D_plus_1 = x[..., :-1], x[..., -1].unsqueeze(-1)
        z = x_1D/(torch.sqrt(torch.clamp(1 - x_D_plus_1**2, min=1e-8, max=1.0)))
        r = x_D_plus_1

        outputs = torch.cat((z, r), dim=-1)
        logabsdet = torch.zeros_like(r, device=r.device).squeeze() #-torch.log(torch.abs(1 - r**2).clamp(min=1e-8, max=1.0)).squeeze()
        '''
        if not torch.allclose(torch.linalg.norm(outputs[..., :-1], dim=-1), torch.ones_like(torch.linalg.norm(outputs[..., :-1], dim=-1)), rtol=1e-1, atol=1e-1):
            import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        '''
        return outputs, logabsdet

    else:
        # split input x = (z, r)
        z, r = x[..., :-1], x[..., -1].unsqueeze(-1)
        batch_dims = x.size()[:-1]
        assert torch.all(torch.linalg.norm(z, dim=-1) - 1. < 1e-5)
        assert torch.all(-1. <= r) and torch.all(r <= 1.)

        z, r = z.view(-1, z.shape[-1]), r.view(-1, r.shape[-1])
        x_1D = torch.bmm(z.unsqueeze(-1), torch.sqrt(torch.clamp(1 - r**2, min=0.0, max=1.0)).unsqueeze(-1)).squeeze(-1)
        x_D_plus_1 = r 

        # unpack outputs
        x_1D, x_D_plus_1 = x_1D.view(*batch_dims, -1), x_D_plus_1.view(*batch_dims, -1)
        r = r.view(*batch_dims, -1)
        
        #import pdb; pdb.set_trace()
        outputs = torch.cat((x_1D, x_D_plus_1), dim=-1)
        logabsdet = torch.zeros_like(r, device=r.device).squeeze() # torch.log(torch.abs(1 - r**2).clamp(min=1e-8, max=1.0)).squeeze()
        return outputs, logabsdet

def sphere_to_cylinder_transform(x, params=None, inverse=False):
    """
    Transform input x from spherical coordinates to cylindrical coordinates or its inverse and the log(abs(det(J))) for normalizing flows.
    """
    eps = 1e-6
    if inverse:
        # split input x = (z, r)
        z, r = x[..., :-1], x[..., -1].unsqueeze(-1)
        batch_dims = x.size()[:-1]
        
        z, r = z.view(-1, z.shape[-1]), r.view(-1, r.shape[-1]).clamp(min=-1.0, max=1.0)
        x_1D = torch.bmm(z.unsqueeze(-1), torch.sqrt(torch.clamp(1 - r**2, min=0.0, max=1.0)).unsqueeze(-1)).squeeze(-1)
        x_D_plus_1 = r

        # unpack outputs
        x_1D, x_D_plus_1 = x_1D.view(*batch_dims, -1), x_D_plus_1.view(*batch_dims, -1)
        r = r.view(*batch_dims, -1)

        outputs = torch.cat((x_1D, x_D_plus_1), dim=-1)
        logabsdet = torch.zeros_like(r, device=r.device).squeeze() # torch.log(torch.abs(1 - r**2).clamp(min=1e-8, max=1.0)).squeeze()
        
        if torch.any(outputs.isinf()) == True:
            import pdb; pdb.set_trace()

        if torch.any(logabsdet.isinf()) == True:
            import pdb; pdb.set_trace()

        if torch.any(outputs.isnan()) == True:
            import pdb; pdb.set_trace()

        if torch.any(logabsdet.isnan()) == True:
            import pdb; pdb.set_trace()

        if not torch.allclose(torch.linalg.norm(outputs, dim=-1), torch.ones_like(torch.linalg.norm(outputs, dim=-1)), rtol=1e-1, atol=1e-1):
            import pdb; pdb.set_trace()
        
        return outputs, logabsdet
    
    else:
        assert torch.all(torch.linalg.norm(x, dim=-1) - 1. < 1e-5)
        x_1D, x_D_plus_1 = x[..., :-1], x[..., -1].unsqueeze(-1)
        z = x_1D/(torch.sqrt((1 - x_D_plus_1**2).clamp(min=1e-8, max=1.0)))
        r = x_D_plus_1

        outputs = torch.cat((z, r), dim=-1)
        logabsdet = torch.zeros_like(r, device=r.device).squeeze() # -torch.log(torch.abs(1 - r**2).clamp(min=1e-8, max=1.0)).squeeze()

        if torch.any(outputs.isinf()) == True:
            import pdb; pdb.set_trace()

        if torch.any(logabsdet.isinf()) == True:
            import pdb; pdb.set_trace()

        if torch.any(outputs.isnan()) == True:
            import pdb; pdb.set_trace()

        if torch.any(logabsdet.isnan()) == True:
            import pdb; pdb.set_trace()
        
        return outputs, logabsdet

def circular_shift_transform(x, params=None, inverse=False):
    """
    Tranform for circular shift.
    """
    bound = math.pi

    if inverse:
        outputs = torch.remainder(x, 2 * bound) - bound
        logabsdet = torch.zeros_like(x)
    else:
        outputs = torch.remainder(x, 2 * bound) - bound
        logabsdet = torch.zeros_like(x)

    return outputs, logabsdet

def symn_to_spdn_transfrom(x, params=None, inverse=False):
    if inverse:
        outputs = logm_taylor(x, n=5)
        I_n = torch.eye(n=x.shape[-1], device=x.device)
        
        A_1 = torch.linalg.solve(torch.linalg.matrix_power(x + I_n, -1), I_n)
        A_2 = torch.linalg.solve(torch.linalg.matrix_power(x + I_n, -2), I_n)
        A_3 = torch.linalg.solve(torch.linalg.matrix_power(x + I_n, -3), I_n) 

        logabsdet = torch.log(torch.abs(torch.linalg.det(2*A_1 - 8*A_2 + 8*A_3)))
        # print(torch.any(A) == 1.)
        # assert not (torch.any(logabsdet) == torch.nan)
        # assert not (torch.any(logabsdet) == -torch.inf)
        # assert not (torch.any(logabsdet) == torch.inf)

    else:
        outputs = torch.linalg.matrix_exp(x)
        if len(x.size()) > 3:
            logabsdet = torch.einsum('sbii', x).T
        else:
            logabsdet = torch.einsum('bii', x)

    return outputs, logabsdet
        
def rn_to_symn_func(x, n=3):  
    diag = x[..., :n]
    off_diag = x[..., n:]

    D = 0.5 * torch.diag_embed(diag)
    tri_indices = torch.tril_indices(row=n, col=n, offset=-1)
    
    D[tri_indices[0], tri_indices[1]] = off_diag
        
    outputs = D + D.T
    return outputs

def symn_to_rn_func(x, n=3):
    diag = torch.diagonal(x, dim1=-2, dim2=-1)
    tril_indices = torch.tril_indices(n, n, -1)
    off_diag = torch.tril(x, diagonal=-1)[tril_indices[0], tril_indices[1]]
    outputs = torch.cat((diag, off_diag), dim=-1)
    # calculate the jacobian
    return outputs

def rn_to_symn_transform(x, n, params=None, inverse=False):
    """
    """
    if inverse:
        sampling = len(x.size()) > 3
        if sampling:
            batch_dims = x.shape[:-2]
            x = x.reshape(-1, *x.shape[-2:])

        func_inv = lambda x: symn_to_rn_func(x, n=n)
        outputs = vmap(symn_to_rn_func)(x) 
        jac = vmap(jacfwd(func_inv))(x)

        tril_indices = torch.tril_indices(n, n)
        N = int(0.5 * n * (n + 1))
        jac_ = jac[:, :, tril_indices[0], tril_indices[1]].reshape(-1, N, N)
        logabsdet = torch.linalg.det(jac_).abs().log()

    else:
        assert x.shape[-1] == 0.5*n*(n+1)
        sampling = len(x.size()) > 2
        if sampling:
            batch_dims = x.shape[:-1]
            x = x.reshape(-1, x.shape[-1])
        
        func_ = lambda x: rn_to_symn_func(x, n=n)
        outputs = vmap(func_)(x)
        jac = vmap(jacfwd(func_))(x)

        tril_indices = torch.tril_indices(n, n)
        N = int(0.5 * n * (n + 1))
        jac_ = jac[:, tril_indices[0], tril_indices[1], :].reshape(-1, N, N)
        logabsdet = torch.linalg.det(jac_).abs().log()

    if sampling:
        output_dims = outputs.size()[1:]
        outputs = outputs.reshape(*batch_dims, *output_dims)
        logabsdet = logabsdet.reshape(*batch_dims)
    return outputs, logabsdet

def test_flow(z, transforms, base_dist):
    z_i1 = z.clone()
    z_i2 = z.clone()

    log_prob_1 = torch.zeros((z.size()[:-1])).to(z.device)
    log_prob_2 = torch.zeros((z.size()[:-1])).to(z.device)

    # log_prob_1 += base_dist.log_prob(z_i).sum(-1)
    for i, t in enumerate(transforms):
        # t_ = transforms[::-1][i]
        z_i1_save = z_i1.clone()#.squeeze()
        z_i1, ldj = t['func'](z_i1, *t['params'], inverse=False)
        z_i1_, ldj_ = t['func'](z_i1, *t['params'], inverse=True)
        # print(ldj, ldj_)

        z_i1_save = z_i1_save.squeeze()
        # z_i1 = z_i1.squeeze()
        # ldj = ldj.squeeze()
        # z_i1_ = z_i1_.squeeze()
        # ldj_ = ldj_.squeeze()
        # print(z_i1_save.type(), z_i1_save.type())
        assert torch.allclose(ldj.squeeze(), -ldj_.squeeze()), (i, (ldj + ldj_).abs().max().item())
        diff = (z_i1_save.squeeze() - z_i1_.squeeze()).abs().max()
        z_diff_idx = (z_i1_save.squeeze() - z_i1_.squeeze()).abs().argmax()
        print(z_diff_idx, z_i1_.shape, z_i1_save.shape, z_i1_.shape, (z_i1_save.squeeze() - z_i1_.squeeze()).shape)
        assert torch.allclose(z_i1_save.squeeze(), z_i1_.squeeze()), (i, (z_i1_save.squeeze() - z_i1_.squeeze()).abs().max().item(), z_i1_save[z_diff_idx], z_i1_[z_diff_idx], z_diff_idx)

        # log_prob -= ldj
        #
        # for i, t in enumerate(transforms[::-1]):
        #     z_i, ldj = t['func'](z_i, *t['params'], inverse=True)
        #     print(i, z_i)
        #     log_prob += ldj
        #     if torch.any(torch.isnan(z_i)) or torch.any(torch.isnan(ldj)):
        #         breakpoint()
        # print(base_dist)
        # print(z_i)
        # print(base_dist.log_prob(z_i))
        # log_prob += base_dist.log_prob(z_i).sum(-1)

    return None

def test_sphere_to_cylinder_transform(inverse=False):
    radius = torch.ones(1)
    base_dist = UniformSphere(radius=radius, dim=3)
    z0_sphere = base_dist.rsample((20,))
    z0_cylinder = torch.rand((20, 3)) - torch.rand((20, 3)) + 1e-6
    # import pdb; pdb.set_trace()
    z0_cylinder[:, :2] /= torch.linalg.norm(z0_cylinder[:, :2], dim=-1, keepdims=True)

    transforms = [{'func': sphere_to_cylinder_transform, 'params': (None, )}]

    if inverse:
        z_i = z0_cylinder
        log_prob = torch.zeros((z_i.size()[:-1])).to(z_i.device)

        for i, t in enumerate(transforms[::-1]):
            z_i, ldj = t['func'](z_i, *t['params'], inverse=True)
            # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
            # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(z_i.isnan()) == False and torch.any(z_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(z_i.isinf()) == False} or NaN: {torch.any(z_i.isnan()) == False}")
            log_prob += ldj

        norm_flag = torch.all(torch.abs(torch.linalg.norm(z_i, dim=-1) - 1.) < 1e-6)
        if norm_flag:
            print(f"Passed all tests!")
        else:
            print(f" Norm is not equal to one")
            import pdb; pdb.set_trace()

    else:
        z_i = z0_sphere
        log_prob = torch.zeros((z_i.size()[:-1])).to(z_i.device)

        for i, t in enumerate(transforms):
            z_i, ldj = t['func'](z_i, *t['params'], inverse=False)
            # print(f"Forward: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
            # print(f"Foward: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(z_i.isnan()) == False and torch.any(z_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(z_i.isinf()) == False} or NaN: {torch.any(z_i.isnan()) == False}")
            log_prob -= ldj

        norm_flag = torch.all(torch.abs(torch.linalg.norm(z_i[:, :2], dim=-1) - 1.) < 1e-5)
        r_flag = torch.all(-1. < z_i[:, -1]) and torch.all(z_i[:, -1]< 1.)
        if norm_flag and r_flag:
            print(f"Passed all tests!")
        else:
            print(f" Norm is not equal to one")
            import pdb; pdb.set_trace()

def test_permutation_transform():
    """"""
    radius = torch.ones(1)
    base_dist = UniformSphere(radius=radius, dim=3)
    z = base_dist.rsample((20,))

    permutation = torch.randperm(n=3)
    transforms = [{'func': permutation_transform, 'params': (permutation, )}]

    z_i = z.clone()
    log_prob = torch.zeros((z_i.size()[:-1])).to(z_i.device)

    for i, t in enumerate(transforms):
        z_i, ldj = t['func'](z_i, *t['params'], inverse=False)
        # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
        assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
        # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
        assert torch.any(z_i.isnan()) == False and torch.any(z_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(z_i.isinf()) == False} or NaN: {torch.any(z_i.isnan()) == False}")
        log_prob -= ldj

    assert torch.all(z_i != z)

    for i, t in enumerate(transforms[::-1]):
            z_i, ldj = t['func'](z_i, *t['params'], inverse=True)
            # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
            # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(z_i.isnan()) == False and torch.any(z_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(z_i.isinf()) == False} or NaN: {torch.any(z_i.isnan()) == False}")
            log_prob += ldj

    assert torch.all(z_i == z)
    assert torch.all(log_prob == 0.)

    norm_flag = torch.all(torch.abs(torch.linalg.norm(z_i, dim=-1) - 1.) < 1e-6)
    if norm_flag:
        print(f"Passed all tests!")
    else:
        print(f" Norm is not equal to one")
        import pdb; pdb.set_trace()

def test_r2_to_theta_transform():
    """"""
    radius = torch.ones(1)
    base_dist = UniformSphere(radius=radius, dim=2)
    z = base_dist.rsample((20,))
    
    theta = math.pi * (torch.rand((20,)) - torch.rand((20,))) 

    transforms = [{'func': r2_to_theta_transform, 'params': (1e-6, )}]

    z_i = z.clone()
    theta_i = theta.clone()
    log_prob = torch.zeros((z_i.size()[:-1])).to(z_i.device)

    for i, t in enumerate(transforms[::-1]):
        theta_i, ldj = t['func'](theta_i, *t['params'], inverse=True)
        # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
        assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
        # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
        assert torch.any(theta_i.isnan()) == False and torch.any(theta_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(theta_i.isinf()) == False} or NaN: {torch.any(theta_i.isnan()) == False}")
        log_prob -= ldj

    assert torch.all(torch.linalg.norm(theta_i, dim=-1) - 1 < 1e-5)
    assert theta_i.size()[-1] == 2

    for i, t in enumerate(transforms):
            z_i, ldj = t['func'](z_i, *t['params'], inverse=False)
            # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
            # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(z_i.isnan()) == False and torch.any(z_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(z_i.isinf()) == False} or NaN: {torch.any(z_i.isnan()) == False}")
            log_prob += ldj

    # assert z_i.size()[-1] == 1

    norm_flag = torch.all(torch.linalg.norm(theta_i, dim=-1) - 1 < 1e-5)
    if norm_flag:
        print(f"Passed all tests!")
    else:
        print(f" Norm is not equal to one")
        import pdb; pdb.set_trace()

def test_circular_rq_spline():
    """
    """
    # compute random samples for angle
    n = 20
    bs = 100
    t = 10
    D_s = 1
    C = 8

    theta = 6 * math.pi * (torch.rand((bs, 1)) - torch.rand((bs, 1))) 
    D_spline = 3*C*D_s + D_s
    weights = torch.rand((bs, D_spline))

    uw, uh, ud = torch.split(weights, [D_s*C, D_s*C, D_s*(C+1)], dim=-1) 
    uw, uh, ud = uw.reshape(-1,D_s,C), uh.reshape(-1,D_s,C), ud.reshape(-1,D_s,C+1)
    transforms = [{'func': circular_rq_spline, 'params': (uw, uh, ud)}]

    theta_i = theta.clone()
    log_prob = torch.zeros((theta_i.size()[:-1])).to(theta_i.device)

    for i, t in enumerate(transforms):
            #import pdb; pdb.set_trace()
            theta_i, ldj = t['func'](theta_i, *t['params'], inverse=False)
            # import pdb; pdb.set_trace()
            # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
            # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
            assert torch.any(theta_i.isnan()) == False and torch.any(theta_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(theta_i.isinf()) == False} or NaN: {torch.any(theta_i.isnan()) == False}")
            log_prob += ldj

    if torch.all(-math.pi <= theta_i) == True and torch.all(theta_i <= math.pi) == True:
        print("All tests passed!")
        #import pdb; pdb.set_trace()

    theta_i_prime = theta_i.clone()

    for i, t in enumerate(transforms[::-1]):
        theta_i_prime, ldj = t['func'](theta_i_prime, *t['params'], inverse=True)
        # print(f"Inverse: Size of ldj {ldj.size()} with value {ldj.mean()} for layer {t['func']} at iteration {i}")
        assert torch.any(ldj.isnan()) == False and torch.any(ldj.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(ldj.isinf()) == False} or NaN: {torch.any(ldj.isnan()) == False}")
        # print(f"Inverse: Size of z_i {z_i.size()} with value {z_i.mean()} for layer {t['func']} at iteration {i}")
        assert torch.any(theta_i.isnan()) == False and torch.any(theta_i.isinf()) == False, AssertionError(f"inverse layer {t['func']} at iteration {i} is Inf: {torch.any(theta_i.isinf()) == False} or NaN: {torch.any(theta_i.isnan()) == False}")
        log_prob -= ldj

    if torch.all(-math.pi <= theta_i) == True and torch.all(theta_i <= math.pi) == True:
        print("All tests passed!")
        import pdb; pdb.set_trace()