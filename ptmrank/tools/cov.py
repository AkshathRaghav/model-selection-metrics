""" 
Credit: https://gist.github.com/xmodar/5ab449acba9df1a26c12060240773110
Why? np.cov(rowvar=False) is not implemented in torch.

Discussion thread in torch: https://github.com/pytorch/pytorch/issues/19037
"""

import torch 

def cov(tensor, rowvar=True, bias=False):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)
    tensor = tensor - tensor.mean(dim=-1, keepdim=True)
    factor = 1 / (tensor.shape[-1] - int(not bool(bias)))
    return factor * tensor @ tensor.transpose(-1, -2).conj()

def corrcoef(tensor, rowvar=True):
    """Get Pearson product-moment correlation coefficients (np.corrcoef)"""
    covariance = cov(tensor, rowvar=rowvar)
    variance = covariance.diagonal(0, -1, -2)
    if variance.is_complex():
        variance = variance.real
    stddev = variance.sqrt()
    covariance /= stddev.unsqueeze(-1)
    covariance /= stddev.unsqueeze(-2)
    if covariance.is_complex():
        covariance.real.clip_(-1, 1)
        covariance.imag.clip_(-1, 1)
    else:
        covariance.clip_(-1, 1)
    return covariance

def cov(tensor, rowvar=True, bias=False, fweights=None, aweights=None):
    """Estimate a covariance matrix (np.cov)"""
    tensor = tensor if rowvar else tensor.transpose(-1, -2)

    weights = fweights
    if aweights is not None:
        if weights is None:
            weights = aweights
        else:
            weights = weights * aweights

    if weights is None:
        mean = tensor.mean(dim=-1, keepdim=True)
    else:
        w_sum = weights.sum(dim=-1, keepdim=True)
        mean = (weights * tensor).sum(dim=-1, keepdim=True) / w_sum

    ddof = int(not bool(bias))
    if weights is None:
        fact = 1 / (tensor.shape[-1] - ddof)
    else:
        if ddof == 0:
            fact = w_sum
        elif aweights is None:
            fact = w_sum - ddof
        else:
            w_sum2 = (weights * aweights).sum(dim=-1, keepdim=True)
            fact = w_sum - w_sum2 / w_sum  # ddof == 1
        # warn if fact <= 0
        fact = weights / fact.relu_()

    tensor = tensor - mean
    return fact * tensor @ tensor.transpose(-1, -2).conj()