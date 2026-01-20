from __future__ import annotations

from smftools.optional_imports import require

torch = require("torch", extra="ml-base", purpose="gradient reversal layers")


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)
