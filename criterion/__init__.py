import math, torch
import torch.nn as nn
from bisect import bisect_right
from torch.optim import Optimizer
import numpy as np
from .ce_klece import KLECE

EPS = 1e-5


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class CrossEntropySoftECE(nn.Module):
    def __init__(self, num_classes, lambda_softece):
        super(CrossEntropySoftECE, self).__init__()
        self.num_classes = num_classes
        self.lambda_softece = lambda_softece
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.cuda()
        inputs = inputs.cuda()
        predictions = torch.argmax(log_probs, dim=-1).cuda()
        softece = get_soft_binning_ece_tensor(predictions, targets,
                                              soft_binning_bins=15,
                                              soft_binning_use_decay=False,
                                              soft_binning_decay_factor=None,
                                              soft_binning_temp=1.1)
        loss = torch.nn.functional.cross_entropy(inputs, targets) + self.lambda_softece * softece
        return loss


class CrossEntropyMMCE(nn.Module):
    def __init__(self, num_classes, lambda_mmce):
        super(CrossEntropyMMCE, self).__init__()
        self.num_classes = num_classes
        self.lambda_mmce = lambda_mmce

    def forward(self, logits, targets, **kwargs):
        ce = torch.nn.functional.cross_entropy(logits, targets)
        mmce = MMCE(ce.device)(logits, targets)
        return ce + (self.lambda_mmce * mmce)


class MMCE(nn.Module):
    """
    Computes MMCE_m loss.
    """

    def __init__(self, device):
        super(MMCE, self).__init__()
        self.device = device

    def torch_kernel(self, matrix):
        return torch.exp(-1.0 * torch.abs(matrix[:, :, 0] - matrix[:, :, 1]) / (0.4))

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1)  # For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

        predicted_probs = torch.nn.functional.softmax(input, dim=1)
        predicted_probs, pred_labels = torch.max(predicted_probs, 1)
        correct_mask = torch.where(torch.eq(pred_labels, target),
                                   torch.ones(pred_labels.shape).to(self.device),
                                   torch.zeros(pred_labels.shape).to(self.device))

        c_minus_r = correct_mask - predicted_probs

        dot_product = torch.mm(c_minus_r.unsqueeze(1),
                               c_minus_r.unsqueeze(0))

        prob_tiled = predicted_probs.unsqueeze(1).repeat(1, predicted_probs.shape[0]).unsqueeze(2)
        prob_pairs = torch.cat([prob_tiled, prob_tiled.permute(1, 0, 2)],
                               dim=2)

        kernel_prob_pairs = self.torch_kernel(prob_pairs)

        numerator = dot_product * kernel_prob_pairs
        # return torch.sum(numerator)/correct_mask.shape[0]**2
        return torch.sum(numerator) / torch.pow(torch.tensor(correct_mask.shape[0]).type(torch.FloatTensor), 2)


def get_soft_binning_ece_tensor(predictions, labels, soft_binning_bins,
                                soft_binning_use_decay,
                                soft_binning_decay_factor, soft_binning_temp):
    """Computes and returns the soft-binned ECE (binned) tensor.
    Soft-binned ECE (binned, l2-norm) is defined in equation (11) in this paper:
    https://arxiv.org/abs/2108.00106. It is a softened version of ECE (binned)
    which is defined in equation (6).
    Args:
        predictions: tensor of predicted confidences of (batch-size,) shape
        labels: tensor of incorrect(0)/correct(1) labels of (batch-size,) shape
        soft_binning_bins: number of bins
        soft_binning_use_decay: whether temp should be determined by decay factor
        soft_binning_decay_factor: approximate decay factor between successive bins
        soft_binning_temp: soft binning temperature
    Returns:
        A tensor of () shape containing a single value: the soft-binned ECE.
    """
    soft_binning_anchors = torch.tensor(np.arange(1.0 / (2.0 * soft_binning_bins), 1.0, 1.0 / soft_binning_bins))
    predictions_tile = torch.tile(torch.unsqueeze(predictions, 1), [1, soft_binning_anchors.shape[0]])
    predictions_tile = torch.unsqueeze(predictions_tile, 2).cuda()
    bin_anchors_tile = torch.tile(torch.unsqueeze(soft_binning_anchors, 0), [predictions.shape[0], 1])
    bin_anchors_tile = torch.unsqueeze(bin_anchors_tile, 2).cuda()

    if soft_binning_use_decay:
        soft_binning_temp = 1 / (
                math.log(soft_binning_decay_factor) * soft_binning_bins * soft_binning_bins)

    predictions_bin_anchors_product = torch.cat([predictions_tile, bin_anchors_tile], dim=2)
    # pylint: disable=g-long-lambda
    predictions_bin_anchors_differences = torch.sum(
        scan(
            fn=lambda _, row: scan(
                fn=lambda _, x: torch.tensor(
                    [-((x[0] - x[1]) ** 2) / soft_binning_temp, 0.]),
                elems=row,
                initializer=0 * torch.ones(predictions_bin_anchors_product.shape[2:])
            ),
            elems=predictions_bin_anchors_product,
            initializer=torch.zeros(predictions_bin_anchors_product.shape[1:])),
        dim=2,
    )
    # pylint: enable=g-long-lambda
    predictions_soft_binning_coeffs = torch.nn.functional.softmax(
        predictions_bin_anchors_differences,
        dim=1,
    )

    sum_coeffs_for_bin = torch.sum(predictions_soft_binning_coeffs, dim=[0])

    intermediate_predictions_reshaped_tensor = torch.reshape(
        torch.repeat_interleave(predictions, torch.tensor(soft_binning_anchors.shape).cuda()),
        predictions_soft_binning_coeffs.shape)
    net_bin_confidence = torch.div(
        torch.sum(
            torch.multiply(intermediate_predictions_reshaped_tensor.cuda(),
                           predictions_soft_binning_coeffs.cuda()),
            dim=0),
        torch.maximum(sum_coeffs_for_bin.cuda(), EPS * torch.ones(sum_coeffs_for_bin.shape).cuda()))

    intermediate_labels_reshaped_tensor = torch.reshape(
        torch.repeat_interleave(labels.cuda(), torch.tensor(soft_binning_anchors.shape).cuda()),
        predictions_soft_binning_coeffs.shape)
    net_bin_accuracy = torch.div(
        torch.sum(
            torch.multiply(intermediate_labels_reshaped_tensor.cuda(),
                           predictions_soft_binning_coeffs.cuda()),
            dim=0),
        torch.maximum(sum_coeffs_for_bin.cuda(), EPS * torch.ones(sum_coeffs_for_bin.shape).cuda()))

    bin_weights = sum_coeffs_for_bin / torch.linalg.norm(sum_coeffs_for_bin, ord=1).item()
    soft_binning_ece = torch.sqrt(
        torch.tensordot(
            torch.square(torch.subtract(net_bin_confidence, net_bin_accuracy)).cuda(),
            bin_weights.cuda(),
            dims=1
        ))

    return soft_binning_ece


def scan(fn, elems, initializer=None):
    res = []
    if initializer is None:
        initializer = elems[0]
    a_ = initializer.clone()

    for i in range(len(elems)):
        res.append(fn(a_, elems[i]).unsqueeze(0))
        a_ = fn(a_, elems[i])

    return torch.cat(res)
