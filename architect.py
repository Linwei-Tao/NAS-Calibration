import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        if type(model) == nn.DataParallel:
            self.model = model.module
        else:
            self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]


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
