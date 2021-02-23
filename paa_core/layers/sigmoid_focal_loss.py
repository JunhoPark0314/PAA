import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
import math

from paa_core import _C

# TODO: Use JIT to replace CUDA implementation in the future.
class _SigmoidFocalLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets, gamma, alpha):
        ctx.save_for_backward(logits, targets)
        num_classes = logits.shape[1]
        ctx.num_classes = num_classes
        ctx.gamma = gamma
        ctx.alpha = alpha

        losses = _C.sigmoid_focalloss_forward(
            logits, targets, num_classes, gamma, alpha
        )
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, d_loss):
        logits, targets = ctx.saved_tensors
        num_classes = ctx.num_classes
        gamma = ctx.gamma
        alpha = ctx.alpha
        d_loss = d_loss.contiguous()
        d_logits = _C.sigmoid_focalloss_backward(
            logits, targets, d_loss, num_classes, gamma, alpha
        )
        return d_logits, None, None, None, None


sigmoid_focal_loss_cuda = _SigmoidFocalLoss.apply


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma[0]
    alpha = alpha[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, sum=True):
        device = logits.device
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum() if sum == True else loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

EPS = 1e-5
class ScheduledSigmoidFocalLoss(nn.Module):
    def __init__(self, score_threholds, min_recall, alpha_bumper, gamma_bumper, alpha=None, gamma=None):
        super(ScheduledSigmoidFocalLoss, self).__init__()
        self.score_thresholds = score_threholds 
        self.min_recall = min_recall 
        self.alpha_bumper = alpha_bumper
        self.gamma_bumper = gamma_bumper
        self.alpha = alpha
        self.gamma = gamma

    def find_best_thr(self, pred, targets):
        log_info = {}
        true = targets == 1

        positive_per_thr = (pred.unsqueeze(-1) >= torch.tensor(self.score_thresholds, device=pred.device))
        true_per_thr = true.unsqueeze(-1).expand_as(positive_per_thr)
        positive_per_thr = positive_per_thr.reshape(-1, len(self.score_thresholds))
        true_per_thr = true_per_thr.reshape(-1, len(self.score_thresholds))

        tp_per_thr = true_per_thr * positive_per_thr

        acc_per_thr = tp_per_thr.sum(dim=0) / true_per_thr.sum(dim=0)
        
        for thr, acc in zip(self.score_thresholds,acc_per_thr):
            log_info["acc_{}".format(thr)] = acc
        
        target_thr_idx = 0
        for i, acc in enumerate(acc_per_thr):
            if acc < self.min_recall:
                target_thr_idx = i
                break

        return self.score_thresholds[target_thr_idx], log_info

    def forward(self, logits, targets, mean=True):
        device = logits.device

        log_info = {}

        one_hot_target = torch.zeros_like(logits)
        pos_inds = targets != 0
        one_hot_target[torch.arange(len(targets))[pos_inds], targets[pos_inds].long() - 1] = 1

        loss_func = sigmoid_focal_loss_jit

        pred = logits.sigmoid()
        curr_thr, thr_log = self.find_best_thr(pred, one_hot_target)
        log_info.update(thr_log)

        true = one_hot_target == 1
        positive = pred >= curr_thr 

        tp = true * positive
        fp = ~true * positive
        fn = true * ~positive

        log_info["score_threshold"] = curr_thr
        log_info["true_positive"] = tp.sum().item()
        log_info["false_positive"] = fp.sum().item()
        log_info["false_negative"] = fn.sum().item()

        if self.alpha == None:
            with torch.no_grad():

                alpha_true = (1 - tp.sum() / true.sum()).clamp(min=0.01, max=0.99)
                alpha_false = (1 - fp.sum() / (fp.sum() + tp.sum() + fn.sum())).clamp(min=0.01, max=0.99)
                temp = 1 - (1 - alpha_false + alpha_true) / 3
                assert temp > 0.0 and temp <= 1.0
        
                p_t = pred * one_hot_target + (1 - pred) * (1 - one_hot_target)
                assert p_t.isfinite().all().item()

                gamma = math.log(1e-2) / math.log(curr_thr)

            log_info["alpha_false"] = alpha_false
            log_info["alpha_true"] = alpha_true
            log_info["gamma"] = gamma

            if true.any().item():
                loss_true = loss_func(logits[true], one_hot_target[true], gamma=gamma, alpha=alpha_true)
            else:
                loss_true = torch.tensor([])
            if (~true).any().item():
                loss_false = loss_func(logits[~true], one_hot_target[~true], gamma=gamma, alpha=alpha_false)
            else:
                loss_false = torch.tensor([])
            loss = torch.cat([loss_true, loss_false]) * temp

        else:
            if logits.is_cuda:
                loss_func = sigmoid_focal_loss_cuda
            else:
                loss_func = sigmoid_focal_loss_cpu
            loss = loss_func(logits, targets, self.gamma, self.alpha)

        assert loss.isfinite().all().item()
        return log_info, loss.sum() if mean == True else loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
