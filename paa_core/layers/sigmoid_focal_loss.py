import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from paa_core.utils.comm import is_main_process
import math
import pprint

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
        self.min_ap_thr = min_recall 
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

        # compute precision_thr at here
        tp_per_thr = true_per_thr * positive_per_thr
        min_thr_tp = tp_per_thr.sum(dim=0) + 1
        average_precision_thr = tp_per_thr.sum(dim=0) / (positive_per_thr.sum(dim=0) * min_thr_tp + 1)

        for thr, ap_thr in zip(self.score_thresholds,average_precision_thr):
            log_info["ap_thr_{:.2f}".format(thr)] = ap_thr
        
        target_thr_idx = 0
        for i, ap_thr in enumerate(average_precision_thr):
            if ap_thr < self.min_ap_thr:
                target_thr_idx = i
                break

        return self.score_thresholds[target_thr_idx], average_precision_thr[target_thr_idx], log_info

    def forward(self, logits, targets, mean=True):
        device = logits.device

        log_info = {}
        loss_func = sigmoid_focal_loss_jit

        with torch.no_grad():
            one_hot_target = torch.zeros_like(logits)
            assert len(one_hot_target) == len(targets)
            pos_inds = targets != 0
            assert (targets >= 0).all().item()
            one_hot_target[torch.arange(len(targets))[pos_inds], targets[pos_inds].long() - 1] = 1

            pred = logits.sigmoid()
            curr_thr, curr_acc, thr_log = self.find_best_thr(pred, one_hot_target)
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

                alpha_true = 1 - tp.sum() / (tp.sum() + fp.sum() + fn.sum())
                alpha_false = 1 - fp.sum() / (tp.sum() + fp.sum() + fn.sum())

                #temp = 1 / gamma
                temp = 1.0
                assert temp > 0.0 and temp <= 1.0
        
                p_t = pred * one_hot_target + (1 - pred) * (1 - one_hot_target)
                assert p_t.isfinite().all().item()
                target_p_t = (p_t[fp].mean() * curr_thr**2 + p_t[fn].mean() * (1 - curr_thr)**2)
                gamma = math.log(0.98) / math.log(1 - target_p_t)

                log_info["alpha_false"] = alpha_false
                log_info["alpha_true"] = alpha_true
                #log_info["alpha"] = alpha
                log_info["gamma"] = gamma
                log_info["temp"] = temp

            if true.any().item():
                loss_true = loss_func(logits[true], one_hot_target[true], gamma=gamma, alpha=alpha_true) * temp
            else:
                loss_true = torch.tensor([])

            if (~true).any().item():
                loss_false = loss_func(logits[~true], one_hot_target[~true], gamma=gamma, alpha=alpha_false) * temp
            else:
                loss_false = torch.tensor([])

            loss = torch.cat([loss_true, loss_false])

            with torch.no_grad():
                if logits.is_cuda:
                    loss_func = sigmoid_focal_loss_cuda
                else:
                    loss_func = sigmoid_focal_loss_cpu
                test_loss = loss_func(logits, targets, 2, 0.25)
            log_info["origin_loss"] = test_loss.sum() / true.sum()
            log_info["curr_loss"] = loss.sum() / true.sum()

        else:
            if logits.is_cuda:
                loss_func = sigmoid_focal_loss_cuda
            else:
                loss_func = sigmoid_focal_loss_cpu
            loss = loss_func(logits, targets, self.gamma, self.alpha)
            log_info["origin_loss"] = loss.sum() / true.sum()

        if is_main_process():
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(log_info)

        assert loss.isfinite().all().item()
        return log_info, loss.sum() if mean == True else loss
        #return log_info, test_loss.sum() if mean == True else loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
