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
        self.gamma = 0

    def find_best_thr(self, pred, targets):
        true = targets == 1

        positive_per_thr = (pred.unsqueeze(-1) >= torch.tensor(self.score_thresholds, device=pred.device))
        true_per_thr = true.unsqueeze(-1).expand_as(positive_per_thr)
        positive_per_thr = positive_per_thr.reshape(-1, len(self.score_thresholds))
        true_per_thr = true_per_thr.reshape(-1, len(self.score_thresholds))

        tp_per_thr = true_per_thr * positive_per_thr

        acc_per_thr = tp_per_thr.sum(dim=0) / true_per_thr.sum(dim=0)
        
        target_thr_idx = 0
        for i, acc in enumerate(acc_per_thr):
            if acc < self.min_recall:
                target_thr_idx = i
                break

        return self.score_thresholds[target_thr_idx]

    def forward(self, logits, targets, mean=True):
        device = logits.device

        """
        if logits.is_cuda:
            loss_func = sigmoid_focal_loss_cuda
        else:
            loss_func = sigmoid_focal_loss_cpu
        """

        if self.alpha == None:
            with torch.no_grad():

                loss_func = sigmoid_focal_loss_jit
                one_hot_target = torch.zeros_like(logits)
                assert len(one_hot_target) == len(targets)
                #assert targets.unique().max() < logits.shape[1]
                pos_inds = targets != 0
                assert (targets >= 0).all().item()
                one_hot_target[torch.arange(len(targets))[pos_inds], targets[pos_inds].long() - 1] = 1

                pred = logits.sigmoid()
                curr_thr = self.find_best_thr(pred, one_hot_target)
            
                true = one_hot_target == 1
                positive = pred >= curr_thr 

                tp = true * positive
                fp = ~true * positive
                fn = true * ~positive

                #precision = tp.sum() / (tp.sum() + fp.sum() + EPS)
                #recall = tp.sum() / (tp.sum() + fn.sum() + EPS)
                #alpha = (1 - ((tp.sum() + fn.sum()) / (tp.sum() + fp.sum() + fn.sum() + EPS))).clamp(min=0.1,max=0.9)
                alpha_true = (1 - tp.sum() / true.sum()).clamp(min=0.01, max=0.99)
                alpha_false = (1 - fp.sum() / (fp.sum() + tp.sum() + fn.sum())).clamp(min=0.01, max=0.99)
                temp = 1 - (1 - alpha_false + alpha_true) / 3
                #tempering = (alpha_true + 1 - alpha_false) / 4.0
                #alpha_true -= tempering
                #alpha_false += tempering
        
                p_t = pred * one_hot_target + (1 - pred) * (1 - one_hot_target)
                assert p_t.isfinite().all().item()

                gamma = math.log(1e-2) / math.log(curr_thr)

            print(curr_thr, tp.sum(), fp.sum(), fn.sum())
            print("target 0's weight factor",1-alpha_false)
            print("target 1's weight factor",alpha_true)
            print("target 0's ignore factor", gamma, ((1 - p_t[~true]) ** gamma).min(), ((1 - p_t[~true]) ** gamma).max())
            print("target 1's ignore factor", gamma, ((1 - p_t[true]) ** gamma).min(), ((1 - p_t[true]) ** gamma).max())

            if true.any().item():
                loss_true = loss_func(logits[true], one_hot_target[true], gamma=gamma, alpha=self.alpha if self.alpha is not None else alpha_true)
                #loss_true = loss_func(logits[true], one_hot_target[true], gamma=gamma_true, alpha=0.25)
                #loss_true = loss_func(logits[true], one_hot_target[true], gamma=2, alpha=0.25)
            else:
                loss_true = torch.tensor([])
            if (~true).any().item():
                loss_false = loss_func(logits[~true], one_hot_target[~true], gamma=gamma, alpha=self.alpha if self.alpha is not None else alpha_false)
                #loss_false = loss_func(logits[~true], one_hot_target[~true], gamma=gamma_false, alpha=0.25)
                #loss_false = loss_func(logits[~true], one_hot_target[~true], gamma=2, alpha=0.25)
            else:
                loss_false = torch.tensor([])
            loss = torch.cat([loss_true, loss_false]) * temp
            #loss = loss_false
            print(loss_false.sum(), loss_true.sum())
        else:
            #loss = loss_func(logits, one_hot_target, gamma=self.gamma, alpha=self.alpha if self.alpha is not None else alpha)
            if logits.is_cuda:
                loss_func = sigmoid_focal_loss_cuda
            else:
                loss_func = sigmoid_focal_loss_cpu

            #loss = loss_func(logits, targets, 2, alpha=0.25)
            loss = loss_func(logits, targets, 2, 0.25)

        #print(alpha,gamma_true,gamma_false)
        assert loss.isfinite().all().item()
        return loss.sum() if mean == True else loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
