from paa_core.utils.comm import is_pytorch_1_1_0_or_later
from paa_core.solver.build import make_optimizer
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class PLController(object):

    def __init__(self, cfg, model, pseudo_labler, pl_optimizer, pl_scheduler):
        self.network_momentum = cfg.SOLVER.MOMENTUM
        self.network_weight_decay = cfg.SOLVER.WEIGHT_DECAY
        self.model = model
        self.pl_module = pseudo_labler
        self.pl_scheduler = pl_scheduler
        self.pl_optimizer = pl_optimizer
        self.pytorch_1_1_0_or_later = is_pytorch_1_1_0_or_later()

    def step(self, input, target, eta, network_optimizer):
        #require_grad_theta = [p for pg in  network_optimizer.param_groups for p in pg["params"] if p.requires_grad]
        require_grad_theta = None
        self.pl_optimizer.zero_grad()
        pl_loss_weight = self._backward_step_unrolled(input, target, eta, network_optimizer, require_grad_theta)
        self.pl_optimizer.step()

        if self.pytorch_1_1_0_or_later:
            self.pl_scheduler.step()
        
        return pl_loss_weight

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input, target, eta, network_optimizer, require_grad_theta):
        loss, latent_vector = self.model(input, target)
        clean_loss = {k:v for k, v in loss.items() if 'clean' in k}
        noise_loss = {k:v for k, v in loss.items() if 'pl' in k}
        theta = _concat(self.model.rpn_parameters())
        #latent_flatten = torch.cat([f.flatten() for f in latent_vector])

        clean_dlatent = _concat(torch.autograd.grad(sum(clean_loss.values()), latent_vector, create_graph=True))
        noise_dlatent = _concat(torch.autograd.grad(sum(noise_loss.values()), latent_vector, create_graph=True))

        clean_dtheta = _concat(torch.autograd.grad(sum(clean_loss.values()), self.model.rpn_parameters(), create_graph=True))
        noise_dtheta = _concat(torch.autograd.grad(sum(noise_loss.values()), self.model.rpn_parameters(), create_graph=True))

        dsim = (clean_dlatent * noise_dlatent).sum() + (noise_dtheta * clean_dtheta).sum()
        ideal_dsim = ((clean_dlatent * clean_dlatent).sum() + (clean_dtheta * clean_dtheta).sum())
        pl_grad = torch.autograd.grad(dsim, self.pl_module.parameters(), create_graph=True)

        for p, g in zip(self.pl_module.parameters(), pl_grad):
            p.grad = -g
        
        return (dsim / ideal_dsim).item()

def build_pl_controller(cfg, model, pseudo_labler, pl_optimizer, pl_scheduler):
    return PLController(cfg, model, pseudo_labler, pl_optimizer, pl_scheduler)
