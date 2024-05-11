import torch
import torch.nn as nn
import torch.optim as optim


class GradNormLoss(nn.Module):
    def __init__(self, num_of_task, alpha=1.5, device='cuda'):
        super(GradNormLoss, self).__init__()
        self.num_of_task = num_of_task
        self.alpha = alpha
        self.device = device
        self.w = nn.Parameter(torch.ones(num_of_task, dtype=torch.float).to(self.device).requires_grad_())
        self.l1_loss = nn.L1Loss()
        self.cnt =0
        self.L_0 = None

    # standard forward pass
    def forward(self, L_t: torch.Tensor):
        # initialize the initial loss `Li_0`
        if self.L_0 is None or self.cnt % 1000 == 0:
            self.L_0 = L_t.detach()  # detach
        self.cnt +=1
        # compute the weighted loss w_i(t) * L_i(t)
        self.L_t = L_t
        self.wL_t = L_t * self.w
        # the reduced weighted loss
        self.total_loss = self.wL_t.sum()
        return self.total_loss

    # additional forward & backward pass
    def additional_forward_and_backward(self, grad_norm_weights: nn.Module):
        # do `optimizer.zero_grad()` outside
        # self.total_loss.backward()
        # in standard backward pass, `w` does not require grad
        # self.w.grad.data = self.w.grad.data * 0.0
        self.w.grad = None

        self.GW_t = []
        for i in range(self.num_of_task):
            # get the gradient of this task loss with respect to the shared parameters
            # if i == self.num_of_task - 1:
            #     require_grad = False
            # else:
            #     require_grad = True
            GiW_t = torch.autograd.grad(
                self.L_t[i], grad_norm_weights.parameters(),
                retain_graph=True)  # , create_graph=True
            # compute the norm
            self.GW_t.append(torch.norm(GiW_t[0] * self.w[i]))
        self.GW_t = torch.stack(self.GW_t)  # do not detatch
        self.bar_GW_t = self.GW_t.detach().mean()
        self.tilde_L_t = (self.L_t / self.L_0).detach()
        self.r_t = self.tilde_L_t / self.tilde_L_t.mean()
        grad_loss = self.l1_loss(self.GW_t, self.bar_GW_t * (self.r_t ** self.alpha))
        if self.w[0] >= 0.25:  # 添加w[0]的下限
            self.w.grad = torch.autograd.grad(grad_loss, self.w, retain_graph=True)[0]
        else:
            self.w.grad.zero_()

        # re-norm
        self.w.data = self.w.data / self.w.data.sum() * self.num_of_task

    def clear_grad(self):
        self.GW_t = None
        self.GW_ti, self.bar_GW_t, self.tilde_L_t, \
        self.r_t, self.L_t, self.wL_t = None, None, None, None, None, None


# This is AN interface.
class GradNormModel:
    def get_grad_norm_weights(self) -> nn.Module:
        raise NotImplementedError(
            "Please implement the method `get_grad_norm_weights`")
