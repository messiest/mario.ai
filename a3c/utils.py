import torch
import torch.nn.functional as F


def ensure_shared_grads(model, shared):
    for param, shared_param in zip(model.parameters(), shared.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def choose_action(model, state, hx, cx):
    model.eval()  # set to eval mode
    _, logits, _ = model.forward((state.unsqueeze(0), (hx, cx)))
    prob = F.softmax(logits, dim=-1).detach()
    action = prob.max(-1, keepdim=True)[1]
    model.train()  # return to training

    return action
