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
    model.train()

    return action


def gae(R, rewards, values, log_probs, entropies, args):
    """Generalized Advantage Estimation"""
    policy_loss = 0
    value_loss = 0
    loss = torch.zeros(1, 1)
    for i in reversed(range(len(rewards))):
        if torch.cuda.is_available():
            loss = loss.cuda()

        R = args.gamma * R + rewards[i]
        if torch.cuda.is_available():
            R = R.cuda()

        advantage = R - values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + args.gamma * values[i + 1].data - values[i].data
        if torch.cuda.is_available():
            delta_t = delta_t.cuda()

        if torch.cuda.is_available():
            loss = loss.cuda() * args.gamma * args.tau + delta_t.cuda()
        else:
            loss = loss.cpu() * args.gamma * args.tau + delta_t.cpu()

        policy_loss = policy_loss - \
                      log_probs[i] * loss - \
                      args.entropy_coef * entropies[i]

    total_loss = policy_loss + args.value_loss_coef * value_loss

    return total_loss
