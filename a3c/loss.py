import torch
import torch.nn.functional as F


def gae(R, rewards, values, log_probs, entropies, args):
    """Generalized Advantage Estimation"""
    policy_loss = 0
    value_loss = 0
    loss = torch.zeros(1, 1)
    for i in reversed(range(len(rewards))):
        if torch.cuda.is_available():
            loss = loss.cuda()

        R = args.gamma * R.data + rewards[i]
        if torch.cuda.is_available():
            R = R.cuda()

        advantage = R - values[i].data
        value_loss = value_loss + 0.5 * advantage.pow(2)

        delta_t = rewards[i] + args.gamma * values[i + 1].data.cpu() - values[i].data.cpu()
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
