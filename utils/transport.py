import os

import torch


def save_checkpoint(model, optimizer, args, n, dir='checkpoints'):
    save_dir = os.path.join(dir, args.env_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    torch.save(
        dict(
            env=args.env_name,
            id=args.model_id,
            step=n,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        ),
        os.path.join(save_dir, f"{args.model_id}_{args.algorithm}_params.tar")
    )
    return True

def restore_checkpoint(file, dir='checkpoints'):
    checkpoint = torch.load(os.path.join(dir, file))
    return checkpoint
