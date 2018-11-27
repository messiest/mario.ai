import os

import torch


def save_checkpoint(model, optimizer, args, n, dir='checkpoints'):
    torch.save(
        dict(
            env=args.env_name,
            id=args.model_id,
            step=n,
            model_state_dict=model.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
        ),
        os.path.join(dir, f"{args.env_name}_{args.model_id}_a3c_params.tar")
    )
    return True

def restore_checkpoint(file, dir='checkpoints'):
    checkpoint = torch.load(os.path.join(dir, file))
    return checkpoint
