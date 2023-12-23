import random
import numpy as np 
import torch
import wandb


def set_global_seed(seed: int) -> None:
    """
    Set global seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_wandb(args):
    wandb.init(
            # Set the project where this run will be logged
            project="sales", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            # name=f"experiment_", 
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.l_rate,
                "model_type": args.model_type.value,
                "epochs": args.num_epoch,
                "batch_size": args.batch_size,
                "loss_type": args.loss_type.name,
        })