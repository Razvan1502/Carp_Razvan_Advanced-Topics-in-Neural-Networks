import wandb
import time
from torch.utils.tensorboard import SummaryWriter



def initialize_logging(logging_config, config_name):
    if logging_config['type'] == "tensorboard":

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_dir = f"runs/{config_name}_{timestamp}"
        return SummaryWriter(log_dir=log_dir)
