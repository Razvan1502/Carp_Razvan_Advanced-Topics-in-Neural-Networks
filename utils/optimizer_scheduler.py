import torch

def configure_optimizer(params, optimizer_config):
    optimizer_name = optimizer_config['name']
    lr = optimizer_config['lr']

    weight_decay = optimizer_config.get('weight_decay', 0.0)
    momentum = optimizer_config.get('momentum', 0.0)
    nesterov = optimizer_config.get('nesterov', False)
    fused = optimizer_config.get('fused', False)

    if optimizer_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def configure_scheduler(optimizer, scheduler_config):
    if scheduler_config['name'] == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config['step_size'],
                                               gamma=scheduler_config['gamma'])
    elif scheduler_config['name'] == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                          factor=scheduler_config.get('factor', 0.1),
                                                          patience=scheduler_config['patience'])
    elif scheduler_config['name'] == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config['T_max'])
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config['name']}")
