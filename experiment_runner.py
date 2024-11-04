import os

config_folder = "configs"

configs = [
    "config_resnet18_sgd_stepLR.json",
    "config_resnet18_sgd_nesterov.json",
    "config_resnet18_adam_stepLR.json",
    "config_resnet18_adamw_stepLR.json",
    "config_resnet18_adamw_cosine.json",
    "config_resnet18_rmsprop.json",
    "config_resnet18_sgd_random_erasing_vertical_flip.json",
    "config_resnet18_adamw_color_jitter_rotation.json"

]

for config in configs:
    config_path = os.path.join(config_folder, config)
    print(f"Running experiment with config: {config_path}")
    os.system(f"python main2.py --config {config_path}")
