import ml_collections

__all__ = ["resnet50_imagenet_config", "vit_b16_imagenet_config"]


# NOTE: still deciding if I want to use ml_collections or just a config file
def vit_b16_imagenet_config():
    """Returns the ViT-B/16 configuration.

    Naming convention: vit-{variant}{patch_size}_{dataset}_config

    Solver parameters derived from the DeiT paper https://arxiv.org/pdf/2012.12877 Table 9
    in Table 9; compares ViT with DeiT; the idea behind DeiT is to be data-efficient
    so anyone can train a ViT model on ImageNet; this paper has a few modifications
    from ViT but I'll try training a ViT model with the DeiT parameters to see if it works
    """
    config = ml_collections.ConfigDict()

    # Training-related parameters
    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 1024
    config.training.effective_batch_size = (
        1024  # batch_size * gradient_accumulation_steps
    )
    config.training.max_norm = 1.0  # max norm for gradient clipping
    config.training.epochs = 300

    # Validation-related parameters
    config.validation = ml_collections.ConfigDict()
    config.validation.batch_size = 64

    # Optimizer params
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "adamw"
    config.optimizer.lr = 0.001  # 0.0005 * (effective_batch_size / 512)
    config.optimizer.weight_decay = 5e-2  # 0.05

    # Scheduler params
    config.step_lr_on = "epochs"  # step the lr after n "epochs" or "steps"
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.name = "warmup_cosine_decay"
    config.lr_scheduler.warmup_steps = 5
    config.lr_scheduler.total_steps = config.training.epochs  # total number of epochs
    config.lr_scheduler.warmup_min_lr = 2e-4  # 1e-6

    return config


def resnet50_imagenet_config():
    """Returns the solver parameters used for Resnet50 on the ImageNet dataset

    Parameters defined in the ResNet paper https://arxiv.org/abs/1512.03385 section 3.4
    """
    config = ml_collections.ConfigDict()

    # Optimizer params
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "sgd"
    config.optimizer.lr = 0.1
    config.optimizer.weight_decay = 1e-4  # 0.0001
    config.optimizer.momentum = 0.9

    # Scheduler params
    config.step_lr_on = "epochs"  # step the lr after n "epochs" or "steps"
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.name = "reduce_lr_on_plateau"
    config.lr_scheduler.mode = "min"
    config.lr_scheduler.factor = 0.1
    config.lr_scheduler.patience = 5  # epochs of no improvement

    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


# def get_l32_config():
#     """Returns the ViT-L/32 configuration."""
#     config = get_l16_config()
#     config.patches.size = (32, 32)
#     return config
