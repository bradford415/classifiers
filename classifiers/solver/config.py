import ml_collections

__all__ = ["resnet50_imagenet_config"]


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


def vit_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = "token"
    config.representation_size = None
    return config


def resnet50_imagenet_config(): 
    """Returns the solver parameters used for Resnet50 on the ImageNet dataset

    Parameters defined in the ResNet paper https://arxiv.org/abs/1512.03385 section 3.4
    """
    config = ml_collections.ConfigDict()

    # TODO: decide if I want this in the config.py or yaml General params; currently these params are not used
    config.train_batch_size = 512
    config.val_batch_size = 128
    config.num_epochs = 90

    # Optimizer params
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "sgd"
    config.optimizer.lr = 0.1
    config.optimizer.weight_decay = 1e-4  # 0.0001
    config.optimizer.momentum = 0.9

    # Scheduler params
    config.step_lr_on = "epochs" # step the lr after n "epochs" or "steps" 
    config.lr_scheduler = ml_collections.ConfigDict()
    config.lr_scheduler.name = "reduce_lr_on_plateau"
    config.lr_scheduler.mode = "min"
    config.lr_scheduler.factor = 0.1
    config.lr_scheduler.patience = 5  # epochs of no improvement

    return config


# def get_l32_config():
#     """Returns the ViT-L/32 configuration."""
#     config = get_l16_config()
#     config.patches.size = (32, 32)
#     return config
