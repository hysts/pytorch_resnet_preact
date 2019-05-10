from yacs.config import CfgNode

config = CfgNode()
config.model = CfgNode()
config.model.name = 'resnet_preact'
config.model.block_type = 'basic'
config.model.depth = 20
config.model.base_channels = 16
config.model.remove_first_relu = False
config.model.add_last_bn = False
config.model.preact_stage = [True, True, True]
config.model.input_shape = (1, 3, 32, 32)
config.model.n_classes = 10

config.run = CfgNode()
config.run.outdir = ''
config.run.seed = 0
config.run.num_workers = 4
config.run.device = 'cuda'
config.run.tensorboard = True

config.train = CfgNode()
config.train.optimizer = 'sgd'
config.train.epochs = 160
config.train.batch_size = 128
config.train.base_lr = 0.1
config.train.weight_decay = 1e-4
config.train.momentum = 0.9
config.train.nesterov = True

config.scheduler = CfgNode()
config.scheduler.name = 'multistep'
config.scheduler.multistep = CfgNode()
config.scheduler.multistep.milestones = [80, 120]
config.scheduler.multistep.lr_decay = 0.1
config.scheduler.cosine = CfgNode()
config.scheduler.cosine.lr_min = 0
config.scheduler.cosine.lr_decay = 0.1


def get_default_config():
    return config.clone()
