from configs import cfg

from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer
import wandb, os
import torch, numpy as np

def init_wandb():
    wandb.login(key='5421ff43bf1e3a6e19103432d161c885d4bbeda8')
    wandb_run = wandb.init(project='HumanNerf', config=cfg, resume=cfg.resume, dir=cfg.logdir)
    wandb.run.name = '/'.join(cfg.logdir.split('/')[-2:])
    wandb.run.save()
    return wandb_run

def main():
    log = Logger()
    log.print_config()
    wandb_run = init_wandb()
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    model = create_network()
    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer, wandb_run)

    train_loader = create_dataloader('train')

    # estimate start epoch
    epoch = trainer.iter // len(train_loader) + 1
    while True:
        if trainer.iter > cfg.train.maxiter:
            break
        
        trainer.train(epoch=epoch,
                      train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()

if __name__ == '__main__':
    main()
