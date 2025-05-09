'''
Script for training.
'''

import sys
import logging
from omegaconf import OmegaConf

import numpy as np

import torch
import lightning as L

from lightning.pytorch.callbacks import EarlyStopping

from src.milleacarinet.datamodule import MilleAcariDataModule
from src.milleacarinet.model import MilleAcariNet, YoloV5Backbone
from src.milleacarinet.utils import create_logger

import warnings
warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)
log = create_logger(log, level=logging.DEBUG)  # Ensure logging level is set to DEBUG

def train(config_file):
    '''
    Train MilleAcariNet
    '''

    config = OmegaConf.load(config_file)
    config = OmegaConf.to_object(config)

    print(config)

    # Seed everything
    if config['seed']:
        log.info('Setting seed')
        L.seed_everything(config['seed'])

    # Instantiate the datamodule
    datamodule = MilleAcariDataModule(**config['datamodule'])
    datamodule.setup(stage='fit')

    log.info(f'Length of training data: {len(datamodule.train_ds)}')
    log.info(f'Length of validation data: {len(datamodule.val_ds)}')

    # Instantiate the model
    backbone = YoloV5Backbone(**config['backbone'])
    model = MilleAcariNet(backbone=backbone, **config['model'])

    # Instantiate the callbacks
    callbacks = [EarlyStopping(monitor='val/loss', patience=5, mode='min')]

    # Instantiate the trainer
    trainer = L.Trainer(**config['trainer'], callbacks=callbacks)

    # Run training
    trainer.fit(model, datamodule)

    # Finalize
    dbb = trainer.predict(model, dataloaders=datamodule.test_dataloader())
    np.save('prediction.npy', np.asarray(dbb))


if __name__=='__main__':
    train(sys.argv[1])