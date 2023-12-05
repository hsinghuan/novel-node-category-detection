import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def train(config):
    L.seed_everything(int(config.seed))


    # root_dir = os.path.join(config.checkpoint_dir, config.model_name)
    # os.makedirs(root_dir, exist_ok=True)

    logger: TensorBoardLogger = hydra.utils.instantiate(config.logger)

    checkpoint_callback = ModelCheckpoint(save_weights_only=True, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", patience=50, mode="min") # for vanilla pu
    trainer: L.Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    detector: L.LightningModule = hydra.utils.instantiate(config.detector, arch_param=config.arch_param)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    trainer.fit(model=detector, datamodule=datamodule)
    trainer.validate(model=detector, datamodule=datamodule)
    trainer.test(model=detector, datamodule=datamodule)