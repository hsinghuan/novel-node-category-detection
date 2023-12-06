import hydra
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def train(config):
    L.seed_everything(int(config.seed))


    # root_dir = os.path.join(config.checkpoint_dir, config.model_name)
    # os.makedirs(root_dir, exist_ok=True)

    logger: TensorBoardLogger = hydra.utils.instantiate(config.logger)

    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoint_dirpath, save_weights_only=True, save_last=True)
    early_stop_callback = EarlyStopping(monitor="val/loss", patience=50, mode="min") # for vanilla pu
    trainer: L.Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback]
    )

    detector: L.LightningModule = hydra.utils.instantiate(config.detector)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    trainer.fit(model=detector, datamodule=datamodule)
    trainer.validate(model=detector, datamodule=datamodule)
    trainer.test(model=detector, datamodule=datamodule)


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def main(config):
    train(config)


if __name__ == "__main__":
    main()