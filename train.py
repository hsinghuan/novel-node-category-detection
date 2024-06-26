import os
import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from src.utils.model_utils import MonitorAfterModelCheckpoint

def train(config):
    L.seed_everything(int(config.seed))


    # root_dir = os.path.join(config.checkpoint_dir, config.model_name)
    # os.makedirs(root_dir, exist_ok=True)

    logger: TensorBoardLogger = hydra.utils.instantiate(config.logger)


    detector: L.LightningModule = hydra.utils.instantiate(config.detector)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    if config.mode == "domain_disc" or config.mode == "random":
        checkpoint_callback = ModelCheckpoint(monitor="val/loss",
                                              save_weights_only=True,
                                              save_last=True)
        early_stop_callback = EarlyStopping(monitor="val/loss", patience=50, mode="min")
        trainer: L.Trainer = hydra.utils.instantiate(
            config.trainer,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback],
            num_sanity_val_steps=0
        )
    elif config.mode == "label_prop":
        checkpoint_callback = ModelCheckpoint()
        trainer: L.Trainer = hydra.utils.instantiate(
            config.trainer,
            logger=logger,
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback]
        )
    else:
        checkpoint_callback = MonitorAfterModelCheckpoint(monitor_after=config.max_epochs // 2,
                                                      monitor="val/loss",
                                                      save_weights_only=True,
                                                      save_last=True)
        # checkpoint_callback = ModelCheckpoint(monitor="val/loss",
        #                                       save_weights_only=True,
        #                                       save_last=True)
        trainer: L.Trainer = hydra.utils.instantiate(
            config.trainer,
            logger=logger,
            callbacks=[checkpoint_callback],
            num_sanity_val_steps=0
        )


    trainer.fit(model=detector, datamodule=datamodule)

    if config.mode == "random":
        trainer.save_checkpoint(os.path.join(checkpoint_callback.dirpath, "0.ckpt"))

    print("log subdir", config.log_subdir)
    if config.log_subdir:  # various hyper-parameter combinations
        ckpt_dirpath = checkpoint_callback.dirpath
        print(os.listdir(ckpt_dirpath))
        ckpt_dirpath_ls = os.listdir(ckpt_dirpath)
        if "last.ckpt" in ckpt_dirpath_ls:
            ckpt_dirpath_ls.remove("last.ckpt")
        if "val_outputs.ckpt" in ckpt_dirpath_ls:
            ckpt_dirpath_ls.remove("val_outputs.ckpt")
        ckpt_path = os.path.join(ckpt_dirpath, ckpt_dirpath_ls[0])
        detector_class: L.LightningModule = hydra.utils.get_class(config.detector._target_)
        detector = detector_class.load_from_checkpoint(ckpt_path)
        val_outputs = trainer.validate(model=detector, datamodule=datamodule)
        torch.save(val_outputs, os.path.join(ckpt_dirpath, "val_outputs.ckpt"))
        print("val outputs saved")


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def main(config):
    train(config)


if __name__ == "__main__":
    main()