import os
import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def train(config):
    L.seed_everything(int(config.seed))


    # root_dir = os.path.join(config.checkpoint_dir, config.model_name)
    # os.makedirs(root_dir, exist_ok=True)

    logger: TensorBoardLogger = hydra.utils.instantiate(config.logger)


    if config.mode == "domain_disc":
        checkpoint_callback = ModelCheckpoint(# dirpath=config.checkpoint_dirpath,
                                              monitor="val/loss",
                                              save_weights_only=True,
                                              save_last=True)
        early_stop_callback = EarlyStopping(monitor="val/loss", patience=50, mode="min") # for vanilla pu
        trainer: L.Trainer = hydra.utils.instantiate(
            config.trainer,
            logger=logger,
            callbacks=[checkpoint_callback, early_stop_callback]
        )
    else:
        checkpoint_callback = ModelCheckpoint(# dirpath=config.checkpoint_dirpath,
                                              monitor="val/loss",
                                              save_weights_only=True,
                                              save_last=True)
        # print(f"checkpoint save last?:{checkpoint_callback.save_last}")
        trainer: L.Trainer = hydra.utils.instantiate(
            config.trainer,
            logger=logger,
            callbacks=[checkpoint_callback]
        )
        # print(f"last model path: {checkpoint_callback.last_model_path}")

    detector: L.LightningModule = hydra.utils.instantiate(config.detector)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.datamodule)

    trainer.fit(model=detector, datamodule=datamodule)
    # print(f"callback metrics: {trainer.callback_metrics}")
    # trainer.validate(model=detector, datamodule=datamodule)
    # trainer.test(model=detector, datamodule=datamodule)
    # evaluate the metric for hyper-parameter selection and save it in ckpt path

    if config.log_subdir:  # various hyper-parameter combinations
        ckpt_dirpath = checkpoint_callback.dirpath
        # ckpt_dirpath = "/".join(ckpt_dirpath.split("/")[:-1])  # drop log_subdir
        print(os.listdir(ckpt_dirpath))
        ckpt_dirpath_ls = os.listdir(ckpt_dirpath)
        # for subdir in os.listdir(ckpt_dirpath):
        #     ckpt_dirpath_ls = os.listdir(os.path.join(ckpt_dirpath, subdir))
        if "last.ckpt" in ckpt_dirpath_ls:
            ckpt_dirpath_ls.remove("last.ckpt")
        if "val_outputs.ckpt" in ckpt_dirpath_ls:
            ckpt_dirpath_ls.remove("val_outputs.ckpt")
        ckpt_path = os.path.join(ckpt_dirpath, ckpt_dirpath_ls[0])
        detector_class: L.LightningModule = hydra.utils.get_class(config.detector._target_)
        detector = detector_class.load_from_checkpoint(ckpt_path)
        val_outputs = trainer.validate(model=detector, datamodule=datamodule)
        torch.save(val_outputs, os.path.join(ckpt_dirpath, "val_outputs.ckpt"))


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def main(config):
    train(config)


if __name__ == "__main__":
    main()