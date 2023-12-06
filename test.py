import os
import hydra
import lightning as L


def test(config):
    ckpt_dirpath = config.checkpoint_dirpath
    ckpt_dirpath_ls = os.listdir(ckpt_dirpath)
    if "last.ckpt" in ckpt_dirpath_ls:
        ckpt_dirpath_ls.remove("last.ckpt")
    ckpt_path = os.path.join(ckpt_dirpath, ckpt_dirpath_ls[0])
    detector_class: L.LightningModule = hydra.utils.get_class(config.detector._target_)
    detector = detector_class.load_from_checkpoint(ckpt_path)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    trainer: L.Trainer = hydra.utils.instantiate(
        config.trainer
    )
    trainer.validate(model=detector, datamodule=datamodule)
    trainer.test(model=detector, datamodule=datamodule)


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def main(config):
    test(config)


if __name__ == "__main__":
    main()