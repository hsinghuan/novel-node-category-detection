import os
import hydra
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from src.utils.model_utils import model_selection_from_val_outputs

def test(config):

    logger: TensorBoardLogger = hydra.utils.instantiate(config.logger)
    ckpt_dirpath = config.checkpoint_dirpath
    if config.log_subdir: # various hyper-parameter combinations
        ckpt_parent_dirpath = "/".join(ckpt_dirpath.split("/")[:-1]) # drop log_subdir
        val_outputs_dict = dict()
        for subdir in os.listdir(ckpt_parent_dirpath):
            ckpt_dirpath = os.path.join(ckpt_parent_dirpath, subdir)
            ckpt_dirpath = os.path.join(ckpt_dirpath, "version_" + str(config.test_ckpt_version), "checkpoints")
            val_outputs = torch.load(os.path.join(ckpt_dirpath, "val_outputs.ckpt"))
            val_outputs_dict[ckpt_dirpath] = val_outputs
        ckpt_dirpath = model_selection_from_val_outputs(val_outputs_dict, config.model_selection_args)
        print(f"Selected model dirpath: {ckpt_dirpath}")
    else:
        ckpt_dirpath = os.path.join(ckpt_dirpath, "version_" + str(config.test_ckpt_version), "checkpoints")
    ckpt_dirpath_ls = os.listdir(ckpt_dirpath)
    if "last.ckpt" in ckpt_dirpath_ls:
        ckpt_dirpath_ls.remove("last.ckpt")
    if "val_outputs.ckpt" in ckpt_dirpath_ls:
        ckpt_dirpath_ls.remove("val_outputs.ckpt")

    ckpt_path = os.path.join(ckpt_dirpath, ckpt_dirpath_ls[0])
    print(f"ckpt_path: {ckpt_path}")
    detector_class: L.LightningModule = hydra.utils.get_class(config.detector._target_)
    detector = detector_class.load_from_checkpoint(ckpt_path)
    datamodule: L.LightningDataModule = hydra.utils.instantiate(config.datamodule)
    trainer: L.Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger
    )

    trainer.validate(model=detector, datamodule=datamodule)
    trainer.test(model=detector, datamodule=datamodule)


@hydra.main(version_base=None, config_path="config/", config_name="config.yaml")
def main(config):
    test(config)


if __name__ == "__main__":
    main()