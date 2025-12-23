import torch
import logging

torch.set_float32_matmul_precision("medium")

from galaxy_datasets import gz2
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.training.finetune import (
    FinetuneableZoobotClassifier,
    get_trainer,
)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

data_dir = "C:/MASTERFOLDER/Programming/testrun3d"
save_dir = "C:/MASTERFOLDER/Programming/test_zoobot_runs"

label_columns = [
    "has-spiral-arms-gz2_yes",
    "has-spiral-arms-gz2_no",
]

num_classes = 2
batch_size = 16       #safe for RTX 3050 4GB
max_epochs = 10       #short for test run

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # --------------------------------------------------
    # load GZ2 catalogue
    # --------------------------------------------------

    train_catalogue, _ = gz2(
        root=data_dir,
        train=True,
        download=True
    )

    test_catalogue, _ = gz2(
        root=data_dir,
        train=False,
        download=True
    )

    print(f"Train size: {len(train_catalogue)}")
    print(f"Test size: {len(test_catalogue)}")

    # --------------------------------------------------
    # image transforms
    # --------------------------------------------------

    transform_cfg = default_view_config()
    transform_cfg.output_size = 128
    transform_cfg.greyscale = True

    transform = get_galaxy_transform(transform_cfg)

    # --------------------------------------------------
    # datamodule
    # --------------------------------------------------

    datamodule = CatalogDataModule(
        label_cols=label_columns,
        catalog=train_catalogue,
        batch_size=batch_size,
        train_transform=transform,
        test_transform=transform,
    )

    # --------------------------------------------------
    # zoobot-3d model
    # --------------------------------------------------

    model = FinetuneableZoobotClassifier(
        name="hf_hub:mwalmsley/zoobot-encoder-convnext_nano",
        num_classes=num_classes,
        label_col="has-spiral-arms-gz2_yes",
        training_mode="head_only",
        greyscale=True,
    )

    # --------------------------------------------------
    # trainer
    # --------------------------------------------------

    trainer = get_trainer(
        save_dir=save_dir,
        accelerator="gpu",
        max_epochs=max_epochs,
    )

    # --------------------------------------------------
    # train sequence
    # --------------------------------------------------

    datamodule.setup("fit")
    batch = next(iter(datamodule.train_dataloader()))

    print("Batch keys:", batch.keys())

    for k, v in batch.items():
        print(f"\nKey: {k}")
        print("  type:", type(v))
        if hasattr(v, "shape"):
            print("  shape:", v.shape)
            print("  dtype:", v.dtype)
            print("  min:", v.min().item() if v.numel() else None)
            print("  max:", v.max().item() if v.numel() else None)

    trainer.fit(model, datamodule)
    print("Zoobot-3D training complete")

    best_ckpt = trainer.checkpoint_callback.best_model_path
    print("Best checkpoint:", best_ckpt)

    model = FinetuneableZoobotClassifier.load_from_checkpoint(best_ckpt)
    model.eval()
    print("Model loaded from checkpoint + set to evaluation mode")