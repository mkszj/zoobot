import torch
import logging
import os

torch.set_float32_matmul_precision("medium")

from galaxy_datasets import gz2
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.training.finetune import (FinetuneableZoobotClassifier, get_trainer,)
from zoobot.pytorch.predictions import predict_on_catalog

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

data_dir = "C:/MASTERFOLDER/Programming/testrun3d"
save_dir = "C:/MASTERFOLDER/Programming/test_zoobot_runs"

#binary label from the vote counts
source_label_column = "has-spiral-arms-gz2_yes"
binary_label_column = "has_spiral_binary"

num_classes = 2
batch_size = 32
max_epochs = 100

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    #load GZ2 catalogue

    train_catalogue, label_cols = gz2(
        root=data_dir,
        train=True,
        download=True
    )

    test_catalogue, _ = gz2(
        root=data_dir,
        train=False,
        download=True
    )

    print(f"\nOriginal train size: {len(train_catalogue)}")
    print(f"Original test size: {len(test_catalogue)}")
    
    #chwck/binarise labels
    
    print(f"\n=== Checking '{source_label_column}' column ===")
    print(f"Data type: {train_catalogue[source_label_column].dtype}")
    print(f"Unique values: {sorted(train_catalogue[source_label_column].unique())[:20]}")
    print(f"Value range: {train_catalogue[source_label_column].min()} to {train_catalogue[source_label_column].max()}")
    print(f"\nValue counts (top 10):")
    print(train_catalogue[source_label_column].value_counts().head(10))
    
    #binary conversion: 1 if votes > threshold, else 0
    threshold = 0
    
    train_catalogue[binary_label_column] = (train_catalogue[source_label_column] > threshold).astype(int)
    test_catalogue[binary_label_column] = (test_catalogue[source_label_column] > threshold).astype(int)
    
    print(f"\n=== Binary label distribution ===")
    print("Train:")
    print(train_catalogue[binary_label_column].value_counts())
    print("\nTest:")
    print(test_catalogue[binary_label_column].value_counts())
    
    #filter out rows with invalid data
    train_catalogue = train_catalogue[train_catalogue[binary_label_column].isin([0, 1])].reset_index(drop=True)
    test_catalogue = test_catalogue[test_catalogue[binary_label_column].isin([0, 1])].reset_index(drop=True)
    
    print(f"\nFiltered train size: {len(train_catalogue)}")
    print(f"Filtered test size: {len(test_catalogue)}")

    #image transforms

    transform_cfg = default_view_config()
    transform_cfg.output_size = 128
    transform_cfg.greyscale = True

    transform = get_galaxy_transform(transform_cfg)

    #datamodule

    datamodule = CatalogDataModule(
        label_cols=[binary_label_column],
        catalog=train_catalogue,
        batch_size=batch_size,
        train_transform=transform,
        test_transform=transform,
    )

    #zoobotmodule

    model = FinetuneableZoobotClassifier(
        name="hf_hub:mwalmsley/zoobot-encoder-convnext_nano",
        num_classes=num_classes,
        label_col=binary_label_column,
        training_mode="head_only",
        greyscale=True,
    )

    #trainermodule

    trainer = get_trainer(
        save_dir=save_dir,
        accelerator="gpu",
        max_epochs=max_epochs,
    )

    #batch inspection

    print("\n=== Inspecting first batch ===")
    datamodule.setup("fit")
    
    batch = next(iter(datamodule.train_dataloader()))
    
    print("Batch keys:", batch.keys())
    
    for k, v in batch.items():
        print(f"\nKey: {k}")
        print("  type:", type(v))
        if hasattr(v, "shape"):
            print("  shape:", v.shape)
            print("  dtype:", v.dtype)
            if v.numel() > 0:
                print("  min:", v.min().item())
                print("  max:", v.max().item())
                if k == binary_label_column:
                    print("  unique values:", v.unique().tolist())

    #trainingmodule

    print("\n=== Starting training ===")
    
    trainer.fit(model, datamodule)
    print("\nTraining complete!")
    
    best_ckpt = trainer.checkpoint_callback.best_model_path
    print(f"Best checkpoint: {best_ckpt}")

    #load/evalute

    model = FinetuneableZoobotClassifier.load_from_checkpoint(best_ckpt)
    model.eval()
    print("Model loaded from checkpoint and set to evaluation mode")
    
    #predictionmodel
    
    print("\n=== Making predictions on test set ===")
    
    #column names for BOTH output classes
    prediction_label_cols = [f'{binary_label_column}_class_0', f'{binary_label_column}_class_1']
    
    preds = predict_on_catalog.predict(
        test_catalogue,
        model,
        inference_transform=transform,
        label_cols=prediction_label_cols,  #2 columns for 2 classes
        save_loc=os.path.join(save_dir, 'predictions.csv'),
        datamodule_kwargs={'batch_size': batch_size},
        trainer_kwargs={'accelerator': 'gpu'}
    )
    
    print("\nPredictions saved!")
    print(f"Prediction shape: {preds.shape}")
    print("\nFirst few predictions:")
    print(preds.head())
    
    #most likely
    preds['predicted_class'] = (preds[f'{binary_label_column}_class_1'] > 0.5).astype(int)
    
    print("\nPrediction summary:")
    print(preds['predicted_class'].value_counts())
    
    #save
    preds.to_csv(os.path.join(save_dir, 'predictions_with_class.csv'), index=False)
    print(f"\nEnhanced predictions saved to: {os.path.join(save_dir, 'predictions_with_class.csv')}")