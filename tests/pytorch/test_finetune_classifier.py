import os

import numpy as np
import pandas as pd
import pytest

from galaxy_datasets import demo_rings
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
from zoobot.pytorch.training.finetune import get_trainer
from zoobot.pytorch.predictions import predict_on_catalog

# pytest version of colab notebook

@pytest.fixture
def tmp_path(tmp_path):
    return tmp_path

@pytest.fixture
def catalogs(tmp_path):
    data_dir = tmp_path / "data_dir"
    data_dir.mkdir()
    train_catalog, _ = demo_rings(root=data_dir, download=True, train=True)

    test_catalog, _ = demo_rings(root=data_dir, download=True, train=False)

    return train_catalog, test_catalog

@pytest.fixture(params=[True, False])
def greyscale(request):
    return request.param

@pytest.fixture(params=['head_only', 'full'])
def training_mode(request):
    return request.param


def test_finetune_classifier(tmp_path, catalogs, greyscale, training_mode):

    transform_cfg = default_view_config()
    transform_cfg.greyscale = greyscale
    transform_cfg.output_size = (64, 64)  # small, for mem
    transform = get_galaxy_transform(transform_cfg)

    datamodule = CatalogDataModule(
        label_cols=['ring'],  # specifying which columns to load as labels with `label_cols`
        catalog=catalogs[0],  # train catalog
        train_transform=transform, 
        test_transform=transform,
        batch_size=32,  # small batch size because our gpu has limited memory
        num_workers=2,  # sets the parallelism for loading data. 2 works well on colab.
    )

    model = FinetuneableZoobotClassifier(
        # arguments for any FinetuneableZoobot class
        # there are many options for customizing finetuning. See the FinetuneableZoobotAbstract docstring.
        name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
        training_mode=training_mode,  # Finetune this many blocks. Set 0 for only the head. Set e.g. 1, 2 to finetune deeper (5 max for convnext).
        learning_rate=1e-5,  # use a low learning rate
        layer_decay=0.8,  # reduce the learning rate from lr to lr^0.8 for each block deeper in the network
        # arguments specific to FinetuneableZoobotClassifier
        num_classes=2,
        label_col='ring',
        greyscale=greyscale,  # convert the model to single channel version
    )

    save_dir = tmp_path / 'finetune_binary_classification'
    save_dir.mkdir()

    trainer = get_trainer(save_dir, devices='auto', max_epochs=1)
    trainer.fit(model, datamodule)

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    finetuned_model = FinetuneableZoobotClassifier.load_from_checkpoint(best_checkpoint)

    _ = predict_on_catalog.predict(
        catalog=catalogs[1],  # test catalog
        model=finetuned_model,
        inference_transform=transform,  # type: ignore
        label_cols=['not_ring', 'ring'],  # name the output columns
        save_loc=os.path.join(tmp_path, 'finetuned_predictions.csv'),
        trainer_kwargs={'accelerator': 'auto'},
        datamodule_kwargs={'num_workers': 2, 'batch_size': 4},
    )