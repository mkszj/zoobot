import pandas as pd
from PIL import Image
import pytest
import torch
import os

from lightning import LightningModule

from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.predictions import predict_on_catalog


@pytest.fixture
def catalog(tmp_path):

    n_galaxies = 256

    df = pd.DataFrame({
        'id_str': [f'galaxy_{i}' for i in range(n_galaxies)],
        'file_loc': [str(tmp_path / f'galaxy_{i}.jpg') for i in range(n_galaxies)]
    })

    for _, galaxy in df.iterrows():
        # Create dummy image files
        loc = galaxy['file_loc']
        img = Image.new('RGB', (64, 64))
        img.save(loc)

    return df

@pytest.fixture
def label_cols():
    return ['not_ring', 'ring']

@pytest.fixture
def model(label_cols):

    class DummyModel(LightningModule):
        def __init__(self):
            super().__init__()

        def forward(self, batch):
            return torch.rand(len(batch['image']), len(label_cols))

    return DummyModel()

def test_predict_on_catalog(tmp_path, catalog, model, label_cols):

    transform_cfg = default_view_config()
    transform_cfg.output_size = (64, 64)  # small, for mem
    transform = get_galaxy_transform(transform_cfg)

    _ = predict_on_catalog.predict(
        catalog=catalog,  # test catalog
        model=model,
        inference_transform=transform,  # type: ignore
        label_cols=label_cols,  # name the output columns
        save_loc=tmp_path / 'finetuned_predictions.csv',
        trainer_kwargs={'accelerator': 'auto'},
        datamodule_kwargs={'num_workers': 2, 'batch_size': 4},
    )
