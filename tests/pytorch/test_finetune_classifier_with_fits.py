import numpy as np
import pytest
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.training.finetune import get_trainer
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier


@pytest.fixture
def tmpdir(tmp_path):
    return tmp_path

@pytest.fixture
def labelled_df(tmp_path):
    galaxies = [{'id_str': f'galaxy_{i}', 'file_loc': os.path.join(tmp_path, f'galaxy_{i}.fits'), 'early_type': np.random.randint(0, 2)} for i in range(256)]

    for galaxy in galaxies:
        # Create dummy FITS files
        fits_path = galaxy['file_loc']
        from astropy.io import fits
        hdu = fits.PrimaryHDU(np.random.rand(100, 100))  # Dummy data
        hdu.writeto(fits_path, overwrite=True)

    return pd.DataFrame(galaxies)

def test_finetune_classifier_with_fits(tmp_path, labelled_df):

    #Fine-tune Zoobot on early-type labels
    #    labels.csv must have columns: id_str,file_loc,early_type (0 or 1)
    # labelled_df = pd.read_csv('tests/data/fits_test/labels_zoobot.csv')
    # adjust paths
    # labelled_df['file_loc'] = labelled_df['file_loc'].apply(lambda x: os.path.join('tests/data/fits_test/images', os.path.basename(x)))
    # labelled_df['file_exists'] = labelled_df['file_loc'].apply(os.path.exists)
    # labelled_df = labelled_df[labelled_df['file_exists']]  # remove rows with missing files
    # print(labelled_df['early_type'].value_counts())
    # labelled_df = labelled_df.drop(columns=['file_exists'])  # remove the helper column
    # labelled_df = labelled_df.sort_values('id_str')[:16]
    # labelled_df.to_csv('tests/data/fits_test/labels_zoobot_subset.csv', index=False)


    transform_cfg = default_view_config()

    transform = get_galaxy_transform(transform_cfg)

    datamodule = CatalogDataModule(
      label_cols=["early_type"],  # specifying which columns to load as labels with `label_cols`
      catalog=labelled_df ,
      train_transform=transform,  # any torchvision Compose transform
      test_transform=transform,  # same transform for inference
      batch_size=4,  # small batch size because our gpu has limited memory
      num_workers=4,  # sets the parallelism for loading data. 2 works well on colab.
    )

    model = FinetuneableZoobotClassifier(
      name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
      training_mode="full",
      learning_rate=1e-5,
      weight_decay=0.,
      layer_decay=0.8,
      num_classes=2,
      label_col="early_type",
      greyscale=True,
    )

    trainer = get_trainer(tmp_path, accelerator='auto', devices='auto', max_epochs=1)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

# if __name__ == "__main__":
    # test_finetune_classifier_with_fits(tmp_path='tests/data/fits_test/tmp')