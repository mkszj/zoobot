# import pytest
import os
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.training.finetune import get_trainer
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier


# # get pytest tmpdir
# @pytest.fixture
# def tmpdir(tmp_path):
#     return tmp_path

def test_finetune_classifier_with_fits(tmp_path):


    #Fine-tune Zoobot on early-type labels
    #    labels.csv must have columns: id_str,file_loc,early_type (0 or 1)
    labelled_df = pd.read_csv('tests/data/fits_test/labels_zoobot.csv')
    # adjust paths
    labelled_df['file_loc'] = labelled_df['file_loc'].apply(lambda x: os.path.join('tests/data/fits_test/images', os.path.basename(x)))
    labelled_df['file_exists'] = labelled_df['file_loc'].apply(os.path.exists)
    labelled_df = labelled_df[labelled_df['file_exists']]  # remove rows with missing files
    print(labelled_df['early_type'].value_counts())

    transform_cfg = default_view_config()

    transform = get_galaxy_transform(transform_cfg)

    datamodule = CatalogDataModule(
      label_cols=["early_type"],  # specifying which columns to load as labels with `label_cols`
      catalog=labelled_df ,
      requested_transform=transform,  # any torchvision Compose transform
      batch_size=32,  # small batch size because our gpu has limited memory
      num_workers=4,  # sets the parallelism for loading data. 2 works well on colab.
    )

    model = FinetuneableZoobotClassifier(
      # arguments for any FinetuneableZoobot class
      # there are many options for customizing finetuning. See the FinetuneableZoobotAbstract docstring.
      name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
      training_mode="full",  # Finetune this many blocks. Set 0 for only the head. Set e.g. 1, 2 to finetune deeper (5 max for convnext).
      learning_rate=1e-5,  # use a low learning rate
      weight_decay=0.,
      layer_decay=0.8,  # reduce the learning rate from lr to lr^0.5 for each block deeper in the network
      # arguments specific to FinetuneableZoobotClassifier
      num_classes=2,
      label_col="early_type",  # the label column to use for classification
      greyscale=True, # equiv to timm_kwargs={'in_chans': 1}. Converts model to single channel version.
    )

    trainer = get_trainer(tmp_path, accelerator='auto', devices='auto', max_epochs=100, precision='32')
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    test_finetune_classifier_with_fits(tmp_path='tests/data/fits_test/tmp')