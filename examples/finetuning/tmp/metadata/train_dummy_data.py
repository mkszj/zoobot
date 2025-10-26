import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

from zoobot.pytorch.training.finetune import FinetuneableZoobotMetadataRegressor
from zoobot.pytorch.training.finetune import get_trainer


def main():

    repo_dir = '/Users/user/repos/zoobot'
    batch_size=16
    num_workers=7

    save_dir = os.path.join(repo_dir, 'examples/finetuning/tmp/metadata/results')


    df_train = pd.read_csv(os.path.join(repo_dir, 'examples/finetuning/tmp/metadata/imgs/train_dataset.csv'))
    df_test = pd.read_csv(os.path.join(repo_dir, 'examples/finetuning/tmp/metadata/imgs/test_dataset.csv'))

    df_train['X'] = df_train['X'].astype(np.float32)
    df_train['y'] = df_train['y'].astype(np.float32)
    df_test['X'] = df_test['X'].astype(np.float32)
    df_test['y'] = df_test['y'].astype(np.float32)

    transform_cfg = default_view_config()
    transform = get_galaxy_transform(transform_cfg)

    datamodule = CatalogDataModule(
    label_cols = ['y', 'X'],
    catalog = df_train,
    train_transform = transform,
    test_transform = transform,
    batch_size= batch_size,
    num_workers= num_workers
    )
    datamodule.setup()

    model = FinetuneableZoobotMetadataRegressor(
        # name='nassm/convnext_nano',
        name='hf_hub:mwalmsley/zoobot-encoder-convnext_pico',
        label_col='y',
        metadata_cols=['X']
    )

    datamodule.setup("fit")

    trainer = get_trainer(save_dir, accelerator="auto", max_epochs=1, precision=32)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()