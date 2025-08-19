import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split

from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.shared.demo_gz_candels import demo_gz_candels
from galaxy_datasets.transforms import default_view_config, minimal_view_config, get_galaxy_transform

from zoobot.pytorch.training import finetune
from zoobot.pytorch.predictions import predict_on_catalog
from zoobot.shared.schemas import gz_candels_ortho_schema
from zoobot.shared.load_predictions import prediction_hdf5_to_summary_parquet



"""
Example for finetuning Zoobot on counts of volunteer responses throughout a complex decision tree (here, GZ CANDELS).
Useful if you are running a Galaxy Zoo campaign with many questions and answers.
Probably you are in the GZ collaboration if so!
Also useful if you are running a simple yes/no citizen science project on e.g. the Zooniverse app

See also:
- finetune_binary_classification.py to finetune on class (0 or 1) labels
"""


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    schema = gz_candels_ortho_schema

    output_size = 128

    # TODO you will want to replace these paths with your own paths
    # I'm being a little lazy and leaving my if/else for local/cluster training here,
    # this is often convenient for debugging
    if os.path.isdir('/share/nas2'):  # run on cluster
        repo_dir = '/share/nas2/walml/repos'
        data_download_dir = '/share/nas2/walml/repos/_data/demo_gz_candels'
        accelerator = 'gpu'
        devices = 1
        batch_size = 64  
        prog_bar = False
        max_galaxies = None
    else:  # test locally
        if os.path.isdir('/Users/user/repos'):  # run on local machine
            repo_dir = '/Users/user/repos'
            data_download_dir = '/Users/user/repos/galaxy-datasets/roots/demo_gz_candels'
            accelerator = 'cpu'
        elif os.path.isdir('/home/walml/repos'):  # run on local machine
            repo_dir = '/home/walml/repos'
            data_download_dir = '/home/walml/repos/galaxy-datasets/roots/demo_gz_candels'
            accelerator = 'gpu'
        devices = None
        batch_size = 32 # 32 with resize=224, 16 at 380
        prog_bar = True
        # max_galaxies = 256
        max_galaxies = None

    # pd.DataFrame with columns 'id_str' (unique id), 'file_loc' (path to image),
    # and label_cols (e.g. smooth-or-featured-cd_smooth) with count responses
    train_and_val_catalog, _ = demo_gz_candels(root=data_download_dir, train=True, download=True)
    test_catalog, _ = demo_gz_candels(root=data_download_dir, train=True, download=True)

    train_catalog, val_catalog = train_test_split(train_and_val_catalog, test_size=0.3)

    train_transform_cfg = default_view_config()
    train_transform_cfg.output_size = output_size
    train_transform = get_galaxy_transform(train_transform_cfg)

    test_transform_cfg = minimal_view_config()
    test_transform_cfg.output_size = output_size
    test_transform = get_galaxy_transform(test_transform_cfg)

    datamodule = CatalogDataModule(
        label_cols=schema.label_cols,
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        train_transform=train_transform,
        test_transform=test_transform
    )

    model = finetune.FinetuneableZoobotTree(
        name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
        schema=schema,
        training_mode='full',  # 'head_only' for just the head'
    )
    
    # TODO set this to wherever you'd like to save your results
    save_dir = f'results/finetune_counts_full_tree_{np.random.randint(1e8)}'

    # can do logger=None or, to use wandb:
    from lightning.pytorch.loggers import WandbLogger
    logger = WandbLogger(project='finetune', name='full_tree_example')

    trainer = finetune.get_trainer(save_dir=save_dir, logger=logger, accelerator=accelerator, max_epochs=1)
    trainer.fit(model, datamodule)

    # now save predictions on test set to evaluate performance
    datamodule_kwargs = {'batch_size': batch_size}
    trainer_kwargs = {'devices': 1, 'accelerator': accelerator}

    csv_loc = os.path.join(save_dir, 'test_predictions.csv')
    predict_on_catalog.predict(
        test_catalog,
        model,
        inference_transform=test_transform,
        label_cols=schema.label_cols,
        save_loc=csv_loc,
        datamodule_kwargs=datamodule_kwargs,
        trainer_kwargs=trainer_kwargs
    )
    logging.info('Predictions saved to {}'.format(csv_loc))  # dirichlet concentrations

    # prediction_hdf5_to_summary_parquet(hdf5_loc=hdf5_loc, save_loc=hdf5_loc.replace('.hdf5', 'summary.parquet'), schema=schema)