import logging
import time
import datetime
from typing import List, Union

import pandas as pd
import torch
import torchvision
import lightning as L

from torchvision.transforms.v2 import Compose

# from zoobot.shared import save_predictions
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule


def predict(catalog: pd.DataFrame, model: L.LightningModule, save_loc: str, label_cols: List[str], inference_transform: Compose, datamodule_kwargs={}, trainer_kwargs={}) -> pd.DataFrame:
    """
    Use trained model to make predictions on a catalog of galaxies.

    Args:
        catalog (pd.DataFrame): catalog of galaxies to make predictions on. Must include `file_loc` and `id_str` columns.
        model (L.LightningModule): with which to make predictions. Probably ZoobotTree, FinetuneableZoobotClassifier, FinetuneableZoobotTree, or ZoobotEncoder.
        save_loc (str): desired name of file recording the predictions
        label_cols (List[str]): columns in the catalog to use as labels. Used to name the output columns.
        datamodule_kwargs (dict, optional): Passed to CatalogDataModule. Use to e.g. add custom augmentations. Defaults to {}.
        trainer_kwargs (dict, optional): Passed to L.Trainer. Defaults to {}.
    """

    image_id_strs = list(catalog['id_str'].astype(str))

    predict_datamodule = CatalogDataModule(
        label_cols=None,  # not using label_cols to load labels, we're only using it to name our predictions
        predict_catalog=catalog,  # no need to specify the other catalogs
        test_transform=inference_transform,  # see galaxy-datasets, e.g. torchvision.transforms.Compose([torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor()])
        **datamodule_kwargs  # e.g. batch_size, num_workers, etc
    )
    # with this stage arg, will only use predict_catalog 
    # crucial to specify the stage, or will error (as missing other catalogs)
    predict_datamodule.setup(stage='predict')  
    # for images in predict_datamodule.predict_dataloader():
        # print(images)
        # print(images.shape)
        # print(images.min(), images.max())
        # exit()
        # import matplotlib.pyplot as plt
        # plt.imshow(images[0].permute(1, 2, 0))
        # plt.show()


    # set up trainer (again)
    trainer = L.Trainer(
        max_epochs=-1,  # does nothing in this context, suppresses warning
        **trainer_kwargs  # e.g. gpus
    )

    logging.info('Beginning predictions')
    start = datetime.datetime.fromtimestamp(time.time())
    logging.info('Starting at: {}'.format(start.strftime('%Y-%m-%d %H:%M:%S')))

    # logging.info(len(trainer.predict(model, predict_datamodule)))

    # trainer.predict gives list of tensors, each tensor being predictions for a batch. Concat on axis 0.
    # range(n_samples) list comprehension repeats this, for dropout-permuted predictions. Stack to create new last axis.
    # final shape (n_galaxies, label_cols)

    predictions: torch.Tensor = torch.cat(trainer.predict(model, predict_datamodule), dim=0)  # in latest version, now a tensor
    logging.info('Predictions complete - {}'.format(predictions.shape))
    
    logging.info(f'Saving predictions to {save_loc}')
    # predictions: pd.DataFrame = pd.concat(trainer.predict(model, predict_datamodule), axis=0)  # in latest version, now a dataframe
    prediction_df = pd.DataFrame(predictions.numpy(), columns=label_cols)  # convert to pandas dataframe
    prediction_df['id_str'] = image_id_strs  # add the id_str column
    prediction_df.to_csv(save_loc, index=False)  # RegressionBaseline returns pandas dataframe, do I want this, or hdf5?
    # probably just dataframe, since I only load to dataframe next anyway, and I never used repeated draws

    # if save_loc.endswith('.csv'):      # save as pandas df
    #     save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)
    # elif save_loc.endswith('.hdf5'):
    #     save_predictions.predictions_to_hdf5(predictions, image_id_strs, label_cols, save_loc)
    # else:
    #     logging.warning('Save format of {} not recognised - assuming csv'.format(save_loc))
    #     save_predictions.predictions_to_csv(predictions, image_id_strs, label_cols, save_loc)

    logging.info(f'Predictions saved to {save_loc}')

    end = datetime.datetime.fromtimestamp(time.time())
    logging.info('Completed at: {}'.format(end.strftime('%Y-%m-%d %H:%M:%S')))
    logging.info('Time elapsed: {}'.format(end - start))

    return prediction_df
