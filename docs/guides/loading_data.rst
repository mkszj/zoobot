
Loading Data
--------------------------


Loading from a table of labels and image paths
==============================================

Zoobot often includes code like:

.. code-block:: python

    # we're loading code from a different package/repository, not Zoobot
    # see github.com/mwalmsley/galaxy-datasets
    from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule

    datamodule = CatalogDataModule(
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        label_cols=['is_cool_galaxy']
        # ...many more options, see below for augmentations
    )

You can pass ``CatalogDataModule`` train, val, test and predict catalogs. Each catalog needs the columns:

* ``file_loc``: the path to the image file
* ``id_str``: a unique identifier for the galaxy
* plus any columns for labels, which you will specify with ``label_cols``. 

CatalogDataModule has attributes like ``.train_dataloader()`` which yield batches like

.. code-block:: python

    {
        'image': transformed tensor (see below) of shape (batch_size, channels, height, width),
        'id_str': tensor of shape (batch_size, 1),
        'is_cool_galaxy': tensor of shape (batch_size, 1)
    }

Lightning's ``Trainer`` object will use these dataloaders to demand data when training, making predictions, and so forth. 
See `<https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html>`_.

Setting ``label_cols=None`` will load the data without labels.


Loading from HuggingFace
==============================================

There is a HuggingFace equivalent to ``CatalogDataModule`` called ``HuggingFaceDataModule``. 
You can find many galaxy datasets from Galaxy Zoo on HuggingFace `here <https://huggingface.co/mwalmsley/datasets>`_.

.. code-block:: python

    from galaxy_datasets.pytorch.galaxy_datamodule import HuggingFaceDataModule
    from datasets import load_dataset as hf_load_dataset

    ds_dict = load_dataset("mwalmsley/gz2")  # returns dict with train and test keys

    datamodule = HuggingFaceDataModule(
        dataset_dict=ds_dict,  # must have train and test keys
        batch_size=32,
        iterable=False  # whether to use IterableDataset (can be faster, no indexed access)
        # many more options...
    )


Standard Augmentations
===============================================

Both ``CatalogDataModule`` and ``HuggingFaceDataModule`` accept ``train_transform`` and ``test_transform`` arguments.
These transforms are applied to the images, and are typically augmentations like rotation, cropping, etc.

I have a standard set of default augmentations you can use:

.. code-block:: python

    from galaxy_datasets.transforms import default_view_config, minimal_view_config, get_galaxy_transform

    # dictionary describing which augmentations to apply
    train_transform_cfg = default_view_config()  
    # get a T.Compose object applying those augmentations
    train_transform = get_galaxy_transform(train_transform_cfg)  

    # another dictionary, with simpler augmentations
    test_transform_cfg = minimal_view_config()  
    test_transform = get_galaxy_transform(test_transform_cfg)

    # you can test the transforms first with
    transformed = train_transform(im)

    # and then use them in the datamodule:
    datamodule = HuggingFaceDataModule(
        dataset_dict=ds_dict,
        batch_size=32,
        train_transform=train_transform, # for train dataloaders
        test_transform=test_transform  # for val and test dataloaders
    )


See `galaxy_datasets.transforms` for more details of the options available.



Loading FITS
===============================================

Where possible, and especially when loading large datasets, I tend to use jpg images. 
These are more convenient at scale and (after adjusting the dynamic range) can look visually identical to FITS.
However, it's often convenient to load FITS images.


.. code-block:: python

    from galaxy_datasets.transforms import default_view_config, get_galaxy_transform

    cfg = default_view_config()
    cfg.flux_to_jpg_like_dynamic_range={
        'arcsinh_q': 1.0, 'percentile_min': 0, 'percentile_max': 99.7
    }
    cfg.pil_to_tensor = False  # fits already load as a tensor
    transform = get_galaxy_transform(cfg)  # ready for datamodule

    # be sure to test it first with
    transformed = transform(im)

    # this works exactly as before
    datamodule = CatalogDataModule(
        train_catalog=train_catalog,
        val_catalog=val_catalog,
        test_catalog=test_catalog,
        batch_size=batch_size,
        label_cols=['is_cool_galaxy'],
        train_transform=transform,  # use the transform we just made
        test_transform=transform,  # same for test
    )

Only single channel FITS images are supported. You will likely also want to load your finetuneable Zoobot model with ``greyscale=True``.

.. code-block:: python

    from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier

    model = FinetuneableZoobotClassifier(
        name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
        greyscale=True,  # converts pretrained model to accept single channel images
        ...
    )


Custom Augmentations
===============================================

You're not limited to these: you can use any torchvision transforms (``T.Compose`` objects). To work well with the Zoobot pretrained models:
* Images should be PyTorch tensors of shape (batch_size, channels, height, width).
* Values should be floats normalized from 0 to 1 (though in practice, Zoobot can handle other ranges provided you use end-to-end finetuning).
* If you are presenting flux values (see FITS below), you should apply a dynamic range rescaling like ``np.arcsinh`` before normalizing to [0, 1].
* Galaxies should appear large and centered in the image.


I Want To Do It All Myself
===========================

Using ``galaxy-datasets`` is optional. Zoobot is designed to work with any PyTorch ``LightningDataModule`` that returns batches of ``({'image': images, 'some_label': labels})``.
And advanced users can pass data to Zoobot's encoder however they like (see :doc:`advanced_finetuning`).

