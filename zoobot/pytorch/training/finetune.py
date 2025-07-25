import logging
import os
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks import LearningRateMonitor

import timm
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import torch
import torch.nn.functional as F
import torchmetrics as tm
from sklearn import linear_model
from sklearn.metrics import accuracy_score

from zoobot.pytorch.estimators import define_model
from zoobot.pytorch.training import losses, schedulers
from zoobot.shared import schemas

# https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/7
# I do this recursively and only for BatchNorm2d (not dropout, which I still want active)


def freeze_batchnorm_layers(model):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            logging.debug("Freezing {} {}".format(child, name))
            child.eval()  # no grads, no param updates, no statistic updates
        else:
            freeze_batchnorm_layers(child)  # recurse


class FinetuneableZoobotAbstract(L.LightningModule):
    """
    Parent class of :class:`FinetuneableZoobotClassifier`, :class:`FinetuneableZoobotRegressor`, :class:`FinetuneableZoobotTree`.
    You cannot use this class directly - you must use the child classes above instead.

    This class defines the shared finetuning args and methods used by those child classes.
    For example:
    * When provided ``name``, it will load the HuggingFace encoder with that name (see below for more).
    * When provided ``learning_rate`` it will set the optimizer to use that learning rate.

    Any FinetuneableZoobot model can be loaded in one of three ways:
    * HuggingFace name e.g. ``FinetuneableZoobotX(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...)``. Recommended.
    * Any PyTorch model in memory e.g. ``FinetuneableZoobotX(encoder=some_model, ...)``
    * ZoobotTree checkpoint e.g. ``FinetuneableZoobotX(zoobot_checkpoint_loc='path/to/zoobot_tree.ckpt', ...)``

    You could subclass this class to solve new finetuning tasks - see :ref:`advanced_finetuning`.

    Args:
        name (str, optional): Name of a model on HuggingFace Hub e.g. ``hf_hub:mwalmsley/zoobot-encoder-convnext_nano``. Defaults to ``None``.
        encoder (torch.nn.Module, optional): Instead of ``name``, use a PyTorch model already loaded in memory. Defaults to ``None``.
        zoobot_checkpoint_loc (str, optional): Instead of ``name``, use a path to ZoobotTree lightning checkpoint to load. Loads with :func:`zoobot.pytorch.training.finetune.load_pretrained_zoobot`. Defaults to ``None``.
        training_mode (str, optional): ``'full'`` to train all parameters, ``'head_only'`` to freeze encoder and only train head. Defaults to ``'full'``.
        layer_decay (float, optional): For each layer below the head, reduce the learning rate by ``layer_decay ** i``. Defaults to ``0.75``.
        weight_decay (float, optional): AdamW weight decay arg (i.e. L2 penalty). Defaults to ``0.05``.
        learning_rate (float, optional): AdamW learning rate arg. Defaults to ``1e-4``.
        head_dropout_prob (float, optional): Probability of dropout before final output layer. Defaults to ``0.5``.
        scheduler_kwargs (dict, optional): Arguments for the optional learning rate scheduler. Defaults to ``None`` (no scheduler).
        timm_kwargs (dict, optional): Additional arguments for ``timm.create_model``.
        greyscale (bool, optional): If ``True``, convert model to single channel version (adds ``{'in_chans': 1}`` to timm kwargs). Defaults to ``False``.
        prog_bar (bool, optional): Print progress bar during finetuning. Defaults to ``True``.
        visualize_images (bool, optional): Upload example images to WandB. Good for debugging but potentially slow. Defaults to ``False``.
        seed (int, optional): Random seed to use. Defaults to ``42``.
    """

    def __init__(
        self,
        # load a pretrained timm encoder saved on huggingface hub
        # (aimed at most users, easiest way to load published models)
        name=None,
        # ...or directly pass any model to use as encoder (if you do this, you will need to keep it around for later)
        # (aimed at tinkering with new architectures e.g. SSL)
        encoder=None,  # use any torch model already loaded in memory (must have .forward() method)
        # load a pretrained zoobottree model and grab the encoder (a timm model)
        # requires the exact same zoobot version used for training, not very portable
        # (aimed at supervised experiments)
        zoobot_checkpoint_loc=None,
        # finetuning settings
        training_mode='full',  # 'full' to train all params, 'head_only' to freeze encoder and only train head
        layer_decay=0.75,
        weight_decay=0.05,
        learning_rate=1e-4,  # 10x lower than typical, you may like to experiment
        head_dropout_prob=0.5,
        # these args are for the optional learning rate scheduler, best not to use unless you've tuned everything else already
        scheduler_kwargs=None,  # e.g. {'name': 'cosine', 'warmup_epochs': 5, 'max_epochs': 100}
        # debugging utils
        timm_kwargs={},  
        greyscale=False,  # sets {'in_chans': 1} for greyscale models
        prog_bar=True,
        visualize_images=False,  # upload examples to wandb, good for debugging
        seed=42,
    ):
        super().__init__()

        # adds every __init__ arg to model.hparams
        # will also add to wandb if using logging=wandb, I think
        # necessary if you want to reload!
        # with warnings.catch_warnings():
        # warnings.simplefilter("ignore")
        # this raises a warning that encoder is already a Module hence saved in checkpoint hence no need to save as hparam
        # true - except we need it to instantiate this class, so it's really handy to have saved as well
        # therefore ignore the warning
        self.save_hyperparameters(ignore=["encoder"])  # never serialise the encoder, way too heavy
        # if you need the encoder to recreate, pass when loading checkpoint e.g.
        # FinetuneableZoobotTree.load_from_checkpoint(loc, encoder=encoder)

        if greyscale:
            logging.warning('Converting encoder to single channel version')
            timm_kwargs['in_chans'] = 1  # convert model to single channel version

        if name is not None:  # will load from Hub
            assert encoder is None, 'Cannot pass both name and encoder to use'
            self.encoder = timm.create_model(name, num_classes=0, pretrained=True, **timm_kwargs)
            self.encoder_dim: int = self.encoder.num_features

        elif zoobot_checkpoint_loc is not None:  # will load from local checkpoint
            assert encoder is None, "Cannot pass both checkpoint to load and encoder to use"
            self.encoder = load_pretrained_zoobot(
                zoobot_checkpoint_loc
            )  # extracts the timm encoder
            self.encoder_dim: int = self.encoder.num_features  # type: ignore

        else:  # passed encoder in-memory directly
            assert (
                zoobot_checkpoint_loc is None
            ), "Cannot pass both checkpoint to load and encoder to use"
            assert encoder is not None, "Must pass either checkpoint to load or encoder to use"
            self.encoder = encoder
            # find out encoder dimension
            if hasattr(self.encoder, 'num_features'):  # timm models generally use this
                self.encoder_dim = self.encoder.num_features
            elif hasattr(self.encoder, 'embed_dim'):  # timm.models.VisionTransformer uses this
                self.encoder_dim = self.encoder.embed_dim 
            else:  # resort to manual estimate
                self.encoder_dim = define_model.get_encoder_dim(self.encoder)

        self.training_mode = training_mode

        self.learning_rate = learning_rate
        self.layer_decay = layer_decay
        self.weight_decay = weight_decay
        self.head_dropout_prob = head_dropout_prob

        self.scheduler_kwargs = scheduler_kwargs

        self.train_loss_metric = tm.MeanMetric()
        self.val_loss_metric = tm.MeanMetric()
        self.test_loss_metric = tm.MeanMetric()

        self.seed = seed
        self.prog_bar = prog_bar
        self.visualize_images = visualize_images

        # Remove ViT head if it exists
        if hasattr(self.encoder, "head") and isinstance(
            self.encoder, timm.models.VisionTransformer
        ):
            # If the encoder has a 'head' attribute, replace it with Identity()
            self.encoder.head = torch.nn.Identity()
            logging.info("Replaced encoder.head with Identity()")

    def configure_optimizers(self):
        """
        Sets up the optimizer and, optionally, a learning rate scheduler.

        When ``self.training_mode == 'head_only'``, only ``self.head`` is optimized (i.e. frozen encoder, linear finetuning).
        When ``self.training_mode == 'full'``, all parameters are optimized.

        Learning rate decay is applied to the encoder only.
        Counterintuitively, a higher learning rate decay value (e.g. ``0.9``) causes less reduction in the learning rate: the learning rate is (from the top encoder layer down) ``lr``, ``lr * layer_decay``, ``lr * layer_decay**2``, ...
        I use timm's definition of layers, which groups some torch layers together.

        Weight decay (aka L2 regularization, penalizing large weights) is applied to both the head and (if relevant) the encoder.

        For schedulers, I use the timm scheduler factory. See https://github.com/rwightman/timm/blob/main/timm/scheduler/scheduler_factory.py#L63.
        self.scheduler_kwargs (passed to the factory) should be a dict with the scheduler name and any additional args, e.g. ``{'name': 'cosine', 'warmup_epochs': 5, 'max_epochs': 100}``.
        """

        logging.info(f"Encoder architecture to finetune: {type(self.encoder)}")


        if hasattr(self.encoder, 'vit'):  # e.g. mae
            logging.info('Encoder has vit attribute, assuming this is timm VisionTransformer')
            model_to_optimize: torch.nn.Module = self.encoder.vit  # type: ignore
        else:
            model_to_optimize: torch.nn.Module = self.encoder

        if hasattr(model_to_optimize, 'pos_embed'):
            logging.info("Encoder has pos_embed, will not train it")
            model_to_optimize.pos_embed.requires_grad_(False)  # don't train pos_embed - typically, not a learnable parameter, despite timm defaults?

        if self.training_mode == 'full':
            logging.info("Training all parameters, not just the head")
            optimizer = create_optimizer_v2(
                model_to_optimize,
                opt='adamw',
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                layer_decay= self.layer_decay
            )
            # add head parameters to optimizer
            optimizer.add_param_group({'params': self.head.parameters(), 'lr': self.learning_rate})  # type: ignore
        elif self.training_mode == 'head_only':
            logging.info("Training only the head, encoder frozen")
            # freeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            # freeze_batchnorm_layers(self.encoder)
            optimizer = create_optimizer_v2(
                self.head,
                opt='adamw',
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                layer_decay=None
            )

        logging.info("Optimizer ready")

        if self.scheduler_kwargs is not None:
            logging.info(f"Using scheduler with kwargs: {self.scheduler_kwargs}")
            # https://github.com/rwightman/timm/blob/main/timm/scheduler/scheduler_factory.py#L63
            scheduler, _ = create_scheduler_v2(optimizer, **self.scheduler_kwargs) # e.g. 'cosine', warmup_epochs=5)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            logging.info("Learning rate scheduler not used")
            logging.info("Manually applying lr_scale to optimizer param groups (because timm scheduler normally does this, but there is no scheduler)")
            for group in optimizer.param_groups:
                group['lr_scale'] = group.get('lr_scale', 1.0)
                group['lr'] *= group['lr_scale']
            return optimizer


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.head(x)  # type:ignore
        return x

    def make_step(self, batch):
        y, y_pred, loss = self.run_step_through_model(batch)
        return self.step_to_dict(y, y_pred, loss)

    def run_step_through_model(self, batch):
      # part of training/val/test for all subclasses
        x, y = self.batch_to_supervised_tuple(batch)
        y_pred = self.forward(x)
        # must be subclasses and specified
        loss = self.loss(y_pred, y)  # type:ignore
        loss.float()
        return y, y_pred, loss
    
    def batch_to_supervised_tuple(self, batch):
        raise NotImplementedError('Must be subclassed to convert batch to supervised tuple (x, y)')

    def step_to_dict(self, y, y_pred, loss):
        return {"loss": loss.mean(), "predictions": y_pred, "labels": y}

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self.make_step(batch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.make_step(batch)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.make_step(batch)

    def predict_step(self, x: torch.Tensor, batch_idx):
        return self.forward(x['image'])  # type: ignore
        # forward = self(batch)
        # and convert to nice pandas dataframe using self.label_col, self.schema, etc

    def on_train_batch_end(self, outputs, batch, batch_idx: int, dataloader_idx=0):
        # v2 docs currently do not show dataloader_idx as train argument so unclear if this will value be updated properly
        # arg is shown for val/test equivalents
        # currently does nothing in Zoobot so inconsequential
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-train-batch-end
        self.train_loss_metric(outputs["loss"])
        self.log(
            "finetuning/train_loss",
            self.train_loss_metric,
            prog_bar=self.prog_bar,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_batch_end(self, outputs: dict, batch, batch_idx: int, dataloader_idx=0):
        self.val_loss_metric(outputs["loss"])
        self.log(
            "finetuning/val_loss",
            self.val_loss_metric,
            prog_bar=self.prog_bar,
            on_step=False,
            on_epoch=True,
        )
        # unique to val batch end
        if self.visualize_images:
            self.upload_images_to_wandb(outputs, batch, batch_idx)

    def on_test_batch_end(self, outputs: dict, batch, batch_idx: int, dataloader_idx=0):
        self.test_loss_metric(outputs["loss"])
        self.log(
            "finetuning/test_loss",
            self.test_loss_metric,
            prog_bar=self.prog_bar,
            on_step=False,
            on_epoch=True,
        )

    # lighting v2. removed validation_epoch_end(self, outputs)
    # now only has *on_*validation_epoch_end(self)
    # replacing by using explicit torchmetric for loss
    # https://github.com/Lightning-AI/lightning/releases/tag/2.0.0

    def upload_images_to_wandb(self, outputs, batch, batch_idx):
        raise NotImplementedError("Must be subclassed")

    @classmethod
    def load_from_name(cls, name: str, **kwargs):
        downloaded_loc = download_from_name(cls.__name__, name)
        return cls.load_from_checkpoint(
            downloaded_loc, **kwargs
        )  # trained on GPU, may need map_location='cpu' if you get a device error



class FinetuneableZoobotClassifier(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a classification problem.

    Any args not listed below are passed to :class:`FinetuneableZoobotAbstract` (for example, ``learning_rate``).
    These are shared between classifier, regressor, and tree models.
    See the docstring of :class:`FinetuneableZoobotAbstract` for more.

    Models can be loaded with ``FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...)``.
    See :class:`FinetuneableZoobotAbstract` for other loading options (e.g. in-memory models or local checkpoints).

    Args:
        label_col (str, optional): Name of the column in the batch dict (e.g. a column in your dataframe) containing the labels. Defaults to ``'label'``.
        num_classes (int): Number of target classes (e.g. ``2`` for binary classification).
        label_smoothing (float, optional): See torch ``cross_entropy_loss`` docs. Defaults to ``0``.
        class_weights (arraylike, optional): See torch ``cross_entropy_loss`` docs. Defaults to ``None``.
        run_linear_sanity_check (bool, optional): Before fitting, use sklearn to fit a linear model. Defaults to ``False``.
    """

    def __init__(
            self,
            num_classes: int,
            label_col: str = 'label',
            label_smoothing=0.,
            class_weights=None,

            run_linear_sanity_check: bool = False,
            **super_kwargs) -> None:

        super().__init__(**super_kwargs)

        self.label_col = label_col

        self.label_col = label_col

        logging.info("Using classification head and cross-entropy loss")
        self.head = LinearHead(
            input_dim=self.encoder_dim,  # type: ignore
            output_dim=num_classes,
            head_dropout_prob=self.head_dropout_prob,
        )
        self.label_smoothing = label_smoothing

        # if isinstance(class_weights, list) or isinstance(class_weights, np.ndarray):
        if class_weights is not None:
            # https://lightning.ai/docs/pytorch/stable/accelerators/accelerator_prepare.html#init-tensors-using-tensor-to-and-register-buffer
            self.register_buffer("class_weights", torch.Tensor(class_weights))
            print(self.class_weights, self.class_weights.device)  # type: ignore
            # can now use self.class_weights in forward pass and will be on correct device (because treated as model parameters)
        else:
            self.class_weights = None

        self.loss = partial(cross_entropy_loss,
                            weight=self.class_weights,
                            label_smoothing=self.label_smoothing)
        logging.info(f'num_classes: {num_classes}')

        if num_classes == 2:
            logging.info("Using binary classification")
            task = "binary"
        else:
            logging.info("Using multi-class classification")
            task = "multiclass"
        self.train_acc = tm.Accuracy(task=task, average="micro", num_classes=num_classes)
        self.val_acc = tm.Accuracy(task=task, average="micro", num_classes=num_classes)
        self.test_acc = tm.Accuracy(task=task, average="micro", num_classes=num_classes)

        self.run_linear_sanity_check = run_linear_sanity_check

    def step_to_dict(self, y, y_pred, loss):
        y_class_preds = torch.argmax(y_pred, axis=1)  # type: ignore
        return {
            "loss": loss.mean(),
            "predictions": y_pred,
            "labels": y,
            "class_predictions": y_class_preds,
        }

    def batch_to_supervised_tuple(self, batch):
        return batch['image'], batch[self.label_col]

    def on_train_batch_end(self, step_output, *args):
        super().on_train_batch_end(step_output, *args)

        self.train_acc(step_output["class_predictions"], step_output["labels"])
        self.log(
            "finetuning/train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar,
        )

    def on_validation_batch_end(self, step_output, *args):
        super().on_validation_batch_end(step_output, *args)

        self.val_acc(step_output["class_predictions"], step_output["labels"])
        self.log(
            "finetuning/val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar,
        )

    def on_test_batch_end(self, step_output, *args) -> None:
        super().on_test_batch_end(step_output, *args)

        self.test_acc(step_output["class_predictions"], step_output["labels"])
        self.log(
            "finetuning/test_acc",
            self.test_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar,
        )

    def predict_step(self, x: Union[list[torch.Tensor], torch.Tensor], batch_idx):
        # overrides abstract version
        x = self.forward(x['image'])  # type: ignore # logits from LinearHead
        # then applies softmax. Only in predict step because we prefer logits for gradient stability during training
        return F.softmax(x, dim=1)

    def upload_images_to_wandb(self, outputs, batch, batch_idx):
      # self.logger is set by L.Trainer(logger=) argument
        if (self.logger is not None) and (batch_idx == 0):
            x, y = batch
            y_pred_softmax = F.softmax(outputs["predictions"], dim=1)
            n_images = 5
            images = [img for img in x[:n_images]]
            captions = [
                f"Ground Truth: {y_i} \nPrediction: {y_p_i}"
                for y_i, y_p_i in zip(y[:n_images], y_pred_softmax[:n_images])
            ]
            self.logger.log_image(key="val_images", images=images, caption=captions)  # type: ignore

        # Sanity check embeddings with linear evaluation first
    def on_train_start(self) -> None:
       if self.run_linear_sanity_check:  # default False
           self.linear_sanity_check()

    def linear_sanity_check(self):
        # only implemented on Zoobot...Classifier as assumes accuracy
        with torch.no_grad():
            embeddings, labels = {"train": [], "val": []}, {"train": [], "val": []}

            # Get validation set embeddings
            for x, y in self.trainer.datamodule.val_dataloader():  # type: ignore
                embeddings["val"] += self.encoder(x.to(self.device)).cpu()
                labels["val"] += y

            # Get train set embeddings
            for x, y in self.trainer.datamodule.train_dataloader():  # type: ignore
                embeddings["train"] += self.encoder(x.to(self.device)).cpu()
                labels["train"] += y

            # this is linear *train* acc but that's okay, simply test of features
            model = linear_model.LogisticRegression(penalty=None, max_iter=200)
            model.fit(embeddings["train"], labels["train"])

            self.log(
                "finetuning/linear_eval/val",
                accuracy_score(labels["val"], model.predict(embeddings["val"])),  # type: ignore
            )
            self.log(
                "finetuning/linear_eval/train",
                accuracy_score(labels["train"], model.predict(embeddings["train"])),  # type: ignore
            )
            # doesn't need to be torchmetric, only happens in one go? but distributed

class FinetuneableZoobotRegressor(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a regression problem.

    Any args not listed below are passed to :class:`FinetuneableZoobotAbstract` (for example, ``learning_rate``).
    These are shared between classifier, regressor, and tree models.
    See the docstring of :class:`FinetuneableZoobotAbstract` for more.

    Models can be loaded with ``FinetuneableZoobotRegressor(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...)``.
    See :class:`FinetuneableZoobotAbstract` for other loading options (e.g. in-memory models or local checkpoints).


    Args:
        label_col (str, optional): Name of the column in the batch dict (e.g. a column in your dataframe) containing the labels. Defaults to ``'label'``.
        loss (str, optional): Loss function to use. Must be one of ``'mse'``, ``'mae'``. Defaults to ``'mse'``.
        unit_interval (bool, optional): If ``True``, use sigmoid activation for the final layer, ensuring predictions between 0 and 1. Defaults to ``False``.
    """

    def __init__(
            self,
            label_col: str = 'label',
            loss: str = 'mse',
            unit_interval: bool = False,
            **super_kwargs) -> None:

        super().__init__(**super_kwargs)

        self.label_col = label_col  # TODO could add MultipleLabelRegressor, Nasser working on this

        self.unit_interval = unit_interval
        if self.unit_interval:
            logging.info("unit_interval=True, using sigmoid activation for finetunng head")
            head_activation = torch.nn.functional.sigmoid
        else:
            head_activation = None

        logging.info("Using classification head and cross-entropy loss")
        self.head = LinearHead(
            input_dim=self.encoder_dim,
            output_dim=1,
            head_dropout_prob=self.head_dropout_prob,
            activation=head_activation,
        )
        if loss in ["mse", "mean_squared_error"]:
            self.loss = mse_loss
        elif loss in ["mae", "mean_absolute_error", "l1", "l1_loss"]:
            self.loss = l1_loss
        else:
            raise ValueError(f"Loss {loss} not recognised. Must be one of mse, mae")

        # rmse metrics. loss is mse already.
        self.train_rmse = tm.MeanSquaredError(squared=False)
        self.val_rmse = tm.MeanSquaredError(squared=False)
        self.test_rmse = tm.MeanSquaredError(squared=False)


    def step_to_dict(self, y, y_pred, loss):
        return {'loss': loss.mean(), 'predictions': y_pred, 'labels': y}
    
    def batch_to_supervised_tuple(self, batch):
        return batch['image'], batch[self.label_col]

    def on_train_batch_end(self, step_output, *args):
        super().on_train_batch_end(step_output, *args)

        self.train_rmse(step_output["predictions"], step_output["labels"])
        self.log(
            "finetuning/train_rmse",
            self.train_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar,
        )

    def on_validation_batch_end(self, step_output, *args):
        super().on_validation_batch_end(step_output, *args)

        self.val_rmse(step_output["predictions"], step_output["labels"])
        self.log(
            "finetuning/val_rmse",
            self.val_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar,
        )

    def on_test_batch_end(self, step_output, *args) -> None:
        super().on_test_batch_end(step_output, *args)

        self.test_rmse(step_output["predictions"], step_output["labels"])
        self.log(
            "finetuning/test_rmse",
            self.test_rmse,
            on_step=False,
            on_epoch=True,
            prog_bar=self.prog_bar,
        )

    def predict_step(self, x: torch.Tensor, batch_idx):
        return self.forward(x['image'])  # type: ignore


class FinetuneableZoobotTree(FinetuneableZoobotAbstract):
    """
    Pretrained Zoobot model intended for finetuning on a decision tree (i.e. GZ-like) problem.
    Uses Dirichlet-Multinomial loss introduced in GZ DECaLS.
    Briefly: predicts a Dirichlet distribution for the probability of a typical volunteer giving each answer,
    and uses the Dirichlet-Multinomial loss to compare the predicted distribution of votes (given k volunteers were asked) to the true distribution.

    Does not produce accuracy or MSE metrics, as these are not relevant for this task. Loss logging only.

    If you're using this, you're probably working on a Galaxy Zoo catalog, and you should Slack Mike!

    Any args not listed below are passed to :class:`FinetuneableZoobotAbstract` (for example, ``learning_rate``).
    These are shared between classifier, regressor, and tree models.
    See the docstring of :class:`FinetuneableZoobotAbstract` for more.

    Models can be loaded with ``FinetuneableZoobotTree(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...)``.
    See :class:`FinetuneableZoobotAbstract` for other loading options (e.g. in-memory models or local checkpoints).

    Args:
        schema (schemas.Schema): Description of the layout of the decision tree. See :class:`zoobot.shared.schemas.Schema`.
    """

    def __init__(self, schema: schemas.Schema, **super_kwargs):
        """Initialize the finetuneable Zoobot tree model.

        Args:
            schema (schemas.Schema): Description of the layout of the decision tree. See :class:`zoobot.shared.schemas.Schema` for examples.
        """

        super().__init__(**super_kwargs)

        logging.info("Using dropout+dirichlet head and dirichlet (count) loss")

        self.schema = schema
        self.output_dim = len(self.schema.label_cols)

        self.head = define_model.get_pytorch_dirichlet_head(
            encoder_dim=self.encoder_dim,
            output_dim=self.output_dim,
            test_time_dropout=False,
            dropout_rate=self.head_dropout_prob,
        )

        self.loss = define_model.get_dirichlet_loss_func(self.schema.question_answer_pairs)

    def batch_to_supervised_tuple(self, batch):
        """
        Converts a batch to a supervised tuple (x, y) for training.
        x is the image, y is the counts of votes for each answer in the schema.
        """
        x = batch['image']
        # y is a dict with keys as label_cols and values as counts
        y = dict([(label_col, batch[label_col]) for label_col in self.schema.label_cols])

        # old version, tensor of shape (batch, answers)
        # y = torch.stack([batch[label_col] for label_col in self.schema.label_cols], dim=1)
        return x, y

    def upload_images_to_wandb(self, outputs, batch, batch_idx):
        raise NotImplementedError

    # other functions are simply inherited from FinetunedZoobotAbstract


class LinearHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, head_dropout_prob=0.5, activation=None):
        """
        Small utility class for a linear head with dropout and optional choice of activation.

        - Apply dropout to features before the final linear layer.
        - Apply a final linear layer.
        - Optionally, apply ``activation`` callable.

        Args:
            input_dim (int): Input dimension of the linear layer (i.e. the encoder output dimension).
            output_dim (int): Output dimension of the linear layer (often e.g. N for N classes, or 1 for regression).
            head_dropout_prob (float, optional): Dropout probability. Defaults to ``0.5``.
            activation (callable, optional): Callable expecting tensor e.g. torch softmax. Defaults to ``None``.
        """
        # input dim is representation dim, output_dim is num classes
        super(LinearHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = torch.nn.Dropout(p=head_dropout_prob)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        """
        Returns logits, as recommended for CrossEntropy loss.

        Args:
            x (torch.Tensor): Encoded representation.

        Returns:
            torch.Tensor: Result (see docstring of LinearHead).
        """
        #
        x = self.dropout(x)
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.output_dim == 1:
            return x.squeeze()
        else:
            return x


def cross_entropy_loss(
    y_pred: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0, weight=None
):
    """
    Calculate cross-entropy loss with optional label smoothing and class weights. No aggregation applied.
    Trivial wrapper of ``torch.nn.functional.cross_entropy`` with ``reduction='none'``.

    Args:
        y_pred (torch.Tensor): Predictions of shape ``(batch, classes)``.
        y (torch.Tensor): Targets of shape ``(batch)``.
        label_smoothing (float, optional): See docstring of ``torch.nn.functional.cross_entropy``. Defaults to ``0.0``.
        weight (arraylike, optional): See docstring of ``torch.nn.functional.cross_entropy``. Defaults to ``None``.

    Returns:
        torch.Tensor: Unreduced cross-entropy loss.
    """

    # added .to(y) to ensure weights are on same device, bit of a hack but self.register_buffer() doesn't work as I expected
    # should be true automatically anyway if passing in self.weights above
    return F.cross_entropy(y_pred, y.long(), label_smoothing=label_smoothing, weight=weight.to(y) if weight is not None else None, reduction='none')  


def mse_loss(y_pred, y):
    """
    Trivial wrapper of ``torch.nn.functional.mse_loss`` with ``reduction='none'``.

    Args:
        y_pred (torch.Tensor): See docstring of ``torch.nn.functional.mse_loss``.
        y (torch.Tensor): See docstring of ``torch.nn.functional.mse_loss``.

    Returns:
        torch.Tensor: See docstring of ``torch.nn.functional.mse_loss``.
    """
    return F.mse_loss(y_pred, y, reduction="none")


def l1_loss(y_pred, y):
    """
    Trivial wrapper of ``torch.nn.functional.l1_loss`` with ``reduction='none'``.

    Args:
        y_pred (torch.Tensor): See docstring of ``torch.nn.functional.l1_loss``.
        y (torch.Tensor): See docstring of ``torch.nn.functional.l1_loss``.

    Returns:
        torch.Tensor: See docstring of ``torch.nn.functional.l1_loss``.
    """
    return F.l1_loss(y_pred, y, reduction="none")


def load_pretrained_zoobot(checkpoint_loc: str) -> torch.nn.Module:
    """
    Load a pretrained Zoobot encoder from a LightningModule checkpoint.

    Args:
        checkpoint_loc (str): Path to saved LightningModule checkpoint, likely of :class:`ZoobotTree`, :class:`FinetuneableZoobotClassifier`, or :class:`FinetuneableZoobotTree`. Must have ``.encoder`` attribute.

    Returns:
        torch.nn.Module: Pretrained PyTorch encoder within that LightningModule.
    """
    if torch.cuda.is_available():
        map_location = None
    else:
        # necessary to load gpu-trained model on cpu
        map_location = torch.device('cpu')

    # changed
    try:
        logging.info('Attempting to load ZoobotTree from checkpoint')
        return define_model.ZoobotTree.load_from_checkpoint(checkpoint_loc, map_location=map_location).encoder # type: ignore
    except TypeError:
        logging.info('Attempting to load FinetuneableZoobotTree from checkpoint')
        return FinetuneableZoobotTree.load_from_checkpoint(checkpoint_loc, map_location=map_location).encoder # type: ignore


def get_trainer(
    save_dir: str,
    file_template="{epoch}",
    save_top_k=1,
    max_epochs=100,
    patience=10,
    devices="auto",
    accelerator="auto",
    logger=None,
    **trainer_kwargs
) -> L.Trainer:
    """
    Convenience wrapper to create a PyTorch Lightning Trainer that carries out the finetuning process.
    Use like so: ``trainer.fit(model, datamodule)``

    ``get_trainer`` args are for common Trainer settings e.g. early stopping, checkpointing, etc. By default:
    - Saves the top-k models based on validation loss
    - Uses early stopping with ``patience`` (i.e. end training if validation loss does not improve after ``patience`` epochs)
    - Monitors the learning rate (useful when using a learning rate scheduler)

    Any extra args not listed below are passed directly to the PyTorch Lightning Trainer.
    Use this to add any custom configuration not covered by the ``get_trainer`` args.
    See https://lightning.ai/docs/pytorch/stable/common/trainer.html

    Args:
        save_dir (str): Folder in which to save checkpoints and logs.
        file_template (str, optional): Custom naming for checkpoint files. See Lightning docs. Defaults to ``"{epoch}"``.
        save_top_k (int, optional): Save the top k checkpoints only. Defaults to ``1``.
        max_epochs (int, optional): Train for up to this many epochs. Defaults to ``100``.
        patience (int, optional): Wait up to this many epochs for decreasing loss before ending training. Defaults to ``10``.
        devices (str, optional): Number of devices for training (typically, num. GPUs). Defaults to ``'auto'``.
        accelerator (str, optional): Which device to use (typically ``'gpu'`` or ``'cpu'``). Defaults to ``'auto'``.
        logger (L.pytorch.loggers.wandb_logger, optional): If ``L.pytorch.loggers.wandb_logger``, track experiment on Weights and Biases. Defaults to ``None``.

    Returns:
        L.Trainer: PyTorch Lightning trainer object for finetuning a model on a GalaxyDataModule.
    """

    checkpoint_callback = ModelCheckpoint(
        monitor="finetuning/val_loss",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
        verbose=True,
        dirpath=os.path.join(save_dir, "checkpoints"),
        filename=file_template,
        save_weights_only=True,
        save_top_k=save_top_k,
    )

    early_stopping_callback = EarlyStopping(
        monitor="finetuning/val_loss", mode="min", patience=patience
    )

    learning_rate_monitor_callback = LearningRateMonitor(logging_interval="epoch")

    # Initialise pytorch lightning trainer
    trainer = L.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            early_stopping_callback,
            learning_rate_monitor_callback,
        ],
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        **trainer_kwargs,
    )

    return trainer


def download_from_name(class_name: str, hub_name: str):
    """
    Download a finetuned model from the HuggingFace Hub by name.
    Used to load pretrained Zoobot models by name, e.g. ``FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...)``.

    Downloaded models are saved to the HuggingFace cache directory for later use (typically ``~/.cache/huggingface``).

    You shouldn't need to call this; it's used internally by the FinetuneableZoobot classes.

    Args:
        class_name (str): One of ``FinetuneableZoobotClassifier``, ``FinetuneableZoobotRegressor``, ``FinetuneableZoobotTree``.
        hub_name (str): e.g. ``mwalmsley/zoobot-encoder-convnext_nano``.

    Returns:
        str: Path to downloaded model (in HuggingFace cache directory). Likely then loaded by Lightning.
    """
    from huggingface_hub import hf_hub_download

    if hub_name.startswith("hf_hub:"):
        logging.info("Passed name with hf_hub: prefix, dropping prefix")
        repo_id = hub_name.split("hf_hub:")[1]
    else:
        repo_id = hub_name
    downloaded_loc = hf_hub_download(repo_id=repo_id, filename=f"{class_name}.ckpt")
    return downloaded_loc
    return downloaded_loc
