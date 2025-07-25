.. _choosing_parameters:

Choosing Parameters
=====================================

All ``FinetuneableZoobot`` classes share a common set of parameters for controlling the finetuning process. These can have a big effect on performance.

Finetuning is fast and easy to experiment with, so we recommend trying different parameters to see what works best for your dataset.
This guide provides some explanation for each option.

We list the key parameters below in rough order of importance. 
See :class:`zoobot.pytorch.training.finetune.FinetuneableZoobotAbstract` for the full list of parameters.

``learning_rate``
...............................

Learning rate sets how fast the model parameters are updated during training.
Zoobot uses the adaptive optimizer ``AdamW``.
Adaptive optimizers adjust the learning rate for each parameter based on the mean and variance of the previous gradients.
This means you don't need to tune the learning rate as carefully as you would with a fixed learning rate optimizer like SGD.
We find a learning rate of ``1e-4`` is a good starting point for most tasks.

If you find the model is not learning, you can try increasing the learning rate.
If you see the model loss is varying wildly, or the train loss decreases much faster than the validation loss (overfitting), you can try decreasing the learning rate.
Using ``training_mode='full'`` (below) often requires a lower learning rate than ``training_mode='head_only'``, as the model will adjust more parameters for each batch.


``training_mode``
...............................

Deep learning models are often divided into an encoder and a head.
The encoder is the part of the model that learns to extract features from the input data, while the head is the (much smaller) part that makes predictions based on those features.
In Zoobot, when you do e.g. ``FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', ...)`` the encoder is this ConvNext model.

The ``training_mode`` parameter controls whether you finetune the whole model (``'full'``, encoder and head) or just the head (``'head_only'``).

Training the whole model is often called end-to-end finetuning, and just the head is often called transfer learning or linear probing.
End-to-end finetuning can give better results, but it often requires more data (or smaller pretrained models) and more careful tuning of the learning rate and other parameters.


``layer_decay``
............................... 

The common intuition for deep learning is that lower layers (near the input) learn simple general features and higher layers (near the output) learn more complex features specific to your task.
It is often useful to adjust the learning rate to be lower for lower layers, which have already been pretrained to recognise simple galaxy features.

Learning rate decay reduces the learning rate by layer.
For example, with ``learning_rate=1e-4`` and ``layer_decay=0.75`` (the default):

* The highest block (group of layers) has a learning rate of ``1e-4 * (0.75 ** 0) = 1e-4``
* The second-highest block has a learning rate of ``1e-4 * (0.75 ** 1) = 7.5e-5``
* The third-highest block has a learning rate of ``1e-4 * (0.75 ** 2) = 5.6e-5``

and so on.

Decreasing ``layer_decay`` will exponentially decrease the learning rate for lower blocks.
Notice that this is a bit counterintuitive: a lower ``layer_decay`` means a faster learning rate decrease for lower blocks.

In the extreme cases:

* Setting ``layer_decay=0`` will disable learning in all blocks except the first block (``0 ** 0 = 1``).
* Setting ``layer_decay=1`` will give all blocks the same learning rate.

The head always uses the full learning rate.

``weight_decay``
...............................

Weight decay is a regularization term that penalizes large weights.
When using Zoobot's default ``AdamW`` optimizer, it is closely related to L2 regularization, though there's some subtlety - see https://arxiv.org/abs/1711.05101.
Increasing weight decay will increase the penalty on large weights, which can help prevent overfitting, but may slow or even stop training.
By default, Zoobot uses a small weight decay of ``0.05``.

The head does not use weight decay.


``head_dropout_prob``
...............................

Dropout is a regularization technique that randomly sets some activations to zero during training.
Similarly to weight decay, dropout can help prevent overfitting.
Zoobot uses a head dropout probability of ``0.5`` by default.


``scheduler_kwargs``
.................................

Gradually reducing the learning rate during training can slightly improve results by finding a better minimum near convergence.
This process is called learning rate scheduling.
Zoobot supports (new in v2.9) the ``timm`` learning rate schedulers.
You can read the available options `here <https://github.com/rwightman/timm/blob/main/timm/scheduler/scheduler_factory.py#L63>`_.
Arguments passed to ``scheduler_kwargs`` are passed to the scheduler constructor.
For example, to use a cosine learning rate schedule, you can set:

.. code-block:: python

    scheduler_kwargs = {
        'sched': 'cosine',
        'num_epochs': 50,
        'decay_rate': 0.1
    }

