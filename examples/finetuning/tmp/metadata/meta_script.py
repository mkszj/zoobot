# Import necessary libraries
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform
from zoobot.pytorch.training.finetune import FinetuneableZoobotMetadataRegressor, get_trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from zoobot.pytorch.predictions import predict_on_catalog
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from matplotlib.lines import Line2D

# Define normalization function
def normalize(array):
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(array.values.reshape(-1, 1))
    return pd.Series(normalized.flatten(), index=array.index, name='norm', dtype='float32'), scaler

# Paths and parameters
path = '/home/nasserm/local/zoobot/examples/finetuning/tmp/metadata/'
label = 'Vm'  # km/s
id_str = 'plateifu'
batch_size = 32
num_workers = 64
save_dir = '/home/nasserm/local/zoobot/examples/finetuning/tmp/metadata/training'
predictions_path = '/home/nasserm/local/zoobot/examples/finetuning/tmp/metadata/training/preds/save.csv'

# Load datasets
train_ = pd.read_csv('rot/train_dataset.csv')
test_ = pd.read_csv('rot/test_dataset.csv')

train_['file_loc'] = path + train_['file_loc'].astype(str)
train_['id_str'] = train_[id_str]
test_['file_loc'] = path + test_['file_loc'].astype(str)
test_['id_str'] = test_[id_str]

# Normalize label column
train_[label] = normalize(np.abs(train_[label]))[0]
test_[label] = normalize(np.abs(test_[label]))[0]

# Data transformations
transform_cfg = default_view_config()
transform = get_galaxy_transform(transform_cfg)

datamodule = CatalogDataModule(
    label_cols=[label, 'nsa_sersic_mass'],
    catalog=train_,
    train_transform=transform,
    test_transform=transform,
    batch_size=batch_size,
    num_workers=num_workers
)

# Define model
model = FinetuneableZoobotMetadataRegressor(
    label_col=label,
    name='nassm/convnext_nano',
    learning_rate=1e-6,
    unit_interval=True,
    metadata_cols=['nsa_sersic_mass']
)

# Setup trainer
csv_logger = CSVLogger(
    save_dir=save_dir,
    name='training_logs',
    version=None
)
trainer = get_trainer(
    save_dir,
    accelerator='auto',
    devices='auto',
    max_epochs=2,
    logger=csv_logger,
    enable_checkpointing=True
)

# Train model
trainer.fit(model, datamodule)

# Load best model
best_checkpoint = trainer.checkpoint_callback.best_model_path
finetuned_model = FinetuneableZoobotMetadataRegressor.load_from_checkpoint(best_checkpoint)

# Make predictions
predictions_result = predict_on_catalog.predict(
    catalog=test_,
    model=finetuned_model,
    inference_transform=transform,
    label_cols=['label'],
    save_loc=predictions_path,
    trainer_kwargs={'accelerator': 'auto', 'devices': 'auto'},
    datamodule_kwargs={'num_workers': 2, 'batch_size': 32},
)

# Plot results
def plot_results():
    def plot_form(ax):
        ax.grid(ls='-.', alpha=0.5, zorder=0)
        ax.tick_params(direction='in')
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_linewidth(2)
            ax.spines[spine].set_color('0.8')
        ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
        ax.minorticks_on()
        plt.rcParams['font.family'] = 'serif'

    def add_metrics_legend(ax, y_true, y_pred):
        def sigfig(x, n=2):
            if x == 0:
                return "0"
            return f"{x:.{n}g}"
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        text = f"MAE = {sigfig(mae, 3)}\n$R^2$ = {sigfig(r2, 2)}"
        handle = Line2D([], [], marker='o', color='w', alpha=0, linestyle='None')
        ax.legend([handle], [text], loc='upper left', fontsize=11, frameon=True,
                  borderpad=0.7, labelspacing=0.3, handlelength=0, handletextpad=0)

    def plot_contours(ax, x, y):
        cmap = sns.light_palette('steelblue', as_cmap=True)
        if len(x) > 10:
            try:
                sns.kdeplot(x=x, y=y, ax=ax, fill=True, cmap=cmap,
                            levels=8, thresh=0.05, alpha=0.25, zorder=0)
                sns.kdeplot(x=x, y=y, ax=ax, color='steelblue',
                            levels=8, thresh=0.05, linewidths=1.2,
                            fill=False, alpha=0.15, zorder=1)
            except Exception:
                pass

    merged = test_.merge(predictions_result, on='id_str')
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_form(ax)

    x = merged['Vm']       # true
    y = merged['label']    # predicted

    plot_contours(ax, x, y)
    sns.scatterplot(x=x, y=y, ax=ax, alpha=0.7, s=25, color='steelblue', edgecolor='none')

    # 1:1 reference line
    ax.plot([x.min()-0.1, x.max()], [x.min()-0.1, x.max()],
            color='black', linestyle='--', linewidth=2)

    # labels and title
    ax.set_xlabel('True $V_m$', fontsize=13)
    ax.set_ylabel('Predicted $V_m$', fontsize=13)
    ax.set_title('Predicted vs True $V_m$', fontsize=14)
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(-0.1, 1)

    # add metrics legend
    add_metrics_legend(ax, x, y)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load datasets
    train_ = pd.read_csv('rot/train_dataset.csv')
    test_ = pd.read_csv('rot/test_dataset.csv')

    train_['file_loc'] = path + train_['file_loc'].astype(str)
    train_['id_str'] = train_[id_str]
    test_['file_loc'] = path + test_['file_loc'].astype(str)
    test_['id_str'] = test_[id_str]

    # Normalize label column
    train_[label] = normalize(np.abs(train_[label]))[0]
    test_[label] = normalize(np.abs(test_[label]))[0]

    # Data transformations
    transform_cfg = default_view_config()
    transform = get_galaxy_transform(transform_cfg)

    datamodule = CatalogDataModule(
        label_cols=[label, 'nsa_sersic_mass'],
        catalog=train_,
        train_transform=transform,
        test_transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Define model
    model = FinetuneableZoobotMetadataRegressor(
        label_col=label,
        name='nassm/convnext_nano',
        learning_rate=1e-6,
        unit_interval=True,
        metadata_cols=['nsa_sersic_mass']
    )

    # Setup trainer
    csv_logger = CSVLogger(
        save_dir=save_dir,
        name='training_logs',
        version=None
    )
    trainer = get_trainer(
        save_dir,
        accelerator='auto',
        devices='auto',
        max_epochs=5,
        logger=csv_logger,
        enable_checkpointing=True
    )

    # Train model
    trainer.fit(model, datamodule)

    # Load best model
    best_checkpoint = trainer.checkpoint_callback.best_model_path
    finetuned_model = FinetuneableZoobotMetadataRegressor.load_from_checkpoint(best_checkpoint)

    # Make predictions
    predictions_result = predict_on_catalog.predict(
        catalog=test_,
        model=finetuned_model,
        inference_transform=transform,
        label_cols=['label'],
        save_loc=predictions_path,
        trainer_kwargs={'accelerator': 'auto', 'devices': 'auto'},
        datamodule_kwargs={'num_workers': 2, 'batch_size': 32},
    )

    # Plot results
    plot_results()

    plt.savefig("pred_vs_true_vm.jpg", dpi=300, bbox_inches='tight')