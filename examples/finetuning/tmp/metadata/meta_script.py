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
import torch
import os
import time

# Define normalization function
def normalize(array):
    scaler = MinMaxScaler()
    arr = array.to_numpy().reshape(-1, 1)
    normalized = scaler.fit_transform(arr)
    return pd.Series(normalized.flatten(), index=array.index, name='norm', dtype='float32'), scaler

# Define data loading function
def load_datasets():
    train_ = pd.read_csv('rot/train_dataset.csv')
    test_ = pd.read_csv('rot/test_dataset.csv')

    train_['file_loc'] = path + train_['file_loc'].astype(str)
    train_['id_str'] = train_[id_str]
    test_['file_loc'] = path + test_['file_loc'].astype(str)
    test_['id_str'] = test_[id_str]

    # Preserve raw label values before normalization so we can inverse-transform later
    train_[f'{label}_raw'] = np.abs(train_[label])
    test_[f'{label}_raw'] = np.abs(test_[label])

    # Normalize label column using scaler fitted on training set
    train_norm_series, scaler = normalize(train_[f'{label}_raw'])
    train_[label] = train_norm_series

    # Apply the same scaler to test set (consistent scaling)
    test_norm = scaler.transform(test_[f'{label}_raw'].to_numpy().reshape(-1, 1)).flatten()
    test_[label] = pd.Series(test_norm, index=test_.index, name='norm', dtype='float32')

    return train_, test_, scaler

# Define model training function
def train_model(train_):
    datamodule = CatalogDataModule(
        label_cols=['Vm', 'nsa_sersic_mass', 'nsa_sersic_flux_F', 'nsa_sersic_flux_N', 'nsa_sersic_flux_u', 'nsa_sersic_flux_g', 'nsa_sersic_flux_r', 'nsa_sersic_flux_i', 'nsa_sersic_flux_z'],
        catalog=train_,
        train_transform=transform,
        test_transform=transform,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = FinetuneableZoobotMetadataRegressor(
        label_col=label,
        name='nassm/convnext_nano',
        learning_rate=1e-6,
        unit_interval=True,
        metadata_cols=['nsa_sersic_mass', 'nsa_sersic_flux_F', 'nsa_sersic_flux_N', 'nsa_sersic_flux_u', 'nsa_sersic_flux_g', 'nsa_sersic_flux_r', 'nsa_sersic_flux_i', 'nsa_sersic_flux_z']
    )

    csv_logger = CSVLogger(
        save_dir=save_dir,
        name='training_logs',
        version=None
    )
    trainer = get_trainer(
        save_dir,
        accelerator='auto',
        devices='auto',
        max_epochs=100,
        logger=csv_logger,
        enable_checkpointing=True
    )

    t0 = time.time()
    trainer.fit(model, datamodule)
    t1 = time.time()
    duration = t1 - t0

    best_checkpoint = trainer.checkpoint_callback.best_model_path
    finetuned_model = FinetuneableZoobotMetadataRegressor.load_from_checkpoint(best_checkpoint)

    # return trainer too so we can access logger outputs for plotting and diagnostics
    return finetuned_model, trainer, duration, best_checkpoint


def save_and_plot_losses(trainer):
    """Locate CSVLogger metrics file, extract train/val loss vs step, save CSV and plot."""
    import os, glob
    # determine logger directory
    log_dir = None
    try:
        log_dir = trainer.logger.log_dir
    except Exception:
        try:
            # fallback: CSVLogger save_dir/name/version pattern
            log_dir = os.path.join(save_dir, 'training_logs')
        except Exception:
            log_dir = None

    if log_dir is None or not os.path.exists(log_dir):
        print(f"No logger directory found at {log_dir}")
        return

    # find metrics csv (CSVLogger writes metrics.csv)
    metrics_files = glob.glob(os.path.join(log_dir, '*.csv'))
    metrics_file = None
    for f in metrics_files:
        # prefer metrics.csv
        if os.path.basename(f).lower().startswith('metrics') or 'metrics' in os.path.basename(f).lower():
            metrics_file = f
            break
    if metrics_file is None and metrics_files:
        metrics_file = metrics_files[0]

    if metrics_file is None:
        print(f"No CSV metrics file found in {log_dir}")
        return

    df = pd.read_csv(metrics_file)
    print(f"Using metrics file: {metrics_file}")
    print(f"Metrics columns: {df.columns.tolist()}")

    # find step column
    step_col = None
    for candidate in ['step', 'global_step', 'epoch']:
        if candidate in df.columns:
            step_col = candidate
            break
    if step_col is None:
        # use row index as step
        df['step_index'] = df.index
        step_col = 'step_index'

    # find train/val loss columns (allow nested names like 'finetuning/train_loss')
    train_loss_col = next((c for c in df.columns if 'train_loss' in c.lower()), None)
    val_loss_col = next((c for c in df.columns if 'val_loss' in c.lower()), None)

    # build out CSV to save
    out_cols = [step_col]
    if train_loss_col:
        out_cols.append(train_loss_col)
    if val_loss_col:
        out_cols.append(val_loss_col)

    if len(out_cols) <= 1:
        print(f"No train/val loss columns found in {metrics_file}, available columns: {df.columns.tolist()}")
        return

    out_df = df[out_cols].copy()

    # ensure numeric and drop rows with NaNs in step or losses
    out_df[step_col] = pd.to_numeric(out_df[step_col], errors='coerce')
    if train_loss_col:
        out_df[train_loss_col] = pd.to_numeric(out_df[train_loss_col], errors='coerce')
    if val_loss_col:
        out_df[val_loss_col] = pd.to_numeric(out_df[val_loss_col], errors='coerce')
    out_df = out_df.dropna(subset=[step_col] + ([train_loss_col] if train_loss_col else []) + ([val_loss_col] if val_loss_col else []))

    # sort by step to ensure correct plotting
    out_df = out_df.sort_values(by=step_col)

    losses_csv = os.path.join(save_dir, 'training_losses.csv')
    out_df.to_csv(losses_csv, index=False)
    print(f"Saved training losses to {losses_csv}")

    # plot with markers and clearer styling
    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    if train_loss_col and train_loss_col in out_df.columns:
        ax.plot(out_df[step_col], out_df[train_loss_col], label='train_loss', marker='o', markersize=4, linewidth=1.2, alpha=0.9)
        plotted = True
    if val_loss_col and val_loss_col in out_df.columns:
        ax.plot(out_df[step_col], out_df[val_loss_col], label='val_loss', marker='s', markersize=4, linewidth=1.2, alpha=0.9)
        plotted = True

    if not plotted:
        print('No valid loss series to plot after cleaning.')
        plt.close(fig)
        return

    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.set_title('Loss vs Training Step')
    ax.grid(True, ls='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    loss_plot = os.path.join(save_dir, 'loss_vs_steps.png')
    plt.savefig(loss_plot, dpi=200)
    plt.close(fig)
    print(f"Saved loss plot to {loss_plot}")


# Define prediction function
def make_predictions(finetuned_model, test_, scaler):
    # predict on the full test set (was using test_[0:5] which produced only 5 predictions)
    predict_catalog = test_.copy()

    predict_datamodule = CatalogDataModule(
        label_cols=['nsa_sersic_mass', 'nsa_sersic_flux_F', 'nsa_sersic_flux_N', 'nsa_sersic_flux_u', 'nsa_sersic_flux_g', 'nsa_sersic_flux_r', 'nsa_sersic_flux_i', 'nsa_sersic_flux_z'],
        predict_catalog=predict_catalog,
        test_transform=transform
    )

    import lightning as L
    trainer = L.Trainer(max_epochs=-1)

    # Run predictions (returns list of tensors per batch), concatenate and convert
    preds_tensor = torch.cat(trainer.predict(finetuned_model, predict_datamodule), dim=0)
    preds_np = preds_tensor.detach().cpu().numpy()

    # Handle shapes: (N,), (N,1), or (N,k)
    if preds_np.ndim == 1:
        primary = preds_np
        extra = None
    elif preds_np.ndim == 2 and preds_np.shape[1] == 1:
        primary = preds_np.flatten()
        extra = None
    elif preds_np.ndim == 2 and preds_np.shape[1] > 1:
        # multiple outputs per sample (keep first as primary and store extras)
        primary = preds_np[:, 0]
        extra = preds_np[:, 1:]
    else:
        # unexpected shape — coerce to 1D primary
        primary = preds_np.flatten()
        extra = None

    # Ensure primary is numpy array
    primary = np.asarray(primary)

    # Inverse-transform predictions to raw units if scaler provided
    try:
        pred_raw = scaler.inverse_transform(primary.reshape(-1, 1)).flatten()
    except Exception:
        pred_raw = primary

    # Build a predictions DataFrame that includes id_str and both normalized and raw predicted label
    predictions_result = pd.DataFrame({
        'id_str': predict_catalog['id_str'].values,
        'label': primary,
        'label_raw': pred_raw
    })

    # If there are extra columns, add them as label_1, label_2, ... (keep them in normalized space)
    if extra is not None:
        for i in range(extra.shape[1]):
            predictions_result[f'label_{i+1}'] = extra[:, i]

    # Save predictions with id column and raw predictions so downstream merging works and is in original units
    # write only id_str and raw prediction to CSV to satisfy request
    predictions_result[['id_str', 'label_raw']].to_csv(predictions_path, index=False)

    return predictions_result

# Plot results
def plot_results(test_, predictions_result, scaler):
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

    # Merge test and predictions on id_str
    merged = test_.merge(predictions_result, on='id_str')

    # True values in original units
    y_true_raw = merged[f'{label}_raw'].values

    # Predictions are stored in normalized space; inverse transform to raw units
    y_pred_norm = merged['label'].values
    try:
        y_pred_un = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
    except Exception:
        # fallback if scaler not compatible
        y_pred_un = y_pred_norm

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_form(ax)

    plot_contours(ax, y_true_raw, y_pred_un)
    sns.scatterplot(x=y_true_raw, y=y_pred_un, ax=ax, alpha=0.7, s=25, color='steelblue', edgecolor='none')

    # 1:1 reference line
    ax.plot([y_true_raw.min()-0.1, y_true_raw.max()], [y_true_raw.min()-0.1, y_true_raw.max()],
            color='black', linestyle='--', linewidth=2)

    # labels and title (units are raw)
    ax.set_xlabel('True $V_m$ (raw)', fontsize=13)
    ax.set_ylabel('Predicted $V_m$ (raw)', fontsize=13)
    ax.set_title('Predicted vs True $V_m$ (raw)', fontsize=14)

    # add metrics legend using raw values
    add_metrics_legend(ax, y_true_raw, y_pred_un)

    plt.tight_layout()
    plt.show()

def write_diagnostics(predictions_result, test_, scaler, duration, best_checkpoint):
    # merge and compute metrics on raw units
    merged = test_.merge(predictions_result, on='id_str')
    y_true_raw = merged[f'{label}_raw'].values

    if 'label_raw' in merged.columns:
        y_pred_un = merged['label_raw'].values
    else:
        y_pred_norm = merged['label'].values
        try:
            y_pred_un = scaler.inverse_transform(y_pred_norm.reshape(-1, 1)).flatten()
        except Exception:
            y_pred_un = y_pred_norm

    mae = mean_absolute_error(y_true_raw, y_pred_un)
    r2 = r2_score(y_true_raw, y_pred_un)

    # collect metadata columns present in dataset
    metadata_cols = [c for c in test_.columns if c not in ['file_loc', 'id_str', label, f'{label}_raw']]

    diagnostics = {
        'MAE': mae,
        'R2': r2,
        'duration_seconds': duration,
        'best_checkpoint': best_checkpoint,
        'num_test_samples': len(merged),
        'metadata_columns': metadata_cols
    }

    diag_path = os.path.join(save_dir, 'diagnostics.txt')
    with open(diag_path, 'w') as f:
        for k, v in diagnostics.items():
            f.write(f"{k}: {v}\n")

    print(f"Wrote diagnostics to {diag_path}")

# Paths and parameters
path = '/home/nasserm/local/zoobot/examples/finetuning/tmp/metadata/'
label = 'Vm'  # km/s
id_str = 'plateifu'
batch_size = 32
num_workers = 64
save_dir = '/home/nasserm/local/zoobot/examples/finetuning/tmp/metadata/training'
predictions_path = '/home/nasserm/local/zoobot/examples/finetuning/tmp/metadata/training/preds/save4.csv'

# Data transformations
transform_cfg = default_view_config()
transform = get_galaxy_transform(transform_cfg)

if __name__ == "__main__":
    train_, test_, scaler = load_datasets()
    finetuned_model, trainer, duration, best_checkpoint = train_model(train_)
    predictions_result = make_predictions(finetuned_model, test_, scaler)
    plot_results(test_, predictions_result, scaler)
    save_and_plot_losses(trainer)
    write_diagnostics(predictions_result, test_, scaler, duration, best_checkpoint)
    plt.savefig("pred_vs_true_vm_4.jpg", dpi=300, bbox_inches='tight')