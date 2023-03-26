""" UNet training script for 2D microscopy SEM dataset
Mostly based on https://github.com/Project-MONAI/tutorials/blob/main/2d_segmentation/torch/unet_training_dict.py
but adapted to create patches from the input images and have a multi-class model output.
This script tries to reproduce most accurately this ivadomed config used in the default SEM model
for ADS: https://github.com/axondeepseg/default-SEM-model/blob/main/model_seg_rat_axon-myelin_sem/model_seg_rat_axon-myelin_sem.json
"""

import logging
import json
import sys
import tempfile
from glob import glob
from pathlib import Path

from tqdm import tqdm

import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import monai
from monai.data import create_test_image_2d, pad_list_data_collate, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandAffined,
    Rand2DElasticd,
    NormalizeIntensityd,
    ResizeWithPadOrCropd
)
from monai.visualize import plot_2d_or_3d_image


working_dir = Path.cwd()
monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
device = "cpu" if not torch.cuda.is_available() else "cuda:0"
if device != "cpu":
    torch.multiprocessing.set_start_method('spawn')

# load data from SEM dataset
data_path = working_dir / 'data_axondeepseg_sem/'
images = sorted(data_path.rglob("*_SEM.png"))
ax_labels = sorted(data_path.rglob("*_SEM_seg-axon-manual.png"))
my_labels = sorted(data_path.rglob("*_SEM_seg-myelin-manual.png"))
train, val, test = [], [], []

# manually selecting samples to reproduce default SEM model training config
for example in zip(images, ax_labels, my_labels):
    fname = str(example[0])
    if 'sub-rat6' in fname:
        test.append(example)
        print(f'Found test image from original config: {fname}')
    elif 'sub-rat1' in fname or 'sub-rat5' in fname:
        val.append(example)
        print(f"Found validation image from original config: {fname}")
    else:
        train.append(example)
print(f"Found {len(train)} samples for training, {len(val)} for validation and {len(test)} for testing.")

train_files = [{"im": str(img), "seg-ax": str(ax), "seg-my": str(my)} for (img, ax, my) in train]
val_files = [{"im": str(img), "seg-ax": str(ax), "seg-my": str(my)} for (img, ax, my) in val]
test_files = [{"im": str(img), "seg-ax": str(ax), "seg-my": str(my)} for (img, ax, my) in test]

# define transforms
train_transforms = Compose(
    [
        LoadImaged(keys=["im", "seg-ax", "seg-my"]),
        EnsureChannelFirstd(keys=["im", "seg-ax", "seg-my"]),
        NormalizeIntensityd(keys="im"),
        # affine and elastic transforms: adapted from ADS default-SEM-model
        # see https://github.com/axondeepseg/default-SEM-model
        # RandAffined(
            # keys=["im", "seg-ax", "seg-my"], 
            # prob=1.0, 
            # rotate_range=np.pi/64, 
            # scale_range=0.05,
            # translate_range=(10, 10),
            # padding_mode="zeros",
            # device=device
        # ),
        # Rand2DElasticd(
            # keys=["im", "seg-ax", "seg-my"],
            # prob=0.5,
            # spacing=(30, 30),
            # magnitude_range=(1, 2),
            # padding_mode="zeros",
            # device=device,
        # ),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["im", "seg-ax", "seg-my"]),
        EnsureChannelFirstd(keys=["im", "seg-ax", "seg-my"]),
        NormalizeIntensityd(keys="im")
    ]
)

# first, load the images
train_data = monai.data.Dataset(data=train_files, transform=train_transforms)
val_data = monai.data.Dataset(data=val_files, transform=val_transforms)

# note that we need a GridPatchDataset instead of a Dataset to stack the patches
# otherwise, the vanilla Dataset class does not support different image sizes
# (which is often the case for microscopy data)
patch_iterator = monai.data.PatchIterd(keys=["im", "seg-ax", "seg-my"], patch_size=(256, 256), mode='constant')
bs = 4
# need num_worker=0 in dataloader for GPU
nw = 0

# some checks
check_ds = monai.data.GridPatchDataset(data=train_data, patch_iter=patch_iterator)
check_loader = DataLoader(check_ds, batch_size=bs, num_workers=nw, collate_fn=list_data_collate)
check_data = monai.utils.misc.first(check_loader)
print(check_data[0]["im"].shape, check_data[0]["seg-ax"].shape, check_data[0]["seg-my"].shape)
loader_size = sum(1 for _ in check_loader)
print(f"Size of the training loader: {loader_size}")
# TODO: NEED RESAMPLING... SEM DATASET PX SIZES RANGE FROM 0.093 um to 0.18 um...........

# training data loader
train_ds = monai.data.GridPatchDataset(data=train_data, patch_iter=patch_iterator)
train_loader = DataLoader(train_ds, batch_size=bs, num_workers=nw, collate_fn=list_data_collate)

# validation data loader
val_ds = monai.data.GridPatchDataset(data=val_data, patch_iter=patch_iterator)
val_loader = DataLoader(val_ds, batch_size=bs, num_workers=nw, collate_fn=list_data_collate)

# not 100% sure about include_background=False
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
#TODO: check if softmax=true is appropriate
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_label = Compose([AsDiscrete()])

# define UNet
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = monai.networks.nets.UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=2,      # NOTE: in her thesis, Marie-Helene mentions out_channels=3
    channels=(64, 128, 256, 512),
    strides=(1, 1, 1),
    act="ReLU",
    dropout=0.2,
    # nn.BatchNorm2d momentum defaults to 0.1
    norm="BATCH",
    # order of conv operations in ivadomed is ReLU -> batch-norm -> dropout
    adn_ordering="AND"
).to(device)

# original config: 150 epochs but stopped at epoch 117 due to early stopping
num_epochs = 117
# TODO: HMM... MAYBE REMOVE SIGMOID FROM THE LOSS? DOESN'T SEEM TO BE APPLIED IN IVADOMED
loss_function = monai.losses.DiceLoss(include_background=False, sigmoid=True)
optimizer = torch.optim.Adam(params=model.parameters(), lr=5e-3)
# original config used CosineAnnealingLR; for more information on how to use it with 
# monai, see https://github.com/Project-MONAI/tutorials/blob/main/modules/learning_rate.ipynb
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=150, eta_min=1e-9)

# training loop
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
writer = SummaryWriter()
for epoch in range(num_epochs):
    print("-" * 10)
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    epoch_loss = 0
    iteration = 0

    with tqdm(total=loader_size) as pbar:
        for batch_data in train_loader:
            iteration += 1
            global_it = loader_size * epoch + iteration

            inputs = batch_data[0]["im"].to(device)
            mask_ax, mask_my = batch_data[0]["seg-ax"].to(device), batch_data[0]["seg-my"].to(device)
            # merge axon and myelin masks into a single label tensor of size (bs, 2, 256, 256)
            labels = torch.cat((mask_ax, mask_my), dim=1)
            plot_2d_or_3d_image(inputs, global_it, writer, index=0, tag="TRAIN-image")
            plot_2d_or_3d_image([labels[0][0]], global_it, writer, index=0, tag="TRAIN-label-ax")
            plot_2d_or_3d_image([labels[0][1]], global_it, writer, index=0, tag="TRAIN-label-my")

            optimizer.zero_grad()
            outputs = model(inputs)
            plot_2d_or_3d_image([outputs[0][0]], global_it, writer, index=0, tag="TRAIN-pred-ax")
            plot_2d_or_3d_image([outputs[0][1]], global_it, writer, index=0, tag="TRAIN-pred-my")
            # note: in ivadomed, a final activation function is applied at the end of the UNet decoder
            # see https://github.com/ivadomed/ivadomed/blob/e101ebea632683d67deab3c50dd6b372207de2a9/ivadomed/models.py#L462
            # this is not the case with the default monai UNet but the sigmoid is applied inside the loss function
            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()
            epoch_loss += loss.item()
            # print(f"\t{iteration}/{loader_size}: train loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), loader_size * epoch + iteration)
            pbar.update(1)

    # CosineAnnealingLR scheduler should be stepped at every epoch
    scheduler.step()
    writer.add_scalar("learning_rate", scheduler.get_last_lr()[-1], epoch+1)
    epoch_loss /= iteration
    epoch_loss_values.append(epoch_loss)
    print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")

    # validation 
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images = val_data[0]["im"].to(device)
                val_labels = (val_data[0]["seg-ax"].to(device), val_data[0]["seg-my"].to(device))
                val_labels = torch.cat(val_labels, dim=1)
                roi_size = (256, 256)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "best_metric_sem_model_dict.pth")
                print("Saved new best model.")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

            writer.add_scalar("val_mean_dice", metric, epoch+1)
            plot_2d_or_3d_image(val_images, epoch+1, writer, index=0, tag="image")
            plot_2d_or_3d_image([val_labels[0][0]], epoch+1, writer, index=0, tag="label-ax")
            plot_2d_or_3d_image([val_labels[0][1]], epoch+1, writer, index=0, tag="label-my")
            # TODO: HMM... MAYBE THIS BELOW IS WRONG; HOW ARE OUTPUT DIMENSIONS ARRANGED? (B,C,H,W)?
            plot_2d_or_3d_image([val_outputs[0][0]], epoch+1, writer, index=0, tag="pred-ax")
            plot_2d_or_3d_image([val_outputs[0][1]], epoch+1, writer, index=0, tag="pred-my")
    
print(f"Training complete. Best metric: {best_metric:.4f} at epoch {best_metric_epoch}")
writer.close()
