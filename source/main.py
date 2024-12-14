import os
from glob import glob

from dataset import FacemapDataset
import torch
from train import get_prediction

from train import train
from models import FaceMapNet
from utils import plot_metrics, plot_keypoints_with_images, KeypointsLoaders


# Assuming train_metrics and val_metrics are available
# train_metrics = [(epoch, loss, accuracy), ...]
# val_metrics = [(epoch, loss, accuracy), ...]

output_directory = "/project/branicio_73/yazdanim/super-marker-tracker/source/log"
loss_type_str = "focal, alpha=0.25, gamma=1.5"  # This will dynamically change the loss type
save_checkpoint = True
checkpoint_path = r"/project/branicio_73/yazdanim/super-marker-tracker/source/log"
checkpoint_filename = f"{loss_type_str}_loss_model.pth"  # Use loss_type in the filename

# Example usage
datadir_train = r"/project/branicio_73/yazdanim/super-marker-tracker/pose_estimation/train"
datadir_eval = r"/project/branicio_73/yazdanim/super-marker-tracker/pose_estimation/eval"
scorer = "All"
loader_train = KeypointsLoaders(datadir_train, scorer)
loader_eval = KeypointsLoaders(datadir_eval, scorer)
landmarks_train = loader_train.load_keypoints_h5()
landmarks_eval = loader_eval.load_keypoints_h5()

# Ensure keypoints data is a NumPy array
landmarks_train_numpy = landmarks_train.numpy()  # Convert PyTorch tensor to NumPy array
landmarks_eval_numpy = landmarks_eval.numpy()  # Convert PyTorch tensor to NumPy array

# Creating the dataset with corrected data type
facemap_dataset_train = FacemapDataset(datadir=datadir_train, keypoints_data=landmarks_train_numpy)
facemap_dataset_eval = FacemapDataset(datadir=datadir_eval, keypoints_data=landmarks_eval_numpy)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_ch = 1
output_ch = 15

channels = [32, 64, 128, 128, 200]

net = FaceMapNet(
    img_ch,
    output_ch,
    None,
    channels,
    device='cuda',
    kernel=3,
    shape=(256, 256),
    num_upsamples=2,
).to(device)

n_epochs = 20
learning_rate = 1e-3
weight_decay = 1e-6
batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    facemap_dataset_train, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    facemap_dataset_eval, batch_size=batch_size, shuffle=False
)

net, train_metrics, val_metrics = train(
    train_dataloader,
    net,
    n_epochs,
    learning_rate,
    weight_decay,
    50,
    val_dataloader,
    save_checkpoint=True,
    checkpoint_path=checkpoint_path,
    checkpoint_filename=checkpoint_filename,
    loss_type_str=loss_type_str,  # Pass loss_type as an argument
)

# Test predictions
pred_keypoints, keypoints, images_original = get_prediction(net, val_dataloader)

# Plot keypoints and save to a file with dynamic name based on loss_type
plot_keypoints_with_images(
    pred_keypoints, keypoints, images_original, 
    output_file=f"/project/branicio_73/yazdanim/super-marker-tracker/source/log/{loss_type_str}_plot.png"
)

# Plot metrics and save to output directory with dynamic loss type in the name
plot_metrics(train_metrics, val_metrics, loss_type_str, output_directory)
