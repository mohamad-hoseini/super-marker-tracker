import os
from glob import glob

import numpy as np
import torch
import h5py

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_metrics(train_metrics, val_metrics, loss_type, output_dir):
    """
    Plots training and validation accuracy and loss metrics per epoch and saves the plots.

    Parameters:
    - train_metrics: List of tuples [(epoch, loss, accuracy)] for training.
    - val_metrics: List of tuples [(epoch, loss, accuracy)] for validation.
    - loss_type: String indicating the type of loss (e.g., 'focal', 'cross_entropy').
    - output_dir: Directory path to save the plots.
    """
    # Extract values for plotting
    epochs_train = [metric[0] for metric in train_metrics]  # Train epochs
    train_losses = [metric[1] for metric in train_metrics]  # Train losses
    train_accuracies = [metric[2] for metric in train_metrics]  # Train accuracies

    epochs_val = [metric[0] for metric in val_metrics]  # Validation epochs
    val_losses = [metric[1] for metric in val_metrics]  # Validation losses
    val_accuracies = [metric[2] for metric in val_metrics]  # Validation accuracies

    # Plot 1: Accuracy vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_train, train_accuracies, label="Train Accuracy", marker="o")
    plt.plot(epochs_val, val_accuracies, label="Validation Accuracy", marker="o", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.title(f"Model Accuracy per Epoch ({loss_type.capitalize()} Loss)")
    plt.legend()

    # Set ticks inside and custom x-axis ticks every 5 epochs
    plt.tick_params(axis='both', direction='in')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Save plot to file
    accuracy_plot_path = f"{output_dir}/{loss_type}_accuracy_plot.png"
    plt.savefig(accuracy_plot_path)
    plt.close()

    # Plot 2: Loss vs Epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_train, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs_val, val_losses, label="Validation Loss", marker="o", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.title(f"Model Loss per Epoch ({loss_type.capitalize()} Loss)")
    plt.legend()

    # Set ticks inside and custom x-axis ticks every 5 epochs
    plt.tick_params(axis='both', direction='in')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(5))

    # Save plot to file
    loss_plot_path = f"{output_dir}/{loss_type}_loss_plot.png"
    plt.savefig(loss_plot_path)
    plt.close()

    # Save train and validation metrics to a text file
    metrics_file_path = f"{output_dir}/{loss_type}_metrics.txt"
    with open(metrics_file_path, 'w') as f:
        f.write("Epoch\tTrain Loss\tTrain Accuracy\tValidation Loss\tValidation Accuracy\n")
        for epoch_train, loss_train, acc_train, epoch_val, loss_val, acc_val in zip(epochs_train, train_losses, train_accuracies, epochs_val, val_losses, val_accuracies):
            f.write(f"{epoch_train}\t{loss_train:.4f}\t{acc_train:.4f}\t{loss_val:.4f}\t{acc_val:.4f}\n")

    print(f"Accuracy and Loss plots saved successfully:\n- {accuracy_plot_path}\n- {loss_plot_path}")
    print(f"Metrics saved to text file: {metrics_file_path}")




class KeypointsLoaders:
    def __init__(self, datadir, scorer="All", bodyparts=None):
        self.datadir = datadir
        self.scorer = scorer
        self.bodyparts = bodyparts or ["eye(back)", "eye(bottom)", "nose(tip)", "whisker(I)", "mouth", "paw"]  # Example bodyparts

    def load_keypoints_h5(self):
        """
        Load landmarks/keypoints from all subdirectories containing .h5 files using h5py.
        Returns
        -------
        landmarks : torch.Tensor
            Tensor containing keypoints for all frames, of shape (total_frames, 15, 2).
        """
        # Find all .h5 files in all subdirectories
        annotation_files = sorted(
            glob(os.path.join(self.datadir, "**", f"CollectedData_{self.scorer}.h5"), recursive=True)
        )
        if len(annotation_files) == 0:
            raise ValueError("No .h5 files found in the directory or subdirectories.")

        # Initialize list for all landmarks
        all_landmarks = []

        for f in annotation_files:
            with h5py.File(f, 'r') as h5_file:
                
                keypoints_data = h5_file['df/block0_values'][:]
                

                # Reshape data (num_frames, 15, 2)
                num_frames, num_coords = keypoints_data.shape
                num_bodyparts = 15  # Example number of bodyparts
                if num_coords != num_bodyparts * 2:
                    raise ValueError(f"Unexpected number of coordinates: {num_coords}. Expected {num_bodyparts * 2}.")
                reshaped_data = keypoints_data.reshape(num_frames, num_bodyparts, 2)

                # Apply label fixes (optional)
                reshaped_data = self.fix_labels(reshaped_data)

                # Add reshaped data to list
                all_landmarks.append(reshaped_data)

        # Combine all landmarks into one array
        all_landmarks = np.concatenate(all_landmarks, axis=0)

        # Convert to tensor
        return torch.tensor(all_landmarks, dtype=torch.float32)

    def fix_labels(self, landmarks_data):
        """
        Placeholder function for renaming bodyparts.
        Here, you can apply similar logic as in fix_labels function.
        """
        for i, landmark_set in enumerate(landmarks_data):
            for j, bodypart in enumerate(self.bodyparts):
                pass
        return landmarks_data
    


    
def plot_keypoints_with_images(pred_keypoints, keypoints, images_original, output_file="keypoints_plot.png"):
    """
    Plots images with overlaid keypoints and saves the result to a file.

    Parameters:
    - pred_keypoints: Predicted keypoints, a numpy array of shape (N, num_keypoints, 2).
    - keypoints: Ground-truth keypoints, a numpy array of shape (N, num_keypoints, 2).
    - images_original: Original images, a numpy array of shape (N, H, W).
    - output_file: Path to save the output plot (default: "keypoints_plot.png").
    """
    num_images = min(25, keypoints.shape[0])  # Limit to a 5x5 grid (max 25 images)
    plt.figure(figsize=(20, 20))
    
    for ind in range(num_images):
        plt.subplot(5, 5, ind + 1)
        
        # Plot the image
        plt.imshow(images_original[ind], cmap='gray')  # Display grayscale image
        
        # Overlay the ground-truth keypoints
        plt.scatter(keypoints[ind, :, 0], keypoints[ind, :, 1], c='red', s=10, label='Ground-truth')
        
        # Overlay the predicted keypoints
        plt.scatter(pred_keypoints[ind, :, 0], pred_keypoints[ind, :, 1], c='green', s=10, label='Predicted')
        
        # Add title and remove axes for better visualization
        plt.title(f"Image {ind + 1}")
        plt.axis('off')
        
        # Add legend only for the first image
        if ind == 0:
            plt.legend()
    
    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()
