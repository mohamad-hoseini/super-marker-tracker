import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
import pose_helpers as pose_helpers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def parse_loss_params(loss_type_str):
    # Split the loss_type string by commas
    parts = [part.strip() for part in loss_type_str.split(',')]
    
    # The first part is the loss type (e.g., "focal")
    loss_type = parts[0]
    
    # Default values for alpha and gamma
    alpha = 0.25
    gamma = 1
    
    # Parse alpha and gamma if provided
    for part in parts[1:]:
        if 'alpha=' in part:
            alpha = float(part.split('=')[1])
        elif 'gamma=' in part:
            gamma = float(part.split('=')[1])
    
    return loss_type, alpha, gamma

def calculate_loss(hm_pred, locx_pred, locy_pred, lbl, xmesh, ymesh, n_factor, sigma, loss_type_str, epoch=None, n_epochs=None):
    loss_type, alpha, gamma = parse_loss_params(loss_type_str)
    lbl_mask = torch.isnan(lbl).sum(axis=-1)
    lbl[lbl_mask > 0] = 0
    lbl_nan = lbl_mask == 0
    lbl_nan = lbl_nan.to(device=device)
    lbl_batch = lbl

    y_true = (lbl_batch[:, :, 0]) / n_factor
    x_true = (lbl_batch[:, :, 1]) / n_factor

    locx = ymesh - x_true.unsqueeze(-1).unsqueeze(-1)
    locy = xmesh - y_true.unsqueeze(-1).unsqueeze(-1)

    hm_true = torch.exp(-(locx**2 + locy**2) / (2 * sigma**2))
    hm_true = (
        10
        * hm_true
        / (1e-3 + hm_true.sum(axis=(-2, -1)).unsqueeze(-1).unsqueeze(-1))
    ).to(dtype=torch.float32)

    mask = (locx**2 + locy**2) ** 0.5 <= sigma

    locx = locx / (2 * sigma)
    locy = locy / (2 * sigma)

    hm_true = hm_true[lbl_nan]
    y_true = y_true[lbl_nan]
    x_true = x_true[lbl_nan]
    locx = locx[lbl_nan]
    locy = locy[lbl_nan]
    mask = mask[lbl_nan]

    hm_pred = hm_pred[lbl_nan]
    locx_pred = locx_pred[lbl_nan]
    locy_pred = locy_pred[lbl_nan]

    if loss_type == "original":
        # Original loss
        hm_loss = ((hm_true - hm_pred).abs()).sum(axis=(-2, -1))
        loc_loss = 0.5 * (
            mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5
        ).sum(axis=(-2, -1))
        loss = hm_loss + loc_loss

    elif loss_type == "bce":
        # Binary cross-entropy for heatmaps
        bce_loss = torch.nn.functional.binary_cross_entropy(torch.sigmoid(hm_pred), hm_true)
        loc_loss = 0.5 * (
            mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5
        ).sum(axis=(-2, -1))
        loss = bce_loss + loc_loss

    elif loss_type == "focal":
        # Focal loss for heatmaps
        alpha = 0.25
        gamma = 1
        focal_loss = -alpha * (1 - hm_pred) ** gamma * hm_true * torch.log(torch.sigmoid(hm_pred) + 1e-7)
        focal_loss = focal_loss.mean()
        loc_loss = 0.5 * (
            mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5
        ).sum(axis=(-2, -1))
        loss = focal_loss + loc_loss

    elif loss_type == "dynamic":
        # Dynamic weighting for heatmap and location losses
        if epoch is None or n_epochs is None:
            raise ValueError("Epoch and n_epochs must be provided for dynamic loss.")
        weight_hm = 0.8 - (0.6 * epoch / n_epochs)
        weight_loc = 0.2 + (0.6 * epoch / n_epochs)
        hm_loss = ((hm_true - hm_pred).abs()).sum(axis=(-2, -1))
        loc_loss = 0.5 * (
            mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5
        ).sum(axis=(-2, -1))
        loss = weight_hm * hm_loss + weight_loc * loc_loss

    elif loss_type == "auto dynamic":
        # Auto Dynamic loss
        log_sigma_hm = torch.nn.Parameter(torch.tensor(0.0))  # log(σ²) for heatmap loss
        log_sigma_loc = torch.nn.Parameter(torch.tensor(0.0))  # log(σ²) for location loss

        # Compute heatmap and location loss
        hm_loss = ((hm_true - hm_pred).abs()).sum(axis=(-2, -1))
        loc_loss = 0.5 * (mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5).sum(axis=(-2, -1))

        # Convert log(σ²) to σ²
        sigma_hm_sq = torch.exp(log_sigma_hm)
        sigma_loc_sq = torch.exp(log_sigma_loc)

        # Compute dynamic loss
        loss = (
            (1 / (2 * sigma_hm_sq)) * hm_loss +
            (1 / (2 * sigma_loc_sq)) * loc_loss +
            0.5 * (log_sigma_hm + log_sigma_loc)
        )

    elif loss_type == "focal auto dynamic":
        # Auto Dynamic loss components
        log_sigma_focal = torch.nn.Parameter(torch.tensor(0.0))  # log(σ²) for focal loss
        log_sigma_loc = torch.nn.Parameter(torch.tensor(0.0))  # log(σ²) for location loss

        # Compute focal and location loss
        # Focal loss for heatmaps
        alpha = 0.25
        gamma = 1
        focal_loss = -alpha * (1 - hm_pred) ** gamma * hm_true * torch.log(torch.sigmoid(hm_pred) + 1e-7)
        focal_loss = focal_loss.mean()
        loc_loss = 0.5 * (mask * ((locx - locx_pred) ** 2 + (locy - locy_pred) ** 2) ** 0.5).sum(axis=(-2, -1))

        # Convert log(σ²) to σ²
        sigma_focal_sq = torch.exp(log_sigma_focal)
        sigma_loc_sq = torch.exp(log_sigma_loc)


        # Combine Auto Dynamic and Focal Loss components
        loss = (
            (1 / (2 * sigma_focal_sq)) * focal_loss +
            (1 / (2 * sigma_loc_sq)) * loc_loss +
            0.5 * (log_sigma_focal + log_sigma_loc) +
            focal_loss
        )

        
        
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    with torch.no_grad():
        Lx = 64
        hm_pred = hm_pred.reshape(hm_pred.shape[0], Lx * Lx)
        locx_pred = locx_pred.reshape(locx_pred.shape[0], Lx * Lx)
        locy_pred = locy_pred.reshape(locy_pred.shape[0], Lx * Lx)

        nn = hm_pred.shape[0]
        imax = torch.argmax(hm_pred, 1)

        x_pred = (
            ymesh.flatten()[imax] - (2 * sigma) * locx_pred[np.arange(nn), imax]
        )
        y_pred = (
            xmesh.flatten()[imax] - (2 * sigma) * locy_pred[np.arange(nn), imax]
        )
        
        
        # Calculate the expected "true" positions based on the mesh
        # Example: Use mesh grid centers or peaks from `hm_pred`
        x_true = xmesh.flatten()[imax]  # Hypothetical target from mesh
        y_true = ymesh.flatten()[imax]  # Hypothetical target from mesh
        
        y_err = (y_true - y_pred).abs()
        x_err = (x_true - x_pred).abs()
        
        accuracy = (y_err + x_err) / 2
        
    return loss.mean(), accuracy.mean()



from torch.optim.lr_scheduler import StepLR  # Import StepLR

def train(
    train_dataloader,
    net,
    n_epochs,
    learning_rate,
    weight_decay,
    ggmax=50,
    val_dataloader=None,
    save_checkpoint=False,
    checkpoint_path=None,
    checkpoint_filename=None,
    loss_type_str='original',
):
    loss_type, alpha, gamma = parse_loss_params(loss_type_str)
    n_factor = 2**4 // (2 ** net.n_upsamples)
    xmesh, ymesh = np.meshgrid(
        np.arange(train_dataloader.dataset.img_size[1] / n_factor),
        np.arange(train_dataloader.dataset.img_size[0] / n_factor),
    )
    ymesh = torch.from_numpy(ymesh).to(device)
    xmesh = torch.from_numpy(xmesh).to(device)

    sigma = 3 * 4 / n_factor

    # Set up optimizer and StepLR scheduler
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.6)  # Decays LR by 0.1 every 5 epochs

    min_test_loss = np.inf
    train_metrics = []
    val_metrics = []

    for epoch in tqdm(range(n_epochs)):
        pose_helpers.set_seed(epoch)

        # Training loop
        train_loss, train_accuracy, n_batches = 0, 0, 0
        for train_batch in train_dataloader:
            net.train()
            images = train_batch["image"].to(device, dtype=torch.float32)
            lbl = train_batch["keypoints"].to(device, dtype=torch.float32)

            hm_pred, locx_pred, locy_pred = net(images)
            loss, _ = calculate_loss(
                hm_pred, locx_pred, locy_pred, lbl, xmesh, ymesh, n_factor, sigma, loss_type, epoch, n_epochs,
            )

            train_loss += loss.item()
            # train_accuracy += accuracy.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_batches += 1

        train_loss /= n_batches

        # Validation loop
        if val_dataloader is not None:
            val_loss, val_accuracy, n_val_batches = 0, 0, 0
            with torch.no_grad():
                net.eval()
                for val_batch in val_dataloader:
                    images = val_batch["image"].to(device, dtype=torch.float32)
                    lbl = val_batch["keypoints"].to(device, dtype=torch.float32)

                    hm_pred, locx_pred, locy_pred = net(images, normalize=True)
                    loss, accuracy = calculate_loss(
                        hm_pred, locx_pred, locy_pred, lbl, xmesh, ymesh, n_factor, sigma, loss_type, epoch, n_epochs,
                    )
                    val_loss += loss.item()
                    val_accuracy += accuracy.item() 
                    n_val_batches += 1

            val_loss /= n_val_batches
            val_accuracy /= n_batches
            val_metrics.append((epoch, val_loss, val_accuracy))

        
        train_accuracy, n_train_batches = 0, 0
        with torch.no_grad():
            net.eval()
            for train_batch in train_dataloader:
                images = train_batch["image"].to(device, dtype=torch.float32)
                lbl = train_batch["keypoints"].to(device, dtype=torch.float32)

                hm_pred, locx_pred, locy_pred = net(images, normalize=True)
                loss, accuracy = calculate_loss(
                    hm_pred, locx_pred, locy_pred, lbl, xmesh, ymesh, n_factor, sigma, loss_type, epoch, n_epochs,
                )
                #train_loss += loss.item()
                train_accuracy += accuracy.item() 
                n_train_batches += 1

        train_accuracy /= n_batches
        train_metrics.append((epoch, train_loss, train_accuracy))


        # Update the learning rate using the scheduler
        scheduler.step()

        # Print epoch summary
        print(
            f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Accuracy {train_accuracy:.2f} "
            f"Validation Loss {val_loss  if val_dataloader else 'N/A':.4f}, "
            f"Validation Accuracy {val_accuracy  if val_dataloader else 'N/A':.2f}, "
            f"LR {scheduler.get_last_lr()[0]:.6f}"
        )

        # Save model checkpoint if required
        if save_checkpoint and val_dataloader and val_loss < min_test_loss:
            min_test_loss = val_loss
            if checkpoint_filename:
                savepath = os.path.join(checkpoint_path, checkpoint_filename)
            else:
                savepath = os.path.join(checkpoint_path, "checkpoint.pth")
            torch.save(net.state_dict(), savepath)
            print(f"Checkpoint saved at {savepath}")

    return net, train_metrics, val_metrics
