import os
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from torch import optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from src.dataset.solarpanel_dataset import SolarPanelDataset
from src.utils.loss_functions import get_loss_function
from src.utils.models import get_model


def binarize_output(outputs, threshold=0.1):
    outputs = torch.sigmoid(outputs)
    return (outputs > threshold).float()


def train_model(epochs, batch_size, lr, model_name, criterion_name):
    print(
        f"Training {model_name} with {criterion_name} loss function for {epochs} epochs with batch size {batch_size} and learning rate {lr}")

    train_images_dir = 'data/train/images'
    train_masks_dir = 'data/train/masks'
    val_images_dir = 'data/val/images'
    val_masks_dir = 'data/val/masks'

    # Get all files in the directory
    train_image_files = [os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir)]
    train_mask_files = [os.path.join(train_masks_dir, f) for f in os.listdir(train_masks_dir)]
    val_image_files = [os.path.join(val_images_dir, f) for f in os.listdir(val_images_dir)]
    val_mask_files = [os.path.join(val_masks_dir, f) for f in os.listdir(val_masks_dir)]

    # Check if the number of images and masks are the same
    assert len(train_image_files) == len(train_mask_files), 'Number of training images and masks are not the same'
    assert len(val_image_files) == len(val_mask_files), 'Number of validation images and masks are not the same'

    # Image transformations
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Mask transformations
    mask_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

    train_dataset = SolarPanelDataset(
        image_dir=train_images_dir,
        mask_dir=train_masks_dir,
        image_transform=image_transforms,
        mask_transform=mask_transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    val_dataset = SolarPanelDataset(
        image_dir=val_images_dir,
        mask_dir=val_masks_dir,
        image_transform=image_transforms,
        mask_transform=mask_transforms
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load a model
    model = get_model(model_name)

    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)
    criterion = get_loss_function(criterion_name)

    # Train model
    train_losses, val_losses, val_iou_scores, val_f1_scores = [], [], [], []
    for epoch in range(epochs):
        model.train()
        running_train_loss, running_val_loss = 0, 0
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)['out']
            train_loss = criterion(outputs, masks)
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        scheduler.step()
        train_losses.append(running_train_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {running_train_loss / len(train_dataloader)}")

        # Validation
        model.eval()
        running_val_iou, running_val_f1 = 0, 0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)['out']
                val_loss = criterion(outputs, masks)
                running_val_loss += val_loss.item()

                # Calculate IoU and F1 score using binarized outputs
                outputs_bin = binarize_output(outputs)
                masks_bin = binarize_output(masks)


                outputs_bin_flat = outputs_bin.view(-1)
                masks_bin_flat = masks_bin.view(-1)
                running_val_iou += jaccard_score(masks_bin_flat.cpu().numpy(), outputs_bin_flat.cpu().numpy(),
                                                 average='binary', zero_division=0)
                running_val_f1 += f1_score(masks_bin_flat.cpu().numpy(), outputs_bin_flat.cpu().numpy(),
                                           average='binary', zero_division=0)

            val_losses.append(running_val_loss / len(val_dataloader))
            val_iou_scores.append(running_val_iou / len(val_dataloader))
            val_f1_scores.append(running_val_f1 / len(val_dataloader))

            print(f"Epoch {epoch + 1}, "
                  f"Train Loss: {running_train_loss / len(train_dataloader)}, "
                  f"Validation Loss: {running_val_loss / len(val_dataloader)}, "
                  f"IoU: {running_val_iou / len(val_dataloader)}, "
                  f"F1 Score: {running_val_f1 / len(val_dataloader)}")

    torch.save(model.state_dict(), f'models/{model_name}_{criterion_name}_e{epochs}_bs{batch_size}_lr{lr}.pth')
    return train_losses, val_losses, val_iou_scores, val_f1_scores
