import os

import torch
from PIL import Image
from torchvision.transforms import transforms

from src.utils.models import load_model_state
from src.utils.visualization import load_and_visualize


def start_test(base_path='models'):
    # Define the directories for the test data
    images_dir = 'data/test/images'
    masks_dir = 'data/test/masks'

    # Get all files in the directory
    image_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
    mask_files = [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)]

    # Ensure we have the same number of images and masks
    if len(image_files) != len(mask_files):
        print("Number of images and masks do not match.")
        return

    # Define any transformations (if needed)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for model_file in os.listdir(base_path):
        if model_file.endswith('.pth'):
            parts = model_file.split('_')
            if len(parts) >= 3:
                model_name = '_'.join(parts[:2])  # Get the model name from the file name
            else:
                continue  # Skip this file if it doesn't match the expected format

            model_path = os.path.join(base_path, model_file)
            print(f"Testing model {model_name} from {model_path}")

            # Load the model
            model = load_model_state(model_name, model_path)

            # Move the model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # Get predictions
            predicted_masks = []
            with torch.no_grad():
                model.eval()  # Set the model to evaluation mode
                for img_path in image_files:
                    img = Image.open(img_path).convert("RGB")
                    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
                    output = model(img)['out']
                    predicted_mask = torch.sigmoid(output).cpu().detach()
                    predicted_masks.append(predicted_mask)

            # Visualize the results
            load_and_visualize(
                images_paths=image_files,
                masks_paths=mask_files,
                predicted_masks=predicted_masks,
                model_path=model_path
            )

