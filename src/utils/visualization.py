import cv2
import matplotlib.pyplot as plt

from src.utils.extract_filename import extract_filename


def load_and_visualize(images_paths, masks_paths, predicted_masks, model_path):
    # Check if the number of images, masks, and predicted masks is equal
    assert len(images_paths) == len(masks_paths) == len(predicted_masks), "The number of images, masks, and predicted masks must be the same."

    num_images = len(images_paths)
    plt.figure(figsize=(15, num_images * 5))

    for i, (image_path, mask_path, predicted_mask) in enumerate(zip(images_paths, masks_paths, predicted_masks)):
        # Load image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)  # 0 to load in grayscale

        # Convert predicted mask to numpy array
        predicted_mask = predicted_mask.squeeze().cpu().numpy()  # Assuming predicted_mask is a tensor

        # Display image
        plt.subplot(num_images, 3, 3 * i + 1)
        plt.imshow(image)
        plt.title(f'Original Image {i + 1}')
        plt.axis('off')

        # Display mask
        plt.subplot(num_images, 3, 3 * i + 2)
        plt.imshow(mask, cmap='gray')
        plt.title(f'Mask {i + 1}')
        plt.axis('off')

        # Display predicted mask
        plt.subplot(num_images, 3, 3 * i + 3)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title(f'Predicted Mask {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    img_name = extract_filename(model_path)
    plt.savefig(f'results/visualization/{img_name}.png')
    plt.show()