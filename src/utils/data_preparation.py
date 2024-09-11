import os
import shutil

from PIL import Image
from sklearn.model_selection import train_test_split


def copy_files(files, src_dir, dest_dir):
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

def prepare_images():
    # Path to the directory with raw images and masks
    raw_images_path = 'data/raw/images'
    raw_masks_path = 'data/raw/masks'
    # Path to the directory with prepared train/val/test sets
    train_images_path = 'data/train/images'
    train_masks_path = 'data/train/masks'
    val_images_path = 'data/val/images'
    val_masks_path = 'data/val/masks'
    test_images_path = 'data/test/images'
    test_masks_path = 'data/test/masks'

    # Create directories if they don't exist
    for path in [train_images_path, train_masks_path, val_images_path, val_masks_path, test_images_path,
                 test_masks_path]:
        os.makedirs(path, exist_ok=True)

    # Get all images and masks
    images = [f for f in os.listdir(raw_images_path) if os.path.isfile(os.path.join(raw_images_path, f))]
    masks = [f for f in os.listdir(raw_masks_path) if os.path.isfile(os.path.join(raw_masks_path, f))]

    # Split images and masks into train, validation and test sets
    train_val_images, test_images = train_test_split(images, test_size=0.1, random_state=42)
    train_images, val_images = train_test_split(train_val_images, test_size=0.222, random_state=42)  # 0.222 * 0.9 ≈ 0.2
    train_val_masks, test_masks = train_test_split(masks, test_size=0.1, random_state=42)
    train_masks, val_masks = train_test_split(train_val_masks, test_size=0.222, random_state=42)

    # Copy images and masks to appropriate directories
    copy_files(train_images, raw_images_path, train_images_path)
    copy_files(train_masks, raw_masks_path, train_masks_path)
    copy_files(val_images, raw_images_path, val_images_path)
    copy_files(val_masks, raw_masks_path, val_masks_path)
    copy_files(test_images, raw_images_path, test_images_path)
    copy_files(test_masks, raw_masks_path, test_masks_path)


def prepare_data():
    prepare_images()

def process_images(source_dir, target_dir, tile_size=(256, 256)):
    # Ako ciljni direktorij ne postoji, stvori ga
    os.makedirs(target_dir, exist_ok=True)

    # Prolazak kroz sve datoteke u izvornom direktoriju
    for filename in os.listdir(source_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg', 'bmp')):  # Provjera da li je datoteka slika
            file_path = os.path.join(source_dir, filename)
            image = Image.open(file_path).convert("RGB")  # Učitavanje i konverzija u RGB
            img_width, img_height = image.size

            # Podijeli sliku na segmente 256x256
            for i in range(0, img_width, tile_size[0]):
                for j in range(0, img_height, tile_size[1]):
                    # Izračunavanje pravokutnika za segment
                    box = (i, j, i + tile_size[0], j + tile_size[1])
                    cropped_img = image.crop(box)

                    # Ako segment nije dovoljno velik, preskoči ga
                    if cropped_img.size[0] != tile_size[0] or cropped_img.size[1] != tile_size[1]:
                        continue

                    # Generiranje imena datoteke za segment
                    segment_filename = f'{filename[:-4]}_{i}_{j}.png'
                    segment_path = os.path.join(target_dir, segment_filename)

                    # Spremanje segmenta
                    cropped_img.save(segment_path)
