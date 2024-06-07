import os
import cv2

base_dir = '../CoinCounter/data/categorised_one_coin'
categories = ['1c', '2c','5c', '10c', '20c', '50c', '1e', '2e']  # Coin categories

# Target size of the images
img_size = (224, 224)

# Function to resize images and rename them
def preprocess_images():
    for category in categories:
        path = os.path.join(base_dir, category)
        if not os.path.exists(path):
            print(f"Directory {path} does not exist")
            continue
        print(f"Processing category: {category}")
        image_count = 1
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error reading image {img_path}")
                    continue
                resized_image = cv2.resize(image, img_size)
                new_img_name = f"{category}_train_{image_count}.jpg"
                new_img_path = os.path.join(path, new_img_name)
                cv2.imwrite(new_img_path, resized_image)
                print(f"Successfully resized and renamed image {img_path} to {new_img_path}")
                image_count += 1
            except Exception as e:
                print(f"Error processing image {img}: {e}")

# Function to check if images are correctly resized
def check_images():
    for category in categories:
        path = os.path.join(base_dir, category)
        if not os.path.exists(path):
            print(f"Directory {path} does not exist")
            continue
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error reading image {img_path}")
                continue
            if image.shape[:2] != img_size:
                print(f"Image {img_path} has incorrect size: {image.shape[:2]}")

preprocess_images()
check_images()