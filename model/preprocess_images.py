import os
import cv2

base_dir = '../data/categorised_one_coin'
categories = ['1c', '2c', '5c' , '10c', '20c', '50c', '1e', '2e']

# Target size of the images
img_size = (224, 224)

# Function to resize images
def preprocess_images():
    for category in categories:
        path = os.path.join(base_dir, category)
        print(f"Processing category: {category}")
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Error reading image {img_path}")
                    continue
                resized_image = cv2.resize(image, img_size)
                cv2.imwrite(img_path, resized_image)
                print(f"Successfully resized image {img_path} to size {img_size}")
            except Exception as e:
                print(f"Error processing image {img}: {e}")

# Function to check if images are correctly resized
def check_images():
    for category in categories:
        path = os.path.join(base_dir, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error reading image {img_path}")
                continue
            if image.shape[:2] != img_size:
                print(f"Image {img_path} has incorrect size: {image.shape[:2]}")

if __name__ == "__main__":
    preprocess_images()
    check_images()
