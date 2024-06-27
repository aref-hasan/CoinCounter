#############################Libraries#################################

import os
import cv2

######################################################################
######################################################################

# base directory where the images are stored
base_dir = '../data/categorised_one_coin'
# list of categories (folders) within the base directory
categories = ['1c', '2c', '5c', '10c', '20c', '50c', '1e', '2e']

# target size of the images (width, height)
img_size = (224, 224)

# function to resize images
def preprocess_images():
    for category in categories:
        path = os.path.join(base_dir, category)  # path to the category folder
        print(f"Processing category: {category}")
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)  # full path to the image
                image = cv2.imread(img_path)  # read the image
                if image is None:
                    print(f"Error reading image {img_path}")
                    continue
                resized_image = cv2.resize(image, img_size)  # resize the image
                cv2.imwrite(img_path, resized_image)  # save the resized image
                print(f"Successfully resized image {img_path} to size {img_size}")
            except Exception as e:
                print(f"Error processing image {img}: {e}")

# function to check if images are correctly resized
def check_images():
    for category in categories:
        path = os.path.join(base_dir, category)  # path to the category folder
        for img in os.listdir(path):
            img_path = os.path.join(path, img)  # full path to the image
            image = cv2.imread(img_path)  # read the image
            if image is None:
                print(f"Error reading image {img_path}")
                continue
            if image.shape[:2] != img_size:  # check if the image size is correct
                print(f"Image {img_path} has incorrect size: {image.shape[:2]}")

if __name__ == "__main__":
    preprocess_images()  # preprocess images: resize them to the target size
    check_images()  # check if the images have been correctly resized
