#############################Libraries################################
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
from roboflow import InferenceHTTPClient

######################################################################
######################################################################


# setup device: use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class index to coin type mapping and their respective values
class_index_to_coin_type = {
    0: ('10ct', 0.10),
    1: ('1ct', 0.01),
    2: ('1euro', 1.00),
    3: ('20ct', 0.20),
    4: ('2ct', 0.02),
    5: ('2euro', 2.00),
    6: ('50ct', 0.50),
    7: ('5ct', 0.05)
}

# function to draw bounding boxes on an image
def draw_bounding_boxes(image_path, detections):
    image = cv2.imread(image_path)
    for detection in detections:
        x, y, width, height = detection['x'], detection['y'], detection['width'], detection['height']
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        color = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# function to load the trained classification model
def load_trained_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load('../models/recognition_model_final.pth'))
    model = model.to(device)
    model.eval()
    return model

# function to predict the class of an image
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        return predicted_idx.item()

# perform inference with roboflow
def perform_inference(image_path):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="E3PZwYOkV9xtmkjnCeLy"
    )
    return CLIENT.infer(image_path, model_id="coin-counting-2/3")

# non-maximum suppression (nms)
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return boxes[keep].tolist(), scores[keep].tolist()

# main function to process an image, detect coins, and classify each coin
def process_and_classify(image_path, classification_model):
    result = perform_inference(image_path)
    detections = result.get('predictions', [])
    detected_coins = []
    total_value = 0.0

    original_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    boxes = [[d['x'] - d['width'] / 2, d['y'] - d['height'] / 2, d['x'] + d['width'] / 2, d['y'] + d['height'] / 2] for d in detections]
    scores = [d['confidence'] for d in detections]

    # apply non-maximum suppression (nms)
    nms_boxes, nms_scores = non_max_suppression(boxes, scores)

    for box in nms_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
        image_tensor = transform(cropped_image).unsqueeze(0).to(device)
        predicted_idx = predict_image(classification_model, image_tensor)
        coin_type, coin_value = class_index_to_coin_type[predicted_idx]
        detected_coins.append(coin_type)
        total_value += coin_value

    draw_bounding_boxes(image_path, [{'x': (box[0] + box[2]) / 2, 'y': (box[1] + box[3]) / 2, 'width': box[2] - box[0], 'height': box[3] - box[1]} for box in nms_boxes])
    return detected_coins, total_value

if __name__ == "__main__":
    classification_model = load_trained_model()

    #image_path = '../data/random/example13.jpg'
    #detected_coins, total_value = process_and_classify(image_path, classification_model)
    #print('Detected coins:', detected_coins)
    #print('Total value: €{:.2f}'.format(total_value))