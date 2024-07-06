import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from inference_sdk import InferenceHTTPClient

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class index to coin type mapping
class_index_to_coin_type = {
    0: '10ct',
    1: '1ct',
    2: '1euro',
    3: '20ct',
    4: '2ct',
    5: '2euro',
    6: '50ct',
    7: '5ct'
}

# Coin value mapping
coin_value_mapping = {
    '1ct': 1,
    '2ct': 2,
    '5ct': 5,
    '10ct': 10,
    '20ct': 20,
    '50ct': 50,
    '1euro': 100,
    '2euro': 200
}

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_path, detections):
    # This function now does nothing.
    pass

# Function to load the trained classification model
def load_trained_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load('coin_recognition_model2.pth', map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Function to predict the class of an image
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        return predicted_idx.item()

# Perform inference with Roboflow
def perform_inference(image_path):
    CLIENT = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="E3PZwYOkV9xtmkjnCeLy"
    )
    return CLIENT.infer(image_path, model_id="coin-counting-2/3")

# Non-Maximum Suppression (NMS)
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

# Main function to process an image, detect coins, and classify each coin
def process_and_classify(image_path, classification_model):
    result = perform_inference(image_path)
    detections = result.get('predictions', [])
    detected_coins = []

    original_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    boxes = [[d['x'] - d['width'] / 2, d['y'] - d['height'] / 2, d['x'] + d['width'] / 2, d['y'] + d['height'] / 2] for d in detections]
    scores = [d['confidence'] for d in detections]

    nms_boxes, nms_scores = non_max_suppression(boxes, scores)

    for box in nms_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
        image_tensor = transform(cropped_image).unsqueeze(0).to(device)
        predicted_idx = predict_image(classification_model, image_tensor)
        predicted_coin_type = class_index_to_coin_type[predicted_idx]
        detected_coins.append(predicted_coin_type)

    return detected_coins

# Function to calculate the sum of coin values
def calculate_coin_sum(coin_list):
    return sum(coin_value_mapping[coin] for coin in coin_list)

# Function to evaluate the model on a list of images
def evaluate_model(ground_truth_data, classification_model):
    predictions = []
    true_sums = ground_truth_data['ground_truth_value'].tolist()
    predicted_sums = []

    for index, row in ground_truth_data.iterrows():
        image_path = row['image_path']
        detected_coins = process_and_classify(image_path, classification_model)
        predicted_sum = calculate_coin_sum(detected_coins)
        predictions.append(predicted_sum)
        predicted_sums.append(predicted_sum)
    
    ground_truth_data['predicted_sum'] = predicted_sums  # Adding predictions as a new column
    return predictions, true_sums, ground_truth_data  # Returning the updated DataFrame

# Calculate precision, recall, F1-score, and accuracy
def calculate_metrics(predictions, true_sums):
    pred_categories = [round(pred * 100) for pred in predictions]
    true_categories = [round(true * 100) for true in true_sums]
    
    precision = precision_score(true_categories, pred_categories, average='weighted', zero_division=1)
    recall = recall_score(true_categories, pred_categories, average='weighted', zero_division=1)
    f1 = f1_score(true_categories, pred_categories, average='weighted', zero_division=1)
    accuracy = accuracy_score(true_categories, pred_categories)
    
    return precision, recall, f1, accuracy

# Load the ground truth data
ground_truth_data = pd.read_csv('image_data_plus_ground_truth.csv', delimiter=';')

# Load the trained classification model
classification_model = load_trained_model()

# Evaluate the model and get the updated DataFrame
predictions, true_sums, updated_data = evaluate_model(ground_truth_data, classification_model)

# Calculate metrics
precision, recall, f1, accuracy = calculate_metrics(predictions, true_sums)

# Print the results
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Save the updated DataFrame with predictions to a new CSV file
updated_data.to_csv('image_data_with_predictions.csv', index=False)
