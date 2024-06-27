import os
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import numpy as np
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, send_file
from inference_sdk import InferenceHTTPClient
import socket
import qrcode
from io import BytesIO
import base64  

app = Flask(__name__)
UPLOAD_FOLDER = 'app/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class index to coin type mapping and their respective values
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


# Function to draw bounding boxes on an image
def draw_bounding_boxes(image_path, output_path, detections):
    image = cv2.imread(image_path)
    for detection in detections:
        x, y, width, height = detection['x'], detection['y'], detection['width'], detection['height']
        start_point = (int(x - width / 2), int(y - height / 2))
        end_point = (int(x + width / 2), int(y + height / 2))
        color = (0, 255, 0)
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imwrite(output_path, image)
    
# Function to load the trained classification model
def load_trained_model():
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8)
    model.load_state_dict(torch.load('../CoinCounter/model/model_results/recognition_model_final.pth'))
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
    total_value = 0.0

    original_image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    boxes = [[d['x'] - d['width'] / 2, d['y'] - d['height'] / 2, d['x'] + d['width'] / 2, d['y'] + d['height'] / 2] for d in detections]
    scores = [d['confidence'] for d in detections]

    # apply Non-Maximum Suppression (NMS)
    nms_boxes, nms_scores = non_max_suppression(boxes, scores)

    for box in nms_boxes:
        xmin, ymin, xmax, ymax = map(int, box)
        cropped_image = original_image.crop((xmin, ymin, xmax, ymax))
        image_tensor = transform(cropped_image).unsqueeze(0).to(device)
        predicted_idx = predict_image(classification_model, image_tensor)
        coin_type, coin_value = class_index_to_coin_type[predicted_idx]
        detected_coins.append(coin_type)
        total_value += coin_value

    result_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + os.path.basename(image_path))
    draw_bounding_boxes(image_path, result_image_path, [{'x': (box[0] + box[2]) / 2, 'y': (box[1] + box[3]) / 2, 'width': box[2] - box[0], 'height': box[3] - box[1]} for box in nms_boxes])
    return detected_coins, total_value, result_image_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/take-photo')
def take_photo():
    return render_template('take_photo.html')

@app.route('/scan-qr-code')
def scan_qr_code():
    return render_template('scan_qr_code.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('file_upload.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('file_upload.html', message='No selected file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            classification_model = load_trained_model()
            detected_coins, total_value, result_image_path = process_and_classify(file_path, classification_model)
            return render_template('result.html', original_image=file.filename, result_image=os.path.basename(result_image_path), detected_coins=detected_coins, total_value=total_value)
    return render_template('upload.html')


@app.route('/phone-page', methods=['GET', 'POST'])
def phone_page():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('phone_page.html', message='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('phone_page.html', message='No selected file')
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            classification_model = load_trained_model()
            detected_coins, total_value, result_image_path = process_and_classify(file_path, classification_model)
            return render_template('result.html', original_image=file.filename, result_image=os.path.basename(result_image_path), detected_coins=detected_coins, total_value=total_value)
    return render_template('phone_page.html')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    image_data = request.form['imageData']
    if image_data:
        # Decode base64 image
        image_data = image_data.split(",")[1]
        image_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_data))
        
        # Save the image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.png')
        image.save(file_path)
        
        classification_model = load_trained_model()
        detected_coins, total_value, result_image_path = process_and_classify(file_path, classification_model)
        return render_template('result.html', original_image='captured_image.png', result_image=os.path.basename(result_image_path), detected_coins=detected_coins, total_value=total_value)
    return 'No image data', 400

@app.route('/qrcode')
def qrcode_page():
    local_ip = get_local_ip()
    port = 5000  # Change this to the actual port if different
    url = f"http://{local_ip}:{port}/phone-page"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)

    img = qr.make_image(fill='black', back_color='white')

    buffer = BytesIO()
    img.save(buffer, 'PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

# Neue Route für die Datei-Upload-Seite
@app.route('/file-upload')
def file_upload_page():
    return render_template('file_upload.html')

# Bestehende Routen bleiben gleich...



# Function to get local IP address
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

if __name__ == "__main__":
    port = 5000
    try:
        app.run(debug=True, host='0.0.0.0', port=port)
    except OSError:
        port = 5001
        app.run(debug=True, host='0.0.0.0', port=port)