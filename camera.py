import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np

def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        confidences = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        sorted_indices = torch.argsort(confidences, descending=True)

        if class_names[sorted_indices[0]] == 'face':
            prediction = class_names[sorted_indices[1]]
            confidence = confidences[sorted_indices[1]].item()
        else:
            prediction = class_names[sorted_indices[0]]
            confidence = confidences[sorted_indices[0]].item()

        return prediction, confidence

def categorize_waste(prediction):
    if prediction in ['paper', 'plastic', 'metal', 'glass']:
        return "Recycle"
    elif prediction in ['biohazard', 'electronic', 'battery']:
        return "Hazard"
    elif prediction == 'biodegradable':
        return "Landfill"
    else:
        return "Unknown"



def view():
    model_path = './pythonProject1/waste_classification_model.pth'
    class_names = ['battery', 'biodegradable', 'biohazard', 'electronic', 'face', 'glass', 'metal', 'paper', 'plastic']

    model = load_model(model_path, len(class_names))

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    item = "Empty"
    type_of_waste = "Empty"
    possible = {name: 0 for name in class_names if name != 'face'}

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Full HD width
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Full HD height

    # Attempt to set higher quality
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
    cam.set(cv2.CAP_PROP_FOCUS, 0)  # Set focus to 0 (may trigger autofocus on some cameras)
    cam.set(cv2.CAP_PROP_BRIGHTNESS, 128)  # Set brightness (adjust value as needed)
    cam.set(cv2.CAP_PROP_CONTRAST, 128)  # Set contrast (adjust value as needed)
    cam.set(cv2.CAP_PROP_SATURATION, 128)  # Set saturation (adjust value as needed)

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)

        # Resize the frame
        frame = cv2.resize(frame, (1100, 800))  # You can adjust this size

        image_tensor = process_image(frame)
        prediction, confidence = predict_image(model, image_tensor, class_names)
        if float(confidence) > 75:
            cv2.putText(frame, f"Material: {prediction}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            item = prediction
            possible[item] = possible.get(item, 0) + confidence
            type_of_waste = categorize_waste(prediction)
        elif float(confidence) > 45:
            cv2.putText(frame, f"Material: {item}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, f"Not Detecting Item", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            type_of_waste = "Unknown"

        cv2.putText(frame, f"Type of Waste: {type_of_waste}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.putText(frame, f"Confidence: {confidence}", (10, 120),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the resized frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cam.release()
