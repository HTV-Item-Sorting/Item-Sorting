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
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        return class_names[predicted.item()], confidence[predicted.item()].item()


def main():
    # Path to your trained model
    model_path = './pythonProject1/waste_classification_model.pth'

    # Define your class names in the order they were during training
    class_names = ['battery', 'biodegradable', 'biohazard', 'electronic', 'glass', 'metal', 'paper', 'plastic']

    # Load the model
    model = load_model(model_path, len(class_names))

    # Open the default camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    window_name = 'Camera'
    cv2.namedWindow(window_name)

    # backSub = cv2.createBackgroundSubtractorMOG2()
    pos = [0 for _ in range(8)]
    item = "Empty"
    possibile = {
        'battery' : 0,
        'biodegradable' : 0,
        'biohazard' : 0,
        'electronic': 0,
        'glass': 0,
        'metal' : 0,
        'paper' : 0,
        'plastic' : 0
    }

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to grab frame")
                break

            frame = cv2.flip(frame, 1)

            # Process the frame for prediction
            image_tensor = process_image(frame)
            prediction, confidence = predict_image(model, image_tensor, class_names)

            # Apply background subtraction
            # fgMask = backSub.apply(frame)

            # Display frame count
            # cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
            # cv2.putText(frame, str(cam.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            # Display prediction
            if float(confidence) > 85:
                cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60),
                #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                item = prediction
                possibile[item] += confidence
            elif float(confidence) > 75:
                cv2.putText(frame, f"Prediction: {item}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"No Item Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



            # Display the captured frame
            # cv2.imshow("Filtered", fgMask)
            cv2.imshow(window_name, frame)

            # Check if window has been closed
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Press 'q' to exit the loop
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Release the capture object
        cam.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()