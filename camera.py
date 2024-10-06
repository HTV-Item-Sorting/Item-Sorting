import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
from ultralytics import YOLO


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

        # If the highest confidence is 'face', return the second highest
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
    # Path to your trained model
    model_path = './pythonProject1/waste_classification_model.pth'
    yolo = YOLO('yolov8s.pt')

    # Define your class names in the order they were during training
    class_names = ['battery', 'biodegradable', 'biohazard', 'electronic', 'face', 'glass', 'metal', 'paper', 'plastic']

    # Load the model
    model = load_model(model_path, len(class_names))

    # Open the default camera
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open camera")
        return

    # window_name = 'Camera'
    # cv2.namedWindow(window_name)

    item = "Empty"
    type_of_waste = "Empty"
    possible = {name: 0 for name in class_names if name != 'face'}

    # try:
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.flip(frame, 1)
        results = yolo.track(frame, conf=0.5)

        # Process the frame for prediction
        # image_tensor = process_image(frame)
        # prediction, confidence = predict_image(model, image_tensor, class_names)

        # Apply background subtraction
        # fgMask = backSub.apply(frame)

        # Display frame count
        # cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        # cv2.putText(frame, str(cam.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        # Colour
        colour = (0, 0, 0)

        # Display prediction
        for result in results:
            for box in result.boxes:
                # print(box.conf[0])
                # if box.conf[0] > 0.3:
                # get coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                origin = (x1, y1)

                # make rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                img = frame[y1:y2, x1:x2]
                image_tensor = process_image(img)
                prediction, confidence = predict_image(model, image_tensor, class_names)
                # Process the frame for prediction
                # cv2.imshow("cropped", img)
                if float(confidence) > 85:
                    cv2.putText(frame, f"Prediction: {prediction}", origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    # cv2.putText(img, f"Confidence: {confidence:.2f}%", origin,
                    #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    item = prediction
                    possible[item] = possible.get(item, 0) + confidence
                    type_of_waste = categorize_waste(prediction)
                elif float(confidence) > 50:
                    cv2.putText(frame, f"Prediction: {item}", origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    # cv2.putText(img, f"Confidence: {confidence:.2f}%", origin,
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"No Item Detected", origin,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
                    # cv2.putText(img, f"Confidence: {confidence:.2f}%", origin,
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Type: {type_of_waste}", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

        # Display the captured frame
        # cv2.imshow("Filtered", fgMask)
        # cv2.imshow(window_name, frame)
        frame_str = cv2.imencode('.jpg', frame)[1].tobytes()
        # print(frame_str)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_str + b'\r\n')

            # # Check if window has been closed
            # if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            #     break
            #
            # # Press 'q' to exit the loop
            # key = cv2.waitKey(1)
            # if key == 27 or key == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            #     break
    # finally:
    #     # Release the capture object
    #     cam.release()
    #     cv2.destroyAllWindows()
    #     cv2.waitKey(1)


# if __name__ == "__main__":
#     view()