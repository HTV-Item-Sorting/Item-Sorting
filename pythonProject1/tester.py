import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models


def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def classify_image(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
    return class_names[predicted.item()], probability[predicted.item()].item()


def display_result(image_path, class_name, probability):
    image = Image.open(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {class_name}\nProbability: {probability:.2f}%")
    plt.show()


def main(image_path, model_path):
    # Define your class names here
    class_names = ['biodegradable', 'cardboard', 'glass', 'metal', 'paper', 'plastic']

    # Load the model
    model = load_model(model_path, len(class_names))

    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Classify the image
    predicted_class, probability = classify_image(model, image_tensor, class_names)

    # Display the result
    print(f"Predicted class: {predicted_class}")
    print(f"Probability: {probability:.2f}%")

    # Show the image with the prediction
    display_result(image_path, predicted_class, probability)


if __name__ == "__main__":
    model_path = "waste_classifier_model.pth"  # Path to your trained model
    image_path = "can.jpg"  # Replace with the path to the image you want to classify
    main(image_path, model_path)