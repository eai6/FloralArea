import torch
from ultralytics import YOLO
import argparse


def main(model_size, data, epochs, device, save):
    # Load a model
    model = YOLO(f'yolov8{model_size}-seg.pt')  # load a pretrained model (recommended for training)

    #model = YOLO('/Users/edwardamoah/Documents/GitHub/FloralArea/runs/segment/train9/weights/best.pt')  # load a custom model

    # Ensure MPS is available and set the device
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #model.to(device)

    # Train the model
    results = model.train(data=data, epochs=epochs, imgsz=640,save=save)
    #results = model.train(data=data, epochs=epochs, device=device, imgsz=640,save=save)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--model_size", type=str, default="m", help="Size of model to train")
    parser.add_argument("--data", type=str, default="/Users/edwardamoah/Documents/GitHub/FloralArea/datasets/FloralArea-11/data.yaml", help="Path to dataset yaml file")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train for")
    parser.add_argument("--device", type=str, default="mps", help="Device to train on")
    parser.add_argument("--save", type=bool, default=True, help="Save model")
    args = parser.parse_args()
    main(args.model_size, args.data, args.epochs, args.device, args.save)
