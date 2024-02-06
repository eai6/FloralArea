from ultralytics import YOLO

import argparse


def main(path):
    # Load a model
    model = YOLO(path)  # load a custom model

    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--path", type=str, default="/Users/edwardamoah/Documents/GitHub/FloralArea/runs/segment/train3/weights/best.pt", help="Size of model to train")
    args = parser.parse_args()
    main(args.path)

