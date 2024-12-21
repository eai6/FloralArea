from ultralytics import YOLO

import argparse


def main(path):
    # Load a model
    model = YOLO(path)  # load a custom model

    # Validate the model
    #metrics = model.val()  # no arguments needed, dataset and settings remembered
    # Enable detailed plots (e.g., PR curve, F1-confidence curve)
    metrics = model.val(save_json=True, save_txt=True, plots=True)

    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--path", type=str, default="/Users/edwardamoah/Documents/GitHub/FloralArea/runs/segment/train7/weights/best.pt", help="Size of model to train")
    args = parser.parse_args()
    main(args.path)

