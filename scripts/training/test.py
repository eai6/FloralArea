from ultralytics import YOLO
from PIL import Image
import argparse
import os


# Load a model
model = YOLO('/Users/edwardamoah/Documents/GitHub/FloralArea/runs/segment/train7/weights/best.pt')  # load a custom model

def main(folder, save_folder, conf):
    images = os.listdir(folder) # load images from a folder
    images_paths = [os.path.join(folder, image) for image in images] # get the full path of each image

    # create a folder inside the save folder to save the masks
    if not os.path.exists(os.path.join(save_folder,str(conf))):
        os.makedirs(os.path.join(save_folder,str(conf)))

    # run the model on each image
    for image in images_paths:
        print(f'Processing {image}')
        #results = model(image) # predict on an image
        results = model.predict(image,  conf=conf, classes=[2])
        
        # generate and save prediction mask
        for r in results:
            im_array = r.plot(labels=False, boxes=True)  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])
            # extract image name
            image = image.split("/")[-1].split(".")[0]
            im.save(f'{save_folder}/{str(conf)}/{image}_mask.jpg')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model")
    parser.add_argument("--input", type=str, default="/Users/edwardamoah/Downloads/Floral Area Photos/FA photos", help="Path to folder containing images")
    parser.add_argument("--output", type=str, default="/Users/edwardamoah/Documents/GitHub/FloralArea/output/experiment3_masks", help="Path to folder to save masks")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    args = parser.parse_args()
    main(args.input, args.output, args.conf)