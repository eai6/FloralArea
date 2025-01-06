import numpy as np
import gradio as gr

from utils import helpers
from cv import yolov8
from cv import img_processing as ip
from cv import runTiles

from PIL import Image


# initialize the model
yolo = yolov8.yolov8('/Users/edwardamoah/Documents/GitHub/FloralArea/models/flower_model.pt', '/Users/edwardamoah/Documents/GitHub/FloralArea/models/reference_object_model.pt')

reference_area = 9.0 # area of the reference object in inches  

def estimateArea(file_location:str, component:str, threshold:float, num_tiles) -> float:
    '''
    Estimate the area of either flower or leaf

    - Input:
    file_location: str: path to the image file
    component: str: component to estimate area for
    threshold: float: threshold for the model

    - Output:
    float: area of the component
    '''
    # run inference on the image
    flower_results = yolo.runInference(file_location, component, threshold)
    # get the mask
    flower_mask = yolo.getMask(flower_results)
    # get the area of the mask
    component_area = ip.getPixelArea(flower_mask)

    # # get area for the reference object
    reference_results = yolo.runInference(file_location, 'reference_object', threshold)
    reference_mask = yolo.getMask(reference_results)
    reference_pixel_area = ip.getPixelArea(reference_mask)
    if reference_pixel_area == 0:
       #raise Exception("Reference object not found in the image")
       reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, 3)

    #component_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles, component)
    #reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles+1, 'reference_object')

    area = (component_area/reference_pixel_area) * reference_area
   
    return area , flower_results[0].plot(conf=False, labels=True, boxes=True, masks=True)

def getArea(input_img):
    
    '''
    Estimate the area of either flower or leaf
    Takes in an image file and returns the area of the component with a  threshold
    '''
    try:
        filename = input_img
        file_location = filename #f"api/data/input/{filename}"
        
        # save audio file temporally
        #helpers.saveUploadfile(file_location, uploaded_file)

        # apply preprocessing to the image
        ip.apply_preprocessing(file_location)

        #
        #yolo.savePlot(file_location, component, threshold)

        # run the model on the image
        area, mask = estimateArea(file_location, "flower", 0.5, 2)

        #helpers.removefile(file_location)

        return area, mask
    except Exception as e:
        print(e)
    

#demo = gr.Interface(getArea, gr.Image(type='filepath'), "textbox")

with gr.Blocks() as demo:
    gr.Markdown("# FloralArea Demo")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type='filepath', label='Upload Flowering Plant')
            process_button = gr.Button("Process")

        with gr.Column():
            text_output = gr.Textbox(label="Area (cm^2)")
            image_output = gr.Image(label="Flower Mask")

    process_button.click(getArea, inputs=image_input, outputs=[text_output, image_output])

if __name__ == "__main__":
    demo.launch()