from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from api.utils import helpers
from api.cv import yolov8
from api.cv import img_processing as ip
from api.cv import runTiles

app = FastAPI()


# initialize the model
yolo = yolov8.yolov8('/Users/edwardamoah/Documents/GitHub/FloralArea/runs/segment/train7/weights/best.pt', '/Users/edwardamoah/Documents/GitHub/FloralArea/runs/segment/train9/weights/best.pt')

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
    results = yolo.runInference(file_location, component, threshold)
    # get the mask
    mask = yolo.getMask(results)
    # get the area of the mask
    component_area = ip.getPixelArea(mask)

    # # get area for the reference object
    results = yolo.runInference(file_location, 'reference_object', threshold)
    mask = yolo.getMask(results)
    reference_pixel_area = ip.getPixelArea(mask)
    if reference_pixel_area == 0:
       #raise Exception("Reference object not found in the image")
       reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, 2)

    #component_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles, component)
    #reference_pixel_area = runTiles.runTilles(yolo, file_location, threshold, num_tiles+1, 'reference_object')

    area = (component_area/reference_pixel_area) * reference_area
   
    return area


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/estimate_area/{component}/{threshold}")
async def estimate_area(component:str, threshold:float, background_tasks:BackgroundTasks, uploaded_file:UploadFile = File(...)):
    '''
    Estimate the area of either flower or leaf
    Takes in an image file and returns the area of the component with a  threshold
    '''

    try:
        filename = f"{helpers.generate_random_file_name()}@{uploaded_file.filename}"
        file_location = f"api/data/input/{filename}"
        
        # save audio file temporally
        helpers.saveUploadfile(file_location, uploaded_file)

        # apply preprocessing to the image
        ip.apply_preprocessing(file_location)

        #yolo.savePlot(file_location, component, threshold)

        # run the model on the image
        area = estimateArea(file_location, component, threshold, 2)

        helpers.removefile(file_location)

        return {
            "area": area,
            "mask": f"api/data/output/{component}_{threshold}_{filename}",
            "unit": "inches^2"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

