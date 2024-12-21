import numpy as np
from PIL import Image
from PIL import ImageOps


def getPixelArea(img:Image) -> int:
    '''
    Get the number of white pixels in an image
    '''
    x,y,d = np.array(img).shape

    count = 0
    img_array = np.array(img)
    for i in range(x):
        for j in range(y):
            if img_array[i][j][0] == 255:
                count += 1
            else:
                continue

    return count



def apply_preprocessing(image_path):
    # Open the image
    image = Image.open(image_path)
    
    # Resize the image to 640x640
    resized_image = image.resize((640, 640))
    
    # Apply contrast stretching
    contrast_stretched_image = ImageOps.autocontrast(resized_image)
    #contrast_stretched_image = resized_image
    
    # Save the image
    contrast_stretched_image.save(image_path)
    
    return contrast_stretched_image