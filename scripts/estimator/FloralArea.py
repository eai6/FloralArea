import skimage.io as skio
from skimage.filters import threshold_triangle
from skimage.filters import try_all_threshold
import matplotlib.pyplot as plt
import os
import numpy as np
import csv

plants = [f for f in os.listdir('.') if os.path.isdir(f)]

planar = ["MA","PI","PM","PT"]

with open("./FloralArea.csv", mode='w') as csv_file:
        fieldnames = ['Filename','plant type', 'Floral area (cm^2)']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

for plant in plants:
    images = [f for f in os.listdir(plant) if f != '.DS_Store']
    
    for image in images:
        img = skio.imread(f"{plant}//{image}", as_gray=True) #should return numpy array
        thresh = threshold_triangle(img)
        binary = img > thresh
        
        mean = binary.mean() #should be average greyscale value
        start = 'size_'
        end = '.png'
        s = image # filename, 'xxxxxxxxsize_12_25.png'
        totalArea = (s.split(start))[1].split(end)[0] # returns '12_25'
        totalArea = float(totalArea.replace("_","."))
        FloralArea = totalArea - (totalArea*mean)
        if plant not in planar:
            FloralArea = FloralArea*2
        with open("./FloralArea.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([image,plant,FloralArea])

    



