#from image_slicer import Tile, slice as slice_image
#from app.scripts import deeplab_model 
from PIL import Image
import numpy as np
from numpy import asarray
import os
from api.cv import img_processing as ip



def slice_image(input_image_path, output_directory, num_tiles):
    """
    Slice the input image into the specified number of tiles.

    Parameters:
    - input_image_path: Path to the input image file.
    - output_directory: Directory to save the sliced tiles.
    - num_tiles: Number of tiles to create.

    Returns:
    - List of paths to the sliced tiles.
    """
    # get extension of input image
    extension = os.path.splitext(input_image_path)[1]
    
    # Open the input image
    img = Image.open(input_image_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get image dimensions
    width, height = img.size

    # Calculate tile size
    tile_width = width // num_tiles
    tile_height = height // num_tiles

    # List to store paths of sliced tiles
    sliced_tiles = []

    # Iterate over rows and columns to slice the image
    for i in range(num_tiles):
        for j in range(num_tiles):
            # Calculate coordinates for cropping each tile
            left = j * tile_width
            upper = i * tile_height
            right = left + tile_width
            lower = upper + tile_height

            # Crop the tile
            tile = img.crop((left, upper, right, lower))

            # Save the tile to the output directory
            tile_path = os.path.join(output_directory, f"tile_{i}_{j}{extension}")
            tile.save(tile_path)

            # Append the path to the list
            sliced_tiles.append(tile_path)

    return sliced_tiles



def combine_tiles(tiles_directory, output_image_path, num_tiles):
    """
    Combine sliced tiles into a single image.

    Parameters:
    - tiles_directory: Directory containing the sliced tiles.
    - output_image_path: Path to save the combined image.
    - num_tiles: Number of tiles per row and column.

    Returns:
    - Path to the combined image.
    """
    # Create a list to store the tiles
    tiles = []

    # get extension of input image
    extension = os.listdir(tiles_directory)[0].split('.')[-1]
    #print(extension)

    # Load each tile and append to the list
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile_path = os.path.join(tiles_directory, f"tile_{i}_{j}.{extension}")
            tile = Image.open(tile_path)
            tiles.append(tile)

    # Calculate the dimensions of the combined image
    tile_width, tile_height = tiles[0].size
    total_width = tile_width * num_tiles
    total_height = tile_height * num_tiles

    # Create a new image with the calculated dimensions
    combined_image = Image.new("RGB", (total_width, total_height))

    # Paste each tile onto the combined image
    for i in range(num_tiles):
        for j in range(num_tiles):
            left = j * tile_width
            upper = i * tile_height
            combined_image.paste(tiles[i * num_tiles + j], (left, upper))

    # Save the combined image
    combined_image.save(output_image_path)

    return output_image_path


# load image
def runTile(yolo, threshold, component,  filename):
    #im = Image.open(filename)
    '''
    return the are of the tile image
    '''
    result = yolo.runInference(filename, component, threshold)
    mask = yolo.getMask(result)
    reference_pixel_area = ip.getPixelArea(mask)

    # save mask
    mask.save(f"api/data/tiles/{filename.split('/')[-1]}_mask.png")

    return reference_pixel_area


def runTilles(yolo, file_path, threshold, num_tiles = 2, component = 'reference_object'):
    
    # slice the image
    #num_tiles = 2
    tiles = slice_image(file_path, 'api/data/tiles', num_tiles)

    total_area = 0
    # run model on each tile
    for i in range(num_tiles):
        for j in range(num_tiles):
            tile_area  = runTile(yolo, threshold, component, tiles[i*num_tiles + j])
            total_area += tile_area

    # remove tiles
    #for tile in tiles:
    #    os.remove(tile)

    # return seg_image
    return total_area