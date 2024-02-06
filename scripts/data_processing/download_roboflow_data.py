import os
import argparse
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main(version):
    os.chdir("/Users/edwardamoah/Documents/GitHub/FloralArea/datasets") # make dynmaic later
    #rf = Roboflow(api_key=os.getenv("ROBOFLOW_KEY"))
    #project = rf.workspace("beevision").project("solitary-bee-hotels")
    #dataset = project.version(version).download("yolov8")
    #!pip install roboflow
    #from roboflow import Roboflow

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_KEY"))
    project = rf.workspace("insectnet-2024").project("floralarea-rhya8")
    dataset = project.version(version).download("yolov8")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow")
    parser.add_argument("--version", type=int, required=True, help="Version of dataset to download")
    args = parser.parse_args()
    main(args.version)


