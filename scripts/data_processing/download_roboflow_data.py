import os
import argparse
from roboflow import Roboflow
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def main(version):
    try:
        folder_path = "/Users/edwardamoah/Documents/GitHub/FloralArea/datasets"  # make dynamic later
        os.makedirs(folder_path, exist_ok=True)
        os.chdir(folder_path)
    except Exception as e:
        print(f"An error occurred while changing the directory: {str(e)}")

    rf = Roboflow(api_key=os.getenv("ROBOFLOW_KEY"))
    project = rf.workspace("insectnet-2024").project("floralarea-rhya8")
    dataset = project.version(version).download("yolov8")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Roboflow")
    parser.add_argument("--version", type=int, required=True, help="Version of dataset to download")
    args = parser.parse_args()
    main(args.version)


