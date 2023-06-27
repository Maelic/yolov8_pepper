# get location from script arg with argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--location", help="location of the dataset")
parser.add_argument("--project", help="project name in roboflow")

args = parser.parse_args()
dest_path = args.location
project = args.project
project = args.project

from roboflow import Roboflow
rf = Roboflow(api_key="Bx7p9SDyJ5AuiCtbgrg2")
project = rf.workspace("robobreizh").project(project)
dataset = project.version(1).download("yolov8", location=dest_path)