# get location from script arg with argparse
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--yaml", help="yaml file location")
parser.add_argument("--dest", help="destination to save the txt file")

args = parser.parse_args()
yaml_file = args.yaml
dest = args.dest

# import yaml
import yaml
with open(yaml_file) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    obj_list = yaml.load(file, Loader=yaml.FullLoader)
    # get the names of the objects from the "names" variable
    names = obj_list["names"]

# write the names to a txt file
with open(dest, "w") as file:
    for name in names:
        file.write(name + "\n")