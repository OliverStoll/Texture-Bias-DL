import os

import yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("src")[0].replace("\\", "/")

config = yaml.safe_load(open(f"{ROOT_DIR}config.yml"))  # load config file to be imported
config["ROOT_DIR"] = ROOT_DIR
