from os import getenv as secret  # noqa: F401

import dotenv
import yaml

dotenv.load_dotenv()
config = yaml.safe_load(open("config.yml"))
