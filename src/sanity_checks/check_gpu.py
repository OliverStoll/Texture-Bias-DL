import os
import torch

from utils.logger import create_logger
from utils.config import CONFIG


log = create_logger("SanityCheck GPU")


def print_gpu_info():
    cmd_str = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    result = os.popen(cmd_str).read().replace("utilization.gpu [%]", "").replace(" ", "").replace("\n", "  ")
    log.debug(f"GPU: {CONFIG['gpu_indexes']}  |  [{result}]")