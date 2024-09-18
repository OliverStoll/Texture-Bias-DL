import os

from utils.logger import create_logger
from utils.config import CONFIG


log = create_logger("SanityCheck GPU")


def print_gpu_info(gpu_index):
    cmd_str = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
    result = os.popen(cmd_str).read().replace("utilization.gpu [%]", "").replace(" ", "").replace("\n", "  ")
    log.debug(f"GPU: {CONFIG.get('gpu_indexes', 'None')}  |  [{result}]")
    # get the GPU utilization of the given GPU index
    try:
        if isinstance(gpu_index, list):
            gpu_index = gpu_index[0]
        utilization = result.strip().replace('%', '').split("  ")[gpu_index]
        utilization = int(utilization)
    except Exception as e:
        log.warning(f"Error checking gpu utilization: {e}")
        utilization = 0
    return utilization