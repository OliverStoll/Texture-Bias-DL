#!/bin/bash

# Check if the argument is provided and is within the valid range
if [[ -z "$1" ]] || [[ ! "$1" =~ ^[0-3]$ ]]; then
    echo "Usage: $0 [GPU_ID]"
    echo "GPU_ID must be between 0 and 3."
    exit 1
fi

# Capture the GPU ID from the input argument
gpu_id=$1

# Initialize an associative array for storing the utilizations for this GPU
declare -A gpu_buffer

# Function to compute the moving average
compute_average() {
    local -n buf=$1  # Use a nameref to refer to the array in the caller
    if [ ${#buf[@]} -gt 0 ]; then
        local sum=0
        for i in "${buf[@]}"; do
            sum=$(($sum + i))
        done
        echo $((sum / ${#buf[@]}))
    else
        echo 0  # Default to 0 if no data
    fi
}

# Clear the terminal initially to have a clean space for output
clear

# Infinite loop to keep the script running
while true; do
    # Fetch GPU utilization percentages
    readarray -t usages < <(nvidia-smi | grep -oP 'P0\s+\d+W / \d+W\s+\|\s+\d+MiB / \d+MiB\s+\|\s+\K\d+')

    # Check if the GPU ID is available in the fetched data
    if [[ -z "${usages[$gpu_id]}" ]]; then
        echo "$gpu_id: Data not available"
        sleep 0.3
        continue
    fi

    # Manage the buffer for this specific GPU
    gpu_buffer[$gpu_id]+="${usages[$gpu_id]} "
    read -a buffer <<< "${gpu_buffer[$gpu_id]}"
    if [ ${#buffer[@]} -gt 20 ]; then
        gpu_buffer[$gpu_id]="${gpu_buffer[$gpu_id]:3}"
    fi

    # Calculate the moving average
    avg=$(compute_average buffer)
    # Calculate number of blocks for the load-meter (scale of 0-35)
    num_blocks=$((avg * 20 / 100))
    # Create the load-meter bar
    load_meter=$(printf '%*s' "$num_blocks" '' | tr ' ' '#')

    # Clear the current printed line (reposition cursor to line start)
    tput cup 0 0

    # Print the load meter for the specified GPU
    echo "$gpu_id: [${load_meter}$(printf '%*s' $((20 - num_blocks)) '')]"

    # Sleep for 0.3 seconds
    sleep 0.3
done
