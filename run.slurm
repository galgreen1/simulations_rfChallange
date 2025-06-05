#!/bin/bash
#SBATCH --job-name=sim_job                  
#SBATCH --output=slurm_simulation.txt       
#SBATCH --error=slurm_simulation.txt        
#SBATCH --time=011:00:00                     

# -----------------------------------------------------

IMAGE_NAME="sim_image:latest"              
CONTAINER_NAME="sim_run_container"           
PROJECT_DIR="$HOME/tmp/simulations_rfChallange"   
SCRIPT_COMMAND="python3 /app/simulations.py"      
TIME_TO_RUN=$((11 * 3600))                   
# -----------------------------------------------------



docker run --init --rm \
  --name "$CONTAINER_NAME" \
  --net host \
  --gpus all \
  -v "$PROJECT_DIR":/app \
  "$IMAGE_NAME" \
  bash -c "$SCRIPT_COMMAND"


(
  sleep "$TIME_TO_RUN"
  if docker ps -q -f name="$CONTAINER_NAME" > /dev/null; then
    echo "Timeout reached ($TIME_TO_RUN seconds). Sending SIGTERM to container…"
    docker exec "$CONTAINER_NAME" kill -TERM 1
  fi
) &


