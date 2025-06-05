#!/bin/bash
#SBATCH --job-name=sim_job                   # שם הג’וב שיופיע ב-squeue
#SBATCH --output=slurm_simulation.txt        # קובץ אליו נשלח כל stdout ו-stderr
#SBATCH --error=slurm_simulation.txt         # (משתמשים בקובץ אחד עבור שניהם)
#SBATCH --time=011:00:00                      # הגבלת זמן:

# -----------------------------------------------------
# הגדרות אישיות – יש לוודא שהתיקיות והקבצים קיימים:
IMAGE_NAME="sim_image:latest"                # השם/תג של ה־Docker image שבנית
CONTAINER_NAME="sim_run_container"           # השם שניתן ל־container בריצה
PROJECT_DIR="$HOME/tmp/simulations_rfChallange"   # הנתיב המלא לתיקיית הפרויקט
SCRIPT_COMMAND="python3 /app/simulations.py"      # סקריפט ה־Python בתוך ה־container
TIME_TO_RUN=$((11 * 3600))                   # 2 שעות בשניות = 7200
# -----------------------------------------------------

# הפעלת ה-Docker container בסינכרון (ללא -d),
# כך שכל הפלט יחזור ישירות ל־stdout, ו־SLURM יאסוף אותו ל־slurm_simulation.txt
docker run --init --rm \
  --name "$CONTAINER_NAME" \
  --net host \
  --gpus all \
  -v "$PROJECT_DIR":/app \
  "$IMAGE_NAME" \
  bash -c "$SCRIPT_COMMAND"

# בלוק רקע לשיגור SIGTERM לקונטיינר במקרה של Timeout
(
  sleep "$TIME_TO_RUN"
  if docker ps -q -f name="$CONTAINER_NAME" > /dev/null; then
    echo "Timeout reached ($TIME_TO_RUN seconds). Sending SIGTERM to container…"
    docker exec "$CONTAINER_NAME" kill -TERM 1
  fi
) &

# אין צורך ב-docker wait מכיוון ש-docker run יחזיר רק בסיום הקונטיינר
