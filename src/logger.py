import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y-%H-%M-%S')}.log"
log_dir = os.path.join(os.getcwd(), "logs")
# Create logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

logs_path = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=logs_path,
    format='%(asctime)s - %(levelname)s - %(lineno)d %(name)s - %(message)s',
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")