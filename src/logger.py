import logging
import os
from datetime import datetime

# Create logs directory and log file path
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)
log_path = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s  %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    logging.info("Logging has started...")
