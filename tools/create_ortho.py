import argparse
import glob
import logging
import os

import dotenv
from pyodm import Node

dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_process(info):
    logger.info(f"Progress: {info.progress:1.2f}%")


def progress(p):
    logger.info(f"Creation progress: {p:1.2f}%")


def process(images, output_path, uuid=None):

    n = Node(
        host=os.getenv("LIGHTNING_URL"), port=443, token=os.getenv("LIGHTNING_TOKEN")
    )
    logger.info("Connected to node")

    logger.info(n.info())

    if uuid is not None:
        task = n.get_task(uuid)
        logger.info("Got existing task")
    else:
        task = n.create_task(
            files=images, progress_callback=progress, options={"dsm": True}
        )
        logger.info("Created task")

    logger.info(task.info())

    task.wait_for_completion(status_callback=log_process)

    task.download_zip(output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--output", help="output path", required=True, type=str)
    parser.add_argument("--task_name", help="task name", type=str)
    parser.add_argument("image_glob", help="images (wildcard)", type=str)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    images = glob.glob(args.image_glob)

    process(images, args.output)
