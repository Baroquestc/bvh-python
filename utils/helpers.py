import json
import logging


def read_json(json_file):
    with open(json_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            yield data


def logger(log_file: str):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging


def setup_logging(logger_file: str):
    """
    Set up logging to a file in the specified folder.

    Args:
        logger_file (str): The path to the log file.

    Returns:
        None
    """
    logging.basicConfig(filename=logger_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Logging started')

    return logging
