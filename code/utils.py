import logging
import json

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    '''
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # Change the value of learning_rate in params
    '''
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


import logging
import json

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    '''
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # Change the value of learning_rate in params
    '''
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def set_logger(log_path):
    ''' Set the logger to log info in terminal and file `log_path`.

    Args:
    log_path: (string) where to log
    '''

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

