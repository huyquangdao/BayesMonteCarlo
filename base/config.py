from abc import ABC
import yaml


class Config(ABC):

    def __init__(self):
        """
        Constructor for class config.
        """
        pass

    def set_params(self, params):
        """
        method that sets parameters for class Config.
        :param params: a dictionary where keys are parameter names and values are the parameter values
        :return: None
        """
        for k, v in params.items():
            setattr(self, k, v)

    def get_params(self):
        """
        Method that returns the parameters for class Config
        :return: a dictionary
        """
        return self.__dict__.items()

    @staticmethod
    def load_config_from_yaml_file(file_path):
        """
        Method that loads the parameters from a yaml configuration file
        :param file_path: the path to the config file
        :return: None
        """
        with open(file_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            return params
