from afs_safe_logger import Logger

import warnings


DEFAULT_LOGGER_NAME = 'main'


class MainLogger:
    """
    Keeps track of loggers using the Singleton design pattern.
    """

    loggers = dict()

    def init(self, path, name=DEFAULT_LOGGER_NAME):
        """

        :param path: Output file for logger.
        :param name: Lookup key for logger.
        :return: A Logger instance specified by `name`.
        """
        if name not in MainLogger.loggers:
            MainLogger.loggers[name] = Logger(path)
        else:
            warnings.warn("Logger with path already initialized.", RuntimeWarning)
        return self.get(name)

    def get(self, name=DEFAULT_LOGGER_NAME):
        """

        :param name: Lookup key for logger.
        :return: A Logger instance specified by `name`.
        """
        return MainLogger.loggers[name]
