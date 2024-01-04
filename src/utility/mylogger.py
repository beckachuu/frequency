"""
Created by khadm on 7/14/2022
Feature: 
"""
'''
Utility for logging
Reference: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
'''
import logging

logging.getLogger('tensorflow').disabled = True


class MyLogger:
    logger = None

    @staticmethod
    def getLog(quiet=False):
        LOG_LEVEL = logging.DEBUG if not quiet else logging.CRITICAL

        if (MyLogger.logger is None):
            LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
            from colorlog import ColoredFormatter
            logging.root.setLevel(LOG_LEVEL)
            formatter = ColoredFormatter(LOGFORMAT)
            stream = logging.StreamHandler()
            stream.setLevel(LOG_LEVEL)
            stream.setFormatter(formatter)

            MyLogger.logger = logging.getLogger('pythonConfig')
            MyLogger.logger.addHandler(stream)

        MyLogger.logger.setLevel(LOG_LEVEL)
        return MyLogger.logger
    