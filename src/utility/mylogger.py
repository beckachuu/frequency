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
    def getLog():
        if (MyLogger.logger is None):
            LOG_LEVEL = logging.DEBUG
            LOGFORMAT = "  %(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s%(reset)s"
            from colorlog import ColoredFormatter
            logging.root.setLevel(LOG_LEVEL)
            formatter = ColoredFormatter(LOGFORMAT)
            stream = logging.StreamHandler()
            stream.setLevel(LOG_LEVEL)
            stream.setFormatter(formatter)

            MyLogger.logger = logging.getLogger('pythonConfig')
            MyLogger.logger.setLevel(LOG_LEVEL)
            MyLogger.logger.addHandler(stream)

            # MyLogger.logger.debug("A quirky message only developers care about")
            # MyLogger.logger.info("Curious users might want to know this")
            # MyLogger.logger.warn("Something is wrong and any user should be informed")
            # MyLogger.logger.error("Serious stuff, this is red for a reason")
            # MyLogger.logger.critical("OH NO everything is on fire")
        return MyLogger.logger