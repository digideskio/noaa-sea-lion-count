"""
Main module
"""

import settings
import data

logger = settings.logger.getChild("main")

def run():
    logger.info("Starting...")

    loader = data.Loader()

if __name__ == "__main__":
    run()
