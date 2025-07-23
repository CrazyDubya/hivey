import logging

import utils

if __name__ == "__main__":
    utils.configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Initializing project structure...")
    utils.initialize_database()  # This will create the data/ directory
    logger.info("Project structure initialization complete.")
    print(
        "Data directory should now exist. You can create data/.gitkeep if needed."  # noqa: E501
    )
