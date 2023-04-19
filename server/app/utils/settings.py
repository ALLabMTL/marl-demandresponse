import os

from pydantic import BaseSettings


class Settings(BaseSettings):
    """
    The Settings class inherits from BaseSettings of the Pydantic library, and is used to store and access the settings for a FastAPI application. It has the following attributes:

    Attributes:
        PORT: An integer representing the port number on which the FastAPI application should run. The default value is 5678.
        SUBDOMAIN: A string representing the subdomain for the FastAPI application. The default value is "marl-demandresponse_PI_4".
        BASE_URL: A string representing the base URL for the FastAPI application. The default value is "http://localhost:5678".
        VERBOSE: A boolean indicating whether or not verbose output should be enabled. The value is True if the environment variable VERBOSE is set, otherwise it is False.
    """

    # FastAPI settings
    PORT = 5678
    SUBDOMAIN = "marl-demandresponse_PI_4"
    BASE_URL = "http://localhost:5678"
    VERBOSE = os.environ.get("VERBOSE") is not None


settings = Settings()
