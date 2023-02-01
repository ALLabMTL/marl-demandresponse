import os

from pydantic import BaseSettings


class Settings(BaseSettings):
    # FastAPI settings
    PORT = 5678
    SUBDOMAIN = "marl-demandresponse_PI_4"
    BASE_URL = "http://localhost:5678"
    VERBOSE = os.environ.get("VERBOSE") is not None


settings = Settings()
