from fastapi import APIRouter

from app.utils.logger import logger

endpointRoute = APIRouter()


@endpointRoute.get("/")
async def read_root():
    logger.debug("Root endpoint called!")
    return {"Hello": "World"}

