from fastapi import APIRouter

from app.utils.logger import logger

endpointRoute = APIRouter()


@endpointRoute.get("/")
async def read_root():
    """
    The router has a single endpoint, a GET request handler for the root route "/".
    When a request is made to this endpoint, the read_root function is called.
    This function logs a message using the logger utility, then returns a JSON response with a greeting message "Hello World".
    """
    logger.debug("Root endpoint called!")
    return {"Hello": "World"}
