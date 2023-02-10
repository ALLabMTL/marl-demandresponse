import os
import random
import sys
from time import sleep

import socketio
from dependency_injector.wiring import Provide, inject
from fastapi import FastAPI

# We do this to be able to have app as the main directory regardless of where we run the code
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.containers import Container
from app.routers.endpoints import endpointRoute
from app.routers.websockets import register_endpoints
from app.services.socket_manager_service import SocketManager
from app.services.training_service import TrainingService
from app.utils.const import Color
from app.utils.logger import logger
from app.utils.settings import settings
from config import config_dict


class ExtendedFastAPI(FastAPI):
    container: Container


@inject
def create_app(
    sio: SocketManager = Provide["socket_manager_service"],
) -> ExtendedFastAPI:
    app = ExtendedFastAPI()

    app.include_router(endpointRoute, prefix="/api")
    asgi_app = socketio.ASGIApp(sio._sio)
    app.mount("/", asgi_app)

    register_endpoints()

    return app


@inject
def app_setup() -> None:
    # TODO: setup parser service, wandb...
    pass


container = Container()

# Wiring allows us to declare where we need to inject dependencies
# Doc: https://python-dependency-injector.ets-labs.org/wiring.html
container.wire(
    modules=[
        __name__,
        "app.routers.endpoints",
        "app.routers.websockets",
    ]
)

app = create_app()
app.container = container


@app.on_event("startup")
async def startup() -> None:
    logger.info(f"Started backend on port {Color.BOLD}{settings.PORT}{Color.END}")
    logger.info(
        f"WS Endpoint available at {Color.BOLD}ws://localhost:{settings.PORT}/{Color.END}"
    )
    app_setup()


@app.on_event("shutdown")
@inject
def shutdown() -> None:
    logger.info(f"Shutting down server...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5678, log_level="debug")