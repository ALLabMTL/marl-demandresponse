from typing import List

from dependency_injector.wiring import Provide, inject

from app.services.socket_manager_service import SocketManager
from app.utils.logger import logger

@inject
def register_endpoints(
    sio: SocketManager = Provide["socket_manager_service"],
) -> None:
    """
    Define endpoints here for them to be included in the socketManager instance
    """

    @sio.on("connect")
    async def connect(sid, *args) -> None:
        logger.debug(f"Client connected with sid {sid}")
        await sio.emit("connected", {"message": f"Client connecté avec le sid: {sid}"})

    @sio.on("disconnect")
    async def disconnect(sid):
        logger.debug(f"Client disconnected with sid {sid}")
        # await sio.emit(
        #     "disconnected", {"message": f"Client disconnected with sid {sid}"}
        # )

    @sio.on("ping")
    async def ping(sid, *args) -> None:
        logger.debug(f"Client ping with sid {sid}")
        await sio.emit("pong", args)

