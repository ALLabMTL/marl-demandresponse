from dependency_injector.wiring import Provide, inject

from app.services.socket_manager_service import SocketManager
from app.utils.logger import logger
from app.services.training_service import TrainingService



@inject
def register_endpoints(
    sio: SocketManager = Provide["socket_manager_service"],
    training_service: TrainingService = Provide("training_service")
) -> None:
    """
    Define endpoints here for them to be included in the socketManager instance
    """

    @sio.on("connect")
    async def connect(sid, *args) -> None:
        logger.debug(f"Client connected with sid {sid}")
        await sio.emit("connected", {"message": f"Client connected with sid: {sid}"})
    
    @sio.on("disconnect")
    async def disconnect(sid, *args):
        logger.debug(f"Client disconnected with sid {sid}")
        training_service.stop = True

    @sio.on("train")
    async def train(sid, *args) -> None:
        logger.debug("Starting experiment")
        training_service.stop = False
        await training_service.train()
    
    @sio.on("stop")
    async def stop_training(sid, *args) -> None:
        training_service.stop = True

