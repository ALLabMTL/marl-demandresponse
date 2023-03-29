from dependency_injector.wiring import Provide, inject

from app.services.experiment_manager import ExperimentManager
from app.services.socket_manager_service import SocketManager
from app.utils.logger import logger


@inject
def register_endpoints(
    sio: SocketManager = Provide["socket_manager_service"],
    experiment_manager: ExperimentManager = Provide["experiment_manager"],
) -> None:
    """
    Define endpoints here for them to be included in the socketManager instance
    """

    @sio.on("connect")
    async def connect(sid, *args) -> None:
        logger.debug(f"Client connected with sid {sid}")
        await sio.emit("connected", {"message": f"Client connected with sid: {sid}"})

    @sio.on("disconnect")
    async def disconnect(sid, *args) -> None:
        logger.debug(f"Client disconnected with sid {sid}")
        experiment_manager.update_experiment_state(True)

    @sio.on("train")
    async def train(sid, *args) -> None:
        logger.debug("Starting experiment")
        experiment_manager.update_experiment_state(False)
        await experiment_manager.start()

    @sio.on("stop")
    async def stop_training(sid, *args) -> None:
        experiment_manager.update_experiment_state(True)

    @sio.on("changeSpeed")
    async def change_speed(sid, speed: str, *args) -> None:
        experiment_manager.change_speed(float(speed))
