from dependency_injector import containers, providers

from app.services.client_manager_service import ClientManagerService
from app.services.socket_manager_service import SocketManager
from app.services.controller_manager import ControllerManager
from app.services.wandb_service import WandbManager
from app.services.metrics_service import Metrics
from app.services.training_manager import TrainingManager
from app.services.experiment_manager import ExperimentManager
from app.services.parser_service import ParserService


class Container(containers.DeclarativeContainer):
    socket_manager_service = providers.Singleton(SocketManager)

    client_manager_service = providers.Singleton(ClientManagerService)

    wandb_service = providers.Singleton(WandbManager)

    parser_service = providers.Singleton(ParserService)

    metrics_service = providers.Singleton(Metrics, wandb_service=wandb_service)

    controller_manager = providers.Singleton(
        ControllerManager,
        socket_manager_service=socket_manager_service,
        client_manager_service=client_manager_service,
        metrics_service=metrics_service,
        parser_service=parser_service,
    )

    training_manager = providers.Singleton(
        TrainingManager,
        socket_manager_service=socket_manager_service,
        client_manager_service=client_manager_service,
        metrics_service=metrics_service,
        parser_service=parser_service,
    )

    experiment_manager = providers.Singleton(
        ExperimentManager,
        training_manager=training_manager,
        controller_manager=controller_manager,
        parser_service=parser_service,
    )
