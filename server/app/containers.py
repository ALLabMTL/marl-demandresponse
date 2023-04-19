from dependency_injector import containers, providers

from app.services.client_manager_service import ClientManagerService
from app.services.controller_manager import ControllerManager
from app.services.experiment_manager import ExperimentManager
from app.services.metrics_service import Metrics
from app.services.socket_manager_service import SocketManager
from app.services.training_manager import TrainingManager
from app.services.wandb_service import WandbManager


class Container(containers.DeclarativeContainer):
    """
    A declarative container for dependency injection.

    Attributes:
        socket_manager_service (providers.Singleton): A singleton provider for the SocketManager service.
        client_manager_service (providers.Singleton): A singleton provider for the ClientManagerService service, which depends on the SocketManager service.
        wandb_service (providers.Singleton): A singleton provider for the WandbManager service.
        metrics_service (providers.Singleton): A singleton provider for the Metrics service, which depends on the WandbManager service.
        controller_manager (providers.Singleton): A singleton provider for the ControllerManager service, which depends on the ClientManagerService, Metrics, and SocketManager services.
        training_manager (providers.Singleton): A singleton provider for the TrainingManager service, which depends on the ClientManagerService and Metrics services.
        experiment_manager (providers.Singleton): A singleton provider for the ExperimentManager service, which depends on the TrainingManager and ControllerManager services.
    """

    socket_manager_service = providers.Singleton(SocketManager)

    client_manager_service = providers.Singleton(
        ClientManagerService, socket_manager_service=socket_manager_service
    )

    wandb_service = providers.Singleton(WandbManager)

    metrics_service = providers.Singleton(Metrics, wandb_service=wandb_service)

    controller_manager = providers.Singleton(
        ControllerManager,
        client_manager_service=client_manager_service,
        metrics_service=metrics_service,
        socket_manager_service=socket_manager_service,
    )

    training_manager = providers.Singleton(
        TrainingManager,
        client_manager_service=client_manager_service,
        metrics_service=metrics_service,
    )

    experiment_manager = providers.Singleton(
        ExperimentManager,
        training_manager=training_manager,
        controller_manager=controller_manager,
    )
