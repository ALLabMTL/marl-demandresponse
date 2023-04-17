from dependency_injector import containers, providers

from app.services.client_manager_service import ClientManagerService
from app.services.socket_manager_service import SocketManager
from app.services.training_service import TrainingService


class Container(containers.DeclarativeContainer):
    socket_manager_service = providers.Singleton(SocketManager)

    client_manager_service = providers.Singleton(ClientManagerService)

    training_service = providers.Singleton(
        TrainingService,
        socket_manager_service=socket_manager_service,
        client_manager_service=client_manager_service,
    )
