from dependency_injector import containers, providers

from services.socket_manager_service import SocketManager
from services.training_service import TrainingService
from services.client_manager_service import ClientManagerService

class Container(containers.DeclarativeContainer):

    socket_manager_service = providers.Singleton(SocketManager)

    client_manager_service = providers.Singleton(
        ClientManagerService,
        socket_manager_service=socket_manager_service
    )
    
    training_service = providers.Singleton(
        TrainingService,
        client_manager_service=client_manager_service,
    )
