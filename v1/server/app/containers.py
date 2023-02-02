from dependency_injector import containers, providers

from app.services.socket_manager_service import SocketManager


class Container(containers.DeclarativeContainer):

    socket_manager_service = providers.Singleton(SocketManager)
