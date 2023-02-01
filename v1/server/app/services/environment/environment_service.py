from parser_service import ParserService


class EnvironmentService:
    environment_properties: dict
    parser_service: ParserService

    def __init__(self, parser_service: ParserService) -> None:
        self.parser_service = parser_service
        parser_service.get_environment_properties()
