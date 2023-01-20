"""Always on controller."""

import typing as t
from .controller import Controller


class AlwaysOnController(Controller):
    """Always on controller."""

    def act(self, obs) -> t.Literal[True]:
        """Act on the environment."""
        return True
