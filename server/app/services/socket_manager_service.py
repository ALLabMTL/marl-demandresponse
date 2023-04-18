import socketio


class SocketManager:
    """
    Integrates SocketIO with FastAPI app.
    Adds `sio` property to FastAPI object (app).
    """

    def __init__(self) -> None:
        """Initialize the SocketManager instance by creating a socketio.AsyncServer instance with the "asgi" async_mode and allowing requests from any origin."""
        self._sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")

    @property
    def on(self):
        """Register a callback to be called when a particular event is received."""
        return self._sio.on

    @property
    def attach(self):
        """Attach the SocketIO server to a Flask or FastAPI app."""
        return self._sio.attach

    @property
    def emit(self):
        """Emit an event to one or more clients."""
        return self._sio.emit

    @property
    def send(self):
        """Send a message to one or more clients."""
        return self._sio.send

    @property
    def call(self):
        """Emit an event and wait for the client to respond."""
        return self._sio.call

    @property
    def close_room(self):
        """Close a room."""
        return self._sio.close_room

    @property
    def get_session(self):
        """Get a session."""
        return self._sio.get_session

    @property
    def save_session(self):
        """Save a session."""
        return self._sio.save_session

    @property
    def session(self):
        """Get the session for the current client."""
        return self._sio.session

    @property
    def disconnect(self):
        """Disconnect a client."""
        return self._sio.disconnect

    @property
    def handle_request(self):
        """Handle a request."""
        return self._sio.handle_request

    @property
    def start_background_task(self):
        """Start a background task."""
        return self._sio.start_background_task

    @property
    def sleep(self):
        """Start a background task."""
        return self._sio.sleep

    @property
    def enter_room(self):
        """Add a client to a room."""
        return self._sio.enter_room

    @property
    def leave_room(self):
        """Remove a client from a room."""
        return self._sio.leave_room
