import { Injectable } from '@angular/core';
import { io, Socket } from 'socket.io-client';
import { environment } from 'src/environments/environment';

@Injectable({
    providedIn: 'root',
})
export class SocketService {

  socket!: Socket;

  isSocketAlive() {
      return this.socket && this.socket.connected;
  }

  connect() {
      this.socket = io(environment.serverUrl, { transports: ['websocket'], upgrade: false });
  }

  disconnect() {
      this.socket.disconnect();
  }

  on<T>(event: string, action: (data: T) => void): void {
      this.socket.on(event, action);
  }

  send<T>(event: string, data?: T): void {
      if (data) {
          this.socket.emit(event, data);
      } else {
          this.socket.emit(event);
      }
  }
}
