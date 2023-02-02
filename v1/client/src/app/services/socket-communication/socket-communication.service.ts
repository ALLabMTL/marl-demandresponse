import { Injectable } from '@angular/core';
import { NotificationService } from '@app/services/notification/notification.service';
import { SocketService } from '@app/services/socket/socket.service';

@Injectable({
  providedIn: 'root'
})
export class SocketCommunicationService {

  constructor(        
    public socketService: SocketService,
    private snackBarService: NotificationService,
  ) { }

  connect(): void {
    if (!this.socketService.isSocketAlive()) {
        this.socketService.connect();
        const timeout = 2000;
        setTimeout(() => {
            if (!this.socketService.isSocketAlive()) {
                const message = 'Error: Connection with server failed';
                const action = '';
                this.snackBarService.openFailureSnackBar(message, action);
            }
        }, timeout);
        this.configureSocket();
    }
}

  configureSocket() {
    this.socketService.on('connect', () => {
        this.snackBarService.openSuccessSnackBar('Connected to server', '');
    });

    this.socketService.on('pong', () => {
      this.snackBarService.openSuccessSnackBar('Server pong', '');
  });
  }

  pingServer() {
    this.socketService.send('ping');
    this.snackBarService.openSuccessSnackBar('Pinging server...', '');
}

}
