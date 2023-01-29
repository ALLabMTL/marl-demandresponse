import { Injectable } from '@angular/core';
import { SnackbarService } from '@app/services/snackbar/snackbar.service';
import { SocketService } from '@app/services/socket/socket.service';

@Injectable({
  providedIn: 'root'
})
export class SocketCommunicationService {

  constructor(        
    public socketService: SocketService,
    private snackBarService: SnackbarService,
  ) { }

  connect(): void {
    if (!this.socketService.isSocketAlive()) {
        this.socketService.connect();
        const timeout = 2000;
        setTimeout(() => {
            if (!this.socketService.isSocketAlive()) {
                const message = 'Erreur: Echec de connexion au serveur';
                const action = '';
                this.snackBarService.openFailureSnackBar(message, action);
            }
        }, timeout);
        this.configureSocket();
    }
}

  configureSocket() {
    this.socketService.on('connect', () => {
        this.snackBarService.openSuccessSnackBar('Serveur connecté', '');
    });

  }

  pingServer() {
    this.socketService.send('ping');
    this.snackBarService.openSuccessSnackBar('Pinging server...', '');
}



}
