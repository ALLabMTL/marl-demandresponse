import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';
import { NotificationService } from '@app/services/notification/notification.service';
import { SocketService } from '@app/services/socket/socket.service';
import { SimulationManagerService } from '../simulation-manager.service';
import { SnackbarMessage } from '@app/classes/snackbar-message';


@Injectable({
  providedIn: 'root'
})
export class SocketCommunicationService {

  constructor(
    public socketService: SocketService,
    private snackBarService: NotificationService,
    private simulationManager: SimulationManagerService
  ) { }

  connect(): void {
    if (!this.socketService.isSocketAlive()) {
      this.socketService.connect();
      const timeout = 2000;
      setTimeout(() => {
        if (!this.socketService.isSocketAlive()) {
          const message = 'Error: cannot connect to server';
          const action = '';
          this.snackBarService.openFailureSnackBar(message, action);
        }
      }, timeout);
      this.configureSocket();

    }
  }

  configureSocket() {
    this.socketService.on('connected', () => {
      this.snackBarService.openSuccessSnackBar('Connected to server', '');
      this.changeSpeed(this.simulationManager.speed)
      this.startTraining()
    });

    this.socketService.on('dataChange', (data: SidenavData) => {
      this.simulationManager.addTimeStep(data)

    });

    this.socketService.on('houseChange', (data: HouseData[]) => {
      this.simulationManager.updateHousesData(data);
    })

    this.socketService.on('stopped', () => {
      // this.simulationManager.resetSimulation();
      this.simulationManager.started = false;
      this.simulationManager.stopped = true;
      this.snackBarService.openSuccessSnackBar('Simulation stopped', '');

    })

    this.socketService.on('error', (data: SnackbarMessage) => {
      this.snackBarService.openFailureSnackBar(data.message, '');
    });

    this.socketService.on('success', (data: SnackbarMessage) => {
      this.snackBarService.openSuccessSnackBar(data.message, '');
    });


  }

  startTraining(): void {
    this.socketService.send('train');
    this.simulationManager.started = true;
    this.simulationManager.stopped = false;
  }

  stopTraining(): void {
    this.snackBarService.openSuccessSnackBar('Stopping simulation...', '');
    this.simulationManager.stopped = true;
    this.socketService.send('stop');
  }

  changeSpeed(speed: number): void {
    this.socketService.send('changeSpeed', speed);
  }
}
