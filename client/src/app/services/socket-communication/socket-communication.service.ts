import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';
import { NotificationService } from '@app/services/notification/notification.service';
import { SocketService } from '@app/services/socket/socket.service';
import { SimulationManagerService } from '../simulation-manager.service';
import { SnackbarMessage } from '@app/classes/snackbar-message';

@Injectable({
  providedIn: 'root',
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
      this.simulationManager.connected = true;
      this.changeSpeed(this.simulationManager.speed);
    });

    this.socketService.on('dataChange', (data: SidenavData) => {
      this.simulationManager.addTimeStep(data);
    });

    this.socketService.on('houseChange', (data: HouseData[]) => {
      this.simulationManager.updateHousesData(data);
    });

    this.socketService.on('stopped', () => {
      this.simulationManager.started = false;
      this.simulationManager.stopped = true;
      this.snackBarService.openSuccessSnackBar('Simulation stopped', '');
    });

    this.socketService.on('paused', () => {
      this.simulationManager.started = false;
      this.simulationManager.stopped = false;
      this.snackBarService.openSuccessSnackBar('Simulation paused', '');
    });

    this.socketService.on('error', (data: SnackbarMessage) => {
      this.snackBarService.openFailureSnackBar(data.message, '');
    });

    this.socketService.on('success', (data: SnackbarMessage) => {
      this.snackBarService.openSuccessSnackBar(data.message, '');
    });

    this.socketService.on('agent', (data: string) => {
      this.simulationManager.agentName = data;
    });

    this.socketService.on('timeStepData', (data: SidenavData) => {
      this.simulationManager.setTimeStep(data);
    });
  }

  startSimulation(): void {
    this.simulationManager.reset();
    this.socketService.send('train');
  }

  stopSimulation(): void {
    this.snackBarService.openSuccessSnackBar('Stopping simulation...', '');
    this.simulationManager.stopped = true;
    this.socketService.send('stop');
  }

  changeSpeed(speed: string): void {
    this.socketService.send('changeSpeed', speed);
  }

  pauseSimulation(): void {
    this.snackBarService.openSuccessSnackBar('Pausing simulation...', '');
    this.simulationManager.paused = true;
    this.socketService.send('pause');
  }

  setTimeStep() {
    this.socketService.send('getSimAtTimeStep', {
      timestep: this.simulationManager.currentTimeStep - 1,
    });
  }
}
