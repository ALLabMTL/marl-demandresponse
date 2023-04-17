import { Component, OnInit } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { SocketCommunicationService } from '@app/services/socket-communication/socket-communication.service';
import { DialogComponent } from '../dialog/dialog.component';
import { StopSimulationDialogComponent } from '../stop-simulation-dialog/stop-simulation-dialog.component';

interface SpeedOption {
  value: number;
  label: string;
}

@Component({
  selector: 'app-timestamp',
  templateUrl: './timestamp.component.html',
  styleUrls: ['./timestamp.component.scss'],
})
export class TimestampComponent {
  speed = '4';
  // speedOptions: number[] = [0, 1, 2, 3, 4];
  // speedOpt: string[] = ["Max", "4x", "2x", "1.5x", "1x"];

  speedOptionsMapped: SpeedOption[] = [
    { value: 0, label: 'Max' },
    { value: 1, label: '4x' },
    { value: 2, label: '2x' },
    { value: 3, label: '1.5x' },
    { value: 4, label: '1x' },
  ];

  constructor(
    public simulationManager: SimulationManagerService,
    public socketCommunication: SocketCommunicationService, 
    public dialog: MatDialog,
  ) { }

  setSpeed(speed: string): void {
    this.simulationManager.speed = speed;
    this.socketCommunication.changeSpeed(this.simulationManager.speed);
  }

  changeTimeStep(): void {
    this.socketCommunication.setTimeStep();
  }

  openDialog(): void {
    this.dialog.open(StopSimulationDialogComponent);
  }
}
