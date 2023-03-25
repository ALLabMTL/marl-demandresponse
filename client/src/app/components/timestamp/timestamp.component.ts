import { Component, OnInit } from '@angular/core';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { SocketCommunicationService } from '@app/services/socket-communication/socket-communication.service';

@Component({
  selector: 'app-timestamp',
  templateUrl: './timestamp.component.html',
  styleUrls: ['./timestamp.component.scss'],
})
export class TimestampComponent {
  speed = '2';
  speedOptions: number[] = [0, 1, 2, 3, 4, 5];
  constructor(
    public simulationManager: SimulationManagerService,
    public socketCommunication: SocketCommunicationService
  ) {}

  setSpeed(speed: string): void {
    this.simulationManager.speed = speed;
    this.socketCommunication.changeSpeed(this.simulationManager.speed);
  }

  changeTimeStep(): void {
    this.socketCommunication.setTimeStep();
  }
}
