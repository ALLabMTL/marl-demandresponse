import { Component } from '@angular/core';
import { SocketCommunicationService } from '@app/services/socket-communication/socket-communication.service';

@Component({
  selector: 'app-stop-simulation-dialog',
  templateUrl: './stop-simulation-dialog.component.html',
  styleUrls: ['./stop-simulation-dialog.component.scss'],
})
export class StopSimulationDialogComponent {
  constructor(public socketCommunication: SocketCommunicationService) {}
}
