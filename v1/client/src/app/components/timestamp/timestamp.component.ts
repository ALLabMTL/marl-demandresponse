import { Component } from '@angular/core';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { SocketCommunicationService } from '@app/services/socket-communication/socket-communication.service';

@Component({
  selector: 'app-timestamp',
  templateUrl: './timestamp.component.html',
  styleUrls: ['./timestamp.component.scss']
})
export class TimestampComponent {

  constructor(public simulationManager: SimulationManagerService, public socketCommunication: SocketCommunicationService) {}

}
