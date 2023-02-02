import { Component } from '@angular/core';
import { SocketCommunicationService } from '@app/services/socket-communication/socket-communication.service';

@Component({
  selector: 'app-informations',
  templateUrl: './informations.component.html',
  styleUrls: ['./informations.component.scss']
})
export class InformationsComponent {
  constructor(public socketService: SocketCommunicationService) {}
}
