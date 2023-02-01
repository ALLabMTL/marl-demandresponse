import { Component, OnInit } from '@angular/core';
import { SocketCommunicationService } from './services/socket-communication/socket-communication.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'demandereponse-client';

  constructor(private socketService: SocketCommunicationService) {}

  ngOnInit(): void {
    this.socketService.connect();
  }

}
