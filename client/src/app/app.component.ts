import { Component, HostListener, OnInit } from '@angular/core';
import { SocketCommunicationService } from './services/socket-communication/socket-communication.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'demandereponse-client';
  height = '100vh';
  minHeight = '100vh';

  constructor(private socketService: SocketCommunicationService) {}

  ngOnInit(): void {
    this.socketService.connect();
  }

  fullscreen = false;

  @HostListener('document:fullscreenchange', ['$event'])
  onFullscreenChange(event: Event) {
    this.fullscreen = !!document.fullscreenElement;
  }

  checkFullscreen() {
    this.fullscreen = !!document.fullscreenElement;
  }

}
