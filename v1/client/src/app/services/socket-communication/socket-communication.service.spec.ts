import { TestBed } from '@angular/core/testing';

import { SocketCommunicationService } from './socket-communication.service';

describe('SocketCommunicationService', () => {
  let service: SocketCommunicationService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SocketCommunicationService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
