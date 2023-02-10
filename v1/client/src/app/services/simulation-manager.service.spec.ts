import { TestBed } from '@angular/core/testing';

import { SimulationManagerService } from './simulation-manager.service';

describe('SimulationManagerService', () => {
  let service: SimulationManagerService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SimulationManagerService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
