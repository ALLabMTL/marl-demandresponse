import { Component } from '@angular/core';
import { SimulationManagerService } from '@app/services/simulation-manager.service';

@Component({
  selector: 'app-informations',
  templateUrl: './informations.component.html',
  styleUrls: ['./informations.component.scss']
})
export class InformationsComponent {
  constructor(public simulationManager: SimulationManagerService) {}
}
