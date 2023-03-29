import { Component, Inject, OnInit } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { HouseData } from '@app/classes/sidenav-data';
import { SharedService } from '@app/services/shared/shared.service';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { GridComponent } from '../grid/grid.component';


@Component({
  selector: 'app-dialog',
  templateUrl: './dialog.component.html',
  styleUrls: ['./dialog.component.scss']
})
export class DialogComponent {
  id: number;
  currentPage: number = 1;
  
  // TODO: Do this in a prettier way
  constructor(@Inject(MAT_DIALOG_DATA) public data: number, public simulationManager: SimulationManagerService, public sharedService: SharedService) {
    this.id = data;
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
  }

}
