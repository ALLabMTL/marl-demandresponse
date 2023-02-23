import { Component, Input, OnInit} from '@angular/core';
import { SharedService } from '@app/services/shared/shared.service';
import {MatDialog} from '@angular/material/dialog';
import { DialogComponent } from '../dialog/dialog.component';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { SidebarComponent } from '../sidebar/sidebar.component';


interface PageData {
  id: number;
  content: string[100];
}

@Component({
  selector: 'app-grid',
  templateUrl: './grid.component.html',
  styleUrls: ['./grid.component.scss']
})

export class GridComponent implements OnInit {

  @Input() pages: PageData[];
  currentPage: number = 1;
  maxPage: number = 35;
  precisionValueSelected: number = 0;

  constructor(public sharedService: SharedService, public dialog: MatDialog, public simulationManager: SimulationManagerService){
    this.pages = [
      { id: 1, content: "Page 1" },
      { id: 2, content: "Page 2" },
      { id: 3, content: "Page 3" },
    ]
  }

  ngOnInit() {
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
    this.currentPage = this.pages[0].id;
    for (let i = 0; i < this.simulationManager.housesData.length; i++) {
      console.log (this.simulationManager.housesData[i]);
    }
    this.sharedService.currentPrecisionValue.subscribe(houseColorPrecisionValue => this.precisionValueSelected = houseColorPrecisionValue);
    //this.houseColor();
    // console.log(this.houseColor(0.40));
  }

  cells = new Array(100).fill(null);

  getColumnWidths() {
    return `repeat(10, ${100 / 10}%)`;
  }

  getRowHeights() {
    return `repeat(10, ${100 / 10}%)`;
  }

  getHvacColor(i: number): string {
    if (this.simulationManager.housesData[i].hvacStatus === 'ON') {
      return 'green';
    } 
    else if(this.simulationManager.housesData[i].hvacStatus === 'OFF'){
      return 'red';
    }
    else{
      return 'gray';
    }
  }

  switchPage(id: number) {
    this.currentPage += 1;
  }

  openDialog(index: number) {
    this.dialog.open(DialogComponent, {
      data :index
    });

  }

}
