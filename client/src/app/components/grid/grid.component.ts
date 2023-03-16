import { Component, Input, OnInit } from '@angular/core';
import { SharedService } from '@app/services/shared/shared.service';
import { MatDialog } from '@angular/material/dialog';
import { DialogComponent } from '../dialog/dialog.component';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { SidebarComponent } from '../sidebar/sidebar.component';
import { HouseData } from '@app/classes/sidenav-data';


interface PageData {
  id: number;
  content: HouseData[];
}

@Component({
  selector: 'app-grid',
  templateUrl: './grid.component.html',
  styleUrls: ['./grid.component.scss']
})

export class GridComponent implements OnInit {

  @Input() pages: PageData[];
  currentPage: number = 1;
  // maxPage: number = 1;
  // housesPerPage: number = 100;
  precisionValueSelected: number = 0;
  nbSquares = 100;
  nbSquarePerLine = Math.sqrt(this.nbSquares);

  columnWidths = `repeat(10, ${100/10}%)`;
  rowHeights = `repeat(10, ${100/10}%)`;

  constructor(public sharedService: SharedService, public dialog: MatDialog, public simulationManager: SimulationManagerService) {
    this.pages = []
  }

  ngOnInit() {
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
    this.sharedService.currentPrecisionValue.subscribe(houseColorPrecisionValue => this.precisionValueSelected = houseColorPrecisionValue);
    this.sharedService.squareNbValue.subscribe(nbSquares => this.nbSquares = nbSquares);
    this.sharedService.columnWidthsValue.subscribe(columnWidths => this.columnWidths = columnWidths);
    this.sharedService.rowHeightsValue.subscribe(rowHeights => this.rowHeights = rowHeights);

    console.log(this.simulationManager.pages, "page");
  }

  cells = new Array(this.nbSquares).fill(null);


  getHvacColor(page: number, i: number): string {
    if (this.simulationManager.pages[page].content[i].hvacStatus === 'ON') {
      return 'green';
    }
    else if (this.simulationManager.pages[page].content[i].hvacStatus === 'OFF') {
      return 'red';
    }
    else {
      return 'gray';
    }
  }

  switchPage(pageNumber: number) {
    this.currentPage = pageNumber;
    this.sharedService.changeCount(pageNumber);
  }

  openDialog(index: number) {
    this.dialog.open(DialogComponent, {
      data: index
    });

  }

}