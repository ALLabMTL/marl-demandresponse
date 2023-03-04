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

  constructor(public sharedService: SharedService, public dialog: MatDialog, public simulationManager: SimulationManagerService) {
    this.pages = []
  }

  ngOnInit() {
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
    // this.maxPage = Math.ceil(this.simulationManager.housesData.length / this.housesPerPage);
    // this.updatePages();
    this.sharedService.currentPrecisionValue.subscribe(houseColorPrecisionValue => this.precisionValueSelected = houseColorPrecisionValue);
    // console.log(this.simulationManager.housesData.length, "my length--------");
    // console.log(this.pages);
    console.log(this.simulationManager.pages, "page");
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
    else if (this.simulationManager.housesData[i].hvacStatus === 'OFF') {
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

  // updatePages() {
  //   const pages = [];
  //   for (let i = 0; i < this.maxPage; i++) {
  //     const startIndex = i * this.housesPerPage;
  //     const endIndex = Math.min(startIndex + this.housesPerPage, this.simulationManager.housesData.length);
  //     const pageContent: HouseData[] = this.simulationManager.housesData.slice(startIndex, endIndex);
  //     pages.push({ id: i + 1, content: pageContent });
  //   }
  //   return pages;
  // }

  openDialog(index: number) {
    this.dialog.open(DialogComponent, {
      data: index
    });

  }

}