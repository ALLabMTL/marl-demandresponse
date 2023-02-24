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
  currentPage = 1;
  maxPage = 35;
  precisionValueSelected = 0;

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

  // houseColor(data: number) {
  //   let upperBound = this.precisionValueSelected;
  //   let middleUpperBound = this.precisionValueSelected/2;
  //   let center = 0
  //   let middleLowerBound = -(this.precisionValueSelected/2)
  //   let lowerBound = -(this.precisionValueSelected);
  //   let boundRange = upperBound - middleUpperBound;
  //   console.log(data);
  //   console.log(upperBound);
  //   console.log(middleUpperBound);
  //   console.log(center);
  //   console.log(middleLowerBound);
  //   console.log(lowerBound);


  //   if(data < lowerBound){
  //     return "rgba(0, 0, 255, 100)";
  //   }
  //   else if (lowerBound <= data && data < middleLowerBound) {
  //     let temp = -(lowerBound - data)/boundRange;
  //     let color = temp * 255;
  //     return "rgba(0," + color + ", 255, 100)";
  //   }
  //   else if (middleLowerBound <= data && data < center) {
  //     let temp = (boundRange + (middleLowerBound - data))/boundRange;
  //     let color = temp * 255;
  //     return "rgba(0, 255," + color + ", 100)";
  //   }
  //   else if (center <= data && data < middleUpperBound) {
  //     let temp = (boundRange - (middleUpperBound - data))/boundRange;
  //     let color = temp * 255;
  //     return "rgba("+ color + ",255, 0, 100)";
  //   }
  //   else if (middleUpperBound <= data && data <= upperBound) {
  //     let temp = (upperBound - data)/boundRange;
  //     let color = temp * 255;
  //     return "rgba(255," + color + ", 0, 100)";
  //   }
  //   else {
  //     return "rgba(255, 0, 0, 100)";
  //   }

  // }

}
