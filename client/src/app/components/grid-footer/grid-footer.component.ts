import { Component, EventEmitter, Input, Output } from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import { SharedService } from '@app/services/shared/shared.service';
import { SimulationManagerService } from '@app/services/simulation-manager.service';


@Component({
  selector: 'app-grid-footer',
  templateUrl: './grid-footer.component.html',
  styleUrls: ['./grid-footer.component.scss']
})
export class GridFooterComponent {
  
  
  @Output() pageChangeFromFooter:EventEmitter<any> = new EventEmitter;
  
  currentPage: number = 1;
  maxPage: number = 35;

  nbSquares = 81;
  nbSquareOptions = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

  constructor(private sharedService: SharedService, public simulationManager: SimulationManagerService){
  }

  ngOnInit() {
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
    this.sharedService.squareNbValue.subscribe(nbSquares => this.nbSquares = nbSquares);
  }

  setSquareNb(): void {

  }

  setNbSquare(): void {
    const square = (<HTMLInputElement>document.getElementById("squareNumber")).value;
    this.sharedService.changeSquareNb(parseInt(square));
  }

  setPageInput(): void {
    const page = (<HTMLInputElement>document.getElementById("myNumber")).value;
    this.sharedService.changeCount(parseInt(page));
  }

  incrementPage(): void {
    if(this.currentPage < this.maxPage){
      this.sharedService.changeCount(this.currentPage + 1);
    }
  }

  decrementPage(): void {
    if(this.currentPage > 1) {
      this.sharedService.changeCount(this.currentPage - 1);
    }
  }
  
}

