import { Component, EventEmitter, Output } from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import { GridComponent } from '../grid/grid.component';


@Component({
  selector: 'app-grid-footer',
  templateUrl: './grid-footer.component.html',
  styleUrls: ['./grid-footer.component.scss']
})
export class GridFooterComponent {
  
  @Output() pageChangeFromFooter:EventEmitter<any> = new EventEmitter;
  
  currentPage: number = 1;
  maxPage: number = 35;

  // constructor(public gridComponent:GridComponent){
  //   super();
  // }

  setPageInput(): void {
    var page = (<HTMLInputElement>document.getElementById("myNumber")).value;
    this.currentPage = parseInt(page);
  }

  incrementPage(): void {
    if(this.currentPage < this.maxPage){
      this.currentPage += 1;
      console.log(this.currentPage);
    }
    // let decrementValue = 1;
    // this.pageChangeFromFooter.emit(decrementValue);
  }

  decrementPage(): void {
    if(this.currentPage > 1) {
      this.currentPage -= 1;
      console.log(this.currentPage);
    }
    // let decrementValue = -1;
    // this.pageChangeFromFooter.emit(decrementValue);
  }
  


}
function output() {
  throw new Error('Function not implemented.');
}

