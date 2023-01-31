import { Component } from '@angular/core';
import {MatIconModule} from '@angular/material/icon';


@Component({
  selector: 'app-grid-footer',
  templateUrl: './grid-footer.component.html',
  styleUrls: ['./grid-footer.component.scss']
})
export class GridFooterComponent {

  currentPage: number = 1;
  maxPage: number = 35;

  incrementPage(): void {
    if(this.currentPage < this.maxPage){
      this.currentPage += 1;
    }
  }

  decrementPage(): void{
    if(this.currentPage > 1) {
      this.currentPage -= 1;
    }
    //(<HTMLInputElement>document.getElementById("myNumber")).value = "16";
  }


}
