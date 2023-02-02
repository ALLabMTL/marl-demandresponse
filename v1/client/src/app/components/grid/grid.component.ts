import { Component, HostListener, Input, OnInit} from '@angular/core';
import { IconService } from '@visurel/iconify-angular';

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

  constructor(){
    this.pages = [
      { id: 1, content: "Page 1" },
      { id: 2, content: "Page 2" },
      { id: 3, content: "Page 3" },
    ]
  }

  ngOnInit() {
    this.currentPage = this.pages[0].id;
  }

  // pageChangeFromFooter(data:any){
  //   console.log(data);
  //   this.currentPage = this.currentPage - data;
  // }

  switchPage(id: number) {
    this.currentPage += 1;
  }

}
