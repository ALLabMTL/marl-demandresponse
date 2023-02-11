import { Component, Input, OnInit} from '@angular/core';
import { SharedService } from '@app/services/shared/shared.service';
import {MatDialog} from '@angular/material/dialog';
import { DialogComponent } from '../dialog/dialog.component';

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

  constructor(private sharedService: SharedService, public dialog: MatDialog){
    this.pages = [
      { id: 1, content: "Page 1" },
      { id: 2, content: "Page 2" },
      { id: 3, content: "Page 3" },
    ]
  }

  ngOnInit() {
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
    this.currentPage = this.pages[0].id;
  }

  switchPage(id: number) {
    this.currentPage += 1;
  }

  openDialog() {
    this.dialog.open(DialogComponent);
  }

}
