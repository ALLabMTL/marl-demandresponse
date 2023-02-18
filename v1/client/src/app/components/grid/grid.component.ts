import { Component, Input, OnInit} from '@angular/core';
import { SharedService } from '@app/services/shared/shared.service';
import {MatDialog} from '@angular/material/dialog';
import { DialogComponent } from '../dialog/dialog.component';
import { SimulationManagerService } from '@app/services/simulation-manager.service';


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

  constructor(private sharedService: SharedService, public dialog: MatDialog, public simulationManager: SimulationManagerService){
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

  cells = new Array(100).fill(null);

  getColumnWidths() {
    return `repeat(10, ${100 / 10}%)`;
  }

  getRowHeights() {
    return `repeat(10, ${100 / 10}%)`;
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
