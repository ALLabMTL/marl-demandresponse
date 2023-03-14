import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';
import { SharedService } from './shared/shared.service';

interface PageData {
  id: number;
  content: HouseData[];
}

@Injectable({
  providedIn: 'root'
})
export class SimulationManagerService {

  sidenavData: SidenavData[];
  propertyNames: string[];
  propertyValues: string[];
  housesData: HouseData[];
  started: boolean;
  stopped: boolean;

  maxPage: number = 1;
  housesPerPage: number = 100;
  pages: PageData[];
  nbSquares = 100;

  // houseData1: HouseData[];
  // houseData2: HouseData[];

  constructor(public sharedService: SharedService) {
    this.sharedService.squareNbValue.subscribe(nbSquares => this.nbSquares = nbSquares);
      this.sidenavData = [];
      this.propertyNames = [];
      this.propertyValues = [];
      this.housesData = [];
      this.started = true;
      this.stopped = true;
      this.pages = [];
  }


  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length -1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length -1])
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data;
    this.pages = [];
    this.maxPage = Math.ceil(this.housesData.length / this.nbSquares);
    for (let i = 0; i < this.maxPage; i++) {
      const startIndex = i * this.nbSquares;
      const endIndex = Math.min(startIndex + this.nbSquares, this.housesData.length);
      const pageContent: HouseData[] = this.housesData.slice(startIndex, endIndex);
      this.pages.push({ id: i + 1, content: pageContent });
    }

  }

  resetSimulation(): void {
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.pages = [];
    this.started = false;
    this.stopped = true;
  }

  // updatePages() {
  //   this.maxPage = Math.ceil(this.housesData.length / this.housesPerPage);
  //   for (let i = 0; i < this.maxPage; i++) {
  //     const startIndex = i * this.housesPerPage;
  //     const endIndex = Math.min(startIndex + this.housesPerPage, this.housesData.length);
  //     const pageContent: HouseData[] = this.housesData.slice(startIndex, endIndex);
  //     this.pages.push({ id: i + 1, content: pageContent });
  //   }
  // }

}
