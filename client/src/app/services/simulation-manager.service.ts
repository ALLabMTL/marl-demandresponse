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
  houseDataFiltered: HouseData[];
  originalHousesData: HouseData[];
  started: boolean;
  stopped: boolean;
  speed: number = 2;

  //sidebar
  isSortingSelected: boolean;
  isHvacEnabled: boolean;
  isTempChecked: boolean;
  isFilteredHvac: boolean;
  isTempFiltered: boolean;
  hvacStatus: string;
  tempSelectRange: { min: number, max: number };
  tempSelectRangeInput: { min: number, max: number };
  minValueSliderInit: number;
  maxValueSliderInit: number;

  sortingOptionSelected: string;
  hvacChosen: HouseData[];
  tempDiffHousesData: HouseData[];
  isHvacChecked: boolean;
  isOnChecked: boolean;
  isOffChecked: boolean;
  isLockoutChecked: boolean;

  maxPage: number = 1;
  housesPerPage: number = 100;
  pages: PageData[];
  currentPage: number = 1;
  nbSquares = 100;

  // houseData1: HouseData[];
  // houseData2: HouseData[];

  constructor(public sharedService: SharedService) {
    this.sharedService.squareNbValue.subscribe(nbSquares => this.nbSquares = nbSquares);
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);


      this.sidenavData = [];
      this.propertyNames = [];
      this.propertyValues = [];
      this.housesData = [];

      this.started = true;
      this.stopped = true;
      this.pages = [];
      this.houseDataFiltered = [];
      this.originalHousesData = [];
      this.maxPage = -1;
      this.housesPerPage = 100;

      this.isSortingSelected = false;
      this.isHvacChecked = false;
      this.isTempChecked = false;
      this.isFilteredHvac = false;
      this.isTempFiltered = false;
      this.hvacStatus = " ";
      this.minValueSliderInit= -1;
      this.maxValueSliderInit= 1;
      this.tempSelectRange = {min: this.minValueSliderInit, max: this.maxValueSliderInit }
      this.tempSelectRangeInput = {min: this.minValueSliderInit, max: this.maxValueSliderInit }

      this.sortingOptionSelected = " ";
      this.hvacChosen = [];
      this.tempDiffHousesData = [];
      this.isHvacEnabled = false;
      this.isOnChecked = false;
      this.isOffChecked = false;
      this.isLockoutChecked = false;
  }


  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length - 1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length - 1])
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data; // every house
    console.log('houses data');
    console.log(this.housesData);
    this.originalHousesData = data;

    if (this.isSortingSelected) {
      this.sortByOptionSelected(this.sortingOptionSelected);
    } 
    if(this.isFilteredHvac) {
      this.filterByHvacStatus(this.isHvacChecked, this.hvacStatus);
    } 

    if (this.isTempFiltered) {
      this.filterByTempDiff();
    } 
    this.updateFilteredHouses();

    this.tempSelectRange.min = this.housesData.length > 0 ?
    Math.min(...this.housesData.map((data) => data.tempDifference)) :
    0;
    
    this.tempSelectRange.min = Number(this.tempSelectRange.min.toFixed(3));

    this.tempSelectRange.max = this.housesData.length > 0 ?
    Math.max(...this.housesData.map((data) => data.tempDifference)) : -1;

    this.tempSelectRange.max = Number(this.tempSelectRange.max.toFixed(3));

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

  updateSpeed(test: Event): void {
    throw new Error('Method not implemented.');
  }
  updateFilteredHouses(): void {
    // housesData will get updated with new values before calling
    this.housesData = this.originalHousesData;

    if (this.isSortingSelected) {
      this.housesData = this.housesData.filter(x => {
        return this.houseDataFiltered.find(y => y.id === x.id) !== undefined;
      });
    }  

    if(this.isFilteredHvac) {
      this.housesData = this.housesData.filter(x => {
        return this.hvacChosen.find(y => y.hvacStatus === x.hvacStatus) !== undefined;
      });        
    }

    if (this.isTempFiltered) {
      this.housesData = this.housesData.filter(x => {
        return this.tempDiffHousesData.find(y => y.tempDifference === x.tempDifference) !== undefined;
      }); 
    }

    // if(this.housesData.length === 0) {
    //   this.sharedService.changeCount(0);
    // }
     if(this.currentPage > this.maxPage) {
      this.sharedService.changeCount(1);
    } else {
      this.sharedService.changeCount(this.currentPage);
    }
  }

  sortByTempDiffIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1 : 1); // a negative, then a comes before b
  }

  sortByTempDiffDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1 : 1); // a negative, then a comes before b

  }

  sortByIndoorTempIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1 : 1); // a negative, then a comes before b
  }

  sortByIndoorTempDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp > b.indoorTemp) ? -1 : 1); // a negative, then a comes before b

  }

  sortByOptionSelected(option: string): void {
    this.sortingOptionSelected = option;

    switch (option) {
      case "indoorTempInc":
        this.sortByIndoorTempIncreasing();
        break;
      case "indoorTempDec":
        this.sortByIndoorTempDecreasing();
        break;
      case "tempDiffInc":
        this.sortByTempDiffIncreasing();
        break;
      case "tempDiffDec":
        this.sortByTempDiffDecreasing();
        break;
      case "noSorting":
        this.removeSorting();
        break;
    }

    this.updateFilteredHouses();
  }



  removeSorting(): void {
    this.isSortingSelected = false;
    this.housesData = this.originalHousesData;
  }

  filterByHvacStatus(checked: boolean, hvac: string): void {
    this.housesData = this.originalHousesData;

    this.hvacStatus = hvac;
    this.isHvacChecked = checked;

    if(this.isHvacChecked) {
      this.hvacChosen = [...this.hvacChosen, ...this.housesData.filter(status => status.hvacStatus == this.hvacStatus)];
    } else { // if un-select manually
      this.hvacChosen = [...this.hvacChosen.filter(x => x.hvacStatus !== this.hvacStatus)];
    }

    this.isFilteredHvac = true;  

    this.updateFilteredHouses(); 
    
    if(this.isOnChecked == false && this.isOffChecked == false && this.isLockoutChecked == false) {
      this.removeHvacFilter();
    }
  }

  removeHvacFilter(): void {
    this.housesData = this.originalHousesData;
    this.isFilteredHvac = false;
  }

  filterByTempDiff(): void {
    this.isTempFiltered = true;
    this.housesData = this.originalHousesData;

    this.tempDiffHousesData = this.housesData.filter((e) => e.tempDifference >= this.tempSelectRangeInput.min && e.tempDifference <= this.tempSelectRangeInput.max)

    this.updateFilteredHouses();
  }

  removeTempDiffFilter(): void {
    this.isTempFiltered = false;
    this.housesData = this.originalHousesData;
  }

}
