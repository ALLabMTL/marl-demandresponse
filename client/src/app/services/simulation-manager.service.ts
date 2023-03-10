import { Injectable } from '@angular/core';
import { MatChip, MatChipSelectionChange } from '@angular/material/chips';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';

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

  maxPage: number;
  housesPerPage: number;
  pages: PageData[];
  pagesFiltered: PageData[];

  //sidebar
  isSortingSelected: boolean;
  isHvacChecked: boolean;
  isTempChecked: boolean;
  isFilteredHvac: boolean;
  isTempFiltered: boolean;
  hvacStatus: string;
  tempDiffValue: number;
  sortingOptionSelected: string;



  // houseData1: HouseData[];
  // houseData2: HouseData[];

  constructor() { 
      this.sidenavData = [];
      this.propertyNames = [];
      this.propertyValues = [];
      this.housesData = [];

      this.started = true;
      this.stopped = true;
      this.pages = [];
      this.pagesFiltered = [];
      this.houseDataFiltered = [];
      this.originalHousesData = [];
      this.maxPage = -1;
      this.housesPerPage = 100;

      this.isSortingSelected = false;
      this.isHvacChecked = false;
      this.isTempChecked = false;
      this.isFilteredHvac = false;
      this.isTempFiltered = false;
      this.hvacStatus = "ON";
      this.tempDiffValue = -1;
      this.sortingOptionSelected = " ";


      // this.houseData1 = [
      //   {    
      //   id: 101,
      //   hvacStatus: "Lockout",
      //   secondsSinceOff: 20,
      //   indoorTemp: 20,
      //   targetTemp: 20,
      //   tempDifference: 1},
      //   {    
      //     id: 102,
      //     hvacStatus: "OFF",
      //     secondsSinceOff: 21,
      //     indoorTemp: 21,
      //     targetTemp: 21,
      //     tempDifference: 2},
      //     {    
      //       id: 103,
      //       hvacStatus: "ON",
      //       secondsSinceOff: 22,
      //       indoorTemp: 22,
      //       targetTemp: 22,
      //       tempDifference: 3},
      // ]

      // this.houseData2 = [
      //   {    
      //   id: 201,
      //   hvacStatus: "Lockout",
      //   secondsSinceOff: 20,
      //   indoorTemp: 20,
      //   targetTemp: 20,
      //   tempDifference: 1},
      // ]
  }


  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length -1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length -1])
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data; // every house
    this.originalHousesData = data;

    if(this.isSortingSelected) {
      this.sortByOptionSelected(this.sortingOptionSelected);
    } else if(this.isFilteredHvac) {
      this.filterByHvacStatus(this.chipSelected)
    } else if(this.isTempFiltered) {
      this.filterByTempDiff();
    } 

    //rajouter les remove

    this.pages = [];
    this.maxPage = Math.ceil(this.housesData.length / this.housesPerPage);
    for (let i = 0; i < this.maxPage; i++) {
      const startIndex = i * this.housesPerPage;
      const endIndex = Math.min(startIndex + this.housesPerPage, this.housesData.length);
      const pageContent: HouseData[] = this.housesData.slice(startIndex, endIndex); // separate the data in pages 
      this.pages.push({ id: i + 1, content: pageContent });
    }
    // this.pages.push({id: 2, content: this.houseData1})
    // this.pages.push({id: 3, content: this.houseData2})
    // this.maxPage = 3;
    console.log(this.pages);
  }

  updateFilteredHouses(): void {
    // housesData will get updated with new values before calling
    this.housesData = this.originalHousesData;

    if(this.isSortingSelected) {
      this.housesData = this.housesData.filter(x => {this.houseDataFiltered.find(y => (x.id === y.id))})

      if(this.isFilteredHvac) {
        this.housesData = this.housesData.filter( x => { this.hvacChosen.find( y => y.hvacStatus == x.hvacStatus)})
      } else if(this.isTempFiltered)
    } 
  }

  separateHousesByPage(): void {
    this.pages = [];
    this.maxPage = Math.ceil(this.houseDataFiltered.length / this.housesPerPage);
    for (let i = 0; i < this.maxPage; i++) {
      const startIndex = i * this.housesPerPage;
      const endIndex = Math.min(startIndex + this.housesPerPage, this.houseDataFiltered.length);
      const pageContent: HouseData[] = this.houseDataFiltered.slice(startIndex, endIndex); // separate the data in pages 
      this.pages.push({ id: i + 1, content: pageContent });
    }

    console.log(this.pagesFiltered);
  }

  resetSimulation(): void {
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.started = false;
    this.stopped = true;
  }


  sortByTempDiffIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1: 1); // a negative, then a comes before b  

  }

  sortByTempDiffDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1: 1); // a negative, then a comes before b

  }

  sortByIndoorTempIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1: 1); // a negative, then a comes before b

  }

  sortByIndoorTempDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp > b.indoorTemp) ? -1: 1); // a negative, then a comes before b

  }

  sortByOptionSelected(option: string): void {
    this.sortingOptionSelected = option;
    
    switch(option) {
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
        break;
    }
    this.updateFilteredHouses();
  }



  removeSorting(): void {
    this.isSortingSelected = false;
    this.housesData = this.originalHousesData;
    //update grid
    //this.restoreHousesData();
  }

  hvacChosen: any[] = [];
  chipSelected: any;

  filterByHvacStatus(chip: MatChipSelectionChange): void {
    this.chipSelected = chip.selected;

    if(chip.selected) {
      this.hvacChosen = [...this.hvacChosen, {...this.housesData.filter(status => status.hvacStatus == this.hvacStatus)}];
    } else {
      this.hvacChosen = [...this.hvacChosen.filter(x => x.hvacStatus !== (this.housesData.filter(status => status.hvacStatus == this.hvacStatus)))];
    }
    this.isFilteredHvac = true;

    this.updateFilteredHouses();
  }

  removeHvacFilter(): void {
    this.isFilteredHvac = false;
    this.housesData = this.originalHousesData;
  }

  
  tempDiffHousesData: HouseData[] = [];
  filterByTempDiff(): void {
    this.isTempFiltered = true;

    this.tempDiffHousesData = this.housesData.filter(houses => houses.tempDifference == this.tempDiffValue);
    this.updateFilteredHouses();

  }

  removeTempDiffFilter(): void {
    this.isTempFiltered = false;
    this.housesData = this.originalHousesData;
  }

  resetActivityFilters(): void {
    this.isSortingSelected = false;
    this.isHvacChecked = false;
    this.isTempChecked = false;
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
