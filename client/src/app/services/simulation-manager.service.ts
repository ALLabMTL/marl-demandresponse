import { Injectable } from '@angular/core';
import { MatChip, MatChipSelectionChange } from '@angular/material/chips';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';

interface PageData {
  id: number;
  content: HouseData[];
}

interface Filter {
  id: number;
  type: string;
  value?: any;
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
  filters: Filter[] = [];
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

    if(this.isSortingSelected) {
      this.sortByOptionSelected(this.sortingOptionSelected);
    }

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

  // sortByTempDiffIncreasing(): void {
  //   this.isFilteredSorted = true;
  //   this.isSortingSelected = true;
  //   if(this.isHvacChecked || this.isTempChecked){
  //     this.houseDataFiltered = this.houseDataFiltered.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1: 1); // a negative, then a comes before b
  //   } else {
  //     this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1: 1); // a negative, then a comes before b  
  //   }
  //   this.filters.push({id: 0, type: "tempDiffInc"});
  //   this.separateHousesByPage();
  // }

  sortByTempDiffIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1: 1); // a negative, then a comes before b  

  }

  // sortByTempDiffDecreasing(): void {
  //   this.isFilteredSorted = true;
  //   this.isSortingSelected = true;
  //   if(this.isHvacChecked || this.isTempChecked){
  //     this.houseDataFiltered = this.houseDataFiltered.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1: 1); // a negative, then a comes before b
  //   } else {
  //     this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1: 1); // a negative, then a comes before b
  //   }
  //   this.filters.push({id: 1, type: "tempDiffDec"});
  //   this.separateHousesByPage();

  // }

  sortByTempDiffDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1: 1); // a negative, then a comes before b

  }

  // sortByIndoorTempIncreasing(): void {
  //   this.isFilteredSorted = true;
  //   this.isSortingSelected = true;
  //   if(this.isHvacChecked || this.isTempChecked){
  //     this.houseDataFiltered = this.houseDataFiltered.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1: 1); // a negative, then a comes before b
  //   } else {
  //     this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1: 1); // a negative, then a comes before b
  //   }
  //   this.filters.push({id: 2, type: "indoorTempInc"});
  //   this.separateHousesByPage();

  // }

  sortByIndoorTempIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1: 1); // a negative, then a comes before b

  }

  // sortByIndoorTempDecreasing(): void {
  //   this.isFilteredSorted = true;
  //   this.isSortingSelected = true;
  //   if(this.isHvacChecked || this.isTempChecked){
  //     this.houseDataFiltered = this.houseDataFiltered.sort((a, b) => (a.indoorTemp > b.indoorTemp) ? -1: 1); // a negative, then a comes before b
  //   } else {
  //     this.houseDataFiltered = this.housesData.sort((a, b) => (a.indoorTemp > b.indoorTemp) ? -1: 1); // a negative, then a comes before b
  //   }
  //   this.filters.push({id: 3, type: "indoorTempDec"});
  //   this.separateHousesByPage();
  // }

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
    this.updateFilteredHouses(this.houseDataFiltered);
  }

  updateFilteredHouses(house: HouseData[]): void {
    this.housesData = this.housesData.filter(x => {house.find(y => (x.id === y.id))})
  }

  removeSorting(): void {
    this.isSortingSelected = false;
    //update grid
    //this.restoreHousesData();
  }

  // filterByHvacStatus(): void {
  //   this.isFilteredSorted = true;
  //   if(this.isSortingSelected || this.isTempChecked) {
  //     if(this.hvacStatus == "ON") {
  //       this.houseDataFiltered = this.houseDataFiltered.filter(status => status.hvacStatus == "ON");
  //     } else if(this.hvacStatus == "Lockout") {
  //       this.houseDataFiltered = this.houseDataFiltered.filter(status => status.hvacStatus == "Lockout");
  //     } else if(this.hvacStatus == "OFF") {
  //       this.houseDataFiltered = this.houseDataFiltered.filter(status => status.hvacStatus == "OFF");
  //     }
  //   } else {
  //     if(this.hvacStatus == "ON") {
  //       this.houseDataFiltered = this.housesData.filter(status => status.hvacStatus == "ON");
  //     } else if(this.hvacStatus == "Lockout") {
  //       this.houseDataFiltered = this.housesData.filter(status => status.hvacStatus == "Lockout");
  //     } else if(this.hvacStatus == "OFF") {
  //       this.houseDataFiltered = this.housesData.filter(status => status.hvacStatus == "OFF");
  //     }

  //   }
  //   this.filters.push({id: 4, type: "hvacStatus", value: this.hvacStatus});
  //   this.separateHousesByPage();
  // }
  hvacChosen: any[] = [];

  filterByHvacStatus(chip: MatChipSelectionChange): void {

    if(chip.selected) {
      this.hvacChosen = [...this.hvacChosen, {...this.housesData.filter(status => status.hvacStatus == this.hvacStatus)}];
    } else {
      this.hvacChosen = [...this.hvacChosen.filter(x => x.hvacStatus !== (this.housesData.filter(status => status.hvacStatus == this.hvacStatus)))];
    }
    this.isFilteredHvac = true;
    // if(this.hvacStatus == "ON") {
    //   this.houseDataFiltered = this.housesData.filter(status => status.hvacStatus == "ON");
    // } else if(this.hvacStatus == "Lockout") {
    //   this.houseDataFiltered = this.housesData.filter(status => status.hvacStatus == "Lockout");
    // } else if(this.hvacStatus == "OFF") {
    //   this.houseDataFiltered = this.housesData.filter(status => status.hvacStatus == "OFF");
    // }
    this.updateFilteredHouses(this.houseDataFiltered);
  }

  removeHvacFilter(): void {
    this.isFilteredHvac = false;
  }

  // filterByTempDiff(): void {
  //   this.isFilteredSorted = true;
  //   if(this.isSortingSelected || this.isHvacChecked) {
  //     this.houseDataFiltered = this.houseDataFiltered.filter(houses => houses.tempDifference == this.tempDiffValue);
  //   } else {
  //     this.houseDataFiltered = this.housesData.filter(houses => houses.tempDifference == this.tempDiffValue);
  //   }
  //   this.filters.push({id: 5, type: "tempDiffValue", value: this.tempDiffValue});
  //   this.separateHousesByPage();
  // }

  
  filterByTempDiff(): void {
    this.isTempFiltered = true;

    this.houseDataFiltered = this.housesData.filter(houses => houses.tempDifference == this.tempDiffValue);
    this.updateFilteredHouses(this.houseDataFiltered);

  }

  removeTempDiffFilter(): void {
    this.isTempFiltered = false;
  }

  resetActivityFilters(): void {
    this.isSortingSelected = false;
    this.isHvacChecked = false;
    this.isTempChecked = false;
  }

  // // called when any toggle is turned off and form field no sorting is selected
  // restoreHousesData(): void {
  //   //if()
  //   //this.simulationManager.housesData = this.simulationManager.originalHousesData;
  //   const filterTypes = this.filters.map(filter => filter.type);

  //   if(this.isSortingSelected && this.isHvacChecked) {
  //     if (filterTypes.includes('hvacStatus') && filterTypes.includes(this.sortingOptionSelected)) {
        
  //       this.resetActivityFilters();

  //        const hvacFilter = this.filters.find(filter => filter.type === 'hvacStatus');
  //        this.hvacStatus = hvacFilter?.value;
  //        this.filterByHvacStatus();

  //        //const sortFilter = this.filters.find(filter => filter.type === this.sortingOptionSelected);
  //        this.sortByOptionSelected(this.sortingOptionSelected);

  //     }
  //   } else if(this.isSortingSelected && this.isTempChecked) {

  //   } else if(this.isTempChecked && this.isHvacChecked){

  //   } else if(this.isHvacChecked) {

  //   } else if(this.isTempChecked) {

  //   } else if(this.isSortingSelected) {

  //   } else {
  //       this.isFilteredSorted = false;
  //   }
  // }


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
