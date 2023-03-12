import { Injectable } from '@angular/core';
import { MatChipSelectionChange } from '@angular/material/chips';
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

  //sidebar
  isSortingSelected: boolean;
  isHvacChecked: boolean;
  isTempChecked: boolean;
  isFilteredHvac: boolean;
  isTempFiltered: boolean;
  hvacStatus: string;
  tempSelectRange : {min: number, max:number};
  sortingOptionSelected: string;
  hvacChosen: HouseData[];
  chipSelected: any;
  tempDiffHousesData: HouseData[];

  constructor() { 
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
      this.tempSelectRange = {min: -3, max:3 }
      this.sortingOptionSelected = " ";
      this.hvacChosen = [];
      this.chipSelected = false;
      this.tempDiffHousesData = [];
  }


  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length -1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length -1])
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data; // every house
    console.log('houses data');
    console.log(this.housesData);
    this.originalHousesData = data;

    if(this.isSortingSelected) {
      this.sortByOptionSelected(this.sortingOptionSelected);
    } 
    if(this.isFilteredHvac) {
      this.filteredByHvacStatus();
    } 

    if(this.isTempFiltered) {
      this.filterByTempDiff();
    } 

    this.pages = [];
    this.maxPage = Math.ceil(this.housesData.length / this.housesPerPage);
    for (let i = 0; i < this.maxPage; i++) {
      const startIndex = i * this.housesPerPage;
      const endIndex = Math.min(startIndex + this.housesPerPage, this.housesData.length);
      const pageContent: HouseData[] = this.housesData.slice(startIndex, endIndex); // separate the data in pages 
      this.pages.push({ id: i + 1, content: pageContent });
    }

    console.log(this.pages);
  }

  resetSimulation(): void {
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.started = false;
    this.stopped = true;
  }

  updateFilteredHouses(): void {
    // housesData will get updated with new values before calling
    this.housesData = this.originalHousesData;

    if(this.isSortingSelected) {
      this.housesData = this.housesData.filter(x => {
        return this.houseDataFiltered.find(y => y.id === x.id) !== undefined;
      });
      console.log('sorting filtered', this.housesData);
    }  

    if(this.isFilteredHvac) {
        this.housesData = this.housesData.filter(x => {
          return this.hvacChosen.find(y => y.hvacStatus === x.hvacStatus) !== undefined;
        });   
        console.log('hvac filtered', this.housesData);
    }

    if(this.isTempFiltered) {
      this.housesData = this.housesData.filter(x => {
        return this.tempDiffHousesData.find(y => y.tempDifference === x.tempDifference) !== undefined;
      });
      
    }
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
        console.log("indoortemp inc");
        break;
      case "indoorTempDec":
        this.sortByIndoorTempDecreasing();
        console.log("indoortemp dec");
        break;
      case "tempDiffInc":
        this.sortByTempDiffIncreasing();
        console.log("tempdiff inc");
        break;
      case "tempDiffDec":
        this.sortByTempDiffDecreasing();
        console.log("tempdiff dec");
        break;
      case "noSorting":
        console.log("no sorting");
        break;
    }

    this.updateFilteredHouses();
  }



  removeSorting(): void {
    this.isSortingSelected = false;
    this.housesData = this.originalHousesData;
  }


  filterByHvacStatus(event: MatChipSelectionChange): void {
    this.chipSelected = event.source.selected;
    this.housesData = this.originalHousesData;
    this.hvacStatus = event.source.value;

    console.log('hvac status:', this.hvacStatus)

    // if(this.chipSelected) {
    //   this.hvacChosen = [...this.hvacChosen, {...this.housesData.filter(status => status.hvacStatus == this.hvacStatus)}];
    // } else {
    //   this.hvacChosen = [...this.hvacChosen.filter(x => x.hvacStatus !== (this.housesData.filter(status => status.hvacStatus == this.hvacStatus)))];
    // }

    if(this.chipSelected) {
      this.hvacChosen = this.housesData.filter(status => status.hvacStatus == this.hvacStatus);
    } else { // if un-select manually
      this.hvacChosen = this.hvacChosen.filter(x => x.hvacStatus !== this.hvacStatus);
    }
    this.isFilteredHvac = true;

    console.log('hvac chosen')
    console.log(this.hvacChosen);
    this.updateFilteredHouses();
  }

  filteredByHvacStatus(): void {
    console.log('hvac status:', this.hvacStatus)
    this.housesData = this.originalHousesData;
    // if(this.chipSelected) {
    //   this.hvacChosen = [...this.hvacChosen, {...this.housesData.filter(status => status.hvacStatus == this.hvacStatus)}];
    // } else {
    //   this.hvacChosen = [...this.hvacChosen.filter(x => x.hvacStatus !== (this.housesData.filter(status => status.hvacStatus == this.hvacStatus)))];
    // }

    if(this.chipSelected) {
      this.hvacChosen = this.housesData.filter(status => status.hvacStatus == this.hvacStatus);
    } else { // if un-select manually
      this.hvacChosen = this.hvacChosen.filter(x => x.hvacStatus !== this.hvacStatus);
    }
    this.isFilteredHvac = true;

    console.log('hvac chosen')
    console.log(this.hvacChosen);
    this.updateFilteredHouses();
  }

  removeHvacFilter(): void {
    this.isFilteredHvac = false;
    this.housesData = this.originalHousesData;
  }

  filterByTempDiff(): void {
    this.isTempFiltered = true;
    this.housesData = this.originalHousesData;

    this.tempDiffHousesData = this.housesData.filter((e) => e.tempDifference >= this.tempSelectRange.min && e.tempDifference <= this.tempSelectRange.max)
    console.log('temp diff filtered', this.tempDiffHousesData)

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

}
