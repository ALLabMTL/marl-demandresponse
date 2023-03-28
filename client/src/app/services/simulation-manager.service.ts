import { Injectable } from '@angular/core';
import { MatChipSelectionChange } from '@angular/material/chips';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';
import { SharedService } from './shared/shared.service';

interface PageData {
  id: number;
  content: HouseData[];
}

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
  isHvacChecked: boolean;
  isTempChecked: boolean;
  isFilteredHvac: boolean;
  isTempFiltered: boolean;
  hvacStatus: string;
  tempSelectRange: { min: number, max: number };
  sortingOptionSelected: string;
  hvacChosen: HouseData[];
  chipSelected: any;
  tempDiffHousesData: HouseData[];
  // isChipSelected: boolean;
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
    this.tempSelectRange = { min: -1, max: 1 }
    this.sortingOptionSelected = " ";
    this.hvacChosen = [];
    this.chipSelected = false;
    this.tempDiffHousesData = [];
    // this.isChipSelected = false;

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
    if (this.isFilteredHvac) {
      this.filteredByHvacStatus();
    }

    if (this.isTempFiltered) {
      this.filterByTempDiff();
    }

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

    if (this.isFilteredHvac) {
      this.housesData = this.housesData.filter(x => {
        return this.hvacChosen.find(y => y.hvacStatus === x.hvacStatus) !== undefined;
      });
    }

    if (this.isTempFiltered) {
      this.housesData = this.housesData.filter(x => {
        return this.tempDiffHousesData.find(y => y.tempDifference === x.tempDifference) !== undefined;
      });

    }
    if (this.housesData.length === 0) {
      this.sharedService.changeCount(0);
    } else {
      this.sharedService.changeCount(1);
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


  filterByHvacStatus(event: MatChipSelectionChange): void {
    this.chipSelected = event.source.selected;
    this.housesData = this.originalHousesData;
    this.hvacStatus = event.source.value;

    if (this.chipSelected) {
      this.hvacChosen = this.housesData.filter(status => status.hvacStatus == this.hvacStatus);
    } else { // if un-select manually
      this.hvacChosen = this.hvacChosen.filter(x => x.hvacStatus !== this.hvacStatus);
    }
    this.isFilteredHvac = true;

    this.updateFilteredHouses();
  }

  filteredByHvacStatus(): void {
    this.housesData = this.originalHousesData;

    if (this.chipSelected) {
      this.hvacChosen = this.housesData.filter(status => status.hvacStatus == this.hvacStatus);
    } else { // if un-select manually
      this.hvacChosen = this.hvacChosen.filter(x => x.hvacStatus !== this.hvacStatus);
    }
    this.isFilteredHvac = true;

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
