import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';
import { Observable, Subject } from 'rxjs';
import { SharedService } from './shared/shared.service';

interface PageData {
  id: number;
  content: HouseData[];
}

@Injectable({
  providedIn: 'root',
})
export class SimulationManagerService {
  sidenavData: SidenavData[] = [];
  sidenavObservable: Subject<SidenavData[]> = new Subject<SidenavData[]>();
  houseDataObservable: Subject<HouseData[][]> = new Subject<HouseData[][]>();

  agentName = '';
  housesBufferData: HouseData[][] = [];
  propertyNames: string[] = [];
  propertyValues: string[] = [];
  housesData: HouseData[] = [];
  houseDataFiltered: HouseData[] = [];
  originalHousesData: HouseData[] = [];
  started = false;
  stopped = true;
  connected: boolean;
  speed = '4';
  nbTimeSteps = 0;
  step = 4;
  currentTimeStep = 0;
  paused = false;

  //sidebar
  isSortingSelected = false;
  isHvacEnabled = false;
  isTempChecked = false;
  isFilteredHvac = false;
  isTempFiltered = false;
  hvacStatus = ' ';
  tempSelectRange: { min: number; max: number } = {
    min: 0,
    max: 0,
  };
  tempSelectRangeInput: { min: number; max: number } = {
    min: 0,
    max: 0,
  };

  sortingOptionSelected = ' ';
  hvacChosen: HouseData[] = [];
  tempDiffHousesData: HouseData[] = [];
  isHvacChecked = false;
  isOnChecked = false;
  isOffChecked = false;
  isLockoutChecked = false;

  maxPage = 1;
  housesPerPage = 100;
  pages: PageData[] = [];
  currentPage = 1;
  nbSquares = 100;

  // houseData1: HouseData[];
  // houseData2: HouseData[];

  constructor(public sharedService: SharedService) {
    this.sharedService.squareNbValue.subscribe(
      (nbSquares) => (this.nbSquares = nbSquares)
    );
    this.sharedService.currentPageCount.subscribe(
      (currentPage) => (this.currentPage = currentPage)
    );
    this.connected = false;
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];

    this.started = false;
    this.stopped = true;
    this.pages = [];
    this.houseDataFiltered = [];
    this.originalHousesData = [];
    this.maxPage = 1;
    this.housesPerPage = 100;

    this.isSortingSelected = false;
    this.isHvacChecked = false;
    this.isTempChecked = false;
    this.isFilteredHvac = false;
    this.isTempFiltered = false;
    this.hvacStatus = ' ';
    this.tempSelectRange = {
      // to get the min max from original houses data
      min: 0,
      max: 0,
    };
    this.tempSelectRangeInput = {
      // interval chosen on interface
      min: 0,
      max: 0,
    };

    this.sortingOptionSelected = ' ';
    this.hvacChosen = [];
    this.tempDiffHousesData = [];
    this.isHvacEnabled = false;
    this.isOnChecked = false;
    this.isOffChecked = false;
    this.isLockoutChecked = false;
  }

  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.sidenavObservable.next(this.sidenavData);
    this.housesBufferData.push(this.housesData);
    this.houseDataObservable.next(this.housesBufferData);
    this.propertyNames = Object.getOwnPropertyNames(
      this.sidenavData[this.sidenavData.length - 1]
    );
    this.propertyValues = Object.values(
      this.sidenavData[this.sidenavData.length - 1]
    );

    this.nbTimeSteps++;
    this.currentTimeStep = this.nbTimeSteps;
    this.setTimeStep(data);
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data; // every house
    this.originalHousesData = data;
    this.updateFilter();
  }

  updateFilter(): void {
    if (this.isSortingSelected) {
      this.sortByOptionSelected(this.sortingOptionSelected);
    }
    if (this.isFilteredHvac) {
      this.filterByHvacStatus(this.isHvacChecked, this.hvacStatus);
    }

    if (this.isTempFiltered) {
      this.filterByTempDiff();
    }
    this.updateFilteredHouses();

    this.tempSelectRange.min =
      this.originalHousesData.length > 0
        ? Math.min(
            ...this.originalHousesData.map((data) => data.tempDifference)
          )
        : 0;

    this.tempSelectRange.min = Number(this.tempSelectRange.min.toFixed(3));

    this.tempSelectRange.max =
      this.originalHousesData.length > 0
        ? Math.max(
            ...this.originalHousesData.map((data) => data.tempDifference)
          )
        : 0;

    this.tempSelectRange.max = Number(this.tempSelectRange.max.toFixed(3));

    this.pages = [];
    this.maxPage = Math.ceil(this.housesData.length / this.nbSquares);
    for (let i = 0; i < this.maxPage; i++) {
      const startIndex = i * this.nbSquares;
      const endIndex = Math.min(
        startIndex + this.nbSquares,
        this.housesData.length
      );
      const pageContent: HouseData[] = this.housesData.slice(
        startIndex,
        endIndex
      );
      this.pages.push({ id: i + 1, content: pageContent });
    }
  }

  setTimeStep(data: SidenavData): void {
    this.propertyNames = Object.getOwnPropertyNames(data);
    this.propertyValues = Object.values(data);
  }

  reset(): void {
    this.started = true;
    if (this.stopped) {
      this.nbTimeSteps = 0;
      this.currentTimeStep = 0;
      this.propertyNames = [];
      this.propertyValues = [];
      this.housesData = [];
      this.sidenavData = [];
      this.housesBufferData = [];
    }
    this.stopped = false;
    this.paused = false;
  }

  updateSpeed(test: Event): void {
    throw new Error('Method not implemented.');
  }
  updateFilteredHouses(): void {
    // housesData will get updated with new values before calling
    this.housesData = this.originalHousesData;

    if (this.isSortingSelected) {
      this.housesData = this.housesData.filter((x) => {
        return this.houseDataFiltered.find((y) => y.id === x.id) !== undefined;
      });
    }

    if (this.isFilteredHvac) {
      this.housesData = this.housesData.filter((x) => {
        return (
          this.hvacChosen.find((y) => y.hvacStatus === x.hvacStatus) !==
          undefined
        );
      });
    }

    if (this.isTempFiltered) {
      this.housesData = this.housesData.filter((x) => {
        return (
          this.tempDiffHousesData.find(
            (y) => y.tempDifference === x.tempDifference
          ) !== undefined
        );
      });
    }

    // if(this.housesData.length === 0) {
    //   this.sharedService.changeCount(0);
    // }
    if (this.currentPage > this.maxPage) {
      this.sharedService.changeCount(1);
    } else {
      this.sharedService.changeCount(this.currentPage);
    }
  }

  sortByTempDiffIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.originalHousesData.sort((a, b) =>
      a.tempDifference < b.tempDifference ? -1 : 1
    ); // a negative, then a comes before b
  }

  sortByTempDiffDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.originalHousesData.sort((a, b) =>
      a.tempDifference > b.tempDifference ? -1 : 1
    ); // a negative, then a comes before b
  }

  sortByIndoorTempIncreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.originalHousesData.sort((a, b) =>
      a.indoorTemp < b.indoorTemp ? -1 : 1
    ); // a negative, then a comes before b
  }

  sortByIndoorTempDecreasing(): void {
    this.isSortingSelected = true;

    this.houseDataFiltered = this.originalHousesData.sort((a, b) =>
      a.indoorTemp > b.indoorTemp ? -1 : 1
    ); // a negative, then a comes before b
  }

  sortByOptionSelected(option: string): void {
    this.sortingOptionSelected = option;

    switch (option) {
      case 'indoorTempInc':
        this.sortByIndoorTempIncreasing();
        break;
      case 'indoorTempDec':
        this.sortByIndoorTempDecreasing();
        break;
      case 'tempDiffInc':
        this.sortByTempDiffIncreasing();
        break;
      case 'tempDiffDec':
        this.sortByTempDiffDecreasing();
        break;
      case 'noSorting':
        this.removeSorting();
        break;
    }

    this.updateFilteredHouses();
  }

  removeSorting(): void {
    this.isSortingSelected = true;
    this.houseDataFiltered = this.originalHousesData.sort((a, b) =>
      a.id < b.id ? -1 : 1
    ); // a negative, then a comes before b
  }

  filterByHvacStatus(checked: boolean, hvac: string): void {
    this.housesData = this.originalHousesData;

    this.hvacStatus = hvac;
    this.isHvacChecked = checked;

    if (this.isHvacChecked) {
      this.hvacChosen = [
        ...this.hvacChosen,
        ...this.housesData.filter(
          (status) => status.hvacStatus == this.hvacStatus
        ),
      ];
    } else {
      // if un-select manually
      this.hvacChosen = [
        ...this.hvacChosen.filter((x) => x.hvacStatus !== this.hvacStatus),
      ];
    }

    this.isFilteredHvac = true;

    this.updateFilteredHouses();

    if (
      this.isOnChecked == false &&
      this.isOffChecked == false &&
      this.isLockoutChecked == false
    ) {
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

    if (
      this.tempSelectRangeInput.min == null ||
      this.tempSelectRangeInput.max == null
    ) {
      this.removeTempDiffFilter();
    } else {
      console.log('temp', this.tempSelectRangeInput.min);

      //if((this.tempSelectRangeInput.min >= this.tempSelectRange.min && this.tempSelectRangeInput.min <= this.tempSelectRange.max) || (this.tempSelectRangeInput.max >= this.tempSelectRange.min && this.tempSelectRangeInput.max <= this.tempSelectRange.max)) {

      this.tempDiffHousesData = this.housesData.filter(
        (e) =>
          e.tempDifference >= this.tempSelectRangeInput.min &&
          e.tempDifference <= this.tempSelectRangeInput.max
      );
      //}
    }

    this.updateFilteredHouses();
  }

  removeTempDiffFilter(): void {
    this.isTempFiltered = false;
    this.housesData = this.originalHousesData;
  }
}
