import { Component } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSliderModule } from '@angular/material/slider';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { AbstractControl, FormControl, Validators } from '@angular/forms';
import { SharedService } from '@app/services/shared/shared.service';
import { SimulationManagerService } from '@app/services/simulation-manager.service';

interface Filter {
  id: number;
  type: string;
  value?: any;
}
@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent {
  isTempChecked = false;
  isHvacChecked = false;
  precisionValueSelected = 0.5;
  negMin = -0.5;
  negMidMin = -0.25;
  mid = 0;
  posMidMax = 0.25;
  posMax = 0.5;
  hvacStatus = "ON";
  isSortingSelected = false;
  tempDiffValue = -1;
  filters: Filter[] = [];
  isFilteredSorted = false; // for the grid 
  sortingOptionSelected = " ";

  numberFormControl = new FormControl('', [Validators.required, Validators.min(0)]);
  
  constructor(public sharedService: SharedService, public simulationManager: SimulationManagerService) {
    this.simulationManager.originalHousesData = this.simulationManager.housesData.slice(); // deep copy
  }

  ngOnInit() {
    this.sharedService.currentPrecisionValue.subscribe(houseColorPrecisionValue => this.precisionValueSelected = houseColorPrecisionValue);
  }

  sortByTempDiffIncreasing(): void {
    this.isFilteredSorted = true;
    this.isSortingSelected = true;
    if(this.isHvacChecked || this.isTempChecked){
      this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1: 1); // a negative, then a comes before b
    } else {
      this.simulationManager.houseDataFiltered = this.simulationManager.housesData.sort((a, b) => (a.tempDifference < b.tempDifference) ? -1: 1); // a negative, then a comes before b  
    }
    this.filters.push({id: 0, type: "tempDiffInc"});
  }

  sortByTempDiffDecreasing(): void {
    this.isFilteredSorted = true;
    this.isSortingSelected = true;
    if(this.isHvacChecked || this.isTempChecked){
      this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1: 1); // a negative, then a comes before b
    } else {
      this.simulationManager.houseDataFiltered = this.simulationManager.housesData.sort((a, b) => (a.tempDifference > b.tempDifference) ? -1: 1); // a negative, then a comes before b
    }
    this.filters.push({id: 1, type: "tempDiffDec"});
  }

  sortByIndoorTempIncreasing(): void {
    this.isFilteredSorted = true;
    this.isSortingSelected = true;
    if(this.isHvacChecked || this.isTempChecked){
      this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1: 1); // a negative, then a comes before b
    } else {
      this.simulationManager.houseDataFiltered = this.simulationManager.housesData.sort((a, b) => (a.indoorTemp < b.indoorTemp) ? -1: 1); // a negative, then a comes before b
    }
    this.filters.push({id: 2, type: "indoorTempInc"});
  }

  sortByIndoorTempDecreasing(): void {
    this.isFilteredSorted = true;
    this.isSortingSelected = true;
    if(this.isHvacChecked || this.isTempChecked){
      this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.sort((a, b) => (a.indoorTemp > b.indoorTemp) ? -1: 1); // a negative, then a comes before b
    } else {
      this.simulationManager.houseDataFiltered = this.simulationManager.housesData.sort((a, b) => (a.indoorTemp > b.indoorTemp) ? -1: 1); // a negative, then a comes before b
    }
    this.filters.push({id: 3, type: "indoorTempDec"});
  }

  sortByOptionSelected(option: string): void {
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
      default:
        break;
    }
  }

  removeSorting(): void {
    this.isSortingSelected = false;
    //update grid
    this.restoreHousesData();
  }

  filterByHvacStatus(): void {
    this.isFilteredSorted = true;
    if(this.isSortingSelected || this.isTempChecked) {
      if(this.hvacStatus == "ON") {
        this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.filter(status => status.hvacStatus == "ON");
      } else if(this.hvacStatus == "Lockout") {
        this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.filter(status => status.hvacStatus == "Lockout");
      } else if(this.hvacStatus == "OFF") {
        this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.filter(status => status.hvacStatus == "OFF");
      }
    } else {
      if(this.hvacStatus == "ON") {
        this.simulationManager.houseDataFiltered = this.simulationManager.housesData.filter(status => status.hvacStatus == "ON");
      } else if(this.hvacStatus == "Lockout") {
        this.simulationManager.houseDataFiltered = this.simulationManager.housesData.filter(status => status.hvacStatus == "Lockout");
      } else if(this.hvacStatus == "OFF") {
        this.simulationManager.houseDataFiltered = this.simulationManager.housesData.filter(status => status.hvacStatus == "OFF");
      }

    }
    this.filters.push({id: 4, type: "hvacStatus", value: this.hvacStatus});
  }

  filterByTempDiff(): void {
    this.isFilteredSorted = true;
    if(this.isSortingSelected || this.isHvacChecked) {
      this.simulationManager.houseDataFiltered = this.simulationManager.houseDataFiltered.filter(houses => houses.tempDifference == this.tempDiffValue);
    } else {
      this.simulationManager.houseDataFiltered = this.simulationManager.housesData.filter(houses => houses.tempDifference == this.tempDiffValue);
    }
    this.filters.push({id: 5, type: "tempDiffValue", value: this.tempDiffValue});
  }


  resetActivityFilters(): void {
    this.isSortingSelected = false;
    this.isHvacChecked = false;
    this.isTempChecked = false;
  }

  // called when toggle turned off and form field no sorting selected
  restoreHousesData(): void {
    //if()
    //this.simulationManager.housesData = this.simulationManager.originalHousesData;
    const filterTypes = this.filters.map(filter => filter.type);

    if(this.isSortingSelected && this.isHvacChecked) {
      if (filterTypes.includes('hvacStatus') && filterTypes.includes(this.sortingOptionSelected)) {
        
        this.resetActivityFilters();

         const hvacFilter = this.filters.find(filter => filter.type === 'hvacStatus');
         this.hvacStatus = hvacFilter?.value;
         this.filterByHvacStatus();

         //const sortFilter = this.filters.find(filter => filter.type === this.sortingOptionSelected);
         this.sortByOptionSelected(this.sortingOptionSelected);

      }
    } else if(this.isSortingSelected && this.isTempChecked) {

    } else if(this.isTempChecked && this.isHvacChecked){

    } else if(this.isHvacChecked) {

    } else if(this.isTempChecked) {

    } else if(this.isSortingSelected) {

    } else {
        this.isFilteredSorted = false;
    }
  }

  formatLabel(value: number): string {
    return `${value}`;
  }

  // tempCheckbox(): void {
  //   this.isTempChecked = !this.isTempChecked;
  // }

  // floorCheckbox(): void {
  //   this.isFloorChecked = !this.isFloorChecked;
  // }

  scaleChartPrecision(): void {
    const precisionValue = (<HTMLInputElement>document.getElementById("precisionValue")).value;
    this.sharedService.changePrecisionValue(parseFloat(precisionValue));
    if(this.precisionValueSelected >= 0){
      this.negMin = -(this.precisionValueSelected);
      this.negMidMin = this.negMin / 2;
      this.posMax = Math.abs(this.precisionValueSelected);
      this.posMidMax = this.posMax / 2
      console.log("precision: %d", this.precisionValueSelected)
    }
  }

}
