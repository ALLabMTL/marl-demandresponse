import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';
import { Observable, Subject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SimulationManagerService {

  sidenavData: SidenavData[];
  sidenavObservable: Subject<SidenavData[]> = new Subject<SidenavData[]>();

  propertyNames: string[];
  propertyValues: string[];
  housesData: HouseData[];
  checkedInfo: string[] = [];
  started: boolean;
  stopped: boolean;
  indexTab: number[] = [];

  constructor() {
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.started = true;
    this.stopped = true;
  }


  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.sidenavObservable.next(this.sidenavData);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length - 1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length - 1]);
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data;
  }

  resetSimulation(): void {
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.started = false;
    this.stopped = true;
  }
/*
  updateChart(event: any, attribute: string): void {
    console.log(attribute.valueOf());
    const dataShown = attribute.valueOf();
    if(event.target.checked){
      this.checkedInfo.push(dataShown);
    }
    else{
      this.checkedInfo=this.checkedInfo.filter(x=>x!==dataShown);
    }
    const len = Math.min(this.propertyNames.length, this.checkedInfo.length);
    for (let i = 0; i < len; i++) {
      for(let y=0; y < len; y++){
        console.log('names', this.propertyNames[i]);
        console.log('checked', this.checkedInfo[y]);
        if (this.propertyNames[i] === this.checkedInfo[y]) {
          this.indexTab.push(i);
          console.log('tab', this.indexTab);
      }
      }
    }
    //this.propertyNames.filter(x)

  }*/

  
 
}
