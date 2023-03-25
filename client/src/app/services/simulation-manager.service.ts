import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';

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
  speed: number;


  constructor() {
    this.sidenavData = [];
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.started = true;
    this.stopped = true;
    this.speed = 2
  }


  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length - 1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length - 1])
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

  updateSpeed(test: Event): void {

  }
}
