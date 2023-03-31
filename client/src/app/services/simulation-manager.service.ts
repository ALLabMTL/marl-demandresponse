import { Injectable } from '@angular/core';
import { SidenavData, HouseData } from '@app/classes/sidenav-data';

@Injectable({
  providedIn: 'root',
})
export class SimulationManagerService {
  agentName: string;
  propertyNames: string[];
  propertyValues: string[];
  housesData: HouseData[];
  started: boolean;
  stopped: boolean;
  speed: string;
  nbTimeSteps: number;
  step: number;
  currentTimeStep: number;
  paused: boolean;

  constructor() {
    this.propertyNames = [];
    this.propertyValues = [];
    this.housesData = [];
    this.started = true;
    this.stopped = true;
    this.paused = false;
    this.speed = '4';
    this.agentName = '';
    this.nbTimeSteps = 0;
    // TODO: send step from server
    this.step = 4;
    this.currentTimeStep = 0;
  }

  addTimeStep(data: SidenavData): void {
    this.nbTimeSteps++;
    this.currentTimeStep = this.nbTimeSteps;
    this.setTimeStep(data);
  }

  updateHousesData(data: HouseData[]): void {
    this.housesData = data;
  }

  setTimeStep(data: SidenavData): void {
    this.propertyNames = Object.getOwnPropertyNames(data);
    this.propertyValues = Object.values(data);
  }
}
