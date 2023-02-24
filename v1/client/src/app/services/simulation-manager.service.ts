import { Injectable } from '@angular/core';
import { SidenavData } from '@app/classes/sidenav-data';

@Injectable({
  providedIn: 'root'
})
export class SimulationManagerService {

  sidenavData: SidenavData[]
  propertyNames: string[]
  propertyValues: string[]
  constructor() { 
    this.sidenavData = []
    this.propertyNames = []
    this.propertyValues = []
  }

  addTimeStep(data: SidenavData): void {
    this.sidenavData.push(data);
    this.propertyNames = Object.getOwnPropertyNames(this.sidenavData[this.sidenavData.length -1]);
    this.propertyValues = Object.values(this.sidenavData[this.sidenavData.length -1])
  }
}
