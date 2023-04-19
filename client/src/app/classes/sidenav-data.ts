export interface SidenavData {
  [key: string]: string;
}

export interface HouseData {
  id: number;
  hvacStatus: string;
  secondsSinceOff: number;
  indoorTemp: number;
  targetTemp: number;
  tempDifference: number;
}
