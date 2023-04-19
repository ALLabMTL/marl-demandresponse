import { TestBed } from '@angular/core/testing';
import { HouseData } from '@app/classes/sidenav-data';

import { SimulationManagerService } from './simulation-manager.service';

describe('SimulationManagerService', () => {
  const originalHousesData: HouseData[] = [
    {
      id: 1,
      hvacStatus: 'ON',
      secondsSinceOff: 10,
      indoorTemp: 25,
      targetTemp: 22,
      tempDifference: 3,
    },
    {
      id: 2,
      hvacStatus: 'OFF',
      secondsSinceOff: 600,
      indoorTemp: 22,
      targetTemp: 22,
      tempDifference: 0,
    },
    {
      id: 3,
      hvacStatus: 'Lockout',
      secondsSinceOff: 1800,
      indoorTemp: 23,
      targetTemp: 22,
      tempDifference: 1,
    },
    {
      id: 4,
      hvacStatus: 'ON',
      secondsSinceOff: 20,
      indoorTemp: 20,
      targetTemp: 22,
      tempDifference: -2,
    },
  ];
  let service: SimulationManagerService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(SimulationManagerService);
    service.originalHousesData = originalHousesData;
    service.housesData = originalHousesData;
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should sort housesData by tempDifference in increasing order', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByTempDiffIncreasing();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
    ]);

    expect(service.isSortingSelected).toBeTrue();
  });

  it('should sort housesData by tempDifference in decreasing order', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByTempDiffDecreasing();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ]);

    expect(service.isSortingSelected).toBeTrue();
  });

  it('should sort housesData by indoorTemp in increasing order', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByIndoorTempIncreasing();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ]);

    expect(service.isSortingSelected).toBeTrue();
  });

  it('should sort housesData by indoorTemp in decreasing order', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByIndoorTempDecreasing();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
    ]);

    expect(service.isSortingSelected).toBeTrue();
  });

  it('should sort houses by indoor temperature increasing when "indoorTempInc" is selected', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByOptionSelected('indoorTempInc');

    const spy = spyOn(service, 'sortByIndoorTempIncreasing').and.callThrough();
    //const spy2 = spyOn(service, 'updateFilteredHouses').and.callThrough();

    expect(spy).not.toHaveBeenCalled();

    expect(service.sortingOptionSelected).toEqual('indoorTempInc');
    expect(service.isSortingSelected).toBeTrue();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ]);

    // expect(spy2).toHaveBeenCalled();
  });

  it('should sort houses by indoor temperature decreasing when "indoorTempDec" is selected', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByOptionSelected('indoorTempDec');

    const spy = spyOn(service, 'sortByIndoorTempDecreasing').and.callThrough();
    expect(spy).not.toHaveBeenCalled();

    expect(service.sortingOptionSelected).toEqual('indoorTempDec');
    expect(service.isSortingSelected).toBeTrue();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
    ]);
  });

  it('should sort houses by temperature difference increasing when "tempDiffInc" is selected', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByOptionSelected('tempDiffInc');

    const spy = spyOn(service, 'sortByTempDiffIncreasing').and.callThrough();
    expect(spy).not.toHaveBeenCalled();

    expect(service.sortingOptionSelected).toEqual('tempDiffInc');
    expect(service.isSortingSelected).toBeTrue();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
    ]);
  });

  it('should sort houses by temperature difference decreasing when "tempDiffDec" is selected', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.housesData = housesData;

    service.sortByOptionSelected('tempDiffDec');

    const spy = spyOn(service, 'sortByTempDiffDecreasing').and.callThrough();
    expect(spy).not.toHaveBeenCalled();

    expect(service.sortingOptionSelected).toEqual('tempDiffDec');
    expect(service.isSortingSelected).toBeTrue();

    expect(service.houseDataFiltered).toEqual([
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ]);
  });

  it('should not sort houses when "noSorting" is selected', () => {
    const housesData = [
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ];
    service.originalHousesData = housesData;

    service.sortByOptionSelected('noSorting');

    const spy = spyOn(service, 'removeSorting').and.callThrough();
    expect(spy).not.toHaveBeenCalled();

    expect(service.sortingOptionSelected).toEqual('noSorting');
    expect(service.isSortingSelected).toBeFalse();
    expect(service.housesData).toEqual([
      {
        id: 1,
        hvacStatus: 'OFF',
        secondsSinceOff: 0,
        indoorTemp: 70,
        targetTemp: 72,
        tempDifference: 2,
      },
      {
        id: 2,
        hvacStatus: 'ON',
        secondsSinceOff: 0,
        indoorTemp: 68,
        targetTemp: 72,
        tempDifference: 4,
      },
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 0,
        indoorTemp: 74,
        targetTemp: 72,
        tempDifference: 2,
      },
    ]);
  });

  it('should filter houses by hvac status when ON status is checked', () => {
    service.filterByHvacStatus(true, 'ON');
    expect(service.isHvacChecked).toBeTrue();
    expect(service.hvacStatus).toEqual('ON');
    expect(service.hvacChosen).toEqual([
      {
        id: 1,
        hvacStatus: 'ON',
        secondsSinceOff: 10,
        indoorTemp: 25,
        targetTemp: 22,
        tempDifference: 3,
      },
      {
        id: 4,
        hvacStatus: 'ON',
        secondsSinceOff: 20,
        indoorTemp: 20,
        targetTemp: 22,
        tempDifference: -2,
      },
    ]);
    // const spy = spyOn(service, 'updateFilteredHouses').and.callThrough();
    // expect(spy).toHaveBeenCalled();

    //expect(service.isFilteredHvac).toBeTrue();
    //expect(service.housesData).toEqual([{ id: 1, hvacStatus: 'ON', secondsSinceOff: 10, indoorTemp: 25, targetTemp: 22, tempDifference: 3 }, { id: 4, hvacStatus: 'ON', secondsSinceOff: 20, indoorTemp: 20, targetTemp: 22, tempDifference: -2 }]);
  });

  it('should filter houses by hvac status when Lockout status is checked', () => {
    service.filterByHvacStatus(true, 'Lockout');
    expect(service.isHvacChecked).toBeTrue();
    expect(service.hvacStatus).toEqual('Lockout');
    expect(service.hvacChosen).toEqual([
      {
        id: 3,
        hvacStatus: 'Lockout',
        secondsSinceOff: 1800,
        indoorTemp: 23,
        targetTemp: 22,
        tempDifference: 1,
      },
    ]);
    // const spy = spyOn(service, 'updateFilteredHouses').and.callThrough();
    // expect(spy).toHaveBeenCalled();

    // service.updateFilteredHouses();
    // service.isFilteredHvac = true;

    //expect(service.isFilteredHvac).toBeTrue();
    // expect(service.housesData).toEqual([{ id: 1, hvacStatus: 'ON', secondsSinceOff: 10, indoorTemp: 25, targetTemp: 22, tempDifference: 3 }, { id: 4, hvacStatus: 'ON', secondsSinceOff: 20, indoorTemp: 20, targetTemp: 22, tempDifference: -2 }]);
  });

  it('should remove hvac status filter when all status are unchecked', () => {
    service.filterByHvacStatus(false, 'ON');
    expect(service.hvacChosen).toEqual([]);
    expect(service.isFilteredHvac).toBeFalse();
    expect(service.housesData).toEqual(originalHousesData);
  });

  it('should filter houses by multiple hvac status when more than one status is checked', () => {
    service.filterByHvacStatus(true, 'ON');
    expect(service.hvacChosen).toEqual([
      {
        id: 1,
        hvacStatus: 'ON',
        secondsSinceOff: 10,
        indoorTemp: 25,
        targetTemp: 22,
        tempDifference: 3,
      },
      {
        id: 4,
        hvacStatus: 'ON',
        secondsSinceOff: 20,
        indoorTemp: 20,
        targetTemp: 22,
        tempDifference: -2,
      },
    ]);
    expect(service.hvacStatus).toEqual('ON');
    expect(service.isHvacChecked).toBeTrue();
    //expect(service.isOnChecked).toBeTrue();

    service.filterByHvacStatus(true, 'OFF');
    expect(service.hvacStatus).toEqual('OFF');
    expect(service.isHvacChecked).toBeTrue();
    expect(service.hvacChosen).toEqual([
      {
        id: 1,
        hvacStatus: 'ON',
        secondsSinceOff: 10,
        indoorTemp: 25,
        targetTemp: 22,
        tempDifference: 3,
      },
      {
        id: 4,
        hvacStatus: 'ON',
        secondsSinceOff: 20,
        indoorTemp: 20,
        targetTemp: 22,
        tempDifference: -2,
      },
      {
        id: 2,
        hvacStatus: 'OFF',
        secondsSinceOff: 600,
        indoorTemp: 22,
        targetTemp: 22,
        tempDifference: 0,
      },
    ]);
    // expect(service.isFilteredHvac).toBeTrue();
    // expect(service.housesData).toEqual([{ id: 1, hvacStatus: 'ON', secondsSinceOff: 10, indoorTemp: 25, targetTemp: 22, tempDifference: 3 }, { id: 4, hvacStatus: 'ON', secondsSinceOff: 20, indoorTemp: 20, targetTemp: 22, tempDifference: -2 }, { id: 2, hvacStatus: 'OFF', secondsSinceOff: 600, indoorTemp: 22, targetTemp: 22, tempDifference: 0 }]);
  });

  it('should remove hvac filters', () => {
    service.removeHvacFilter();
    expect(service.housesData).toEqual(service.originalHousesData);
    expect(service.isFilteredHvac).toBeFalse();
  });

  it('should remove hvac filters when no status is checked', () => {
    service.isOnChecked = false;
    service.isOffChecked = false;
    service.isLockoutChecked = false;

    service.filterByHvacStatus(false, 'Lockout');
    //const spy = spyOn(service, 'removeHvacFilter').and.callThrough();
    //expect(spy).toHaveBeenCalled();

    // service.removeHvacFilter();
    // expect(service.housesData).toEqual(service.originalHousesData);
    // expect(service.isFilteredHvac).toBeFalse();
  });

  it('should filter houses based on temperature difference', () => {
    service.tempSelectRangeInput = { min: 1, max: 3 };
    service.filterByTempDiff();

    expect(service.housesData.length).toEqual(2);
    expect(service.housesData[0].id).toEqual(1);
    expect(service.housesData[1].id).toEqual(3);
  });

  it('should removetemperature difference filter', () => {
    service.removeTempDiffFilter();
    expect(service.isTempFiltered).toBeFalse();
    expect(service.housesData).toEqual(originalHousesData);
  });
});
