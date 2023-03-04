import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { SimulationManagerService } from '../simulation-manager.service';

@Injectable({
  providedIn: 'root'
})
export class SharedService {

  private currentPage = new BehaviorSubject(1);
  currentPageCount = this.currentPage.asObservable();

  private precisionValueSelected = new BehaviorSubject(0.5);
  currentPrecisionValue = this.precisionValueSelected.asObservable();

  constructor() { }

  changeCount(currentPage: number) {
    this.currentPage.next(currentPage);
  }

  changePrecisionValue(currentPrecisionValue: number) {
    this.precisionValueSelected.next(currentPrecisionValue);
  }

  houseColor(data: number) {
    const upperBound = this.precisionValueSelected.value;
    const middleUpperBound = upperBound / 2;
    const center = 0
    const middleLowerBound = -middleUpperBound;
    const lowerBound = -upperBound;
    const boundRange = upperBound - middleUpperBound;

    if (data < lowerBound) {
      return "rgba(0, 0, 255, 100)";
    }
    else if (lowerBound <= data && data < middleLowerBound) {
      const temp = -(lowerBound - data) / boundRange;
      const color = temp * 255;
      return "rgba(0," + color + ", 255, 100)";
    }
    else if (middleLowerBound <= data && data < center) {
      const temp = (boundRange + (middleLowerBound - data)) / boundRange;
      const color = temp * 255;
      return "rgba(0, 255," + color + ", 100)";
    }
    else if (center <= data && data < middleUpperBound) {
      const temp = (boundRange - (middleUpperBound - data)) / boundRange;
      const color = temp * 255;
      return "rgba(" + color + ",255, 0, 100)";
    }
    else if (middleUpperBound <= data && data <= upperBound) {
      const temp = (upperBound - data) / boundRange;
      const color = temp * 255;
      return "rgba(255," + color + ", 0, 100)";
    }
    else {
      return "rgba(255, 0, 0, 100)";
    }

  }

}
