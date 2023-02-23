import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

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
    let upperBound = this.precisionValueSelected.value;
    let middleUpperBound = upperBound / 2;
    let center = 0
    let middleLowerBound = -middleUpperBound;
    let lowerBound = -upperBound;
    let boundRange = upperBound - middleUpperBound;
    console.log(data);
    console.log(upperBound);
    console.log(middleUpperBound);
    console.log(center);
    console.log(middleLowerBound);
    console.log(lowerBound);


    if (data < lowerBound) {
      return "rgba(0, 0, 255, 100)";
    }
    else if (lowerBound <= data && data < middleLowerBound) {
      let temp = -(lowerBound - data) / boundRange;
      let color = temp * 255;
      return "rgba(0," + color + ", 255, 100)";
    }
    else if (middleLowerBound <= data && data < center) {
      let temp = (boundRange + (middleLowerBound - data)) / boundRange;
      let color = temp * 255;
      return "rgba(0, 255," + color + ", 100)";
    }
    else if (center <= data && data < middleUpperBound) {
      let temp = (boundRange - (middleUpperBound - data)) / boundRange;
      let color = temp * 255;
      return "rgba(" + color + ",255, 0, 100)";
    }
    else if (middleUpperBound <= data && data <= upperBound) {
      let temp = (upperBound - data) / boundRange;
      let color = temp * 255;
      return "rgba(255," + color + ", 0, 100)";
    }
    else {
      return "rgba(255, 0, 0, 100)";
    }

  }

}
