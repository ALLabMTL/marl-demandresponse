import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class SharedService {

  private currentPage = new BehaviorSubject(1);
  currentPageCount = this.currentPage.asObservable();

  constructor() { }

  changeCount(currentPage: number) {
    this.currentPage.next(currentPage);
  }
}
