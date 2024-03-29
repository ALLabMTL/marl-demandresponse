import { Component, EventEmitter, Output } from '@angular/core';
import { SharedService } from '@app/services/shared/shared.service';
import { SimulationManagerService } from '@app/services/simulation-manager.service';

@Component({
  selector: 'app-grid-footer',
  templateUrl: './grid-footer.component.html',
  styleUrls: ['./grid-footer.component.scss'],
})
export class GridFooterComponent {
  @Output() pageChangeFromFooter: EventEmitter<any> = new EventEmitter();

  currentPage = 1;
  maxPage = 35;

  constructor(
    private sharedService: SharedService,
    public simulationManager: SimulationManagerService
  ) {}

  ngOnInit() {
    this.sharedService.currentPageCount.subscribe(
      (currentPage) => (this.currentPage = currentPage)
    );
  }

  setPageInput(): void {
    const page = (<HTMLInputElement>document.getElementById('myNumber')).value;
    let pageNb = parseInt(page);
    if (pageNb > this.simulationManager.maxPage) {
      pageNb = this.simulationManager.maxPage;
    }
    if (pageNb < 1) {
      pageNb = 1;
    }
    this.sharedService.changeCount(pageNb);
  }

  incrementPage(): void {
    if (this.currentPage < this.maxPage) {
      this.sharedService.changeCount(this.currentPage + 1);
    }
  }

  decrementPage(): void {
    if (this.currentPage > 1) {
      this.sharedService.changeCount(this.currentPage - 1);
    }
  }
}
