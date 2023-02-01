import { Component } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSliderModule } from '@angular/material/slider';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';


@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent {
  isTempChecked = false;
  isFloorChecked = false;
  precisionValueSelected = 2;
  negMin = -0.5;
  negMidMin = -0.25;
  mid = 0;
  posMidMax = 0.25;
  posMax = 0.5;

  formatLabel(value: number): string {
    return `${value}`;
  }

  tempCheckbox(): void {
    this.isTempChecked = !this.isTempChecked;
  }

  floorCheckbox(): void {
    this.isFloorChecked = !this.isFloorChecked;
  }

  scaleChartPrecision(): void {
    this.negMin = -(this.precisionValueSelected);
    this.negMidMin = this.negMin / 2;
    this.posMax = this.precisionValueSelected;
    this.posMidMax = this.posMax / 2
    console.log("precision: %d", this.precisionValueSelected)
  }
}
