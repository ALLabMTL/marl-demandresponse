import { Component } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSliderModule } from '@angular/material/slider';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { AbstractControl, FormControl, Validators } from '@angular/forms';

@Component({
  selector: 'app-sidebar',
  templateUrl: './sidebar.component.html',
  styleUrls: ['./sidebar.component.scss']
})
export class SidebarComponent {
  isTempChecked = false;
  isHvacChecked = false;
  precisionValueSelected = 0.5;
  negMin = -0.5;
  negMidMin = -0.25;
  mid = 0;
  posMidMax = 0.25;
  posMax = 0.5;

  numberFormControl = new FormControl('', [Validators.required, Validators.min(0)]);

  formatLabel(value: number): string {
    return `${value}`;
  }

  tempCheckbox(): void {
    this.isTempChecked = !this.isTempChecked;
  }

  // floorCheckbox(): void {
  //   this.isFloorChecked = !this.isFloorChecked;
  // }

  scaleChartPrecision(): void {
    if(this.precisionValueSelected >= 0){
      this.negMin = -(this.precisionValueSelected);
      this.negMidMin = this.negMin / 2;
      this.posMax = Math.abs(this.precisionValueSelected);
      this.posMidMax = this.posMax / 2
      console.log("precision: %d", this.precisionValueSelected)
    }
  }
}
