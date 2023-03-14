import { Component } from '@angular/core';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSliderModule } from '@angular/material/slider';
import { MatInputModule } from '@angular/material/input';
import { MatIconModule } from '@angular/material/icon';
import { AbstractControl, FormControl, Validators } from '@angular/forms';
import { SharedService } from '@app/services/shared/shared.service';

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

  constructor(public sharedService: SharedService) { }

  ngOnInit() {
    this.sharedService.currentPrecisionValue.subscribe(houseColorPrecisionValue => this.precisionValueSelected = houseColorPrecisionValue);
  }

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
    const precisionValue = (<HTMLInputElement>document.getElementById("precisionValue")).value;
    this.sharedService.changePrecisionValue(parseFloat(precisionValue));
    if (this.precisionValueSelected >= 0) {
      this.negMin = -(this.precisionValueSelected);
      this.negMidMin = this.negMin / 2;
      this.posMax = Math.abs(this.precisionValueSelected);
      this.posMidMax = this.posMax / 2
      // console.log("precision: %d", this.precisionValueSelected)
    }
  }

}
