import { FormsModule } from '@angular/forms';
import { MatSelectModule } from '@angular/material/select';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatCheckbox, MatCheckboxModule } from '@angular/material/checkbox';
import { MatDividerModule } from '@angular/material/divider';
import { MatSlider, MatSliderModule } from '@angular/material/slider';
import { MatSlideToggleModule } from '@angular/material/slide-toggle';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SidebarComponent } from './sidebar.component';

describe('SidebarComponent', () => {
  let component: SidebarComponent;
  let fixture: ComponentFixture<SidebarComponent>;
  let spyOnSimulationService: SimulationManagerService;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ SidebarComponent ],
      //imports: [ MatSlideToggleModule, MatSliderModule, MatDividerModule, MatCheckboxModule, MatFormFieldModule, MatSelectModule, FormsModule ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(SidebarComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  // beforeEach(() => {
  //   spyOnSimulationService = jasmine.createSpyObj('SimulationManagerService', ['removeTempDiffFilter', 'removeHvacFilter']);
  //   TestBed.configureTestingModule({
  //     declarations: [SidebarComponent],
  //     providers: [
  //         { provide: SimulationManagerService, useValue: spyOnSimulationService },
  //     ],
  //     imports: [ MatSlideToggleModule, MatSliderModule, MatDividerModule, MatCheckboxModule, MatFormFieldModule, MatSelectModule, FormsModule ]
  //   }).compileComponents();
  //   // fixture = TestBed.createComponent(SidebarComponent);
  //   // component = fixture.componentInstance;
  //   // fixture.detectChanges();
  //   spyOnSimulationService = TestBed.inject(SimulationManagerService) as jasmine.SpyObj<SimulationManagerService>;
  // })

  it('should create', () => {
    expect(component).toBeTruthy();
  });


  // it('should reset the slider', () => {
  //   // set some initial values
  //   component.simulationManager.minValueSliderInit = 10;
  //   component.simulationManager.maxValueSliderInit = 20;

  //   // call the resetSlider method
  //   component.resetSlider();

  //   // check that the slider values have been reset to their initial values
  //   expect(component.simulationManager.tempSelectRange.min).toEqual(component.simulationManager.minValueSliderInit);
  //   expect(component.simulationManager.tempSelectRange.max).toEqual(component.simulationManager.maxValueSliderInit);
  //   expect(spyOnSimulationService.removeTempDiffFilter()).toHaveBeenCalled();
  // });

  // it('should reset the HVAC filter', () => {
  //   // set some initial values
  //   component.onChecked.checked = true;
  //   component.offChecked.checked = true;
  //   component.lockoutChecked.checked = true;

  //   // call the resetHvacFilter method
  //   component.resetHvacFilter();

  //   // check that all checkboxes have been unchecked
  //   expect(component.onChecked.checked).toBeFalsy();
  //   expect(component.offChecked.checked).toBeFalsy();
  //   expect(component.lockoutChecked.checked).toBeFalsy();
  //   expect(spyOnSimulationService.removeHvacFilter()).toHaveBeenCalled();

  // });
});
