import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StopSimulationDialogComponent } from './stop-simulation-dialog.component';

describe('StopSimulationDialogComponent', () => {
  let component: StopSimulationDialogComponent;
  let fixture: ComponentFixture<StopSimulationDialogComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StopSimulationDialogComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(StopSimulationDialogComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
