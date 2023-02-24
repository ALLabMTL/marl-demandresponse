import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GridFooterComponent } from './grid-footer.component';

describe('GridFooterComponent', () => {
  let component: GridFooterComponent;
  let fixture: ComponentFixture<GridFooterComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ GridFooterComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(GridFooterComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
