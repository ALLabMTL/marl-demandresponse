import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppMaterialModule } from './modules/material.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { AppRoutingModule } from './modules/app-routing.module';
import { HeaderComponent } from './components/header/header.component';
import { MatToolbarModule } from '@angular/material/toolbar';
import { SidebarComponent } from './components/sidebar/sidebar.component';
import { GridComponent } from './components/grid/grid.component';
import { GridFooterComponent } from './components/grid-footer/grid-footer.component';
import { InformationsComponent } from './components/informations/informations.component';
import { GraphsComponent } from './components/graphs/graphs.component';
import { TimestampComponent } from './components/timestamp/timestamp.component';
import { MatIconModule } from '@angular/material/icon';
import {MatGridListModule} from '@angular/material/grid-list';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { MatSliderModule } from '@angular/material/slider';
import { MatInputModule } from '@angular/material/input';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { DialogComponent } from './components/dialog/dialog.component';
import { MatDialogModule } from '@angular/material/dialog';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SidebarComponent,
    GridComponent,
    GridFooterComponent,
    InformationsComponent,
    GraphsComponent,
    TimestampComponent,
    DialogComponent,
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    MatToolbarModule,
    MatIconModule,
    MatGridListModule,
    AppMaterialModule,
    MatFormFieldModule,
    MatSelectModule,
    MatCheckboxModule,
    MatSliderModule,
    MatInputModule,
    MatIconModule,
    FormsModule, 
    ReactiveFormsModule,
    AppRoutingModule,
    AppMaterialModule,
    MatDialogModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }

// providers: [SidebarComponent],
//   exports: [MatFormFieldModule, MatSliderModule, MatCheckboxModule, MatFormFieldModule, MatSelectModule, FormsModule],
