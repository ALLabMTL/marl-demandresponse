import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { HeaderComponent } from './components/header/header.component';
import {MatToolbarModule} from '@angular/material/toolbar';
import { SidebarComponent } from './components/sidebar/sidebar.component';
import { GridComponent } from './components/grid/grid.component';
import { GridFooterComponent } from './components/grid-footer/grid-footer.component';
import { InformationsComponent } from './components/informations/informations.component';
import { GraphsComponent } from './components/graphs/graphs.component';
import { TimestampComponent } from './components/timestamp/timestamp.component';

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    SidebarComponent,
    GridComponent,
    GridFooterComponent,
    InformationsComponent,
    GraphsComponent,
    TimestampComponent
  ],
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    MatToolbarModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
