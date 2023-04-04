import { Component, Inject, OnInit, ViewChild } from '@angular/core';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { HouseData } from '@app/classes/sidenav-data';
import { SharedService } from '@app/services/shared/shared.service';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { GridComponent } from '../grid/grid.component';
import { Chart, ChartConfiguration, ChartOptions } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';
import zoomPlugin from 'chartjs-plugin-zoom';
Chart.register(zoomPlugin);


@Component({
  selector: 'app-dialog',
  templateUrl: './dialog.component.html',
  styleUrls: ['./dialog.component.scss']
})
export class DialogComponent {
  @ViewChild("a", { static: false }) // Works?
  chartOne!: BaseChartDirective

  id: number;
  currentPage: number = 1;

  // TODO: Do this in a prettier way
  constructor(@Inject(MAT_DIALOG_DATA) public data: number, public simulationManager: SimulationManagerService, public sharedService: SharedService) {
    this.id = data;
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
  }


  ngAfterViewInit(): void {
    // we HAVE to go though a subscribe because we need to call chart.update() to update the chart
    this.simulationManager.houseDataObservable.subscribe((data) => {
      {
        //First graph
        const categories = ['indoor_temp'];
        let datasets = categories.map((category) => {
          console.log(data)
          console.log(this.id)
          return {
            data: data.map((e) => e.find((f) => f.id === this.id)?.indoorTemp).filter((val) => val !== undefined) as number[],
            label: category,
            fill: false,
            tension: 0,
            borderColor: ['blue'],
            backgroundColor: ['blue'],
            pointBackgroundColor: 'black',
            pointRadius: 0,
            pointHoverRadius: 15,
          }
        }
        );
        console.log("patate", datasets);
        this.lineChartData.datasets = datasets;
        //this.lineChartData.datasets = datasets;
        this.lineChartData.labels = Array.from(Array(data.length).keys());
      };
      this.chartOne.chart?.update('none');
    })

  }

  //First Graph
  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'Actual temperature',
        fill: false,
        tension: 0,
        borderColor: 'black',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ]
  };


  //Graphs options
  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    display: true,
    align: 'center',

    plugins: {
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true,
          },
          mode: 'x',
        },
        pan: {
          enabled: true,
          mode: 'xy',
        }
      },
      legend: {
        display: true,
        labels: {
          color: 'black',
          boxWidth: 5,
          boxHeight: 5,
        }
      }
    }
  } as ChartOptions<'line'>;

}
