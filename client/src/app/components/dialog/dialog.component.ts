import { Component, Inject, OnInit, ViewChild, ViewChildren, QueryList } from '@angular/core';
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
  @ViewChild("a", { static: false })
  chartOne!: BaseChartDirective;
  @ViewChild("b", { static: false })
  chartTwo!: BaseChartDirective;
  @ViewChild("c", { static: false })
  chartThree!: BaseChartDirective;
  @ViewChildren(BaseChartDirective)
  charts!: QueryList<BaseChartDirective>;

  id: number;
  currentPage = 1;

  // TODO: Do this in a prettier way
  constructor(@Inject(MAT_DIALOG_DATA) public data: number, public simulationManager: SimulationManagerService, public sharedService: SharedService) {
    this.id = data;
    this.sharedService.currentPageCount.subscribe(currentPage => this.currentPage = currentPage);
  }

  private updateGraphsFromHousesData = (data: HouseData[][]) => {
    {
      //First graph
      const categories = ['Actual temperature'];
      const datasets = categories.map((category) => {
        return {
          data: data.map((e) => e.find((f) => f.id === this.id)?.indoorTemp).filter((val) => val !== undefined) as number[],
          label: category,
          fill: false,
          tension: 0,
          borderColor: ['teal'],
          backgroundColor: ['teal'],
          pointBackgroundColor: 'white',
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHitRadius: 10
        };
      }
      );
      this.lineChartData.datasets = datasets;
      this.lineChartData.labels = Array.from(Array(data.length).keys());
    }
    {
      //Second graph
      const categories = ['Temperature difference'];
      const datasets = categories.map((category) => {
        return {
          data: data.map((e) => e.find((f) => f.id === this.id)?.tempDifference).filter((val) => val !== undefined) as number[],
          label: category,
          fill: false,
          tension: 0,
          borderColor: ['yellow'],
          backgroundColor: ['yellow'],
          pointBackgroundColor: 'white',
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHitRadius: 10
        };
      }
      );
      this.lineChartData2.datasets = datasets;
      this.lineChartData2.labels = Array.from(Array(data.length).keys());
    }
    {
      //Third graph
      const categories = ['HVAC status'];
      const hvacStatusToInteger = (hvacStatus: string | undefined) => {
        switch (hvacStatus) {
          case 'OFF':
            return 0;
          case 'ON':
            return 1;
          case 'Lockout':
            return 0.5;
          default:
            return undefined;
        }
      };

      const datasets = categories.map((category) => {
        return {
          data: data.map((e) => e.find((f) => f.id === this.id)?.hvacStatus).filter((val) => val !== undefined),
          label: category,
          fill: false,
          tension: 0,
          borderColor: ['green'],
          backgroundColor: ['green'],
          pointBackgroundColor: 'white',
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHitRadius: 10
        };
      }
      );
      this.lineChartData3.datasets = datasets as unknown as ChartConfiguration<'line'>['data']['datasets'];
      this.lineChartData3.labels = Array.from(Array(data.length).keys());
    }
    this.chartOne.chart?.update('none');
    this.chartTwo.chart?.update('none');
    this.chartThree.chart?.update('none');
  };


  ngAfterViewInit(): void {
    // we HAVE to go though a subscribe because we need to call chart.update() to update the chart
    this.simulationManager.houseDataObservable.subscribe(this.updateGraphsFromHousesData);
    this.updateGraphsFromHousesData(this.simulationManager.housesBufferData);

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
        borderColor: 'white',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ]
  };

  //Second Graph
  public lineChartData2: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'Temperature difference',
        fill: false,
        tension: 0,
        borderColor: 'white',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ]
  };

  //Third Graph
  public lineChartData3: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'HVAC status',
        fill: false,
        tension: 0, 
        borderColor: 'white',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ],
  };


  //Graphs options
  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    display: true,
    align: 'center',

    scales: {
      y: 
        {
          ticks: {
            color: "white"
          }
        },
        x: 
        {
          ticks: {
            color: "white"
          }
        }
    },

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
          color: 'white',
          boxWidth: 5,
          boxHeight: 5,
        }
      }
    }
  } as ChartOptions<'line'>;


  //Graphs options
  public lineChartOptions3: ChartOptions<'line'> = {
    responsive: true,
    display: true,
    align: 'center',

    scales: {
      y: {
        type: 'category',
        labels: ["ON", "Lockout", "OFF"],
        ticks: {
          color: "white"
        }
      },
      x: 
        {
          ticks: {
            color: "white"
          }
        },
    },

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
          color: 'white',
          boxWidth: 5,
          boxHeight: 5,
        }
      }
    }
  } as ChartOptions<'line'>;

  public lineChartLegend = true;

  public resetZoomGraph(index: number): void {
    // The code(this.charts.get(index)!.chart as any).resetZoom() is likely used to reset the zoom level of a chart.this.charts.get(index)
    // is used to get a reference to the chart object at the specified index. .chart as 
    // any is used to cast the chart object to any type, which allows accessing the resetZoom() method.
    // The resetZoom() method is then called to reset the zoom level of the chart.
    (this.charts.get(index)!.chart as any).resetZoom();
  }

}
