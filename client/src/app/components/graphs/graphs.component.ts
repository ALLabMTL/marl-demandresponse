import { Component, OnInit, ViewChild, ViewChildren, QueryList } from '@angular/core';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { Chart, ChartConfiguration, ChartOptions } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';
import zoomPlugin from 'chartjs-plugin-zoom';
Chart.register(zoomPlugin);

@Component({
  selector: 'app-graphs',
  templateUrl: './graphs.component.html',
  styleUrls: ['./graphs.component.scss']
})
export class GraphsComponent implements OnInit {
  constructor(
    public sms: SimulationManagerService,
  ) {
  }

  @ViewChildren(BaseChartDirective)
  charts!: QueryList<BaseChartDirective>

  ngOnInit(): void {
    // we HAVE to go though a subscribe because we need to call chart.update() to update the chart
    this.sms.sidenavObservable.subscribe((data) => {
      {
        // get the categories
        const categories = ['Average temperature difference (°C)'];
        let datasets = categories.map((category) => {
          return {
            data: data.map((elem) => Number(elem[category])),
            label: category,
            fill: true,
            tension: 0,
            borderColor: 'black',
            backgroundColor: 'rgba(255,0,0,0.3)',
          }
        }
        );
        this.lineChartData.datasets = datasets;
        this.lineChartData.labels = Array.from(Array(data.length).keys());
      };
      {
        // get the categories
        const categories = ['Average indoor temperature (°C)'];
        let datasets = categories.map((category) => {
          return {
            data: data.map((elem) => Number(elem[category])),
            label: category,
            fill: true,
            tension: 0,
            borderColor: 'black',
            backgroundColor: 'rgba(255,0,0,0.3)',
          }
        }
        );
        this.lineChartData2.datasets = datasets;
        this.lineChartData2.labels = Array.from(Array(data.length).keys());
      };
        this.charts.forEach((e) => e.chart!.update("none"));
    })
  }

  //First Graph
  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'Average temperature difference (°C)',
        fill: false,
        tension: 0,
        borderColor: 'black',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ]
  };

  //Second graph
  public lineChartData2: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [],
        label: 'Average indoor temperature (°C)',
        fill: false,
        tension: 0,
        borderColor: 'black',
        backgroundColor: 'rgba(0,0,255,0.3)'
      }
    ],

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
          /*drag:{
            enabled: true,
          },*/
          mode: 'x',
        },
        pan:{
          enabled:true,
          mode:'xy',
        }
      }
    }
  } as ChartOptions<'line'>;

  public lineChartLegend = true;
}
