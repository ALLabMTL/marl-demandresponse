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
      this.charts.forEach((e) => e.chart!.update("none"))
    })
  }

  //TODO let the user choose from which timesetp to which timestep they should see the data; aka let the user zoom in the graph
  //TODO same thing with y axis, let the user choose the min and max values

  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [], //TODO this is also placeholder data
        label: 'Average temperature difference (°C)', //TODO this is also placeholder data
        fill: false,
        tension: 0,
        borderColor: 'black',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ]
  };

  //second graph
  public lineChartData2: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [], //TODO this is also placeholder data
        label: 'Average indoor temperature (°C)', //TODO this is also placeholder data
        fill: false,
        tension: 0,
        borderColor: 'black',
        backgroundColor: 'rgba(0,0,255,0.3)'
      }
    ],

  };

  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      zoom: {
        zoom: {
          wheel: {
            enabled: true,
          },
          pinch: {
            enabled: true
          },
          mode: 'x',
        }
      }
    }
  } as ChartOptions<'line'>;

  public lineChartLegend = true;
}
