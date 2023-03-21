import { Component, OnInit, ViewChild } from '@angular/core';
import { SimulationManagerService } from '@app/services/simulation-manager.service';
import { Chart, ChartConfiguration, ChartOptions } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';

@Component({
  selector: 'app-graphs',
  templateUrl: './graphs.component.html',
  styleUrls: ['./graphs.component.scss']
})
export class GraphsComponent implements OnInit {
  constructor(
    public sms: SimulationManagerService
  ) { }

  @ViewChild(BaseChartDirective)
  lineChart!: BaseChartDirective;

  ngOnInit(): void {
    // we HAVE to go though a subscribe because we need to call chart.update() to update the chart
    this.sms.sidenavObservable.subscribe((data) => {
      // get the categories
      //const categories = Object.keys(data[0]);
      const categories = ['Average temperature difference (°C)'];
      const datasets = categories.map((category) => {
        return {
          data: data.map((elem) => Number(elem[category])),
          label: category,
          fill: false,
          tension: 0,
          borderColor: 'black',
          backgroundColor: 'rgba(255,0,0,0.3)'
        }
      }
      );

      // const number_data = data.map((elem) => Number(elem["Average indoor temperature (°C)"]));
      this.lineChartData.datasets = datasets;
      // labels are range from 0 to number_data.length
      this.lineChartData.labels = Array.from(Array(data.length).keys());
      this.lineChart.chart!.update('none');
    })


    // second graph
    this.sms.sidenavObservable.subscribe((data) => {
      const secondGraph = ['Average indoor temperature (°C)'];
      const secondDatasets = secondGraph.map((category) => {
        return {
          data: data.map((elem) => Number(elem[category])),
          label: category,
          fill: false,
          tension: 0,
          borderColor: 'black',
          backgroundColor: 'rgba(255,0,0,0.3)'
        }
      }
      );
      this.lineChartData2.datasets = secondDatasets;
      this.lineChartData2.labels = Array.from(Array(data.length).keys());
      this.lineChart.chart!.update('none');
    })
  }

 // setInterval():void{

 // }

  //TODO let the user choose from which timesetp to which timestep they should see the data; aka let the user zoom in the graph
  //TODO same thing with y axis, let the user choose the min and max values

  public lineChartData: ChartConfiguration<'line'>['data'] = {
    labels: [],
    datasets: [
      {
        data: [], //TODO this is also placeholder data
        label: 'Series A', //TODO this is also placeholder data
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
        label: 'Series A', //TODO this is also placeholder data
        fill: false,
        tension: 0,
        borderColor: 'black',
        backgroundColor: 'rgba(255,0,0,0.3)'
      }
    ]
  };

  public lineChartOptions: ChartOptions<'line'> = {
    responsive: true
  };

  public lineChartLegend = true;

}
