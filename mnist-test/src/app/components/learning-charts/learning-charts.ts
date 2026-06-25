import { CommonModule } from '@angular/common';
import {
  ChangeDetectorRef,
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnInit,
  SimpleChanges,
  ViewChild,
  ViewEncapsulation,
  inject,
} from '@angular/core';
import { Chart, registerables } from 'chart.js';
import { TrainingProgress } from '../../services/training.service';

Chart.register(...registerables);

@Component({
  selector: 'app-learning-charts',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './learning-charts.html',
  encapsulation: ViewEncapsulation.None,
})
export class LearningChartsComponent implements OnInit, OnChanges {
  private cdr = inject(ChangeDetectorRef);
  @Input() progress!: TrainingProgress;
  @Input() epochs = 5;

  @ViewChild('chartCanvas', { static: false }) chartCanvas!: ElementRef<HTMLCanvasElement>;
  private chart: Chart | null = null;
  public chartData = {
    labels: [] as string[],
    loss: [] as number[],
    valLoss: [] as number[],
    acc: [] as number[],
    valAcc: [] as number[],
  };

  ngOnInit() {
    this.resetChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['progress']) {
      const progressVal = changes['progress'].currentValue as TrainingProgress;
      if (progressVal) {
        if (progressVal.isTraining && progressVal.epoch > 0) {
          this.updateChart(progressVal);
        } else if (!progressVal.isTraining && progressVal.epoch === 0) {
          this.resetChart();
        }
        this.cdr.detectChanges();
      }
    }
  }

  public resetChart() {
    this.chartData = { labels: [], loss: [], valLoss: [], acc: [], valAcc: [] };
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }

  private initChart() {
    if (!this.chartCanvas) return;
    const ctx = this.chartCanvas.nativeElement.getContext('2d')!;

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.chartData.labels,
        datasets: [
          {
            label: 'Pérdida (Loss)',
            data: this.chartData.loss,
            borderColor: '#ff4d6d',
            backgroundColor: 'rgba(255, 77, 109, 0.1)',
            yAxisID: 'y',
            tension: 0.3,
          },
          {
            label: 'Val Pérdida (Val Loss)',
            data: this.chartData.valLoss,
            borderColor: '#ff9f1c',
            backgroundColor: 'rgba(255, 159, 28, 0.1)',
            yAxisID: 'y',
            tension: 0.3,
          },
          {
            label: 'Precisión (Acc)',
            data: this.chartData.acc,
            borderColor: '#4cc9f0',
            backgroundColor: 'rgba(76, 201, 240, 0.1)',
            yAxisID: 'y1',
            tension: 0.3,
          },
          {
            label: 'Val Precisión (Val Acc)',
            data: this.chartData.valAcc,
            borderColor: '#7209b7',
            backgroundColor: 'rgba(114, 9, 183, 0.1)',
            yAxisID: 'y1',
            tension: 0.3,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#e0e0e0' },
          },
        },
        scales: {
          x: {
            ticks: { color: '#a0a0a0' },
            grid: { color: '#333' },
          },
          y: {
            type: 'linear',
            display: true,
            position: 'left',
            ticks: { color: '#ff9f1c' },
            grid: { color: '#333' },
            title: { display: true, text: 'Pérdida', color: '#ff9f1c' },
          },
          y1: {
            type: 'linear',
            display: true,
            position: 'right',
            ticks: { color: '#4cc9f0' },
            grid: { drawOnChartArea: false },
            title: { display: true, text: 'Precisión', color: '#4cc9f0' },
          },
        },
      },
    });
  }

  private updateChart(progress: TrainingProgress) {
    const epochLabel = `Época ${progress.epoch}`;
    if (this.chartData.labels.includes(epochLabel)) {
      const idx = this.chartData.labels.indexOf(epochLabel);
      this.chartData.loss[idx] = progress.loss;
      this.chartData.valLoss[idx] = progress.valLoss;
      this.chartData.acc[idx] = progress.acc;
      this.chartData.valAcc[idx] = progress.valAcc;
    } else {
      this.chartData.labels.push(epochLabel);
      this.chartData.loss.push(progress.loss);
      this.chartData.valLoss.push(progress.valLoss);
      this.chartData.acc.push(progress.acc);
      this.chartData.valAcc.push(progress.valAcc);
    }

    if (this.chart) {
      this.chart.update('none');
    } else {
      this.initChart();
    }
  }
}
