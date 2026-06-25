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
    loss: [] as { x: number; y: number }[],
    valLoss: [] as { x: number; y: number }[],
    acc: [] as { x: number; y: number }[],
    valAcc: [] as { x: number; y: number }[],
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
    this.chartData = {
      loss: [],
      valLoss: [],
      acc: [],
      valAcc: [],
    };
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
            type: 'linear',
            ticks: {
              color: '#a0a0a0',
              callback: function(value: any) {
                const val = Number(value);
                if (val === 0) return '0';
                if (Number.isInteger(val)) {
                  return `Época ${val}`;
                }
                return '';
              },
            },
            grid: { color: '#333' },
            min: 0,
            max: this.epochs,
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
    if (!this.chart) {
      this.initChart();
    }

    const batchesPerEpoch = progress.batchesPerEpoch || 938;
    const epoch = progress.epoch;
    const batch = progress.batch;

    if (progress.isEpochEnd) {
      const xVal = epoch;

      // Añadimos el valor de validación
      if (!this.chartData.valLoss.some((p) => p.x === xVal)) {
        this.chartData.valLoss.push({ x: xVal, y: progress.valLoss });
        this.chartData.valAcc.push({ x: xVal, y: progress.valAcc });
      }

      // También el punto de entrenamiento al final de la época
      if (!this.chartData.loss.some((p) => p.x === xVal)) {
        this.chartData.loss.push({ x: xVal, y: progress.loss });
        this.chartData.acc.push({ x: xVal, y: progress.acc });
      }
    } else if (batch > 0) {
      // Actualización intermedia por lote
      if (batch % 10 === 0 || batch === batchesPerEpoch) {
        const xVal = epoch - 1 + batch / batchesPerEpoch;

        if (!this.chartData.loss.some((p) => p.x === xVal)) {
          this.chartData.loss.push({ x: xVal, y: progress.loss });
          this.chartData.acc.push({ x: xVal, y: progress.acc });
        }
      }
    }

    if (this.chart) {
      this.chart.update('none');
    }
  }
}
