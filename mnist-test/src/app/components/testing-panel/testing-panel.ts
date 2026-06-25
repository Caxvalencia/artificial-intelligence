import {
  Component,
  ElementRef,
  OnInit,
  AfterViewInit,
  ViewChild,
  ViewEncapsulation,
  inject,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { TrainingService } from '../../services/training.service';

@Component({
  selector: 'app-testing-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './testing-panel.html',
  encapsulation: ViewEncapsulation.None,
})
export class TestingPanelComponent implements OnInit, AfterViewInit {
  public trainingService = inject(TrainingService);

  // Drawing Canvas
  @ViewChild('drawingCanvas', { static: false }) drawingCanvas!: ElementRef<HTMLCanvasElement>;
  private ctx!: CanvasRenderingContext2D;
  private isDrawing = false;
  private rawDrawnPixels: any = new Float32Array(784); // 28x28 grayscale [0, 1]

  // Hopfield Canvas Reference
  @ViewChild('hopfieldCanvas', { static: false }) hopfieldCanvas!: ElementRef<HTMLCanvasElement>;

  // Live predictions
  predictions: number[] = new Array(10).fill(0);
  predictedDigit: number | null = null;

  // UI state
  activeTab: 'model' | 'hopfield' = 'model';
  hopfieldNoise = 0.2;
  hopfieldSteps = 5;

  ngOnInit() {
    this.predictions.fill(0);
  }

  ngAfterViewInit() {
    this.initDrawingCanvas();
  }

  ensureCanvasInitialized(): boolean {
    if (this.ctx) return true;
    if (!this.drawingCanvas) return false;
    const canvas = this.drawingCanvas.nativeElement;
    this.ctx = canvas.getContext('2d', { willReadFrequently: true })!;
    return true;
  }

  initDrawingCanvas() {
    if (this.ensureCanvasInitialized()) {
      this.clearCanvas();
    }
  }

  clearCanvas() {
    if (!this.ensureCanvasInitialized()) return;
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, 280, 280);
    this.isDrawing = false;
    this.rawDrawnPixels.fill(0);
    this.predictions.fill(0);
    this.predictedDigit = null;
    this.drawScaledHopfield(this.rawDrawnPixels);
  }

  onMouseDown(e: MouseEvent) {
    this.isDrawing = true;
    this.draw(e);
  }

  onMouseMove(e: MouseEvent) {
    if (!this.isDrawing) return;
    this.draw(e);
  }

  onMouseUp() {
    this.isDrawing = false;
    this.processDrawing();
  }

  onTouchStart(e: TouchEvent) {
    this.isDrawing = true;
    this.drawTouch(e);
    e.preventDefault();
  }

  onTouchMove(e: TouchEvent) {
    if (!this.isDrawing) return;
    this.drawTouch(e);
    e.preventDefault();
  }

  onTouchEnd(e: TouchEvent) {
    this.isDrawing = false;
    this.processDrawing();
    e.preventDefault();
  }

  private draw(e: MouseEvent) {
    if (!this.ensureCanvasInitialized()) return;
    const canvas = this.drawingCanvas.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.beginPath();
    this.ctx.arc(x, y, 12, 0, Math.PI * 2);
    this.ctx.fill();
  }

  private drawTouch(e: TouchEvent) {
    if (e.touches.length === 0) return;
    if (!this.ensureCanvasInitialized()) return;
    const canvas = this.drawingCanvas.nativeElement;
    const rect = canvas.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    const y = e.touches[0].clientY - rect.top;

    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.beginPath();
    this.ctx.arc(x, y, 12, 0, Math.PI * 2);
    this.ctx.fill();
  }

  private processDrawing() {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d')!;

    tempCtx.drawImage(this.drawingCanvas.nativeElement, 0, 0, 28, 28);
    const imgData = tempCtx.getImageData(0, 0, 28, 28);

    for (let i = 0; i < 784; i++) {
      this.rawDrawnPixels[i] = imgData.data[i * 4] / 255;
    }

    this.runPrediction();
  }

  runPrediction() {
    if (!this.trainingService.model) {
      console.warn('No hay modelo para predecir.');
      return;
    }
    try {
      const result = this.trainingService.predict(this.rawDrawnPixels);
      this.predictions = result.probabilities;
      this.predictedDigit = result.prediction;
    } catch (e: any) {
      console.error('Error en predicción:', e);
      this.trainingService.logMessages$.next([
        `[${new Date().toLocaleTimeString()}] Error predicción: ${e.message || e}`,
        ...this.trainingService.logMessages$.value.slice(0, 49),
      ]);
    }
  }

  // --- HOPFIELD NETWORK LOGIC ---
  applyHopfieldNoise() {
    const noisy = new Float32Array(this.rawDrawnPixels);
    for (let i = 0; i < noisy.length; i++) {
      if (Math.random() < this.hopfieldNoise) {
        noisy[i] = noisy[i] > 0.5 ? 0 : 1;
      }
    }
    this.rawDrawnPixels = noisy;
    this.drawScaledDrawing(noisy);
    this.runPrediction();
  }

  reconstructHopfield() {
    const reconstructed = this.trainingService.reconstructWithHopfield(
      this.rawDrawnPixels,
      this.hopfieldSteps,
    );
    this.rawDrawnPixels = reconstructed as Float32Array;
    this.drawScaledDrawing(reconstructed);
    this.runPrediction();
  }

  private drawScaledDrawing(pixels: Float32Array) {
    if (!this.ensureCanvasInitialized()) return;
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, 280, 280);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const val = pixels[y * 28 + x];
        const hex = Math.floor(val * 255)
          .toString(16)
          .padStart(2, '0');
        this.ctx.fillStyle = `#${hex}${hex}${hex}`;
        this.ctx.fillRect(x * 10, y * 10, 10, 10);
      }
    }
    this.drawScaledHopfield(pixels);
  }

  private drawScaledHopfield(pixels: Float32Array) {
    if (!this.hopfieldCanvas) return;
    const canvas = this.hopfieldCanvas.nativeElement;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, 140, 140);

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        const val = pixels[y * 28 + x];
        const hex = Math.floor(val * 255)
          .toString(16)
          .padStart(2, '0');
        ctx.fillStyle = `#${hex}${hex}${hex}`;
        ctx.fillRect(x * 5, y * 5, 5, 5);
      }
    }
  }
}
