import { CommonModule } from '@angular/common';
import {
  Component,
  EventEmitter,
  Input,
  OnInit,
  Output,
  ViewEncapsulation,
  inject,
  ChangeDetectorRef,
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LayerDescription } from '../../services/model-builder.service';
import { TrainingProgress, TrainingService } from '../../services/training.service';

@Component({
  selector: 'app-training-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './training-panel.html',
  encapsulation: ViewEncapsulation.None,
})
export class TrainingPanelComponent implements OnInit {
  public trainingService = inject(TrainingService);
  private cdr = inject(ChangeDetectorRef);

  @Input() layers: LayerDescription[] = [];
  @Input() selectedLayer: LayerDescription | null = null;
  @Input() optimizer = 'adam';
  @Input() learningRate = 0.01;
  @Input() lossFunction = 'categoricalCrossentropy';
  @Input() epochs = 5;
  @Input() batchSize = 64;
  @Input() trainingProgress!: TrainingProgress;
  @Input() isLoaded = false;

  @Output() selectedLayerChange = new EventEmitter<LayerDescription | null>();
  @Output() optimizerChange = new EventEmitter<string>();
  @Output() learningRateChange = new EventEmitter<number>();
  @Output() lossFunctionChange = new EventEmitter<string>();
  @Output() epochsChange = new EventEmitter<number>();
  @Output() batchSizeChange = new EventEmitter<number>();
  @Output() compileModel = new EventEmitter<void>();
  @Output() startTraining = new EventEmitter<void>();
  @Output() exportModel = new EventEmitter<void>();

  // Logs local state
  logs: string[] = [];

  ngOnInit() {
    // Subscribe to training logs
    this.trainingService.logMessages$.subscribe((messages) => {
      this.logs = messages;
      this.cdr.detectChanges();
    });
  }

  // --- ACTIONS ---
  onCompileModel() {
    this.compileModel.emit();
  }

  onStartTraining() {
    this.startTraining.emit();
  }

  onExportModel() {
    this.exportModel.emit();
  }
}
