import { CommonModule } from '@angular/common';
import { Component, OnInit, ViewEncapsulation, inject, ChangeDetectorRef } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HeaderComponent } from './components/header/header';
import { LearningChartsComponent } from './components/learning-charts/learning-charts';
import { TestingPanelComponent } from './components/testing-panel/testing-panel';
import { TrainingPanelComponent } from './components/training-panel/training-panel';
import { WorkspaceComponent } from './components/workspace/workspace';
import { MnistDataService } from './services/mnist-data.service';
import { LayerDescription, ModelBuilderService } from './services/model-builder.service';
import { TrainingProgress, TrainingService } from './services/training.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    HeaderComponent,
    WorkspaceComponent,
    TrainingPanelComponent,
    TestingPanelComponent,
    LearningChartsComponent,
  ],
  templateUrl: './app.html',
  styleUrl: './app.css',
  encapsulation: ViewEncapsulation.None,
})
export class App implements OnInit {
  public dataService = inject(MnistDataService);
  public builderService = inject(ModelBuilderService);
  public trainingService = inject(TrainingService);
  private cdr = inject(ChangeDetectorRef);

  // Hyperparameters
  epochs = 5;
  batchSize = 64;
  learningRate = 0.01;
  optimizer = 'adam';
  lossFunction = 'categoricalCrossentropy';
  backendSelection: 'webgpu' | 'webgl' | 'cpu' = 'webgl';

  // Configured layers (nodes)
  layers: LayerDescription[] = [];
  selectedLayer: LayerDescription | null = null;

  // UI state
  mainTab: 'train' | 'test' = 'train';
  leftTab: 'canvas' | 'charts' = 'canvas';
  trainingProgress: TrainingProgress = {
    epoch: 0,
    batch: 0,
    loss: 0,
    acc: 0,
    valLoss: 0,
    valAcc: 0,
    isTraining: false,
  };

  ngOnInit() {
    // Load dataset automatically
    this.dataService.load().then(() => {
      this.loadPreset('cnn');
      this.trainingService.trainHopfield();
    });

    // Subscribe to progress
    this.trainingService.progress$.subscribe((progress) => {
      this.trainingProgress = progress;
      this.cdr.detectChanges();
    });

    // Sync backend selection
    setTimeout(() => {
      this.backendSelection = this.trainingService.currentBackend as any;
    }, 1000);
  }

  // Pre-configured architecture presets
  loadPreset(type: 'cnn' | 'dense' | 'transformer' | 'hybrid' | 'innovative') {
    let presetLayers: LayerDescription[] = [];
    if (type === 'cnn') {
      presetLayers = this.builderService.getDefaultCNNConfig();
    } else if (type === 'dense') {
      presetLayers = this.builderService.getDefaultDenseConfig();
    } else if (type === 'transformer') {
      presetLayers = this.builderService.getDefaultTransformerConfig();
    } else if (type === 'hybrid') {
      presetLayers = this.builderService.getDefaultHybridConfig();
    } else if (type === 'innovative') {
      presetLayers = this.builderService.getDefaultInnovativeConfig();
    }

    // Set horizontal sequence layout coordinates and linear connections
    this.layers = presetLayers.map((layer, i) => ({
      ...layer,
      x: 50 + i * 210,
      y: 120 + (i % 2) * 50,
      inputs: i > 0 ? [presetLayers[i - 1].id] : [],
    }));

    // Auto-select first node
    if (this.layers.length > 0) {
      this.selectedLayer = this.layers[0];
    } else {
      this.selectedLayer = null;
    }

    this.compileModel();
  }

  addLayer(
    type: 'dense' | 'conv2d' | 'maxPool2d' | 'flatten' | 'reshape' | 'attention' | 'dropout' | 'concatenate' | 'add',
  ) {
    const id = Date.now().toString();
    let config: any = {};

    switch (type) {
      case 'dense':
        config = { units: 64, activation: 'relu', kernelInitializer: 'varianceScaling' };
        break;
      case 'conv2d':
        config = {
          filters: 16,
          kernelSize: 3,
          strides: 1,
          activation: 'relu',
          kernelInitializer: 'varianceScaling',
        };
        break;
      case 'maxPool2d':
        config = { poolSize: [2, 2], strides: [2, 2] };
        break;
      case 'dropout':
        config = { rate: 0.25 };
        break;
      case 'reshape':
        config = { targetShape: [28, 28, 1] };
        break;
      case 'attention':
        config = { units: 32 };
        break;
      case 'concatenate':
        config = { axis: -1 };
        break;
      case 'add':
        config = {};
        break;
    }

    let newX = 50;
    let newY = 120;
    let insertIndex = this.layers.length;
    let inputs: string[] = [];

    if (this.selectedLayer) {
      const selectedIndex = this.layers.findIndex((l) => l.id === this.selectedLayer!.id);
      if (selectedIndex !== -1) {
        insertIndex = selectedIndex + 1;
        const selNode = this.layers[selectedIndex];
        newX = (selNode.x || 0) + 210;
        newY = selNode.y || 120;
        inputs = [selNode.id];

        // Shift subsequent nodes
        for (let i = insertIndex; i < this.layers.length; i++) {
          if (this.layers[i].x !== undefined) {
            this.layers[i].x! += 210;
          }
        }
      }
    } else if (this.layers.length > 0) {
      const lastNode = this.layers[this.layers.length - 1];
      newX = (lastNode.x || 0) + 210;
      newY = lastNode.y || 120;
      inputs = [lastNode.id];
    }

    const newLayer: LayerDescription = { id, type, config, x: newX, y: newY, inputs };
    this.layers.splice(insertIndex, 0, newLayer);
    this.selectedLayer = newLayer;
    this.compileModel();
  }

  removeLayer(index: number) {
    const removed = this.layers[index];
    this.layers.splice(index, 1);

    // Clean up connections pointing to the removed node
    this.layers.forEach((layer) => {
      if (layer.inputs) {
        layer.inputs = layer.inputs.filter((id) => id !== removed.id);
      }
    });

    // Auto-select adjacent layer if current selected is deleted
    if (this.selectedLayer?.id === removed.id) {
      this.selectedLayer = this.layers.length > 0 ? this.layers[Math.max(0, index - 1)] : null;
    }

    this.compileModel();
  }

  onConnectionMade(outputId: string, inputId: string) {
    const inputLayer = this.layers.find((l) => l.id === inputId);
    if (!inputLayer) return;

    if (!inputLayer.inputs) {
      inputLayer.inputs = [];
    }

    // Si es una capa de fusión, permitir múltiples entradas.
    // De lo contrario, reemplazar la conexión actual por la nueva.
    if (inputLayer.type === 'concatenate' || inputLayer.type === 'add') {
      if (!inputLayer.inputs.includes(outputId)) {
        inputLayer.inputs.push(outputId);
      }
    } else {
      inputLayer.inputs = [outputId];
    }

    this.compileModel();
  }

  onConnectionCleared(layerId: string) {
    const layer = this.layers.find((l) => l.id === layerId);
    if (layer) {
      layer.inputs = [];
      this.compileModel();
    }
  }

  selectLayer(layer: LayerDescription) {
    this.selectedLayer = layer;
  }

  compileModel() {
    try {
      this.trainingService.compileModel(
        this.layers,
        this.optimizer,
        this.learningRate,
        this.lossFunction,
      );
    } catch (e) {
      // Handled in trainingService log stream
    }
  }

  onBackendSelectionChange(backend: 'webgpu' | 'webgl' | 'cpu') {
    this.backendSelection = backend;
    this.trainingService.setBackend(this.backendSelection);
  }

  async startTraining() {
    this.leftTab = 'charts';
    await this.trainingService.trainModel(this.epochs, this.batchSize);
  }

  exportModel() {
    this.trainingService.exportModel();
  }
}
