import {
  Component,
  EventEmitter,
  Input,
  Output,
  HostListener,
  ViewEncapsulation,
  ViewChild,
  ElementRef,
  inject,
} from '@angular/core';
import { CommonModule, DOCUMENT } from '@angular/common';
import { LayerDescription } from '../../services/model-builder.service';

@Component({
  selector: 'app-workspace',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './workspace.html',
  encapsulation: ViewEncapsulation.None,
})
export class WorkspaceComponent {
  private document = inject(DOCUMENT);

  @Input() layers: LayerDescription[] = [];
  @Input() selectedLayer: LayerDescription | null = null;

  @Output() layerSelected = new EventEmitter<LayerDescription>();
  @Output() layerRemoved = new EventEmitter<number>();
  @Output() layerAdded = new EventEmitter<
    'dense' | 'conv2d' | 'maxPool2d' | 'flatten' | 'dropout' | 'attention' | 'concatenate' | 'add'
  >();
  @Output() presetLoaded = new EventEmitter<
    'cnn' | 'dense' | 'transformer' | 'hybrid' | 'innovative'
  >();
  @Output() connectionMade = new EventEmitter<{ outputLayerId: string; inputLayerId: string }>();
  @Output() connectionCleared = new EventEmitter<{ layerId: string }>();

  @ViewChild('workspaceEl') workspaceEl!: ElementRef<HTMLDivElement>;

  // Node Drag and Drop
  public activeDragNode: LayerDescription | null = null;
  private dragStartX = 0;
  private dragStartY = 0;
  private nodeStartX = 0;
  private nodeStartY = 0;

  // Connection Drag and Drop
  public activeDragConnection: {
    sourceLayer: LayerDescription;
    type: 'input' | 'output';
  } | null = null;
  public dragConnectionX = 0;
  public dragConnectionY = 0;
  public hoverTargetNode: LayerDescription | null = null;

  // Workspace Zoom & Panning
  public zoom = 1.0;
  public panX = 0;
  public panY = 0;
  public isPanning = false;
  private panStartX = 0;
  private panStartY = 0;
  private startPanX = 0;
  private startPanY = 0;

  // Listen globally to mouse events to ensure smooth dragging outside workspace
  @HostListener('document:mousemove', ['$event'])
  onWorkspaceMouseMove(e: MouseEvent) {
    if (this.activeDragNode) {
      const dx = e.clientX - this.dragStartX;
      const dy = e.clientY - this.dragStartY;
      this.activeDragNode.x = this.nodeStartX + dx;
      this.activeDragNode.y = this.nodeStartY + dy;
    } else if (this.activeDragConnection) {
      const coords = this.getWorkspaceCoords(e.clientX, e.clientY);
      this.dragConnectionX = coords.x;
      this.dragConnectionY = coords.y;

      // Find hover node target
      const element = this.document.elementFromPoint(e.clientX, e.clientY);
      const nodeCard = element?.closest('.node-card');
      const nodeId = nodeCard?.getAttribute('data-node-id');
      const target = this.layers.find((l) => l.id === nodeId) || null;

      // Prevent connecting to self
      if (target && target.id !== this.activeDragConnection.sourceLayer.id) {
        this.hoverTargetNode = target;
      } else {
        this.hoverTargetNode = null;
      }
    } else if (this.isPanning) {
      const dx = e.clientX - this.panStartX;
      const dy = e.clientY - this.panStartY;
      this.panX = this.startPanX + dx;
      this.panY = this.startPanY + dy;
    }
  }

  @HostListener('document:mouseup')
  onWorkspaceMouseUp() {
    if (this.activeDragConnection) {
      const source = this.activeDragConnection.sourceLayer;
      const target = this.hoverTargetNode;

      if (target) {
        let outputLayerId: string | null = null;
        let inputLayerId: string | null = null;

        if (this.activeDragConnection.type === 'output') {
          outputLayerId = source.id;
          inputLayerId = target.id;
        } else {
          outputLayerId = target.id;
          inputLayerId = source.id;
        }

        if (outputLayerId && inputLayerId) {
          this.connectionMade.emit({ outputLayerId, inputLayerId });
        }
      } else {
        // Soltado sobre espacio vacío: desconectar
        if (this.activeDragConnection.type === 'input') {
          this.connectionCleared.emit({ layerId: source.id });
        }
      }
    }

    this.activeDragNode = null;
    this.activeDragConnection = null;
    this.hoverTargetNode = null;
    this.isPanning = false;
  }

  startNodeDrag(e: MouseEvent, layer: LayerDescription) {
    this.activeDragNode = layer;
    this.dragStartX = e.clientX;
    this.dragStartY = e.clientY;
    this.nodeStartX = layer.x || 0;
    this.nodeStartY = layer.y || 0;
    this.layerSelected.emit(layer);
    e.stopPropagation();
    e.preventDefault();
  }

  startConnectionDrag(e: MouseEvent, layer: LayerDescription, type: 'input' | 'output') {
    this.activeDragConnection = {
      sourceLayer: layer,
      type: type,
    };
    const coords = this.getWorkspaceCoords(e.clientX, e.clientY);
    this.dragConnectionX = coords.x;
    this.dragConnectionY = coords.y;
    this.hoverTargetNode = null;
    e.stopPropagation();
    e.preventDefault();
  }

  private getWorkspaceCoords(clientX: number, clientY: number): { x: number; y: number } {
    if (!this.workspaceEl) return { x: clientX, y: clientY };
    const rect = this.workspaceEl.nativeElement.getBoundingClientRect();
    const localX = clientX - rect.left;
    const localY = clientY - rect.top;
    return {
      x: (localX - this.panX) / this.zoom,
      y: (localY - this.panY) / this.zoom,
    };
  }

  getTempConnectionPath(): string {
    if (!this.activeDragConnection) return '';
    const node = this.activeDragConnection.sourceLayer;
    const w = 180;
    const h = 76;

    let x1 = 0;
    let y1 = 0;
    if (this.activeDragConnection.type === 'output') {
      x1 = (node.x || 0) + w;
      y1 = (node.y || 0) + h / 2;
    } else {
      x1 = node.x || 0;
      y1 = (node.y || 0) + h / 2;
    }

    const x2 = this.dragConnectionX;
    const y2 = this.dragConnectionY;

    const dx = Math.max(40, Math.abs(x2 - x1) * 0.5);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }

  startWorkspacePan(e: MouseEvent) {
    // Only pan on left click
    if (e.button !== 0) return;

    const target = e.target as HTMLElement;
    // Check if we clicked on the background grid, the node-workspace container itself, or the connection lines area
    if (
      target.classList.contains('node-workspace') ||
      target.classList.contains('grid-background') ||
      target.tagName.toLowerCase() === 'svg'
    ) {
      this.isPanning = true;
      this.panStartX = e.clientX;
      this.panStartY = e.clientY;
      this.startPanX = this.panX;
      this.startPanY = this.panY;
      e.preventDefault();
    }
  }

  onWorkspaceWheel(e: WheelEvent) {
    if (e.ctrlKey || e.metaKey) {
      // Zoom centrado en la posición del puntero del ratón
      const zoomFactor = 1.1;
      const oldZoom = this.zoom;
      let newZoom = this.zoom;

      if (e.deltaY < 0) {
        newZoom = Math.min(2.0, this.zoom * zoomFactor);
      } else {
        newZoom = Math.max(0.4, this.zoom / zoomFactor);
      }

      const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const workspaceX = (mouseX - this.panX) / oldZoom;
      const workspaceY = (mouseY - this.panY) / oldZoom;

      this.zoom = newZoom;
      this.panX = mouseX - workspaceX * newZoom;
      this.panY = mouseY - workspaceY * newZoom;

      e.preventDefault();
    } else {
      // Desplazamiento (Pan) estándar
      this.panX -= e.deltaX;
      this.panY -= e.deltaY;
      e.preventDefault();
    }
  }

  zoomIn() {
    this.zoom = Math.min(2.0, this.zoom + 0.15);
  }

  zoomOut() {
    this.zoom = Math.max(0.4, this.zoom - 0.15);
  }

  resetZoom() {
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;
  }

  resetView() {
    this.resetZoom();
  }

  selectLayer(layer: LayerDescription) {
    this.layerSelected.emit(layer);
  }

  removeLayer(index: number) {
    this.layerRemoved.emit(index);
  }

  addLayer(type: 'dense' | 'conv2d' | 'maxPool2d' | 'flatten' | 'dropout' | 'attention' | 'concatenate' | 'add') {
    this.layerAdded.emit(type);
  }

  loadPreset(type: 'cnn' | 'dense' | 'transformer' | 'hybrid' | 'innovative') {
    this.presetLoaded.emit(type);
    this.resetView(); // Local reset
  }

  getNodeConnectionPath(parentId: string, childId: string): string {
    const nodeA = this.layers.find((l) => l.id === parentId);
    const nodeB = this.layers.find((l) => l.id === childId);
    if (!nodeA || !nodeB) return '';

    const w = 180; // Node width
    const h = 76; // Node height

    const x1 = (nodeA.x || 0) + w;
    const y1 = (nodeA.y || 0) + h / 2;

    const x2 = nodeB.x || 0;
    const y2 = (nodeB.y || 0) + h / 2;

    const dx = Math.max(40, Math.abs(x2 - x1) * 0.5);
    return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
  }

  getNodeTooltip(type: string): string {
    switch (type) {
      case 'dense':
        return 'Capa Dense: Neuronas completamente conectadas. Mapea y procesa características complejas para la clasificación.';
      case 'conv2d':
        return 'Capa Conv2D: Convolución espacial 2D. Aplica filtros para extraer patrones visuales como bordes, curvas y texturas.';
      case 'maxPool2d':
        return 'Capa MaxPool2D: Submuestreo. Reduce la dimensionalidad espacial reteniendo los valores más importantes.';
      case 'flatten':
        return 'Capa Flatten: Aplanamiento. Convierte mapas tridimensionales a un formato lineal 1D para conectar con capas densas.';
      case 'dropout':
        return 'Capa Dropout: Regularización estocástica. Apaga neuronas al azar para evitar sobreajuste y co-adaptación.';
      case 'reshape':
        return 'Capa Reshape: Redefinición dimensional. Cambia la forma espacial del tensor (ej. de vector 1D a grilla 2D/3D).';
      case 'attention':
        return 'Capa Attention: Autoatención atencional. Relaciona dinámicamente la importancia de partes distantes del tensor.';
      case 'concatenate':
        return 'Capa Concatenate: Une múltiples tensores entrantes a lo largo del último eje (e.g. combina características de diferentes ramas).';
      case 'add':
        return 'Capa Add: Suma elemento a elemento múltiples tensores entrantes (deben tener exactamente las mismas dimensiones).';
      default:
        return '';
    }
  }
}
