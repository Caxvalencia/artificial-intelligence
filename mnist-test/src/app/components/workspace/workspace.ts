import {
  Component,
  EventEmitter,
  Input,
  Output,
  HostListener,
  ViewEncapsulation,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { LayerDescription } from '../../services/model-builder.service';

@Component({
  selector: 'app-workspace',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './workspace.html',
  encapsulation: ViewEncapsulation.None,
})
export class WorkspaceComponent {
  @Input() layers: LayerDescription[] = [];
  @Input() selectedLayer: LayerDescription | null = null;

  @Output() layerSelected = new EventEmitter<LayerDescription>();
  @Output() layerRemoved = new EventEmitter<number>();
  @Output() layerAdded = new EventEmitter<
    'dense' | 'conv2d' | 'maxPool2d' | 'flatten' | 'dropout' | 'attention'
  >();
  @Output() presetLoaded = new EventEmitter<'cnn' | 'dense' | 'transformer'>();

  // Node Drag and Drop
  public activeDragNode: LayerDescription | null = null;
  private dragStartX = 0;
  private dragStartY = 0;
  private nodeStartX = 0;
  private nodeStartY = 0;

  // Workspace Panning
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
    } else if (this.isPanning) {
      const dx = e.clientX - this.panStartX;
      const dy = e.clientY - this.panStartY;
      this.panX = this.startPanX + dx;
      this.panY = this.startPanY + dy;
    }
  }

  @HostListener('document:mouseup')
  onWorkspaceMouseUp() {
    this.activeDragNode = null;
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
    this.panX -= e.deltaX;
    this.panY -= e.deltaY;
    e.preventDefault();
  }

  resetView() {
    this.panX = 0;
    this.panY = 0;
  }

  selectLayer(layer: LayerDescription) {
    this.layerSelected.emit(layer);
  }

  removeLayer(index: number) {
    this.layerRemoved.emit(index);
  }

  addLayer(type: 'dense' | 'conv2d' | 'maxPool2d' | 'flatten' | 'dropout' | 'attention') {
    this.layerAdded.emit(type);
  }

  loadPreset(type: 'cnn' | 'dense' | 'transformer') {
    this.presetLoaded.emit(type);
    this.resetView(); // Local reset
  }

  getConnectionPath(idx: number): string {
    const nodeA = this.layers[idx];
    const nodeB = this.layers[idx + 1];
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
}
