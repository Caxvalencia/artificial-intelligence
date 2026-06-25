import { Component, EventEmitter, Input, Output, ViewEncapsulation } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './header.html',
  encapsulation: ViewEncapsulation.None,
})
export class HeaderComponent {
  @Input() isLoaded = false;
  @Input() loadProgress = 0;
  @Input() backendSelection: 'webgpu' | 'webgl' | 'cpu' = 'webgl';
  @Input() isWebGPUSupported = false;

  @Output() backendSelectionChange = new EventEmitter<'webgpu' | 'webgl' | 'cpu'>();

  onBackendChange(value: 'webgpu' | 'webgl' | 'cpu') {
    this.backendSelectionChange.emit(value);
  }
}
