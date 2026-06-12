/* global Backpropagation */

const scenarios = [
  {
    id: 'xor',
    icon: 'XOR',
    title: 'Compuerta XOR',
    short: 'Decisión no lineal',
    description:
      'XOR solo activa la salida cuando las dos entradas son diferentes. Demuestra por qué una capa oculta puede resolver problemas no lineales.',
    axes: ['Entrada A', 'Entrada B'],
    layers: [4, 1],
    seed: 11,
    data: [
      { input: [0.12, 0.12], output: 0, name: 'Ambas apagadas' },
      { input: [0.12, 0.88], output: 1, name: 'Solo B activa' },
      { input: [0.88, 0.12], output: 1, name: 'Solo A activa' },
      { input: [0.88, 0.88], output: 0, name: 'Ambas activas' }
    ]
  },
  {
    id: 'and',
    icon: 'AND',
    title: 'Compuerta AND',
    short: 'Regla lineal',
    description:
      'La salida solo se activa cuando ambas condiciones se cumplen. Es una frontera sencilla que sirve para comparar con XOR.',
    axes: ['Condición A', 'Condición B'],
    layers: [3, 1],
    seed: 7,
    data: [
      { input: [0.12, 0.12], output: 0, name: 'Ninguna condición' },
      { input: [0.12, 0.88], output: 0, name: 'Solo condición B' },
      { input: [0.88, 0.12], output: 0, name: 'Solo condición A' },
      { input: [0.88, 0.88], output: 1, name: 'Ambas condiciones' }
    ]
  },
  {
    id: 'fraud',
    icon: '₿',
    title: 'Riesgo de fraude',
    short: 'Fintech',
    description:
      'Clasifica transacciones usando importe relativo y anomalía geográfica. Es una simplificación visual de un sistema de detección de riesgo.',
    axes: ['Importe', 'Anomalía geográfica'],
    layers: [5, 1],
    seed: 23,
    data: [
      { input: [0.1, 0.14], output: 0, name: 'Compra habitual' },
      { input: [0.24, 0.3], output: 0, name: 'Compra local' },
      { input: [0.43, 0.16], output: 0, name: 'Importe medio conocido' },
      { input: [0.74, 0.82], output: 1, name: 'Importe alto y ubicación nueva' },
      { input: [0.91, 0.6], output: 1, name: 'Importe extremo' },
      { input: [0.38, 0.91], output: 1, name: 'Ubicación muy anómala' },
      { input: [0.68, 0.36], output: 1, name: 'Patrón de riesgo' }
    ]
  },
  {
    id: 'retention',
    icon: 'CRM',
    title: 'Riesgo de abandono',
    short: 'Retención',
    description:
      'Estima qué clientes podrían abandonar según su baja actividad y cantidad de incidencias. Ayuda a priorizar acciones de retención.',
    axes: ['Baja actividad', 'Incidencias'],
    layers: [5, 1],
    seed: 31,
    data: [
      { input: [0.12, 0.14], output: 0, name: 'Cliente activo' },
      { input: [0.27, 0.38], output: 0, name: 'Uso saludable' },
      { input: [0.48, 0.2], output: 0, name: 'Menos actividad, sin problemas' },
      { input: [0.63, 0.72], output: 1, name: 'Señales de abandono' },
      { input: [0.86, 0.42], output: 1, name: 'Actividad crítica' },
      { input: [0.4, 0.88], output: 1, name: 'Muchas incidencias' },
      { input: [0.88, 0.86], output: 1, name: 'Riesgo máximo' }
    ]
  }
];

const state = {
  scenario: scenarios[0],
  model: null,
  testPoint: null,
  animation: null
};

const elements = {
  scenarioList: document.querySelector('#scenario-list'),
  scenarioTitle: document.querySelector('#scenario-title'),
  scenarioDescription: document.querySelector('#scenario-description'),
  epochs: document.querySelector('#epochs'),
  epochsValue: document.querySelector('#epochs-value'),
  learningRate: document.querySelector('#learning-rate'),
  learningRateValue: document.querySelector('#learning-rate-value'),
  trainButton: document.querySelector('#train-button'),
  decisionCanvas: document.querySelector('#decision-canvas'),
  lossCanvas: document.querySelector('#loss-canvas'),
  predictionList: document.querySelector('#prediction-list'),
  metricLoss: document.querySelector('#metric-loss'),
  metricAccuracy: document.querySelector('#metric-accuracy'),
  metricEpochs: document.querySelector('#metric-epochs'),
  metricStatus: document.querySelector('#metric-status'),
  livePill: document.querySelector('#live-pill'),
  mapTitle: document.querySelector('#map-title')
};

function formatInteger(value) {
  return new Intl.NumberFormat('es-ES').format(value);
}

function formatDecimal(value, digits = 3) {
  return new Intl.NumberFormat('es-ES', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(value);
}

function renderScenarioButtons() {
  elements.scenarioList.innerHTML = scenarios
    .map(
      (scenario) => `
        <button class="scenario-button ${scenario.id === state.scenario.id ? 'active' : ''}"
          type="button" data-scenario="${scenario.id}">
          <span class="scenario-icon">${scenario.icon}</span>
          <span>${scenario.title}<small>${scenario.short}</small></span>
        </button>
      `
    )
    .join('');

  elements.scenarioList.querySelectorAll('small').forEach((element) => {
    element.style.display = 'block';
    element.style.marginTop = '2px';
    element.style.color = '#8a948f';
    element.style.fontSize = '9px';
    element.style.fontWeight = '600';
  });
}

function selectScenario(id) {
  state.scenario = scenarios.find((scenario) => scenario.id === id);
  state.model = null;
  state.testPoint = null;
  cancelAnimationFrame(state.animation);
  renderScenarioButtons();
  elements.scenarioTitle.textContent = state.scenario.title;
  elements.scenarioDescription.textContent = state.scenario.description;
  elements.mapTitle.textContent = `${state.scenario.axes[0]} vs. ${state.scenario.axes[1]}`;
  resetMetrics();
  drawDecisionMap();
  drawLossChart([]);
  renderPredictions();
}

function resetMetrics() {
  elements.metricLoss.textContent = '—';
  elements.metricAccuracy.textContent = '—';
  elements.metricEpochs.textContent = '—';
  elements.metricStatus.textContent = 'Esperando entrenamiento';
  elements.livePill.className = 'live-pill';
  elements.livePill.innerHTML = '<i></i>Sin entrenar';
}

function trainModel() {
  elements.trainButton.disabled = true;
  elements.trainButton.innerHTML = '<span class="button-icon">●</span> Entrenando...';
  elements.livePill.className = 'live-pill training';
  elements.livePill.innerHTML = '<i></i>Entrenando';

  requestAnimationFrame(() => {
    const epochs = Number(elements.epochs.value);
    const learningRate = Number(elements.learningRate.value);
    const model = new Backpropagation({
      epochs,
      learningRate,
      momentum: 0.72,
      seed: state.scenario.seed,
      shuffle: true,
      targetLoss: 0.0008
    });

    state.scenario.layers.forEach((size) => model.addLayer(size));
    model.learn(state.scenario.data);
    state.model = model;
    state.testPoint = null;

    updateMetrics();
    drawDecisionMap();
    renderPredictions();
    animateLossChart(model.history.loss);

    elements.trainButton.disabled = false;
    elements.trainButton.innerHTML = '<span class="button-icon">↻</span> Entrenar de nuevo';
  });
}

function updateMetrics() {
  const loss = state.model.history.loss.at(-1);
  const correct = state.scenario.data.filter((sample) => {
    return Math.round(state.model.process(sample.input)[0]) === sample.output;
  }).length;
  const accuracy = (correct / state.scenario.data.length) * 100;

  elements.metricLoss.textContent = formatDecimal(loss, loss < 0.001 ? 5 : 3);
  elements.metricAccuracy.textContent = `${Math.round(accuracy)}%`;
  elements.metricEpochs.textContent = formatInteger(state.model.history.epochs);
  elements.metricStatus.textContent = state.model.history.stoppedEarly
    ? 'Objetivo alcanzado antes del límite'
    : 'Límite de épocas completado';
}

function canvasContext(canvas) {
  return canvas.getContext('2d');
}

function drawDecisionMap() {
  const canvas = elements.decisionCanvas;
  const context = canvasContext(canvas);
  const width = canvas.width;
  const height = canvas.height;
  const padding = 54;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f7faf7';
  context.fillRect(0, 0, width, height);

  if (state.model) {
    const cells = 38;
    const cellWidth = plotWidth / cells;
    const cellHeight = plotHeight / cells;

    for (let x = 0; x < cells; x++) {
      for (let y = 0; y < cells; y++) {
        const output = state.model.process([x / (cells - 1), 1 - y / (cells - 1)])[0];
        const red = [255, 133, 95];
        const green = [24, 166, 106];
        const mixed = red.map((channel, index) =>
          Math.round(channel * (1 - output) + green[index] * output)
        );
        context.fillStyle = `rgba(${mixed.join(',')}, 0.22)`;
        context.fillRect(
          padding + x * cellWidth,
          padding + y * cellHeight,
          Math.ceil(cellWidth) + 1,
          Math.ceil(cellHeight) + 1
        );
      }
    }
  }

  drawGrid(context, padding, plotWidth, plotHeight);

  state.scenario.data.forEach((sample) => {
    const x = padding + sample.input[0] * plotWidth;
    const y = padding + (1 - sample.input[1]) * plotHeight;
    drawPoint(context, x, y, sample.output, false);
  });

  if (state.testPoint) {
    drawPoint(context, state.testPoint.x, state.testPoint.y, state.testPoint.output, true);
  }

  context.fillStyle = '#77827c';
  context.font = '600 12px Inter, sans-serif';
  context.textAlign = 'center';
  context.fillText(state.scenario.axes[0], width / 2, height - 13);
  context.save();
  context.translate(16, height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText(state.scenario.axes[1], 0, 0);
  context.restore();
}

function drawGrid(context, padding, plotWidth, plotHeight) {
  context.strokeStyle = 'rgba(70, 95, 84, 0.12)';
  context.lineWidth = 1;
  context.font = '10px Inter, sans-serif';
  context.fillStyle = '#89938e';
  context.textAlign = 'center';

  for (let index = 0; index <= 4; index++) {
    const x = padding + (plotWidth / 4) * index;
    const y = padding + (plotHeight / 4) * index;
    context.beginPath();
    context.moveTo(x, padding);
    context.lineTo(x, padding + plotHeight);
    context.moveTo(padding, y);
    context.lineTo(padding + plotWidth, y);
    context.stroke();
    context.fillText(String(index * 25), x, padding + plotHeight + 18);
  }
}

function drawPoint(context, x, y, output, isTest) {
  context.beginPath();
  context.arc(x, y, isTest ? 10 : 8, 0, Math.PI * 2);
  context.fillStyle = output >= 0.5 ? '#18a66a' : '#ff855f';
  context.fill();
  context.lineWidth = isTest ? 4 : 3;
  context.strokeStyle = isTest ? '#17221f' : '#ffffff';
  context.stroke();

  if (isTest) {
    context.fillStyle = '#17221f';
    context.font = '800 11px Inter, sans-serif';
    context.textAlign = 'center';
    context.fillText(`${Math.round(output * 100)}%`, x, y - 17);
  }
}

function drawLossChart(loss, progress = 1) {
  const canvas = elements.lossCanvas;
  const context = canvasContext(canvas);
  const width = canvas.width;
  const height = canvas.height;
  const padding = 54;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;

  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f7faf7';
  context.fillRect(0, 0, width, height);
  drawGrid(context, padding, plotWidth, plotHeight);

  if (loss.length === 0) {
    context.fillStyle = '#89938e';
    context.font = '600 13px Inter, sans-serif';
    context.textAlign = 'center';
    context.fillText('Entrena el modelo para visualizar la curva', width / 2, height / 2);
    return;
  }

  const count = Math.max(2, Math.floor(loss.length * progress));
  const visible = loss.slice(0, count);
  const maxLoss = Math.max(...loss);
  const minLoss = Math.min(...loss);
  const range = Math.max(maxLoss - minLoss, 0.000001);

  context.beginPath();
  visible.forEach((value, index) => {
    const x = padding + (index / (loss.length - 1)) * plotWidth;
    const normalized = (value - minLoss) / range;
    const y = padding + (1 - normalized) * plotHeight;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = '#08754a';
  context.lineWidth = 4;
  context.lineJoin = 'round';
  context.stroke();

  context.lineTo(
    padding + ((visible.length - 1) / (loss.length - 1)) * plotWidth,
    height - padding
  );
  context.lineTo(padding, height - padding);
  context.closePath();
  const gradient = context.createLinearGradient(0, padding, 0, height - padding);
  gradient.addColorStop(0, 'rgba(24, 166, 106, 0.25)');
  gradient.addColorStop(1, 'rgba(24, 166, 106, 0)');
  context.fillStyle = gradient;
  context.fill();

  context.fillStyle = '#77827c';
  context.font = '600 12px Inter, sans-serif';
  context.textAlign = 'center';
  context.fillText('Épocas', width / 2, height - 13);
  context.save();
  context.translate(16, height / 2);
  context.rotate(-Math.PI / 2);
  context.fillText('Pérdida', 0, 0);
  context.restore();
}

function animateLossChart(loss) {
  cancelAnimationFrame(state.animation);
  const start = performance.now();
  const duration = 900;

  function frame(now) {
    const progress = Math.min((now - start) / duration, 1);
    drawLossChart(loss, 1 - Math.pow(1 - progress, 3));

    if (progress < 1) {
      state.animation = requestAnimationFrame(frame);
    } else {
      elements.livePill.className = 'live-pill training';
      elements.livePill.innerHTML = '<i></i>Entrenado';
    }
  }

  state.animation = requestAnimationFrame(frame);
}

function renderPredictions() {
  const items = state.scenario.data.map((sample) => {
    const output = state.model ? state.model.process(sample.input)[0] : 0.5;
    const predicted = Math.round(output);
    const confidence = Math.max(output, 1 - output);
    return `
      <div class="prediction-item">
        <span class="prediction-name">${sample.name}</span>
        <span class="prediction-value">${state.model ? `Clase ${predicted} · ${Math.round(confidence * 100)}%` : 'Pendiente'}</span>
        <span class="confidence-bar"><span style="width: ${state.model ? confidence * 100 : 0}%"></span></span>
      </div>
    `;
  });
  elements.predictionList.innerHTML = items.join('');
}

elements.scenarioList.addEventListener('click', (event) => {
  const button = event.target.closest('[data-scenario]');
  if (button) selectScenario(button.dataset.scenario);
});

elements.epochs.addEventListener('input', () => {
  elements.epochsValue.value = formatInteger(Number(elements.epochs.value));
});

elements.learningRate.addEventListener('input', () => {
  elements.learningRateValue.value = formatDecimal(Number(elements.learningRate.value), 2);
});

elements.trainButton.addEventListener('click', trainModel);

elements.decisionCanvas.addEventListener('click', (event) => {
  if (!state.model) return;
  const rect = elements.decisionCanvas.getBoundingClientRect();
  const scaleX = elements.decisionCanvas.width / rect.width;
  const scaleY = elements.decisionCanvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;
  const padding = 54;
  const plotWidth = elements.decisionCanvas.width - padding * 2;
  const plotHeight = elements.decisionCanvas.height - padding * 2;

  if (x < padding || x > padding + plotWidth || y < padding || y > padding + plotHeight) return;

  const input = [(x - padding) / plotWidth, 1 - (y - padding) / plotHeight];
  state.testPoint = { x, y, output: state.model.process(input)[0] };
  drawDecisionMap();
});

selectScenario('xor');
trainModel();
