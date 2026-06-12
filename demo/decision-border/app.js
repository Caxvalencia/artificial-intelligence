/* global Perceptron */

const canvas = document.querySelector('#border-canvas');
const context = canvas.getContext('2d');
const samples = [];
const toolButtons = document.querySelectorAll('[data-mode]');
let mode = '0';
let model = null;
let testPoint = null;

function train() {
  model = null;
  const classes = new Set(samples.map((sample) => sample.output));
  if (samples.length < 2 || classes.size < 2) return;

  try {
    const candidate = new Perceptron();
    candidate.setMaxEpochs(2000);
    samples.forEach((sample) => candidate.addData(sample.input, sample.output));
    candidate.learn();
    model = candidate;
    setStatus(
      `La frontera separa correctamente los ${samples.length} ejemplos.`,
      false,
      candidate.trainingStats
    );
  } catch (error) {
    setStatus('Estos puntos no pueden separarse con una sola línea.', true);
  }
}

function setStatus(message, error = false, stats = null) {
  const status = document.querySelector('#status-message');
  status.textContent = message;
  status.className = `status-message${error ? ' error' : ''}`;
  document.querySelector('#sample-count').textContent = samples.length;
  document.querySelector('#epoch-count').textContent = stats?.epochs ?? '—';
  document.querySelector('#convergence').textContent = stats?.converged ? 'Sí' : error ? 'No' : '—';
}

function draw() {
  const width = canvas.width;
  const height = canvas.height;
  const padding = 44;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f7faf7';
  context.fillRect(0, 0, width, height);

  if (model) {
    const cells = 42;
    for (let x = 0; x < cells; x++) {
      for (let y = 0; y < cells; y++) {
        const prediction = model.process([x / cells, 1 - y / cells]);
        context.fillStyle = prediction ? 'rgba(24,166,106,.13)' : 'rgba(255,133,95,.13)';
        context.fillRect(
          padding + (x * plotWidth) / cells,
          padding + (y * plotHeight) / cells,
          plotWidth / cells + 1,
          plotHeight / cells + 1
        );
      }
    }
  }

  context.strokeStyle = 'rgba(70,95,84,.13)';
  context.lineWidth = 1;
  for (let index = 0; index <= 5; index++) {
    const x = padding + (plotWidth / 5) * index;
    const y = padding + (plotHeight / 5) * index;
    context.beginPath();
    context.moveTo(x, padding);
    context.lineTo(x, padding + plotHeight);
    context.moveTo(padding, y);
    context.lineTo(padding + plotWidth, y);
    context.stroke();
  }

  if (model && Math.abs(model.weights[1]) > 0.00001) {
    const yAt = (x) => -(model.weights[0] * x + model.threshold) / model.weights[1];
    context.beginPath();
    context.moveTo(padding, padding + (1 - yAt(0)) * plotHeight);
    context.lineTo(padding + plotWidth, padding + (1 - yAt(1)) * plotHeight);
    context.strokeStyle = '#17221f';
    context.lineWidth = 3;
    context.stroke();
  }

  samples.forEach((sample) =>
    drawPoint(sample.input, sample.output, false, padding, plotWidth, plotHeight)
  );
  if (testPoint) drawPoint(testPoint.input, testPoint.output, true, padding, plotWidth, plotHeight);
}

function drawPoint(input, output, test, padding, plotWidth, plotHeight) {
  const x = padding + input[0] * plotWidth;
  const y = padding + (1 - input[1]) * plotHeight;
  context.beginPath();
  context.arc(x, y, test ? 11 : 8, 0, Math.PI * 2);
  context.fillStyle = test ? '#4c67e8' : output ? '#18a66a' : '#ff855f';
  context.fill();
  context.lineWidth = 3;
  context.strokeStyle = '#fff';
  context.stroke();
}

function renderSamples() {
  document.querySelector('#sample-list').innerHTML = samples
    .slice(-8)
    .reverse()
    .map(
      (sample) =>
        `<div class="sample-row"><span>${sample.input.map((value) => value.toFixed(2)).join(' · ')}</span><strong>Clase ${sample.output}</strong></div>`
    )
    .join('');
}

canvas.addEventListener('click', (event) => {
  const rect = canvas.getBoundingClientRect();
  const padding = 44;
  const x =
    ((event.clientX - rect.left) * (canvas.width / rect.width) - padding) /
    (canvas.width - padding * 2);
  const y =
    1 -
    ((event.clientY - rect.top) * (canvas.height / rect.height) - padding) /
      (canvas.height - padding * 2);
  if (x < 0 || x > 1 || y < 0 || y > 1) return;

  if (mode === 'test') {
    if (!model) return setStatus('Entrena primero con puntos de ambas clases.', true);
    testPoint = { input: [x, y], output: model.process([x, y]) };
  } else {
    samples.push({ input: [x, y], output: Number(mode) });
    testPoint = null;
    train();
    renderSamples();
  }
  draw();
});

toolButtons.forEach((button) =>
  button.addEventListener('click', () => {
    mode = button.dataset.mode;
    toolButtons.forEach((item) => item.classList.toggle('active', item === button));
  })
);

document.querySelector('#preset-button').addEventListener('click', () => {
  samples.splice(
    0,
    samples.length,
    { input: [0.18, 0.2], output: 0 },
    { input: [0.28, 0.42], output: 0 },
    { input: [0.42, 0.25], output: 0 },
    { input: [0.62, 0.7], output: 1 },
    { input: [0.78, 0.58], output: 1 },
    { input: [0.82, 0.84], output: 1 }
  );
  train();
  renderSamples();
  draw();
});

document.querySelector('#clear-button').addEventListener('click', () => {
  samples.length = 0;
  model = null;
  testPoint = null;
  setStatus('Añade puntos de ambas clases para comenzar el entrenamiento.');
  renderSamples();
  draw();
});

setStatus('Añade puntos de ambas clases para comenzar el entrenamiento.');
draw();
