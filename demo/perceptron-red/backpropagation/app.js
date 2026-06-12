/* global Backpropagation */

const dataset = [
  { input: [0.12, 0.12], output: 0, name: 'Ambas apagadas' },
  { input: [0.12, 0.88], output: 1, name: 'Solo B activa' },
  { input: [0.88, 0.12], output: 1, name: 'Solo A activa' },
  { input: [0.88, 0.88], output: 0, name: 'Ambas activas' }
];
const canvas = document.querySelector('#loss-canvas');
const context = canvas.getContext('2d');

function format(value, digits = 3) {
  return new Intl.NumberFormat('es-ES', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits
  }).format(value);
}

function train() {
  const model = new Backpropagation({
    epochs: Number(document.querySelector('#epochs').value),
    learningRate: Number(document.querySelector('#rate').value),
    momentum: 0.72,
    seed: 11,
    shuffle: true,
    targetLoss: 0.0008
  });
  model.addLayer(4).addLayer(1).learn(dataset);
  const predictions = dataset.map((sample) => ({ ...sample, value: model.process(sample.input)[0] }));
  const accuracy = predictions.filter((sample) => Math.round(sample.value) === sample.output).length;
  document.querySelector('#loss').textContent = format(model.history.loss.at(-1), 5);
  document.querySelector('#accuracy').textContent = `${Math.round((accuracy / dataset.length) * 100)}%`;
  document.querySelector('#used-epochs').textContent = model.history.epochs.toLocaleString('es-ES');
  document.querySelector('#prediction-list').innerHTML = predictions
    .map(
      (sample) =>
        `<div class="sample-row"><span>${sample.name} · esperado ${sample.output}</span><strong>${format(sample.value, 4)}</strong></div>`
    )
    .join('');
  drawLoss(model.history.loss);
}

function drawLoss(loss) {
  const width = canvas.width;
  const height = canvas.height;
  const padding = 58;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const max = Math.max(...loss);
  const min = Math.min(...loss);
  const range = max - min || 1;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f7faf7';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = 'rgba(70,95,84,.13)';
  for (let index = 0; index <= 5; index++) {
    const y = padding + (plotHeight / 5) * index;
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.stroke();
  }
  context.beginPath();
  loss.forEach((value, index) => {
    const x = padding + (index / Math.max(1, loss.length - 1)) * plotWidth;
    const y = padding + (1 - (value - min) / range) * plotHeight;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = '#08754a';
  context.lineWidth = 4;
  context.stroke();
  context.fillStyle = '#77827c';
  context.font = '600 11px Inter, sans-serif';
  context.textAlign = 'center';
  context.fillText('Épocas', width / 2, height - 18);
}

document.querySelector('#epochs').addEventListener('input', (event) => {
  document.querySelector('#epochs-output').value = Number(event.target.value).toLocaleString('es-ES');
});
document.querySelector('#rate').addEventListener('input', (event) => {
  document.querySelector('#rate-output').value = format(Number(event.target.value), 2);
});
document.querySelector('#train-button').addEventListener('click', train);
train();
