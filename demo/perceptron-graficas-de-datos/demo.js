/* global Perceptron */

const canvas = document.querySelector('#weights-canvas');
const context = canvas.getContext('2d');
const datasets = {
  or: [
    { input: [0, 0], output: 0 },
    { input: [0, 1], output: 1 },
    { input: [1, 0], output: 1 },
    { input: [1, 1], output: 1 }
  ],
  and: [
    { input: [0, 0], output: 0 },
    { input: [0, 1], output: 0 },
    { input: [1, 0], output: 0 },
    { input: [1, 1], output: 1 }
  ]
};
let history = [];
let model;

function train() {
  history = [];
  model = new Perceptron(undefined, captureWeights);
  const dataset = datasets[document.querySelector('#gate').value];
  dataset.forEach((sample) => model.addData(sample.input, sample.output));
  model.learn();
  captureWeights();
  render(dataset);
}

function captureWeights() {
  if (!model?.weights) return;
  history.push([model.weights[0], model.weights[1], model.threshold]);
}

function render(dataset) {
  const correct = dataset.filter((sample) => model.process(sample.input) === sample.output).length;
  document.querySelector('#epochs').textContent = model.trainingStats.epochs;
  document.querySelector('#updates').textContent = Math.max(0, history.length - 1);
  document.querySelector('#accuracy').textContent =
    `${Math.round((correct / dataset.length) * 100)}%`;
  document.querySelector('#weight-list').innerHTML = [
    ['Peso A', model.weights[0]],
    ['Peso B', model.weights[1]],
    ['Umbral', model.threshold]
  ]
    .map(
      ([name, value]) =>
        `<div class="weight-row"><span>${name}</span><strong>${value.toFixed(4)}</strong></div>`
    )
    .join('');
  document.querySelector('#formula').textContent =
    `salida = binaria(${model.weights[0].toFixed(3)} × A ${signed(model.weights[1])} × B ${signed(model.threshold)})`;
  drawChart();
}

function signed(value) {
  return `${value < 0 ? '−' : '+'} ${Math.abs(value).toFixed(3)}`;
}

function drawChart() {
  const width = canvas.width;
  const height = canvas.height;
  const padding = 58;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const values = history.flat();
  const min = Math.min(...values, -0.1);
  const max = Math.max(...values, 0.1);
  const range = max - min || 1;
  const colors = ['#18a66a', '#ff855f', '#4c67e8'];
  const labels = ['Peso A', 'Peso B', 'Umbral'];

  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f7faf7';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = 'rgba(70,95,84,.13)';
  context.lineWidth = 1;
  for (let index = 0; index <= 5; index++) {
    const y = padding + (plotHeight / 5) * index;
    context.beginPath();
    context.moveTo(padding, y);
    context.lineTo(width - padding, y);
    context.stroke();
  }

  labels.forEach((label, series) => {
    context.beginPath();
    history.forEach((snapshot, index) => {
      const x = padding + (index / Math.max(1, history.length - 1)) * plotWidth;
      const y = padding + (1 - (snapshot[series] - min) / range) * plotHeight;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.strokeStyle = colors[series];
    context.lineWidth = 3;
    context.stroke();
    context.fillStyle = colors[series];
    context.font = '700 11px Inter, sans-serif';
    context.fillText(label, padding + series * 85, 28);
  });

  context.fillStyle = '#77827c';
  context.font = '600 11px Inter, sans-serif';
  context.textAlign = 'center';
  context.fillText('Ajustes realizados', width / 2, height - 18);
}

document.querySelector('#train-button').addEventListener('click', train);
document.querySelector('#gate').addEventListener('change', train);
train();
