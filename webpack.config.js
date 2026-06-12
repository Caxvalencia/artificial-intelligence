const path = require('path');

let defaultConfig = {
  entry: './src/index.ts',
  devtool: 'inline-source-map',
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/
      }
    ]
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js']
  },
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
    libraryTarget: 'umd',
    umdNamedDefine: true
  }
};

let demoConfig = Object.assign({}, defaultConfig, {
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'demo/dist'),
    libraryTarget: 'umd',
    umdNamedDefine: true
  }
});

module.exports = [defaultConfig, demoConfig];
