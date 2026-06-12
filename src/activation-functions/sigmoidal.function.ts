export namespace Sigmoidal {
  export function activation(synapse) {
    if (synapse >= 0) {
      const exponential = Math.exp(-synapse);

      return 1 / (1 + exponential);
    }

    const exponential = Math.exp(synapse);

    return exponential / (1 + exponential);
  }

  export function prime(synapse) {
    const output = activation(synapse);

    return output * (1 - output);
  }
}
