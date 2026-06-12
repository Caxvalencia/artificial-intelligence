export namespace Binary {
  export function activation(synapse) {
    return synapse >= 0 ? 1 : 0;
  }

  export function prime() {
    throw new Error(
      'Binary activation is not differentiable and cannot be used with backpropagation'
    );
  }
}
