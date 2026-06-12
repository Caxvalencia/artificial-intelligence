export namespace Binary {
  export function activation(synapse) {
    return synapse >= 0 ? 1 : 0;
  }
}
