import { assert } from 'chai';
import { suite, test } from '@testdeck/mocha';

import {
  ActivationFunction,
  ActivationFunctionType
} from '../activation-functions/activation-function';

@suite
export class ActivationFunctionTest {
  @test
  public binaryActivation() {
    const binary = ActivationFunction.init(ActivationFunctionType.BINARY);

    assert.equal(binary.activation(-1), 0);
    assert.equal(binary.activation(0), 1);
    assert.equal(binary.activation(1), 1);
    assert.throws(
      () => binary.prime(0),
      'Binary activation is not differentiable and cannot be used with backpropagation'
    );
  }

  @test
  public reluActivationAndDerivative() {
    const relu = ActivationFunction.init(ActivationFunctionType.RELU);

    assert.equal(relu.activation(-2), 0);
    assert.equal(relu.activation(0), 0);
    assert.equal(relu.activation(2), 2);
    assert.equal(relu.prime(-2), 0);
    assert.equal(relu.prime(0), 0);
    assert.equal(relu.prime(2), 1);
  }

  @test
  public sigmoidalActivationAndDerivative() {
    const sigmoidal = ActivationFunction.init(ActivationFunctionType.SIGMOIDAL);

    assert.closeTo(sigmoidal.activation(0), 0.5, 1e-12);
    assert.closeTo(sigmoidal.prime(0), 0.25, 1e-12);
  }

  @test
  public hyperbolicTangentActivationAndDerivative() {
    const hyperbolicTangent = ActivationFunction.init(ActivationFunctionType.HYPERBOLIC_TANGENT);

    assert.closeTo(hyperbolicTangent.activation(0), 0, 1e-12);
    assert.closeTo(hyperbolicTangent.prime(0), 1, 1e-12);
  }
}
