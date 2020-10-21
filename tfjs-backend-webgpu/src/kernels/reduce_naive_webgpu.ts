/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import {backend_util, util} from '@tensorflow/tfjs-core';
import {getCoordsDataType, getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class ReduceNaiveProgram implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatch: [number, number, number];
  workGroupSize: [number, number, number];
  variableNames = ['x'];

  constructor(
      reduceInfo: backend_util.ReduceInfo, reduceType: 'max'|'min'|'sum') {
    const inputShape = [reduceInfo.batchSize, reduceInfo.inSize];
    const [outputShape] =
        backend_util.computeOutAndReduceShapes(inputShape, [1]);
    this.outputShape = outputShape.length === 0 ? [1] : outputShape;
    const size = util.sizeFromShape(this.outputShape);

    this.workGroupSize = [128, 1, 1];
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);

    const minmaxOp = `
          if (candidate ${reduceType === 'min' ? '<' : '>'} bestValue
          && !isnan(candidate))
          {  bestValue = candidate; }
      `;
    const sumOp = ' bestValue += candidate; ';

    const outputCoordsType = getCoordsDataType(this.outputShape.length);

    this.userCode = `
      #define DIV_CEIL(x, y) (((x) - 1) / (y) + 1)
      const int WorkGroupSize = int(gl_WorkGroupSize.x);

      int getOffset() {
        const ${outputCoordsType} outputCoords = getOutputCoords();
        int offset = ${
        this.outputShape.length === 1 ?
            'outputCoords' :
            'outputCoords[0]'} * ${getShapeCoords(inputShape)}[1];
        return offset;
      }
      void main() {
        const int flatOutputIndex = int(gl_GlobalInvocationID.x);
        if (flatOutputIndex < ${size})
        {
        const int offset= getOffset();
        ${
        reduceType === 'sum' ? 'float bestValue = 0;' :
                               'float bestValue = x[offset];'}
        const int Length = ${
        inputShape.length === 1 ? `${getShapeCoords(inputShape)}` :
                                  `${getShapeCoords(inputShape)}[1]`};
        for (int i = 0; i < Length; ++i) {
            float candidate = x[offset + i];
            ${(reduceType === 'max' || reduceType === 'min') ? minmaxOp : sumOp}
        }
        setOutput(flatOutputIndex, bestValue);
        }
      }
    `;
  }
}
