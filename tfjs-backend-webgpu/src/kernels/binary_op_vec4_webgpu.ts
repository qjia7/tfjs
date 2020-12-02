/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
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

import {getDispatchLayoutFromLogicalShape} from '../webgpu_texture_util';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';

// import {computeDispatch} from '../webgpu_util';

import {WebGPUProgram} from './webgpu_program';

export class BinaryOpVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[]};
  dispatchLayoutTexture: {x: number[], y: number[]};
  dispatch: [number, number, number];
  variableNames = ['A', 'B'];
  workPerThread = 4;
  workGroupSize: [number, number, number];
  isVec4 = true;

  constructor(
      op: string, aShape: number[], bShape: number[],
      usePackedTexture = false) {
    // TODO(jiajia.qin@intel.com): Heuristically select a good work group size.
    const workGroupSizeX = 32;
    this.outputShape = backend_util.assertAndGetBroadcastShape(aShape, bShape);
    if (usePackedTexture == false) {
      this.workGroupSize = [workGroupSizeX, 1, 1];
      this.dispatchLayout = flatDispatchLayout(this.outputShape);
      this.dispatch = computeDispatch(
          this.dispatchLayout, this.outputShape, this.workGroupSize,
          [1, this.workPerThread, 1]);
    } else {
      this.workGroupSize = [workGroupSizeX, 4, 1];
      const dispatchLayout2 =
          getDispatchLayoutFromLogicalShape(this.outputShape, true);
      this.dispatchLayoutTexture = {x: dispatchLayout2.x, y: dispatchLayout2.y};
      this.dispatch = computeDispatch(
          this.dispatchLayoutTexture, this.outputShape, this.workGroupSize);
    }
    const size = util.sizeFromShape(this.outputShape) / this.workPerThread;
    const fitShape = false;  // = size % workGroupSizeX === 0;

    let sampleA, sampleB, sampleResult;
    if (usePackedTexture) {
      sampleA = `vec4 a = getAAtOutCoords()`;
      sampleB = `vec4 b = getBAtOutCoords()`;
      sampleResult = `setOutput(binaryOperation(a, b))`;
    } else {
      sampleA = `vec4 a = A[index]`;
      sampleB = `vec4 b = B[index]`;
      sampleResult = `setOutput(index, binaryOperation(a, b))`;
    }

    if (fitShape) {
      this.userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        ${sampleA};
        ${sampleB};
        ${sampleResult};
      }
    `;
    } else {
      this.userCode = `
      vec4 binaryOperation(vec4 a, vec4 b) {
        ${op}
      }

      void main() {
        int index = int(gl_GlobalInvocationID.x);
        // TODO(tetxure): if (index < ${size})
        {
          ${sampleA};
          ${sampleB};
          ${sampleResult};
        }
      }
    `;
    }
  }
}
