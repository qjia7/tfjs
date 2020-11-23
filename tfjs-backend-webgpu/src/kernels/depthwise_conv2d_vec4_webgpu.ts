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
import {getShapeCoords} from '../shader_preprocessor';
import {computeDispatch, flatDispatchLayout} from '../webgpu_util';
import {WebGPUProgram} from './webgpu_program';

export class DepthwiseConv2DVec4Program implements WebGPUProgram {
  outputShape: number[];
  shaderKey: string;
  userCode: string;
  dispatchLayout: {x: number[], y?: number[], z?: number[]};
  dispatch: [number, number, number];
  variableNames = ['x', 'W'];
  uniforms = 'ivec2 filterDims, pad, stride, dilation, inDims;';
  isVec4 = true;
  // This is an experimental value.
  workGroupSize: [number, number, number] = [256, 1, 1];

  constructor(convInfo: backend_util.Conv2DInfo) {
    this.outputShape = convInfo.outShape;
    this.dispatchLayout = flatDispatchLayout(this.outputShape);
    this.dispatch = computeDispatch(
        this.dispatchLayout, this.outputShape, this.workGroupSize);
    const channelMul = convInfo.outChannels / convInfo.inChannels;

    util.assert(
        convInfo.dataFormat === 'channelsLast',
        () => 'TODO: NCHW is unimplemented');

    this.userCode = `
      void writeResult(int batch, int row, int col, int chan, vec4 value) {
        ivec4 coord = ivec4(batch, row, col, chan);
        if (coordsInBounds(coord, ${getShapeCoords(this.outputShape)})) {
          setOutput(batch, row, col, chan, value);
        }
      }

      void main() {
        int flatIndex = int(gl_GlobalInvocationID.x) * 4;
        ivec4 coords = getCoordsFromFlatIndex(flatIndex);
        int batch = coords[0];
        ivec2 xRCCorner = coords.yz * stride - pad;
        int d2 = coords[3];
        int d1 = d2 / ${channelMul};
        int q = d2 - d1 * ${channelMul};

        int xRCorner = xRCCorner.x;
        int xCCorner = xRCCorner.y;

        // Convolve x(?, ?, d1) with w(:, :, d1, q) to get y(yR, yC, d2).
        // ? = to be determined. : = across all values in that axis.
        vec4 dotProd = vec4(0.0);
        for (int wR = 0; wR < filterDims[0]; wR++) {
          int xR = xRCorner + wR * dilation[0];

          if (xR < 0 || xR >= inDims[0]) {
            continue;
          }

          for (int wC = 0; wC < filterDims[1]; wC++) {
            int xC = xCCorner + wC * dilation[1];

            if (xC < 0 || xC >= inDims[1]) {
              continue;
            }

            vec4 xVal = getX(batch, xR, xC, d1);
            vec4 wVal = getW(wR, wC, d1, q);
            dotProd += xVal * wVal;
          }
        }
        writeResult(batch, coords[1], coords[2], d2, dotProd);
      }
    `;
  }
}
