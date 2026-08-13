import assert from "node:assert/strict";
import {
  applyBasis,
  buildOperatorBasis,
  determinant3,
} from "../src/slai_mi/ui/pose_hub/static/pose-calibration.mjs";

const close = (actual, expected, tolerance = 1e-9) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert.ok(Math.abs(value - expected[index]) < tolerance, `${value} != ${expected[index]}`));
};

const standard = buildOperatorBasis({
  origin: [0, 0, 0],
  forwardPoint: [0, 0, -0.2],
  rightPoint: [0.2, 0, 0],
});
close(applyBasis(standard.matrix, [0.2, 0, 0]), [0.2, 0, 0]);
close(applyBasis(standard.matrix, [0, 0, -0.2]), [0, 0.2, 0]);
close(applyBasis(standard.matrix, [0, 0.2, 0]), [0, 0, 0.2]);
assert.ok(Math.abs(determinant3(standard.matrix) - 1) < 1e-9);

const diagonal = Math.SQRT1_2 * 0.2;
const rotated = buildOperatorBasis({
  origin: [1, 2, 3],
  forwardPoint: [1 + diagonal, 2, 3 - diagonal],
  rightPoint: [1 + diagonal, 2, 3 + diagonal],
});
close(applyBasis(rotated.matrix, [diagonal, 0, diagonal]), [0.2, 0, 0]);
close(applyBasis(rotated.matrix, [diagonal, 0, -diagonal]), [0, 0.2, 0]);

assert.throws(
  () => buildOperatorBasis({ origin: [0, 0, 0], forwardPoint: [0, 0, -0.2], rightPoint: [0, 0, -0.15] }),
  /different directions/,
);

console.log("pose calibration tests passed");
