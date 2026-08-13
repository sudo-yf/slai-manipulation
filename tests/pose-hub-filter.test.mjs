import assert from "node:assert/strict";
import {
  OneEuroQuaternion,
  OneEuroVector3,
} from "../src/slai_mi/ui/pose_hub/static/pose-filter.mjs";

const position = new OneEuroVector3();
assert.deepEqual(position.filter([0, 0, 0], 0), [0, 0, 0]);
let filteredPosition;
for (let frame = 1; frame <= 30; frame += 1) {
  filteredPosition = position.filter([frame / 30, 0, 0], frame / 60);
}
assert.ok(filteredPosition[0] > 0.9 && filteredPosition[0] <= 1, `unexpected position ${filteredPosition[0]}`);
assert.equal(filteredPosition[1], 0);

const rotation = new OneEuroQuaternion();
assert.deepEqual(rotation.filter([0, 0, 0, 1], 0), [0, 0, 0, 1]);
let filteredRotation;
for (let frame = 1; frame <= 30; frame += 1) {
  const angle = Math.PI * frame / 60;
  filteredRotation = rotation.filter([0, 0, Math.sin(angle / 2), Math.cos(angle / 2)], frame / 60);
}
assert.ok(Math.abs(Math.hypot(...filteredRotation) - 1) < 1e-9);
assert.ok(filteredRotation[2] > 0.64, `unexpected rotation ${filteredRotation}`);

rotation.reset();
assert.deepEqual(rotation.filter([0, 0, 0, -1], 10), [0, 0, 0, -1]);
console.log("pose filter tests passed");
