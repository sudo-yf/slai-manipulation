const EPSILON = 1e-9;

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function length(vector) {
  return Math.hypot(...vector);
}

function normalize(vector, label) {
  const magnitude = length(vector);
  if (!Number.isFinite(magnitude) || magnitude < EPSILON) {
    throw new Error(`${label} direction is too small`);
  }
  return vector.map(value => value / magnitude);
}

export function applyBasis(matrix, vector) {
  return [
    dot(matrix.slice(0, 3), vector),
    dot(matrix.slice(3, 6), vector),
    dot(matrix.slice(6, 9), vector),
  ];
}

export function determinant3(matrix) {
  const [a, b, c, d, e, f, g, h, i] = matrix;
  return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

export function buildOperatorBasis({
  origin,
  forwardPoint,
  rightPoint,
  worldUp = [0, 1, 0],
  minimumDistance = 0.1,
}) {
  const forwardDelta = subtract(forwardPoint, origin);
  const rightDelta = subtract(rightPoint, origin);
  const forwardDistance = length(forwardDelta);
  const rightDistance = length(rightDelta);
  if (forwardDistance < minimumDistance || rightDistance < minimumDistance) {
    throw new Error(`move at least ${Math.round(minimumDistance * 100)} cm in each direction`);
  }

  const measuredForward = normalize(forwardDelta, "forward");
  const measuredRight = normalize(rightDelta, "right");
  const angleDeg = Math.acos(Math.max(-1, Math.min(1, dot(measuredForward, measuredRight)))) * 180 / Math.PI;
  if (angleDeg < 50 || angleDeg > 130) {
    throw new Error("forward and right movements must be clearly different directions");
  }

  const zAxis = normalize(cross(measuredRight, measuredForward), "up");
  const upAlignment = dot(zAxis, normalize(worldUp, "world up"));
  if (upAlignment < 0.6) {
    throw new Error("keep both calibration movements approximately level");
  }
  const xAxis = normalize(cross(measuredForward, zAxis), "right");
  const yAxis = normalize(cross(zAxis, xAxis), "forward");
  const matrix = [...xAxis, ...yAxis, ...zAxis];
  if (determinant3(matrix) < 0.999) {
    throw new Error("calibration did not produce a right-handed frame");
  }

  return { matrix, forwardDistance, rightDistance, angleDeg, upAlignment };
}
