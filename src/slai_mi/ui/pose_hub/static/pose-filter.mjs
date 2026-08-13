const TAU = 2 * Math.PI;

function alpha(cutoff, dt) {
  const value = TAU * Math.max(cutoff, 1e-6) * dt;
  return value / (value + 1);
}

function normalizeQuaternion(value) {
  const magnitude = Math.hypot(...value);
  if (!Number.isFinite(magnitude) || magnitude < 1e-9) throw new Error("invalid quaternion");
  return value.map(component => component / magnitude);
}

function quaternionDot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

function slerp(a, b, amount) {
  let target = b;
  let cosine = quaternionDot(a, target);
  if (cosine < 0) {
    target = target.map(value => -value);
    cosine = -cosine;
  }
  if (cosine > 0.9995) {
    return normalizeQuaternion(a.map((value, index) => value + amount * (target[index] - value)));
  }
  const angle = Math.acos(Math.max(-1, Math.min(1, cosine)));
  const scale = Math.sin(angle);
  const left = Math.sin((1 - amount) * angle) / scale;
  const right = Math.sin(amount * angle) / scale;
  return a.map((value, index) => left * value + right * target[index]);
}

export class OneEuroVector3 {
  constructor({ minCutoff = 2.5, beta = 5, derivativeCutoff = 1.5 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.reset();
  }

  reset() {
    this.previousTime = null;
    this.previousRaw = null;
    this.previousFiltered = null;
    this.previousDerivative = [0, 0, 0];
  }

  filter(value, timestamp) {
    const raw = value.map(Number);
    if (raw.length !== 3 || raw.some(component => !Number.isFinite(component)) || !Number.isFinite(timestamp)) {
      throw new Error("invalid vector sample");
    }
    if (this.previousTime === null || timestamp <= this.previousTime || timestamp - this.previousTime > 0.5) {
      this.previousTime = timestamp;
      this.previousRaw = raw;
      this.previousFiltered = raw;
      this.previousDerivative = [0, 0, 0];
      return [...raw];
    }
    const dt = timestamp - this.previousTime;
    const derivativeAlpha = alpha(this.derivativeCutoff, dt);
    const derivative = raw.map((component, index) => (component - this.previousRaw[index]) / dt);
    const filteredDerivative = derivative.map((component, index) => this.previousDerivative[index] + derivativeAlpha * (component - this.previousDerivative[index]));
    const filtered = raw.map((component, index) => {
      const cutoff = this.minCutoff + this.beta * Math.abs(filteredDerivative[index]);
      const amount = alpha(cutoff, dt);
      return this.previousFiltered[index] + amount * (component - this.previousFiltered[index]);
    });
    this.previousTime = timestamp;
    this.previousRaw = raw;
    this.previousFiltered = filtered;
    this.previousDerivative = filteredDerivative;
    return [...filtered];
  }
}

export class OneEuroQuaternion {
  constructor({ minCutoff = 2.5, beta = 0.8, derivativeCutoff = 1.5 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.derivativeCutoff = derivativeCutoff;
    this.reset();
  }

  reset() {
    this.previousTime = null;
    this.previousRaw = null;
    this.previousFiltered = null;
    this.previousSpeed = 0;
  }

  filter(value, timestamp) {
    let raw = normalizeQuaternion(value.map(Number));
    if (!Number.isFinite(timestamp)) throw new Error("invalid quaternion timestamp");
    if (this.previousTime === null || timestamp <= this.previousTime || timestamp - this.previousTime > 0.5) {
      this.previousTime = timestamp;
      this.previousRaw = raw;
      this.previousFiltered = raw;
      this.previousSpeed = 0;
      return [...raw];
    }
    if (quaternionDot(this.previousRaw, raw) < 0) raw = raw.map(component => -component);
    const dt = timestamp - this.previousTime;
    const cosine = Math.max(-1, Math.min(1, Math.abs(quaternionDot(this.previousRaw, raw))));
    const speed = 2 * Math.acos(cosine) / dt;
    const derivativeAlpha = alpha(this.derivativeCutoff, dt);
    const filteredSpeed = this.previousSpeed + derivativeAlpha * (speed - this.previousSpeed);
    const amount = alpha(this.minCutoff + this.beta * filteredSpeed, dt);
    const filtered = slerp(this.previousFiltered, raw, amount);
    this.previousTime = timestamp;
    this.previousRaw = raw;
    this.previousFiltered = filtered;
    this.previousSpeed = filteredSpeed;
    return [...filtered];
  }
}
