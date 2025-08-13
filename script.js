// Three.js r140 via import map defined in index.html
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// -------- DOM --------
const container = document.getElementById('canvas-container');
const playBtn = document.getElementById('playBtn');
const toggleAxesBtn = document.getElementById('toggleAxesBtn');
const resetBtn = document.getElementById('resetBtn');
const fileInput = document.getElementById('fileInput');
const exportBtn = document.getElementById('exportBtn');
const audioEl = document.getElementById('audioEl');

// -------- Audio setup --------
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
const audioSource = audioContext.createMediaElementSource(audioEl);
const analyser = audioContext.createAnalyser();
analyser.fftSize = 1024; // 512 freq bins
const defaultSmoothing = 0.85;
analyser.smoothingTimeConstant = defaultSmoothing;
audioSource.connect(analyser);
analyser.connect(audioContext.destination);

const freqBinCount = analyser.frequencyBinCount; // 512
const freqData = new Uint8Array(freqBinCount);

function flushAnalyser() {
  // Temporarily disable smoothing and pull a few frames to drop residual state
  analyser.smoothingTimeConstant = 0;
  for (let i = 0; i < 4; i++) analyser.getByteFrequencyData(freqData);
  freqData.fill(0);
}

// -------- 3D Scene --------
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setClearColor(0x000000, 1);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// Camera angled for a perspective similar to the reference video
const camera = new THREE.PerspectiveCamera(
  55,
  container.clientWidth / container.clientHeight,
  0.1,
  2000
);
camera.position.set(90, 55, 120);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.minDistance = 30;
controls.maxDistance = 4000;
controls.maxPolarAngle = Math.PI * 0.49;
// Mouse mappings: middle to pan, right to orbit
controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
controls.mouseButtons.RIGHT = THREE.MOUSE.ROTATE;
// Avoid browser context menu on right-click when orbiting
renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

// Lighting to bring out surface relief
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(1, 2, 1);
scene.add(dir);

// No vertical reference grid; will add a ground grid below instead

// -------- Spectrogram surface --------
// X axis: time (scrolling towards +X). Z axis: frequency bins. Y axis: amplitude.
const pointsPerSlice = 128; // downsample frequency bins for performance
const sliceStride = Math.floor(freqBinCount / pointsPerSlice);
// Visual shaping across frequency axis: spread lows, compress empty highs
const activeFrequencyFraction = 0.75; // show first ~75% of rows; collapse the rest
const frequencyExponent = 0.65; // <1 spreads low/mid frequencies
const noiseFloor = 0.03; // ignore very small magnitudes

const numSlicesInitial = 1024; // initial capacity
const width = 400; // initial visual width used to derive slice spacing
const depth = 80; // z extent (frequency)
const heightScale = 0.6; // amplitude scale

// Constant spacing along X so model can grow indefinitely while keeping scale
const sliceSpacing = width / (numSlicesInitial - 1);
let capacity = numSlicesInitial; // current columns capacity
let currentSliceIndex = 0; // next X index to fill

function createSurfaceGeometry(columnCapacity) {
  const g = new THREE.PlaneGeometry(sliceSpacing * (columnCapacity - 1), depth, columnCapacity - 1, pointsPerSlice - 1);
  g.rotateX(-Math.PI / 2);
  // Anchor the left edge at x = 0 so growth always extends to the right
  g.translate((sliceSpacing * (columnCapacity - 1)) / 2, 0, 0);
  return g;
}

let geometry = createSurfaceGeometry(capacity);

// Starting positions: x from -W/2 .. +W/2, z per non-linear mapping
let positionAttr = geometry.attributes.position;
// Custom color per vertex for a heat map effect
let colors = new Float32Array(positionAttr.count * 3);
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

const material = new THREE.MeshLambertMaterial({
  side: THREE.DoubleSide,
  vertexColors: true,
  emissive: new THREE.Color(0x0),
});

const surface = new THREE.Mesh(geometry, material);
surface.position.x = 0; // center over ground grid
scene.add(surface);

// Removed helper ground grid

// Precompute Z positions per frequency row with nonlinear mapping and collapse of highs
const zMin = -depth / 2;
const zMax = depth / 2;
const activeRows = Math.max(2, Math.floor(pointsPerSlice * activeFrequencyFraction));
const zRowPositions = new Float32Array(pointsPerSlice);
for (let z = 0; z < pointsPerSlice; z++) {
  if (z < activeRows) {
    const t = z / (activeRows - 1);
    const tExp = Math.pow(t, frequencyExponent);
    zRowPositions[z] = zMin + tExp * (zMax - zMin);
  } else {
    zRowPositions[z] = zMax; // collapse unused high frequencies
  }
}
// Apply Z coordinates for the current geometry based on zRowPositions
function applyZRowsToGeometry(geo, columnCapacity) {
  const posAttr = geo.attributes.position;
  const arr = posAttr.array;
  for (let z = 0; z < pointsPerSlice; z++) {
    const rowStart = z * columnCapacity;
    const zVal = zRowPositions[z];
    for (let x = 0; x < columnCapacity; x++) {
      const i3 = (rowStart + x) * 3;
      arr[i3 + 2] = zVal;
    }
  }
  posAttr.needsUpdate = true;
}
applyZRowsToGeometry(geometry, capacity);

// Maintain a continuous scroll by shifting historic columns left
let isCapturing = false;
let hasEnded = false;
const capturedSlices = []; // array of Float32Array(length: pointsPerSlice)
const baseThickness = 4; // export base thickness (units ~ same as scene)
const backThickness = 6; // small back slab thickness along -Z for printability

function expandGeometry(newCapacity) {
  const oldGeometry = geometry;
  const oldPos = positionAttr;
  const oldColors = colors;
  const oldCapacity = capacity;

  const newGeo = createSurfaceGeometry(newCapacity);
  applyZRowsToGeometry(newGeo, newCapacity);
  let newPos = newGeo.attributes.position;
  let newColors = new Float32Array(newPos.count * 3);
  newGeo.setAttribute('color', new THREE.BufferAttribute(newColors, 3));

  // Copy existing Y values and vertex colors into the new geometry
  for (let z = 0; z < pointsPerSlice; z++) {
    for (let x = 0; x < Math.min(currentSliceIndex, newCapacity); x++) {
      const oldIdx = z * oldCapacity + x;
      const newIdx = z * newCapacity + x;
      newPos.setY(newIdx, oldPos.getY(oldIdx));
      const oc = oldIdx * 3;
      const nc = newIdx * 3;
      newColors[nc + 0] = oldColors[oc + 0];
      newColors[nc + 1] = oldColors[oc + 1];
      newColors[nc + 2] = oldColors[oc + 2];
    }
  }
  // Dispose old geometry to free GPU memory
  oldGeometry.dispose();
  // Swap in the new geometry
  geometry = newGeo;
  surface.geometry = geometry;
  positionAttr = geometry.attributes.position;
  colors = newColors;
  capacity = newCapacity;
  positionAttr.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;

  // Rebuild axes at the new right edge
  buildAxesAndTicks();
}

function updateSurfaceFromFrequencies() {
  analyser.getByteFrequencyData(freqData);

  // Normalize and map to 0..1 with a subtle curve
  const newSlice = new Float32Array(pointsPerSlice);
  for (let i = 0; i < pointsPerSlice; i++) {
    const srcIndex = Math.min(freqBinCount - 1, i * sliceStride);
    let bin = freqData[srcIndex] / 255;
    if (bin < noiseFloor) bin = 0; // enforce zero floor everywhere during rebuilds
    // Slight emphasis on mid-highs to echo the modem's chirps
    const emphasized = Math.pow(bin, 1.2);
    newSlice[i] = emphasized;
  }

  // Persist the full-resolution slice for export later
  if (isCapturing) {
    capturedSlices.push(newSlice.slice());
  }

  // Grow geometry if needed
  if (currentSliceIndex >= capacity) {
    const nextCapacity = Math.ceil(capacity * 1.6);
    expandGeometry(nextCapacity);
  }

  // For each z-row (frequency), write this slice at the current column
  for (let z = 0; z < pointsPerSlice; z++) {
    const vIndex = z * capacity + currentSliceIndex;
    const y = newSlice[z] * depth * heightScale;
    positionAttr.setY(vIndex, y);
    const color = referenceColorRamp(newSlice[z]);
    const cIdx = vIndex * 3;
    colors[cIdx + 0] = color.r;
    colors[cIdx + 1] = color.g;
    colors[cIdx + 2] = color.b;
  }

  currentSliceIndex++;
  positionAttr.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
}

// Color ramp designed to match the reference look:
// - Lower amplitude → warm reds/oranges/yellows
// - Higher amplitude → cool cyans/blues
function referenceColorRamp(t) {
  const v = Math.min(1, Math.max(0, t));
  const ramp = [
    { s: 0.00, c: [0.55, 0.00, 0.00] }, // deep red
    { s: 0.15, c: [1.00, 0.13, 0.00] }, // red-orange
    { s: 0.30, c: [1.00, 0.75, 0.00] }, // amber
    { s: 0.45, c: [1.00, 1.00, 0.00] }, // yellow
    { s: 0.65, c: [0.12, 0.86, 1.00] }, // cyan
    { s: 0.82, c: [0.00, 0.46, 1.00] }, // blue
    { s: 1.00, c: [0.00, 0.12, 0.70] }, // deep blue
  ];

  // Find segment
  for (let i = 0; i < ramp.length - 1; i++) {
    const a = ramp[i];
    const b = ramp[i + 1];
    if (v >= a.s && v <= b.s) {
      const t01 = (v - a.s) / (b.s - a.s);
      return {
        r: a.c[0] + (b.c[0] - a.c[0]) * t01,
        g: a.c[1] + (b.c[1] - a.c[1]) * t01,
        b: a.c[2] + (b.c[2] - a.c[2]) * t01,
      };
    }
  }
  const last = ramp[ramp.length - 1].c;
  return { r: last[0], g: last[1], b: last[2] };
}

// -------- Minimal tick marks (Frequency and Amplitude) --------
// A small axes group that places frequency ticks along Z at y≈0 and
// amplitude ticks along Y at the back-left corner. Labels are sprites.
const axesGroup = new THREE.Group();
scene.add(axesGroup);

function disposeObjectRecursive(object3d) {
  object3d.traverse((node) => {
    if (node.geometry) node.geometry.dispose();
    if (node.material) {
      // Handle sprite/mesh materials (single or array)
      const materials = Array.isArray(node.material) ? node.material : [node.material];
      for (const mat of materials) {
        if (mat.map) mat.map.dispose();
        mat.dispose && mat.dispose();
      }
    }
  });
}

function clearAxesGroup() {
  while (axesGroup.children.length) {
    const child = axesGroup.children.pop();
    disposeObjectRecursive(child);
  }
}

function createTextSprite(text, options = {}) {
  const {
    fontSize = 64,
    textColor = '#e5e7eb',
    padding = 8,
    shadowColor = 'rgba(0,0,0,0.8)',
    worldHeight = 6,
  } = options;

  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  const font = `${fontSize}px Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial`;
  context.font = font;
  const metrics = context.measureText(text);
  const widthPx = Math.ceil(metrics.width + padding * 2);
  const heightPx = Math.ceil(fontSize + padding * 2);
  canvas.width = widthPx;
  canvas.height = heightPx;

  context.font = font;
  context.textBaseline = 'middle';
  context.textAlign = 'left';
  context.fillStyle = textColor;
  context.shadowColor = shadowColor;
  context.shadowBlur = 4;
  context.fillText(text, padding, heightPx / 2);

  const texture = new THREE.CanvasTexture(canvas);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);

  const spriteWorldHeight = worldHeight;
  const spriteWorldWidth = (widthPx / heightPx) * spriteWorldHeight;
  sprite.scale.set(spriteWorldWidth, spriteWorldHeight, 1);
  return sprite;
}

function formatHzLabel(hz) {
  if (hz >= 1000) {
    const kilo = hz / 1000;
    return (Math.abs(kilo - Math.round(kilo)) < 1e-6) ? `${Math.round(kilo)}k` : `${kilo.toFixed(1)}k`;
  }
  return `${Math.round(hz)}`;
}

function buildAxesAndTicks() {
  clearAxesGroup();

  // Place axes on the left side of the mesh at a fixed world X (slightly left of x=0)
  const axisX = -2; // just left of surface's anchored left edge (x=0)
  const backZ = zMin;
  const frontActiveZ = zRowPositions[Math.max(0, activeRows - 1)];
  const amplitudeMaxY = depth * heightScale;

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0x9ca3af });

  // Amplitude (Y) axis line at back-left corner
  const yAxisGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(axisX, 0, backZ),
    new THREE.Vector3(axisX, amplitudeMaxY, backZ),
  ]);
  axesGroup.add(new THREE.Line(yAxisGeometry, lineMaterial));

  // Frequency (Z) axis line along base at y=0
  const zAxisGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(axisX, 0, backZ),
    new THREE.Vector3(axisX, 0, frontActiveZ),
  ]);
  axesGroup.add(new THREE.Line(zAxisGeometry, lineMaterial));

  // Frequency tick marks and labels
  const nyquistHz = audioContext.sampleRate / 2;
  const desiredFreqHz = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000];
  const freqTickPoints = [];
  for (const hz of desiredFreqHz) {
    if (hz <= 0 || hz > nyquistHz) continue;
    // Convert target Hz -> downsampled row index -> Z position
    const binIndex = (hz / nyquistHz) * freqBinCount; // 0..freqBinCount
    const rowIndex = Math.round(binIndex / sliceStride);
    if (rowIndex < 0 || rowIndex >= activeRows) continue; // only in active band
    const zAtRow = zRowPositions[rowIndex];
    freqTickPoints.push(new THREE.Vector3(axisX - 1.0, 0, zAtRow));
    freqTickPoints.push(new THREE.Vector3(axisX, 0, zAtRow));

    const label = createTextSprite(formatHzLabel(hz), { worldHeight: 5 });
    label.position.set(axisX - 1.25, 0.01, zAtRow);
    // Anchor so text sits to the left of the tick (right-aligned)
    label.center.set(1, 0.5);
    label.renderOrder = 2;
    axesGroup.add(label);
  }
  if (freqTickPoints.length) {
    const freqTicksGeometry = new THREE.BufferGeometry().setFromPoints(freqTickPoints);
    axesGroup.add(new THREE.LineSegments(freqTicksGeometry, lineMaterial));
  }

  // Amplitude tick marks and labels (approximate dB scale 0..80)
  const amplitudeDbTicks = [0, 20, 40, 60, 80];
  const ampTickPoints = [];
  for (const db of amplitudeDbTicks) {
    const unit = Math.max(0, Math.min(1, db / 80));
    const yAtDb = unit * amplitudeMaxY;
    ampTickPoints.push(new THREE.Vector3(axisX - 1.2, yAtDb, backZ));
    ampTickPoints.push(new THREE.Vector3(axisX, yAtDb, backZ));

    const label = createTextSprite(`${db}`, { worldHeight: 5 });
    label.position.set(axisX - 0.3, yAtDb, backZ);
    // Anchor to right-middle so text hugs the axis from the left side
    label.center.set(1, 0.5);
    label.renderOrder = 2;
    axesGroup.add(label);
  }
  if (ampTickPoints.length) {
    const ampTicksGeometry = new THREE.BufferGeometry().setFromPoints(ampTickPoints);
    axesGroup.add(new THREE.LineSegments(ampTicksGeometry, lineMaterial));
  }

  // Axis titles
  const freqTitle = createTextSprite('Frequency (Hz)', { worldHeight: 5.5 });
  freqTitle.position.set(axisX - 3.2, 0.01, (backZ + frontActiveZ) / 2);
  freqTitle.center.set(0.5, 0.5);
  freqTitle.renderOrder = 2;
  axesGroup.add(freqTitle);

  const ampTitle = createTextSprite('Amplitude (dB)', { worldHeight: 5.5 });
  ampTitle.position.set(axisX - 3.2, amplitudeMaxY, backZ);
  ampTitle.center.set(0.5, 0);
  ampTitle.renderOrder = 2;
  axesGroup.add(ampTitle);
}

// Build initial axes now that geometry mapping is known
buildAxesAndTicks();

// Place the camera so the view centers between the axes (left) and the surface (center),
// backed up and slightly orbiting to the right similar to the reference angle.
function positionCameraOverview() {
  // Axes are on the left near x=0; model grows to the right.
  const targetX = (0 + sliceSpacing * 40) / 2; // slight bias into the surface
  const targetY = depth * heightScale * 0.35;
  const targetZ = 0;

  // Set the look target first
  controls.target.set(targetX, targetY, targetZ);

  // Back the camera off and orbit a bit to the right of the target
  const amplitudeMaxY = depth * heightScale;
  const dx = sliceSpacing * 120;  // right of target
  const dy = amplitudeMaxY * 1.6;  // above target
  const dz = sliceSpacing * 300;  // forward from target
  camera.position.set(targetX + dx, targetY + dy, targetZ + dz);
  controls.update();
  controls.saveState && controls.saveState();
}

positionCameraOverview();

// Ensure the surface is visible before audio plays
resetVisualization();

// Zooms out to fit the full model width while preserving the current angle.
function frameWholeModel() {
  const slices = Math.max(currentSliceIndex, capturedSlices.length);
  const totalWidth = Math.max(1, slices - 1) * sliceSpacing;
  const targetX = totalWidth * 0.55; // bias a bit into the model
  const targetY = depth * heightScale * 0.35;
  const targetZ = 0;
  controls.target.set(targetX, targetY, targetZ);

  const amplitudeMaxY = depth * heightScale;
  const dz = Math.max(sliceSpacing * 300, totalWidth * 1.1);
  const dx = sliceSpacing * 140; // small right offset
  const dy = amplitudeMaxY * 1.7;
  camera.position.set(targetX + dx, targetY + dy, targetZ + dz);
  controls.update();
}

// -------- Animation loop --------
let isRendering = true;
function animate() {
  if (!isRendering) return;
  requestAnimationFrame(animate);
  if (!audioEl.paused && !hasEnded) {
    updateSurfaceFromFrequencies();
  }
  controls.update();
  renderer.render(scene, camera);
}
animate();

// -------- Events --------
let didSetInitialView = false;
function resize() {
  const { clientWidth, clientHeight } = container;
  renderer.setSize(clientWidth, clientHeight);
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  // Ensure the camera is set to the intended overview once after layout is known
  if (!didSetInitialView && typeof positionCameraOverview === 'function') {
    positionCameraOverview();
    didSetInitialView = true;
  }
}
window.addEventListener('resize', resize);
resize();

// Axes toggle
let axesVisible = true;
function setAxesVisibility(visible) {
  axesVisible = visible;
  axesGroup.visible = visible;
  if (toggleAxesBtn) toggleAxesBtn.textContent = visible ? 'Hide Axes' : 'Show Axes';
}
if (toggleAxesBtn) {
  toggleAxesBtn.addEventListener('click', () => setAxesVisibility(!axesVisible));
}

function setPlayButtonState(isPlaying) {
  if (isPlaying) {
    playBtn.textContent = 'Pause';
    playBtn.classList.remove('bg-emerald-600', 'hover:bg-emerald-500');
    playBtn.classList.add('bg-yellow-500', 'hover:bg-yellow-400');
  } else {
    playBtn.textContent = 'Play';
    playBtn.classList.remove('bg-yellow-500', 'hover:bg-yellow-400');
    playBtn.classList.add('bg-emerald-600', 'hover:bg-emerald-500');
  }
}

function setExportAvailability(enabled) {
  if (!exportBtn) return;
  exportBtn.disabled = !enabled;
}

playBtn.addEventListener('click', async () => {
  if (audioEl.paused) {
    if (audioContext.state === 'suspended') await audioContext.resume();
    try { await audioEl.play(); } catch (e) { /* ignore */ }
    isCapturing = true;
    setPlayButtonState(true);
  } else {
    audioEl.pause();
    isCapturing = false;
    setPlayButtonState(false);
    // keep export availability as-is
  }
});

resetBtn.addEventListener('click', () => {
  audioEl.pause();
  audioEl.currentTime = 0;
  // Ensure browser seeks to start for both default and uploaded sources
  try { audioEl.load(); } catch { }
  // Clear analyser buffers to avoid a spike on the first frame
  flushAnalyser();
  setPlayButtonState(false);
  hasEnded = false;
  isCapturing = false;
  capturedSlices.length = 0;
  hardResetVisualization();
  setExportAvailability(false);
});

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  audioEl.src = url;
  // Fresh start for a newly uploaded file
  hasEnded = false;
  isCapturing = false;
  capturedSlices.length = 0;
  hardResetVisualization();
  flushAnalyser();
  try { audioEl.currentTime = 0; audioEl.load(); } catch { }
  if (audioContext.state === 'suspended') await audioContext.resume();
  audioEl.play();
  isCapturing = true;
  setExportAvailability(false);
});

// Drag & drop support
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('drop', (e) => {
  e.preventDefault();
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  audioEl.src = url;
  // Fresh start for a newly dropped file
  hasEnded = false;
  isCapturing = false;
  capturedSlices.length = 0;
  hardResetVisualization();
  flushAnalyser();
  try { audioEl.currentTime = 0; audioEl.load(); } catch { }
  audioEl.play();
  isCapturing = true;
  setExportAvailability(false);
});

// Auto-start visualization when audio starts
audioEl.addEventListener('play', () => {
  if (audioContext.state === 'suspended') audioContext.resume();
  setPlayButtonState(true);
  // Enable export once at least one slice lands
  setExportAvailability(true);
});

audioEl.addEventListener('ended', () => {
  hasEnded = true;
  isCapturing = false;
  finalizeModelGeometry();
  // back the camera up a bit so the entire model is in view
  frameWholeModel();
  setPlayButtonState(false);
});

exportBtn.addEventListener('click', () => {
  if (capturedSlices.length < 2) {
    // If user exports early, synthesize from current displayed geometry
    // by sampling visible vertices so export always works
    synthesizeCapturedFromSurface();
  }
  const { objText, mtlText } = buildOBJFromCaptured();
  downloadTextAsFile('dialup_spectrogram.mtl', mtlText);
  downloadTextAsFile('dialup_spectrogram.obj', objText);
});

// Rebuild the surface geometry to display the full captured model when playback finishes
function finalizeModelGeometry() {
  const slices = capturedSlices.length;
  if (slices < 2) return;
  const finalWidth = sliceSpacing * (slices - 1);
  const newGeo = new THREE.PlaneGeometry(finalWidth, depth, slices - 1, pointsPerSlice - 1);
  newGeo.rotateX(-Math.PI / 2);
  // Anchor left edge at x=0 so the model does not shift on completion
  newGeo.translate(finalWidth / 2, 0, 0);

  const pos = newGeo.attributes.position;
  const col = new Float32Array(pos.count * 3);
  newGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));

  const z0 = -depth / 2;
  const dz = depth / (pointsPerSlice - 1);
  // Fill Y and color from captured data
  for (let z = 0; z < pointsPerSlice; z++) {
    for (let x = 0; x < slices; x++) {
      const idx = z * slices + x;
      const amp = capturedSlices[x][z];
      const y = amp * depth * heightScale;
      pos.setY(idx, y);
      const color = referenceColorRamp(amp);
      const cIdx = idx * 3;
      col[cIdx + 0] = color.r;
      col[cIdx + 1] = color.g;
      col[cIdx + 2] = color.b;
    }
  }
  pos.needsUpdate = true;
  newGeo.attributes.color.needsUpdate = true;
  surface.geometry.dispose();
  geometry = newGeo;
  surface.geometry = geometry;
}

// If user exports before capture has enough slices, sample the on-screen geometry
function synthesizeCapturedFromSurface() {
  const pos = geometry.attributes.position;
  // Determine current visible slices from geometry's segments along X
  const totalVertices = pos.count;
  const slices = Math.max(2, Math.floor(totalVertices / pointsPerSlice));
  capturedSlices.length = 0;
  for (let x = 0; x < slices; x++) {
    const slice = new Float32Array(pointsPerSlice);
    for (let z = 0; z < pointsPerSlice; z++) {
      const idx = z * slices + x;
      const y = pos.getY(idx);
      slice[z] = Math.max(0, Math.min(1, y / (depth * heightScale)));
    }
    capturedSlices.push(slice);
  }
}

// Build a watertight OBJ + MTL. Colors are assigned per-face via materials.
function buildOBJFromCaptured() {
  const slices = capturedSlices.length;
  const activeRowsCount = Math.max(2, Math.floor(pointsPerSlice * activeFrequencyFraction));
  // Use only the active frequency band plus one front boundary row to avoid degenerate cells
  const bins = activeRowsCount + 1;
  const dx = sliceSpacing;
  const dz = depth / (bins - 1);
  const exportWidth = dx * (slices - 1);
  const x0 = -exportWidth / 2;
  const z0 = -depth / 2;
  const backZ = z0 - backThickness;

  const v = [];
  const f = [];
  const vx = []; const vy = []; const vz = [];
  const indexTop = Array.from({ length: slices }, () => new Array(bins));
  const yTop = Array.from({ length: slices }, () => new Array(bins));

  // Material palette (quantized RGB -> material)
  const materials = new Map(); // key -> { name, r, g, b }
  function materialNameForColor(color) {
    const ir = Math.round(Math.max(0, Math.min(1, color.r)) * 255);
    const ig = Math.round(Math.max(0, Math.min(1, color.g)) * 255);
    const ib = Math.round(Math.max(0, Math.min(1, color.b)) * 255);
    const key = `${ir}_${ig}_${ib}`;
    if (!materials.has(key)) {
      const name = `c_${key}`;
      materials.set(key, { name, r: ir / 255, g: ig / 255, b: ib / 255 });
    }
    return materials.get(key).name;
  }
  const baseColor = { r: 0.7, g: 0.7, b: 0.7 };
  const baseMat = materialNameForColor(baseColor);

  function pushV(x, y, z, r, g, b) {
    v.push(`v ${x.toFixed(5)} ${y.toFixed(5)} ${z.toFixed(5)}`);
    vx.push(x); vy.push(y); vz.push(z);
    return v.length; // 1-based index
  }

  function emitFace(i1, i2, i3, desired, matName) {
    // Orient triangle so its normal generally points along 'desired'
    const a = i1 - 1, b = i2 - 1, c = i3 - 1;
    const abx = vx[b] - vx[a], aby = vy[b] - vy[a], abz = vz[b] - vz[a];
    const acx = vx[c] - vx[a], acy = vy[c] - vy[a], acz = vz[c] - vz[a];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    const dot = nx * desired.x + ny * desired.y + nz * desired.z;
    f.push(`usemtl ${matName}`);
    if (dot >= 0) {
      f.push(`f ${i1} ${i2} ${i3}`);
    } else {
      f.push(`f ${i1} ${i3} ${i2}`);
    }
  }

  const OUT_TOP = { x: 0, y: 1, z: 0 };
  const OUT_BOTTOM = { x: 0, y: -1, z: 0 };
  const OUT_FRONT = { x: 0, y: 0, z: 1 };
  const OUT_BACK = { x: 0, y: 0, z: -1 };
  const OUT_LEFT = { x: -1, y: 0, z: 0 };
  const OUT_RIGHT = { x: 1, y: 0, z: 0 };

  // Top vertices with color (clamp near-flat to exact 0 to enable merging)
  const flatEps = 0.02 * depth * heightScale;
  for (let i = 0; i < slices; i++) {
    const x = x0 + dx * i;
    const slice = capturedSlices[i];
    for (let j = 0; j < bins; j++) {
      // Use same nonlinear Z placement used in the live view
      let z;
      if (j < activeRowsCount) {
        const t = j / (activeRowsCount - 1);
        const tExp = Math.pow(t, frequencyExponent);
        z = z0 + tExp * depth;
      } else {
        z = z0 + depth;
      }
      const amp = j < activeRowsCount ? slice[j] : 0; // collapse highs
      let y = amp * depth * heightScale;
      if (y < flatEps) y = 0;
      yTop[i][j] = y;
      const c = referenceColorRamp(amp);
      indexTop[i][j] = pushV(x, y, z, c.r, c.g, c.b);
    }
  }

  // Bottom grid vertices at y = -baseThickness matching the top grid segmentation
  // This guarantees shared edges with all side walls for a watertight mesh
  const indexBottom = Array.from({ length: slices }, () => new Array(bins));
  for (let i = 0; i < slices; i++) {
    const x = x0 + dx * i;
    for (let j = 0; j < bins; j++) {
      let z;
      if (j < activeRowsCount) {
        const t = j / (activeRowsCount - 1);
        const tExp = Math.pow(t, frequencyExponent);
        z = z0 + tExp * depth;
      } else {
        z = z0 + depth;
      }
      indexBottom[i][j] = pushV(x, -baseThickness, z, baseColor.r, baseColor.g, baseColor.b);
    }
  }

  // Removed rear slab and zero-height shelf to avoid overlapping/co-planar faces.

  // Triangulate top surface with coplanar merge on near-flat regions
  // Build a boolean grid of flat cells
  const flatCell = Array.from({ length: slices - 1 }, () => new Array(bins - 1).fill(false));
  for (let i = 0; i < slices - 1; i++) {
    for (let j = 0; j < bins - 1; j++) {
      const y00 = yTop[i][j];
      const y10 = yTop[i + 1][j];
      const y01 = yTop[i][j + 1];
      const y11 = yTop[i + 1][j + 1];
      flatCell[i][j] = y00 === 0 && y10 === 0 && y01 === 0 && y11 === 0;
    }
  }
  const visited = Array.from({ length: slices - 1 }, () => new Array(bins - 1).fill(false));
  for (let j = 0; j < bins - 1; j++) {
    for (let i = 0; i < slices - 1; i++) {
      if (visited[i][j]) continue;
      if (!flatCell[i][j]) {
        // Emit normal two triangles for this single non-flat cell
        const a = indexTop[i][j];
        const bIdx = indexTop[i + 1][j];
        const cIdx = indexTop[i + 1][j + 1];
        const d = indexTop[i][j + 1];
        // Face colors from average height (approx amplitude)
        const scale = depth * heightScale;
        const t1 = (yTop[i][j] + yTop[i + 1][j] + yTop[i + 1][j + 1]) / (3 * scale);
        const t2 = (yTop[i][j] + yTop[i + 1][j + 1] + yTop[i][j + 1]) / (3 * scale);
        const c1 = referenceColorRamp(Math.max(0, Math.min(1, t1)));
        const c2 = referenceColorRamp(Math.max(0, Math.min(1, t2)));
        emitFace(a, bIdx, cIdx, OUT_TOP, materialNameForColor(c1));
        emitFace(a, cIdx, d, OUT_TOP, materialNameForColor(c2));
        visited[i][j] = true;
        continue;
      }
      // Flat cell: grow a rectangle (i..iW, j..jH)
      let iW = i;
      while (iW + 1 < slices - 1 && flatCell[iW + 1][j] && !visited[iW + 1][j]) iW++;
      let jH = j;
      let grow = true;
      while (grow && jH + 1 < bins - 1) {
        for (let k = i; k <= iW; k++) {
          if (!flatCell[k][jH + 1] || visited[k][jH + 1]) { grow = false; break; }
        }
        if (grow) jH++;
      }
      // Emit two triangles covering the rectangle
      const a = indexTop[i][j];
      const bIdx = indexTop[iW + 1][j];
      const cIdx = indexTop[iW + 1][jH + 1];
      const d = indexTop[i][jH + 1];
      emitFace(a, bIdx, cIdx, OUT_TOP, baseMat);
      emitFace(a, cIdx, d, OUT_TOP, baseMat);
      // Mark visited
      for (let jj = j; jj <= jH; jj++) {
        for (let ii = i; ii <= iW; ii++) visited[ii][jj] = true;
      }
    }
  }

  // Close the front seam at the nonlinear-mapping boundary to avoid gaps
  const jBoundary = Math.max(0, Math.min(bins - 2, activeRowsCount - 1));
  for (let i = 0; i < slices - 1; i++) {
    const a0 = indexTop[i][jBoundary];
    const a1 = indexTop[i + 1][jBoundary];
    const b0 = indexTop[i][jBoundary + 1];
    const b1 = indexTop[i + 1][jBoundary + 1];
    emitFace(a0, a1, b1, OUT_FRONT, baseMat);
    emitFace(a0, b1, b0, OUT_FRONT, baseMat);
  }

  // Triangulate the bottom surface cell-by-cell to match the top grid
  for (let i = 0; i < slices - 1; i++) {
    for (let j = 0; j < bins - 1; j++) {
      const a = indexBottom[i][j];
      const b = indexBottom[i + 1][j];
      const c = indexBottom[i + 1][j + 1];
      const d = indexBottom[i][j + 1];
      emitFace(a, b, c, OUT_BOTTOM, baseMat);
      emitFace(a, c, d, OUT_BOTTOM, baseMat);
    }
  }

  // Side walls: connect top borders to bottom grid
  // Back edge z = z0 → vertical wall using bottom grid row j=0
  for (let i = 0; i < slices - 1; i++) {
    const aTop = indexTop[i][0];
    const bTop = indexTop[i + 1][0];
    const aBot = indexBottom[i][0];
    const bBot = indexBottom[i + 1][0];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_BACK, baseMat);
    emitFace(aTop, bBot, bTop, OUT_BACK, baseMat);
  }
  // Back edge is closed directly by the vertical wall from top edge down to base back edge
  // (emitted above), avoiding extra overlapping geometry.

  // Removed side faces for the deleted back shelf.
  // Front edge z = z0 + depth → vertical wall using bottom grid row j=bins-1
  for (let i = 0; i < slices - 1; i++) {
    const aTop = indexTop[i][bins - 1];
    const bTop = indexTop[i + 1][bins - 1];
    const aBot = indexBottom[i][bins - 1];
    const bBot = indexBottom[i + 1][bins - 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_FRONT, baseMat);
    emitFace(aTop, bBot, bTop, OUT_FRONT, baseMat);
  }
  // Side walls (left and right) with per-row bottoms to avoid giant triangles
  // Precompute side bottoms along Z
  const sideBottomLeft = new Array(bins);
  const sideBottomRight = new Array(bins);
  const activeRows = bins - 1; // since we truncated to active band + boundary
  for (let j = 0; j < bins; j++) {
    let z;
    if (j < activeRows) {
      const t = j / (activeRows - 1);
      const tExp = Math.pow(t, frequencyExponent);
      z = z0 + tExp * depth;
    } else {
      z = z0 + depth;
    }
    sideBottomLeft[j] = indexBottom[0][j];
    sideBottomRight[j] = indexBottom[slices - 1][j];
  }
  // Left wall
  for (let j = 0; j < bins - 1; j++) {
    const aTop = indexTop[0][j];
    const bTop = indexTop[0][j + 1];
    const aBot = sideBottomLeft[j];
    const bBot = sideBottomLeft[j + 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_LEFT, baseMat);
    emitFace(aTop, bBot, bTop, OUT_LEFT, baseMat);
  }
  // Ensure seam closed at top/bottom of left wall
  emitFace(indexTop[0][bins - 2], indexTop[0][bins - 1], sideBottomLeft[bins - 1], OUT_LEFT, baseMat);
  emitFace(indexTop[0][bins - 2], sideBottomLeft[bins - 1], sideBottomLeft[bins - 2], OUT_LEFT, baseMat);
  // Right wall
  for (let j = 0; j < bins - 1; j++) {
    const aTop = indexTop[slices - 1][j];
    const bTop = indexTop[slices - 1][j + 1];
    const aBot = sideBottomRight[j];
    const bBot = sideBottomRight[j + 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_RIGHT, baseMat);
    emitFace(aTop, bBot, bTop, OUT_RIGHT, baseMat);
  }
  emitFace(indexTop[slices - 1][bins - 2], sideBottomRight[bins - 2], indexTop[slices - 1][bins - 1], OUT_RIGHT, baseMat);
  emitFace(indexTop[slices - 1][bins - 1], sideBottomRight[bins - 2], sideBottomRight[bins - 1], OUT_RIGHT, baseMat);

  // No extra projection behind the rear plane; clipped at z0

  const header = [
    '# Dial-up spectrogram export',
    'mtllib dialup_spectrogram.mtl',
    'g spectrogram',
  ];
  const objText = header.concat(v).concat(f).join('\n');

  // Build MTL text from used materials
  const mtlLines = ['# Materials'];
  // Ensure baseMat defined
  const baseKey = `${Math.round(baseColor.r * 255)}_${Math.round(baseColor.g * 255)}_${Math.round(baseColor.b * 255)}`;
  if (!materials.has(baseKey)) materials.set(baseKey, { name: baseMat, r: baseColor.r, g: baseColor.g, b: baseColor.b });
  for (const { name, r, g, b } of materials.values()) {
    mtlLines.push(`newmtl ${name}`);
    mtlLines.push(`Kd ${r.toFixed(6)} ${g.toFixed(6)} ${b.toFixed(6)}`);
    mtlLines.push('Ka 0 0 0');
    mtlLines.push('Ks 0 0 0');
    mtlLines.push('illum 1');
    mtlLines.push('d 1');
    mtlLines.push('');
  }
  const mtlText = mtlLines.join('\n');
  return { objText, mtlText };
}

function downloadTextAsFile(filename, text) {
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Clear the current visualization back to a flat plane and base color
function resetVisualization() {
  const posArray = geometry.attributes.position.array;
  const baseColor = { r: 0.15, g: 0.15, b: 0.15 }; // neutral gray, not red
  for (let i = 0; i < positionAttr.count; i++) {
    posArray[i * 3 + 1] = 0; // y
    const ci = i * 3;
    colors[ci + 0] = baseColor.r;
    colors[ci + 1] = baseColor.g;
    colors[ci + 2] = baseColor.b;
  }
  positionAttr.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
  // Also clear any residual captured slices to avoid pre-populating export
  capturedSlices.length = 0;
}

// Full reset: re-create the surface to initial capacity and clear indices
function hardResetVisualization() {
  currentSliceIndex = 0;
  capacity = numSlicesInitial;
  if (surface.geometry) surface.geometry.dispose();
  geometry = createSurfaceGeometry(capacity);
  applyZRowsToGeometry(geometry, capacity);
  positionAttr = geometry.attributes.position;
  colors = new Float32Array(positionAttr.count * 3);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  surface.geometry = geometry;
  resetVisualization();
  buildAxesAndTicks();
  // Optionally recenters the view near the beginning
  positionCameraOverview();
}

// Cleanup on hot reload
window.addEventListener('beforeunload', () => {
  isRendering = false;
  try { audioEl.pause(); } catch { }
});


