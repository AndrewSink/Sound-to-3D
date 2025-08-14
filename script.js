// Three.js r140 via import map defined in index.html
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// -------- DOM --------
const container = document.getElementById('canvas-container');
const playBtn = document.getElementById('playBtn');
const toggleAxesBtn = document.getElementById('toggleAxesBtn');
const flyCamBtn = document.getElementById('flyCamBtn');
const resetBtn = document.getElementById('resetBtn');
const fileInput = document.getElementById('fileInput');
const exportBtn = document.getElementById('exportBtn');
const exportStlBtn = document.getElementById('exportStlBtn');
const audioEl = document.getElementById('audioEl');
const recordBtn = document.getElementById('recordBtn');

// -------- Audio setup --------
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
const audioSource = audioContext.createMediaElementSource(audioEl);
const analyser = audioContext.createAnalyser();
analyser.fftSize = 1024; // 512 freq bins
const defaultSmoothing = 0.85;
analyser.smoothingTimeConstant = defaultSmoothing;
audioSource.connect(analyser);
// Route media element directly to output so the analyser does not feed speakers (for mic capture)
audioSource.connect(audioContext.destination);

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
// Enable local clipping so we can preview trims with planes
renderer.localClippingEnabled = true;
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

// Surface smoothing to avoid knife-edges and produce a printable, continuous wave
// - Temporal smoothing spreads peaks across adjacent slices (along X)
// - Frequency smoothing blends neighboring bins (along Z)
const timeSmoothWindow = 7;    // odd size preferred (number of slices)
const freqSmoothWindow = 5;    // odd size preferred (number of bins)
const timeSigma = timeSmoothWindow / 3; // Gaussian sigma
const freqSigma = freqSmoothWindow / 3; // Gaussian sigma
let sliceHistory = []; // recent raw slices (Float32Array)

function makeGaussianKernel(size, sigma) {
  const n = Math.max(1, Math.floor(size) | 1); // force odd
  const s2 = sigma > 0 ? (2 * sigma * sigma) : 1;
  const half = (n - 1) / 2;
  const k = new Float32Array(n);
  let sum = 0;
  for (let i = -half, j = 0; j < n; i++, j++) {
    const w = Math.exp(-(i * i) / s2);
    k[j] = w; sum += w;
  }
  for (let j = 0; j < n; j++) k[j] /= sum;
  return k;
}

function reflectedIndex(i, n) {
  // Reflect at boundaries: -1->1, n->n-2, etc.
  if (n <= 1) return 0;
  const period = 2 * (n - 1);
  let m = i % period; if (m < 0) m += period;
  return m <= (n - 1) ? m : period - m;
}

const Kt = makeGaussianKernel(timeSmoothWindow, timeSigma);
const Kz = makeGaussianKernel(freqSmoothWindow, freqSigma);

function smoothAcrossFrequency(src) {
  const n = src.length;
  const half = (Kz.length - 1) / 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let j = -half, k = 0; k < Kz.length; j++, k++) {
      const idx = reflectedIndex(i + j, n);
      acc += src[idx] * Kz[k];
    }
    out[i] = acc;
  }
  return out;
}

function smoothAcrossTimeWithHistory(newSlice) {
  sliceHistory.push(newSlice);
  if (sliceHistory.length > Kt.length) sliceHistory.shift();
  // Weighted average of available history, normalized to available weights
  const nAvail = sliceHistory.length;
  const half = (Kt.length - 1) / 2;
  // Center the kernel at the newest slice
  const out = new Float32Array(newSlice.length);
  let weightSum = 0;
  for (let o = 0; o < nAvail; o++) {
    const kIndex = Kt.length - 1 - o; // newest aligns with kernel center
    const w = Kt[Math.max(0, Math.min(Kt.length - 1, kIndex))];
    weightSum += w;
  }
  const norm = weightSum > 0 ? (1 / weightSum) : 1;
  for (let i = 0; i < out.length; i++) {
    let acc = 0; let wsum = 0; // track exact weight sum of available terms
    for (let o = 0; o < nAvail; o++) {
      const kIndex = Kt.length - 1 - o;
      const w = Kt[Math.max(0, Math.min(Kt.length - 1, kIndex))];
      acc += sliceHistory[nAvail - 1 - o][i] * w;
      wsum += w;
    }
    out[i] = wsum > 0 ? (acc / wsum) : newSlice[i];
  }
  return out;
}

function processSliceForSurface(rawSlice) {
  const tSmoothed = smoothAcrossTimeWithHistory(rawSlice);
  const zSmoothed = smoothAcrossFrequency(tSmoothed);
  return zSmoothed;
}

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
// Hide the legacy deformable surface; we'll render per-slice modules instead
surface.visible = false;

// Enable per-slice geometry generation (append-only modules each frame)
// Legacy deformable surface path is fully disabled
const USE_PER_SLICE_GEOMETRY = true;

// Group to hold per-slice modules (top strip + helper geometry spans)
const slicesGroup = new THREE.Group();
scene.add(slicesGroup);
let sliceModules = [];
let lastSliceAmps = new Float32Array(pointsPerSlice); // initialized to zeros

function clearSliceModules() {
  while (slicesGroup.children.length) {
    const child = slicesGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material && child.material.dispose && child.material !== wallsMaterial && child.material !== material) {
      child.material.dispose();
    }
  }
  sliceModules = [];
  lastSliceAmps = new Float32Array(pointsPerSlice);
}

function appendSliceModule(prevAmps, currAmps) {
  const index = sliceModules.length; // 0-based module index
  const xPrev = worldLeftOffsetX + index * sliceSpacing;
  const xCurr = xPrev + sliceSpacing;

  // Top strip between prev and current across Z (quad strip)
  const topPositions = new Float32Array((pointsPerSlice * 2) * 3);
  const topColors = new Float32Array((pointsPerSlice * 2) * 3);
  const topIndices = new Uint32Array((pointsPerSlice - 1) * 6);
  for (let z = 0; z < pointsPerSlice; z++) {
    const zWorld = zRowPositions[z];
    const yPrev = prevAmps[z] * depth * heightScale;
    const yCurr = currAmps[z] * depth * heightScale;
    const v0 = z * 2;     // (xPrev, yPrev)
    const v1 = v0 + 1;    // (xCurr, yCurr)
    // v0
    topPositions[3 * v0 + 0] = xPrev;
    topPositions[3 * v0 + 1] = yPrev;
    topPositions[3 * v0 + 2] = zWorld;
    const c0 = referenceColorRamp(prevAmps[z]);
    topColors[3 * v0 + 0] = c0.r; topColors[3 * v0 + 1] = c0.g; topColors[3 * v0 + 2] = c0.b;
    // v1
    topPositions[3 * v1 + 0] = xCurr;
    topPositions[3 * v1 + 1] = yCurr;
    topPositions[3 * v1 + 2] = zWorld;
    const c1 = referenceColorRamp(currAmps[z]);
    topColors[3 * v1 + 0] = c1.r; topColors[3 * v1 + 1] = c1.g; topColors[3 * v1 + 2] = c1.b;
  }
  let ti = 0;
  for (let z = 0; z < pointsPerSlice - 1; z++) {
    const a = z * 2;     // prev(z)
    const b = a + 1;     // curr(z)
    const c = a + 2;     // prev(z+1)
    const d = c + 1;     // curr(z+1)
    // triangles: a,b,d and a,d,c
    topIndices[ti++] = a; topIndices[ti++] = b; topIndices[ti++] = d;
    topIndices[ti++] = a; topIndices[ti++] = d; topIndices[ti++] = c;
  }
  const topGeo = new THREE.BufferGeometry();
  topGeo.setAttribute('position', new THREE.BufferAttribute(topPositions, 3));
  topGeo.setAttribute('color', new THREE.BufferAttribute(topColors, 3));
  topGeo.setIndex(new THREE.BufferAttribute(topIndices, 1));
  topGeo.computeVertexNormals();
  const topMat = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide });
  const topMesh = new THREE.Mesh(topGeo, topMat);
  slicesGroup.add(topMesh);

  // Helper geometry per slice
  const helper = [];
  // Base span plane
  {
    const baseGeo = new THREE.PlaneGeometry(sliceSpacing, depth, 1, 1);
    baseGeo.rotateX(-Math.PI / 2);
    baseGeo.translate((xPrev + xCurr) / 2, -baseThickness, 0);
    const mesh = new THREE.Mesh(baseGeo, wallsMaterial);
    slicesGroup.add(mesh); helper.push(mesh);
  }
  // Right wall at xCurr
  {
    const vcount = pointsPerSlice * 2;
    const positions = new Float32Array(vcount * 3);
    const indices = new Uint32Array((pointsPerSlice - 1) * 6);
    for (let z = 0; z < pointsPerSlice; z++) {
      const zW = zRowPositions[z];
      const yTop = currAmps[z] * depth * heightScale;
      const it = z * 2;
      const ib = it + 1;
      positions[3 * it + 0] = xCurr; positions[3 * it + 1] = yTop; positions[3 * it + 2] = zW;
      positions[3 * ib + 0] = xCurr; positions[3 * ib + 1] = -baseThickness; positions[3 * ib + 2] = zW;
    }
    let pi = 0;
    for (let z = 0; z < pointsPerSlice - 1; z++) {
      const a = z * 2, b = a + 2, c = a + 1, d = b + 1;
      indices[pi++] = a; indices[pi++] = c; indices[pi++] = d;
      indices[pi++] = a; indices[pi++] = d; indices[pi++] = b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, wallsMaterial);
    slicesGroup.add(mesh); helper.push(mesh);
  }
  // Back span (z = zMin) across xPrev..xCurr using row 0
  {
    const yPrev = prevAmps[0] * depth * heightScale;
    const yCurr = currAmps[0] * depth * heightScale;
    const geo = new THREE.BufferGeometry();
    const posArr = new Float32Array(4 * 3);
    const idxArr = new Uint16Array([0, 1, 3, 0, 3, 2]);
    posArr.set([xPrev, yPrev, zMin, xCurr, yCurr, zMin, xPrev, -baseThickness, zMin, xCurr, -baseThickness, zMin]);
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, wallsMaterial);
    slicesGroup.add(mesh); helper.push(mesh);
  }
  // Front span (z = zMax) across xPrev..xCurr using last row
  {
    const yPrev = prevAmps[pointsPerSlice - 1] * depth * heightScale;
    const yCurr = currAmps[pointsPerSlice - 1] * depth * heightScale;
    const geo = new THREE.BufferGeometry();
    const posArr = new Float32Array(4 * 3);
    const idxArr = new Uint16Array([0, 1, 3, 0, 3, 2]);
    posArr.set([xPrev, yPrev, zMax, xCurr, yCurr, zMax, xPrev, -baseThickness, zMax, xCurr, -baseThickness, zMax]);
    geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
    geo.setIndex(new THREE.BufferAttribute(idxArr, 1));
    geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, wallsMaterial);
    slicesGroup.add(mesh); helper.push(mesh);
  }
  sliceModules.push({ topMesh, helper });
}

// Limit drawing of the deformable surface to only the filled region
function updateSurfaceDrawRange() {
  if (!geometry || !geometry.index) return;
  const segmentsX = capacity - 1;
  const segmentsZ = pointsPerSlice - 1;
  const filledCols = getFilledColumns();
  const filledSegX = Math.max(0, Math.min(segmentsX, filledCols - 1));
  const triangles = filledSegX * segmentsZ * 2;
  const indicesCount = triangles * 3;
  geometry.setDrawRange(0, indicesCount);
}

// -------- Real-time watertight extrusions (base + 4 walls) --------
const extrusionsGroup = new THREE.Group();
scene.add(extrusionsGroup);
let baseMesh = null;
let backWallMesh = null;
let frontWallMesh = null;
let leftWallMesh = null;
let rightWallMesh = null;
const wallsMaterial = new THREE.MeshLambertMaterial({ color: 0xb3b3b3, side: THREE.DoubleSide, toneMapped: false });
// Ensure all mesh materials render double-sided to avoid culling artifacts
material.side = THREE.DoubleSide;
wallsMaterial.side = THREE.DoubleSide;
const topStripMaterial = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, toneMapped: false });

let unifiedMesh = null; // single watertight mesh built at finalize

function clearUnifiedMesh() {
  if (unifiedMesh) {
    if (unifiedMesh.geometry) unifiedMesh.geometry.dispose();
    if (unifiedMesh.material) {
      const mats = Array.isArray(unifiedMesh.material) ? unifiedMesh.material : [unifiedMesh.material];
      for (const m of mats) m.dispose && m.dispose();
    }
    scene.remove(unifiedMesh);
    unifiedMesh = null;
  }
}

function clearExtrusions() {
  while (extrusionsGroup.children.length) {
    const m = extrusionsGroup.children.pop();
    if (m.geometry) m.geometry.dispose();
    if (m.material && m.material !== wallsMaterial) m.material.dispose && m.material.dispose();
  }
  baseMesh = backWallMesh = frontWallMesh = leftWallMesh = rightWallMesh = null;
}

function getFilledColumns() {
  // Number of populated time columns (slices) on screen
  const filledByStream = currentSliceIndex; // columns written into the geometry during playback
  const filledByFinalize = capturedSlices.length > 0 ? capturedSlices.length : 0; // after finalize/trim
  const filled = Math.max(filledByStream, filledByFinalize);
  return Math.max(0, filled);
}

function ensureExtrusionMeshes() {
  if (!baseMesh) {
    baseMesh = new THREE.Mesh(new THREE.BufferGeometry(), wallsMaterial);
    extrusionsGroup.add(baseMesh);
  }
  if (!backWallMesh) {
    backWallMesh = new THREE.Mesh(new THREE.BufferGeometry(), wallsMaterial);
    extrusionsGroup.add(backWallMesh);
  }
  if (!frontWallMesh) {
    frontWallMesh = new THREE.Mesh(new THREE.BufferGeometry(), wallsMaterial);
    extrusionsGroup.add(frontWallMesh);
  }
  if (!leftWallMesh) {
    leftWallMesh = new THREE.Mesh(new THREE.BufferGeometry(), wallsMaterial);
    extrusionsGroup.add(leftWallMesh);
  }
  if (!rightWallMesh) {
    rightWallMesh = new THREE.Mesh(new THREE.BufferGeometry(), wallsMaterial);
    extrusionsGroup.add(rightWallMesh);
  }
}

function updateExtrusionsRealtime() {
  const filledCols = getFilledColumns();
  if (filledCols < 2) { clearExtrusions(); return; }
  ensureExtrusionMeshes();
  const pos = geometry.attributes.position;
  const leftX = worldLeftOffsetX;
  const width = (filledCols - 1) * sliceSpacing;

  // Base slab: simple plane at y = -baseThickness
  {
    const bx = leftX + width / 2;
    const by = -baseThickness;
    const bz = 0;
    const geo = new THREE.PlaneGeometry(width, depth, 1, 1);
    geo.rotateX(-Math.PI / 2);
    geo.translate(bx, by, bz);
    baseMesh.geometry.dispose();
    baseMesh.geometry = geo;
  }

  // Helper to build a wall along X (z fixed)
  function buildWallAlongX(targetMesh, rowZIndex, zValue) {
    const vcount = (filledCols) * 2; // top+bottom per column
    const positions = new Float32Array(vcount * 3);
    const indices = new Uint32Array((filledCols - 1) * 6);
    for (let x = 0; x < filledCols; x++) {
      const worldX = leftX + x * sliceSpacing;
      const yTop = pos.getY(rowZIndex * capacity + x);
      const iTop = x * 2;
      const iBot = iTop + 1;
      positions[3 * iTop + 0] = worldX;
      positions[3 * iTop + 1] = yTop;
      positions[3 * iTop + 2] = zValue;
      positions[3 * iBot + 0] = worldX;
      positions[3 * iBot + 1] = -baseThickness;
      positions[3 * iBot + 2] = zValue;
    }
    let idx = 0;
    for (let x = 0; x < filledCols - 1; x++) {
      const a = x * 2;      // top(x)
      const b = a + 2;      // top(x+1)
      const c = a + 1;      // bot(x)
      const d = b + 1;      // bot(x+1)
      indices[idx++] = a; indices[idx++] = c; indices[idx++] = d;
      indices[idx++] = a; indices[idx++] = d; indices[idx++] = b;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    if (targetMesh.geometry) targetMesh.geometry.dispose();
    targetMesh.geometry = geo;
  }

  // Helper to build a wall along Z (x fixed)
  function buildWallAlongZ(targetMesh, colXIndex, xValue) {
    const vcount = (pointsPerSlice) * 2; // per row top+bottom
    const positions = new Float32Array(vcount * 3);
    const indices = new Uint32Array((pointsPerSlice - 1) * 6);
    for (let z = 0; z < pointsPerSlice; z++) {
      const yTop = pos.getY(z * capacity + colXIndex);
      const worldZ = zRowPositions[z];
      const iTop = z * 2;
      const iBot = iTop + 1;
      positions[3 * iTop + 0] = xValue;
      positions[3 * iTop + 1] = yTop;
      positions[3 * iTop + 2] = worldZ;
      positions[3 * iBot + 0] = xValue;
      positions[3 * iBot + 1] = -baseThickness;
      positions[3 * iBot + 2] = worldZ;
    }
    let idx = 0;
    for (let z = 0; z < pointsPerSlice - 1; z++) {
      const a = z * 2;
      const b = a + 2;
      const c = a + 1;
      const d = b + 1;
      indices[idx++] = a; indices[idx++] = d; indices[idx++] = c;
      indices[idx++] = a; indices[idx++] = b; indices[idx++] = d;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    geo.computeVertexNormals();
    if (targetMesh.geometry) targetMesh.geometry.dispose();
    targetMesh.geometry = geo;
  }

  // Back/front walls
  buildWallAlongX(backWallMesh, 0, zMin);
  buildWallAlongX(frontWallMesh, pointsPerSlice - 1, zMax);
  // Left/right walls using first and last available columns
  buildWallAlongZ(leftWallMesh, 0, leftX);
  buildWallAlongZ(rightWallMesh, Math.max(0, filledCols - 1), leftX + width);
}

// -------- Reference grid (optional) --------
const GRID_ENABLED = false; // Turn off the grid entirely
const gridGroup = new THREE.Group();
scene.add(gridGroup);
gridGroup.visible = GRID_ENABLED;

function clearGrid() {
  if (!GRID_ENABLED) return;
  while (gridGroup.children.length) {
    const child = gridGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose && child.material.dispose();
  }
}

function buildReferenceGrid(widthOverride, clipStartX = 0, clipEndX = 0) {
  if (!GRID_ENABLED) return;
  clearGrid();
  const widthWorld = (typeof widthOverride === 'number') ? widthOverride : sliceSpacing * (capacity - 1);
  const z0 = zMin;
  const z1 = zMax;
  const y = 0.0; // coincide with spectrogram base plane (y=0)

  // Choose roughly even spacing that scales with size
  const xDiv = Math.max(6, Math.round(widthWorld / (sliceSpacing * 64)) * 8);
  const zDiv = 10;
  const dx = widthWorld / xDiv;
  const dz = (z1 - z0) / zDiv;
  const startX = Math.max(0, Math.min(widthWorld, clipStartX));
  const endX = Math.max(0, Math.min(widthWorld, clipEndX));

  const segments = [];

  function appendRange(xA, xB) {
    const a = Math.max(0, Math.min(widthWorld, xA));
    const b = Math.max(0, Math.min(widthWorld, xB));
    if (b <= a) return;
    // Lines along X (vary Z)
    for (let i = 0; i <= zDiv; i++) {
      const z = z0 + dz * i;
      segments.push(a, y, z, b, y, z);
    }
    // Lines along Z (vary X)
    for (let i = 0; i <= xDiv; i++) {
      const x = dx * i;
      if (x >= a && x <= b) {
        segments.push(x, y, z0, x, y, z1);
      }
    }
  }

  if (clipEndX > 0 && endX > startX) {
    // Two visible regions: [0, startX] and [endX, width]
    appendRange(0, startX);
    appendRange(endX, widthWorld);
  } else {
    // Single visible region: [startX, width]
    appendRange(startX, widthWorld);
  }

  const pos = new Float32Array(segments);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x3f3f46 });
  // Keep grid color constant and draw independent of depth (we clip under the model)
  mat.depthTest = true;
  mat.depthWrite = true;
  mat.toneMapped = false;
  mat.transparent = false;
  const lines = new THREE.LineSegments(geo, mat);
  gridGroup.add(lines);
}

// Tracks the world-space X where the current spectrogram geometry starts
let worldLeftOffsetX = 0;

function getCurrentMaxX() {
  const slices = Math.max(currentSliceIndex, capturedSlices.length);
  return Math.max(0, (Math.max(2, slices) - 1) * sliceSpacing);
}

// Trim functionality removed

// Trim functionality removed

// Trim functionality removed

// Dragging logic
// Trim functionality removed

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
// Build the reference grid once on initial load so it's visible before playback
buildReferenceGrid(undefined, 0);

// Maintain a continuous scroll by shifting historic columns left
let isCapturing = false;
let hasEnded = false;
const capturedSlices = []; // array of Float32Array(length: pointsPerSlice)
const baseThickness = 4; // export base thickness (units ~ same as scene)
const backThickness = 6; // small back slab thickness along -Z for printability

// -------- Recording state (microphone) --------
let isRecording = false;
let micStream = null;
let micSource = null;
let mediaRecorder = null;
let recordedChunks = [];

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
  // Rebuild reference grid to match new width
  buildReferenceGrid(sliceSpacing * (newCapacity - 1));
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

  // Apply smoothing across time (with history) and frequency to avoid knife-edges
  const processed = processSliceForSurface(newSlice);

  // Persist the smoothed slice for export later
  if (isCapturing) {
    capturedSlices.push(processed.slice());
  }

  // Grow backing buffers if needed to maintain z mapping for export/sampling
  if (currentSliceIndex >= capacity) {
    const nextCapacity = Math.ceil(capacity * 1.6);
    expandGeometry(nextCapacity);
    // Keep legacy surface hidden after growth
    surface.visible = false;
  }

  // For each z-row (frequency), write this slice at the current column
  for (let z = 0; z < pointsPerSlice; z++) {
    const vIndex = z * capacity + currentSliceIndex;
    const y = processed[z] * depth * heightScale;
    positionAttr.setY(vIndex, y);
    const color = referenceColorRamp(processed[z]);
    const cIdx = vIndex * 3;
    colors[cIdx + 0] = color.r;
    colors[cIdx + 1] = color.g;
    colors[cIdx + 2] = color.b;
  }

  currentSliceIndex++;
  positionAttr.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;

  // Per-slice path only: append module for the new slice (smoothed)
  appendSliceModule(lastSliceAmps, processed);
  lastSliceAmps = processed;
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

// Frequency axis world X position (used for axis placement and frequency grid clipping)
const frequencyAxisX = -2;

// Infinite grid that renders only on the left side of the frequency axis (x <= frequencyAxisX)
const freqGridGroup = new THREE.Group();
scene.add(freqGridGroup);

function clearFrequencyGrid() {
  while (freqGridGroup.children.length) {
    const child = freqGridGroup.children.pop();
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose && child.material.dispose();
  }
}

function buildFrequencyAxisGrid() {
  clearFrequencyGrid();
  const size = 100000; // effectively infinite at our scene scale
  const geo = new THREE.PlaneGeometry(size, size, 1, 1);
  geo.rotateX(-Math.PI / 2);
  // Place below the helper base so it never clips with geometry (extra 0.5 units down)
  geo.translate(0, -baseThickness - 0.55, 0);
  // Compute Z clipping range to show only between 0 Hz and 15k on the axis
  const nyquistHz = audioContext.sampleRate / 2;
  function zAtHz(hz) {
    const clamped = Math.max(0, Math.min(nyquistHz, hz));
    const binIndex = (clamped / nyquistHz) * freqBinCount; // 0..freqBinCount
    const rowIndex = Math.round(binIndex / sliceStride);
    const idx = Math.max(0, Math.min(pointsPerSlice - 1, rowIndex));
    // Constrain to active band so we don't go into the collapsed region
    const activeIdx = Math.max(0, Math.min(activeRows - 1, idx));
    return zRowPositions[activeIdx];
  }
  const zClipMin = zAtHz(0);
  const zClipMax = zAtHz(15000);
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uMinor: { value: 4.0 },
      uMajor: { value: 20.0 },
      uColorMinor: { value: new THREE.Color(0x3f3f46) },
      uColorMajor: { value: new THREE.Color(0x3f3f46) },
      uOpacity: { value: 0.65 },
      uClipX: { value: frequencyAxisX - 1e-4 }, // include the axis line (right side)
      uZMin: { value: Math.min(zClipMin, zClipMax) },
      uZMax: { value: Math.max(zClipMin, zClipMax) },
    },
    transparent: true,
    depthTest: true,
    depthWrite: false,
    toneMapped: false,
    side: THREE.DoubleSide,
    extensions: { derivatives: true },
    vertexShader: `
      varying vec3 vWorld;
      void main(){
        vec4 w = modelMatrix * vec4(position, 1.0);
        vWorld = w.xyz;
        gl_Position = projectionMatrix * viewMatrix * w;
      }
    `,
    fragmentShader: `
      varying vec3 vWorld;
      uniform float uMinor;
      uniform float uMajor;
      uniform vec3 uColorMinor;
      uniform vec3 uColorMajor;
      uniform float uOpacity;
      uniform float uClipX;
      uniform float uZMin;
      uniform float uZMax;

      float gridFactor(vec2 p, float step, float width){
        vec2 q = p / step;
        vec2 w = abs(fract(q - 0.5) - 0.5) / fwidth(q);
        float line = min(w.x, w.y);
        return 1.0 - smoothstep(width, width + 1.0, line);
      }

      void main(){
        // Show only on the RIGHT side of the frequency axis
        if (vWorld.x < uClipX) discard;
        // Restrict to 0..15k Hz span on Z
        if (vWorld.z < uZMin || vWorld.z > uZMax) discard;
        vec2 p = vWorld.xz;
        float minor = gridFactor(p, uMinor, 0.8);
        float major = gridFactor(p, uMajor, 1.2);
        float a = max(major, minor * 0.6) * uOpacity;
        vec3 col = mix(uColorMinor, uColorMajor, step(minor, major));
        gl_FragColor = vec4(col, a);
      }
    `,
  });
  const mesh = new THREE.Mesh(geo, mat);
  // Render the grid before opaque geometry; since it doesn't write depth,
  // later draws (spectrogram/base) will fully occlude it.
  mesh.renderOrder = -100;
  freqGridGroup.add(mesh);
}

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
    return `${Math.round(kilo)}k`;
  }
  return `${Math.round(hz)}`;
}

function buildAxesAndTicks() {
  clearAxesGroup();

  // Place axes on the left side of the mesh at a fixed world X (slightly left of x=0)
  const axisX = frequencyAxisX; // just left of surface's anchored left edge (x=0)
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

  // Frequency tick marks and labels (reduced low-frequency clutter)
  const nyquistHz = audioContext.sampleRate / 2;
  const desiredFreqHz = [1000, 2000, 5000, 10000, 15000];
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

    const label = createTextSprite(formatHzLabel(hz), { worldHeight: 3.5 });
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

    const label = createTextSprite(`${db}`, { worldHeight: 3.5 });
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
  const freqTitle = createTextSprite('Frequency (Hz)', { worldHeight: 4.5 });
  freqTitle.position.set(axisX - 3.2, 0.01, (backZ + frontActiveZ) / 2);
  freqTitle.center.set(0.5, 0.5);
  freqTitle.renderOrder = 2;
  axesGroup.add(freqTitle);

  const ampTitle = createTextSprite('Amplitude (dB)', { worldHeight: 4.5 });
  ampTitle.position.set(axisX - 3.2, amplitudeMaxY, backZ);
  ampTitle.center.set(0.5, 0);
  ampTitle.renderOrder = 2;
  axesGroup.add(ampTitle);
}

// Build initial axes now that geometry mapping is known
buildAxesAndTicks();
buildFrequencyAxisGrid();

// Place the camera so the view centers between the axes (left) and the surface (center),
// backed up and slightly orbiting to the right similar to the reference angle.
function positionCameraOverview() {
  // Straight-on view, grid framed slightly higher on screen
  // Nudge left on small/mobile screens so axes are in frame on load
  const isMobileNarrow = Math.min(window.innerWidth || 0, document.documentElement.clientWidth || 0) <= 640;
  const targetX = isMobileNarrow ? sliceSpacing * 45 : sliceSpacing * 120;
  const targetY = depth * heightScale * .2; // slightly higher center so grid sits more centered
  const targetZ = 0;
  controls.target.set(targetX, targetY, targetZ);

  const amplitudeMaxY = depth * heightScale;
  const dx = 0;                  // no rightward orbit -> straight-on
  const dy = amplitudeMaxY * 1.2; // modest elevation
  const dz = sliceSpacing * 260;  // pull back to frame front area
  camera.position.set(targetX + dx, targetY + dy, targetZ + dz);
  controls.update();
  controls.saveState && controls.saveState();
}

// Only set the initial overview once on load; do not reset camera on play
positionCameraOverview();

// Initialize visualization state (keep legacy surface hidden)
resetVisualization();

// (Removed initial procedural preview per request)

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
let flyCamActive = false;
let flyCamAnchor = null; // { rel: THREE.Vector3, relLen: number, dxRight: number, targetY: number, targetZ: number }

function enableFlyCam(active) {
  flyCamActive = !!active;
  if (flyCamBtn) flyCamBtn.textContent = flyCamActive ? 'Stop Tracking' : 'Tracking Camera';
  if (flyCamActive) {
    // Capture current relative camera offset to target so angle is preserved
    const rel = new THREE.Vector3().subVectors(camera.position, controls.target);
    const slices = Math.max(currentSliceIndex, capturedSlices.length);
    const totalWidth = Math.max(1, slices - 1) * sliceSpacing;
    const rightX = worldLeftOffsetX + totalWidth;
    flyCamAnchor = {
      rel: rel.clone(),
      relLen: rel.length(),
      dxRight: controls.target.x - rightX, // keep same X offset from the growing right edge
      targetY: controls.target.y,
      targetZ: controls.target.z,
    };
    // Disable manual orbit/pan while flying
    controls.enableRotate = false;
    controls.enablePan = false;
    controls.enableZoom = false;
  } else {
    controls.enableRotate = true;
    controls.enablePan = true;
    controls.enableZoom = true;
  }
}
function animate() {
  if (!isRendering) return;
  requestAnimationFrame(animate);
  if ((isRecording || (!audioEl.paused && !hasEnded))) {
    updateSurfaceFromFrequencies();
    // Only per-slice modules are rendered during growth; keep legacy surface hidden
    surface.visible = false;
  }
  // Update fly cam position to maintain angle and pull back as width grows
  if (flyCamActive && flyCamAnchor) {
    const slices = Math.max(currentSliceIndex, capturedSlices.length);
    const totalWidth = Math.max(1, slices - 1) * sliceSpacing;
    const rightX = worldLeftOffsetX + totalWidth;
    const desiredTarget = new THREE.Vector3(
      rightX + flyCamAnchor.dxRight,
      flyCamAnchor.targetY,
      flyCamAnchor.targetZ
    );
    // Smoothly move target to follow the right edge
    controls.target.lerp(desiredTarget, 0.12);
    const desiredPos = controls.target.clone().add(flyCamAnchor.rel);
    camera.position.lerp(desiredPos, 0.12);
    controls.update();
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

if (flyCamBtn) {
  flyCamBtn.addEventListener('click', () => enableFlyCam(!flyCamActive));
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
  if (exportStlBtn) exportStlBtn.disabled = !enabled;
}

// Trim functionality removed: keep a no-op for calls from existing flows
function setTrimAvailability() { }

function setRecordButtonState(isRec) {
  if (!recordBtn) return;
  if (isRec) {
    recordBtn.textContent = 'Stop Recording';
    recordBtn.classList.remove('bg-rose-600', 'hover:bg-rose-500');
    recordBtn.classList.add('bg-yellow-500', 'hover:bg-yellow-400');
  } else {
    recordBtn.textContent = 'Record Audio';
    recordBtn.classList.remove('bg-yellow-500', 'hover:bg-yellow-400');
    recordBtn.classList.add('bg-rose-600', 'hover:bg-rose-500');
  }
}

function setPlayAvailability(enabled) {
  if (!playBtn) return;
  playBtn.disabled = !enabled;
  if (enabled) {
    playBtn.classList.remove('opacity-40', 'cursor-not-allowed');
  } else {
    playBtn.classList.add('opacity-40', 'cursor-not-allowed');
  }
}

function setRecordAvailability(enabled) {
  if (!recordBtn) return;
  recordBtn.disabled = !enabled;
  if (enabled) {
    recordBtn.classList.remove('opacity-40', 'cursor-not-allowed');
  } else {
    recordBtn.classList.add('opacity-40', 'cursor-not-allowed');
  }
}

// Trim functionality removed: keep a no-op for calls from existing flows
function setControlsForTrimMode() { }

// Disable specific controls while recording: Play, Load Audio, Hide Axes, Reset
function setControlsForRecording(active) {
  const toToggle = [playBtn, toggleAxesBtn, resetBtn, fileInput];
  for (const el of toToggle) {
    if (!el) continue;
    el.disabled = !!active;
  }
  const toStyle = [playBtn, toggleAxesBtn, resetBtn];
  for (const el of toStyle) {
    if (!el) continue;
    if (active) el.classList.add('opacity-40', 'cursor-not-allowed');
    else el.classList.remove('opacity-40', 'cursor-not-allowed');
  }
  const fileLabel = document.querySelector('label[for="fileInput"]');
  if (fileLabel) {
    if (active) fileLabel.classList.add('opacity-40', 'cursor-not-allowed', 'pointer-events-none');
    else fileLabel.classList.remove('opacity-40', 'cursor-not-allowed', 'pointer-events-none');
  }
}

async function startRecording() {
  try {
    if (audioContext.state === 'suspended') await audioContext.resume();
    // Pause any element playback
    try { audioEl.pause(); } catch { }

    // Request mic
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Feed mic only into analyser (not speakers)
    micSource = audioContext.createMediaStreamSource(micStream);
    micSource.connect(analyser);

    // Reset visualization state
    hasEnded = false;
    isCapturing = true;
    capturedSlices.length = 0;
    hardResetVisualization();
    flushAnalyser();
    setExportAvailability(false);
    setControlsForRecording(true);
    setTrimAvailability(false);

    // Recorder for saving to an audio file
    recordedChunks = [];
    let mimeType = '';
    if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
      mimeType = 'audio/webm;codecs=opus';
    } else if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm')) {
      mimeType = 'audio/webm';
    } else if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/mp4')) {
      mimeType = 'audio/mp4';
    }
    mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : undefined);
    mediaRecorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => {
      const type = mediaRecorder.mimeType || (mimeType || 'audio/webm');
      const blob = new Blob(recordedChunks, { type });
      const url = URL.createObjectURL(blob);
      audioEl.src = url;
      try { audioEl.load(); } catch { }
      isCapturing = false;
      isRecording = false;
      disconnectMic();
      // Do not rebuild/finalize geometry here; keep current per-slice modules
      if (capturedSlices.length < 2) synthesizeCapturedFromSurface();
      frameWholeModel();
      setRecordButtonState(false);
      setExportAvailability(true);
      setTrimAvailability(true);
      setControlsForRecording(false);
    };
    mediaRecorder.start();

    isRecording = true;
    setRecordButtonState(true);
  } catch (err) {
    console.error('Recording failed', err);
    // Best-effort cleanup
    disconnectMic();
    isRecording = false;
    setRecordButtonState(false);
  }
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    try { mediaRecorder.stop(); } catch { }
  } else {
    // No recorder; still cleanup mic
    disconnectMic();
    isRecording = false;
    setRecordButtonState(false);
  }
}

function stopRecordingIfActive() {
  if (isRecording) stopRecording();
}

function disconnectMic() {
  try { if (micSource) micSource.disconnect(); } catch { }
  if (micStream) {
    try { micStream.getTracks().forEach((t) => t.stop()); } catch { }
  }
  micSource = null;
  micStream = null;
}

playBtn.addEventListener('click', async () => {
  if (isRecording) return; // ignore play while recording

  if (!audioEl.paused) {
    // Currently playing → Pause without resetting
    audioEl.pause();
    isCapturing = false;
    setPlayButtonState(false);
    return;
  }

  // From a paused/stopped state, start from the beginning and clear the viz
  try { audioEl.currentTime = 0; audioEl.load(); } catch { }
  hasEnded = false;
  isCapturing = false;
  capturedSlices.length = 0;
  hardResetVisualization();
  // Clear analyser state so the first slice doesn't contain previous data
  flushAnalyser();
  if (audioContext.state === 'suspended') await audioContext.resume();
  try { await audioEl.play(); } catch (e) { /* ignore */ }
  isCapturing = true;
  setPlayButtonState(true);
});

resetBtn.addEventListener('click', () => {
  stopRecordingIfActive();
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
  // Trim functionality removed
  // Trim functionality removed; ensure any trim-specific disables are cleared (no-op)
  // After reset, ensure no recording-specific disables remain
  setControlsForRecording(false);
  // Stop fly cam on reset
  enableFlyCam(false);
});

fileInput.addEventListener('change', async (e) => {
  stopRecordingIfActive();
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
  // Trim functionality removed
});

// Drag & drop support
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('drop', (e) => {
  e.preventDefault();
  stopRecordingIfActive();
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
  // Trim functionality removed
});

// Auto-start visualization when audio starts
audioEl.addEventListener('play', () => {
  if (audioContext.state === 'suspended') audioContext.resume();
  setPlayButtonState(true);
  // Enable export once at least one slice lands
  setExportAvailability(true);
  // Trim functionality removed
});

audioEl.addEventListener('ended', () => {
  hasEnded = true;
  isCapturing = false;
  // Only move the camera to frame the current model; do not rebuild or finalize geometry
  frameWholeModel();
  setPlayButtonState(false);
  // Trim functionality removed
});

exportBtn.addEventListener('click', () => {
  if (capturedSlices.length < 2) {
    // If user exports early, synthesize from current displayed geometry
    // by sampling visible vertices so export always works
    synthesizeCapturedFromSurface();
  }
  const { objText, mtlText } = buildOBJFromCaptured();
  downloadTextAsFile('3D_spectrogram.mtl', mtlText);
  downloadTextAsFile('3D_spectrogram.obj', objText);
});

if (exportStlBtn) {
  exportStlBtn.addEventListener('click', () => {
    if (capturedSlices.length < 2) {
      // Synthesize from on-screen geometry to always have something to export
      synthesizeCapturedFromSurface();
    }
    const stlBuffer = buildSTLFromCaptured();
    const blob = new Blob([stlBuffer], { type: 'application/vnd.ms-pki.stl' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = '3D_spectrogram.stl';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });
}

if (recordBtn) {
  recordBtn.addEventListener('click', async () => {
    if (!isRecording) {
      await startRecording();
    } else {
      stopRecording();
    }
  });
}

// Trim functionality removed

// Rebuild the surface geometry to display the full captured model when playback finishes
function finalizeModelGeometry(leftOffsetX = 0) {
  const slices = capturedSlices.length;
  if (slices < 2) return;
  const finalWidth = sliceSpacing * (slices - 1);

  // Remove legacy/preview content
  clearExtrusions();
  clearUnifiedMesh();
  clearSliceModules();

  // Build unified watertight geometry buffers
  // We will assemble: top surface (colored) + base slab + back/front + left/right walls
  const activeRowsCount = Math.max(2, Math.floor(pointsPerSlice * activeFrequencyFraction));
  const bins = activeRowsCount + 1; // include front boundary
  const dx = sliceSpacing;
  const x0 = leftOffsetX;

  // Estimate counts conservatively
  const vTop = slices * bins;
  const vBottom = slices * bins;
  const vWallsApprox = slices * 2 * 4 + bins * 2 * 2; // rough
  const maxVertices = (vTop + vBottom + vWallsApprox) * 2; // safety margin

  const positions = [];
  const colorsArr = [];
  const indices = [];
  const groups = []; // { start, count, materialIndex }
  const MATERIAL_TOP = 0; // colored
  const MATERIAL_BASE = 1; // gray helper

  function pushVertex(x, y, z, c) {
    positions.push(x, y, z);
    if (c) colorsArr.push(c.r, c.g, c.b); else colorsArr.push(0.7, 0.7, 0.7);
    return (positions.length / 3) - 1;
  }

  function addQuad(a, b, c, d, materialIndex) {
    const start = indices.length;
    // Two triangles: a,b,d and a,d,c
    indices.push(a, b, d, a, d, c);
    const count = indices.length - start;
    groups.push({ start, count, materialIndex });
  }

  // Precompute Z per active row with same mapping as live view
  const zAt = new Array(bins);
  for (let j = 0; j < bins; j++) {
    let z;
    if (j < activeRowsCount) {
      const t = j / (activeRowsCount - 1);
      const tExp = Math.pow(t, frequencyExponent);
      z = zMin + tExp * (zMax - zMin);
    } else {
      z = zMax;
    }
    zAt[j] = z;
  }

  // Build top surface
  const vIndexTop = []; // [slices][bins]
  for (let i = 0; i < slices; i++) {
    vIndexTop[i] = new Array(bins);
    const x = x0 + dx * i;
    const slice = capturedSlices[i];
    for (let j = 0; j < bins; j++) {
      const z = zAt[j];
      const amp = (j < activeRowsCount) ? slice[j] : 0;
      const y = amp * depth * heightScale;
      const c = referenceColorRamp(amp);
      vIndexTop[i][j] = pushVertex(x, y, z, c);
    }
  }
  // Bottom base grid vertices at y = -baseThickness
  const vIndexBottom = [];
  for (let i = 0; i < slices; i++) {
    vIndexBottom[i] = new Array(bins);
    const x = x0 + dx * i;
    for (let j = 0; j < bins; j++) {
      const z = zAt[j];
      vIndexBottom[i][j] = pushVertex(x, -baseThickness, z, null);
    }
  }

  // Top faces
  for (let i = 0; i < slices - 1; i++) {
    for (let j = 0; j < bins - 1; j++) {
      const a = vIndexTop[i][j];
      const b = vIndexTop[i + 1][j];
      const c = vIndexTop[i + 1][j + 1];
      const d = vIndexTop[i][j + 1];
      addQuad(a, b, d, c, MATERIAL_TOP);
    }
  }
  // Bottom faces (base slab top)
  for (let i = 0; i < slices - 1; i++) {
    for (let j = 0; j < bins - 1; j++) {
      const a = vIndexBottom[i][j];
      const b = vIndexBottom[i + 1][j];
      const c = vIndexBottom[i + 1][j + 1];
      const d = vIndexBottom[i][j + 1];
      addQuad(a, b, d, c, MATERIAL_BASE);
    }
  }
  // Back (z = zMin) and Front (z = zMax) walls
  const jBack = 0;
  const jFront = bins - 1;
  for (let i = 0; i < slices - 1; i++) {
    // Back
    addQuad(vIndexTop[i][jBack], vIndexTop[i + 1][jBack], vIndexBottom[i][jBack], vIndexBottom[i + 1][jBack], MATERIAL_BASE);
    // Front
    addQuad(vIndexTop[i][jFront], vIndexTop[i + 1][jFront], vIndexBottom[i][jFront], vIndexBottom[i + 1][jFront], MATERIAL_BASE);
  }
  // Left and Right walls along Z
  for (let j = 0; j < bins - 1; j++) {
    // Left
    addQuad(vIndexTop[0][j], vIndexTop[0][j + 1], vIndexBottom[0][j], vIndexBottom[0][j + 1], MATERIAL_BASE);
    // Right
    addQuad(vIndexTop[slices - 1][j], vIndexTop[slices - 1][j + 1], vIndexBottom[slices - 1][j], vIndexBottom[slices - 1][j + 1], MATERIAL_BASE);
  }

  // Build unified BufferGeometry
  const posArr = new Float32Array(positions);
  const colArr = new Float32Array(colorsArr);
  const idxArr = (posArr.length / 3 > 65535) ? new Uint32Array(indices) : new Uint16Array(indices);
  const unifiedGeo = new THREE.BufferGeometry();
  unifiedGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
  unifiedGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
  unifiedGeo.setIndex(new THREE.BufferAttribute(idxArr, 1));
  unifiedGeo.computeVertexNormals();
  // Define groups
  unifiedGeo.clearGroups();
  for (const g of groups) unifiedGeo.addGroup(g.start, g.count, g.materialIndex);

  // Build mesh with groups: [colored top, gray base]
  const materials = [topStripMaterial, wallsMaterial];
  unifiedMesh = new THREE.Mesh(unifiedGeo, materials);
  scene.add(unifiedMesh);

  // Hide legacy surface
  surface.visible = false;
  worldLeftOffsetX = leftOffsetX;
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
  // Apply trim range to captured slices if trim is active
  let slices = capturedSlices.length;
  let startSlice = 0;
  let endSlice = slices - 1;
  // Trim is disabled in this build; export the full captured range
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
    // Export with Z-up and match on-screen left/right: (X=x, Y=-z, Z=y+baseThickness)
    const X = x;
    const Y = -z; // flip Y to avoid mirrored appearance in DCCs
    const Z = y + baseThickness; // puts the flat base at Z=0
    v.push(`v ${X.toFixed(5)} ${Y.toFixed(5)} ${Z.toFixed(5)}`);
    vx.push(X); vy.push(Y); vz.push(Z);
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

  // With Z-up export and Y flipped, outward directions adjust accordingly
  const OUT_TOP = { x: 0, y: 0, z: 1 };
  const OUT_BOTTOM = { x: 0, y: 0, z: -1 };
  const OUT_FRONT = { x: 0, y: -1, z: 0 };
  const OUT_BACK = { x: 0, y: 1, z: 0 };
  const OUT_LEFT = { x: -1, y: 0, z: 0 };
  const OUT_RIGHT = { x: 1, y: 0, z: 0 };

  // Top vertices with color (clamp near-flat to exact 0 to enable merging)
  const flatEps = 0.02 * depth * heightScale;
  for (let i = 0; i < slices; i++) {
    const x = x0 + dx * i;
    const slice = capturedSlices[startSlice + i];
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

  // Bottom boundary vertices only (no interior grid). We keep vertices along
  // the 4 rectangle edges so walls can share edges and the base can be a single quad.
  const bottomBack = new Array(slices);   // j = 0
  const bottomFront = new Array(slices);  // j = bins-1
  const bottomLeft = new Array(bins);     // i = 0
  const bottomRight = new Array(bins);    // i = slices-1

  const backZPlane = z0;           // equals zMin
  const frontZPlane = z0 + depth;  // equals zMax

  for (let i = 0; i < slices; i++) {
    const x = x0 + dx * i;
    bottomBack[i] = pushV(x, -baseThickness, backZPlane, baseColor.r, baseColor.g, baseColor.b);
    bottomFront[i] = pushV(x, -baseThickness, frontZPlane, baseColor.r, baseColor.g, baseColor.b);
  }
  for (let j = 0; j < bins; j++) {
    let z;
    if (j < activeRowsCount) {
      const t = j / (activeRowsCount - 1);
      const tExp = Math.pow(t, frequencyExponent);
      z = z0 + tExp * depth;
    } else {
      z = frontZPlane;
    }
    bottomLeft[j] = pushV(x0, -baseThickness, z, baseColor.r, baseColor.g, baseColor.b);
    bottomRight[j] = pushV(x0 + dx * (slices - 1), -baseThickness, z, baseColor.r, baseColor.g, baseColor.b);
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

  // Triangulate the bottom surface as a single quad using 4 corner vertices
  const bl = bottomBack[0];
  const br = bottomBack[slices - 1];
  const fr = bottomFront[slices - 1];
  const fl = bottomFront[0];
  emitFace(bl, br, fr, OUT_BOTTOM, baseMat);
  emitFace(bl, fr, fl, OUT_BOTTOM, baseMat);

  // Side walls: connect top borders to bottom grid
  // Back edge z = z0 → vertical wall using bottom boundary row j=0
  for (let i = 0; i < slices - 1; i++) {
    const aTop = indexTop[i][0];
    const bTop = indexTop[i + 1][0];
    const aBot = bottomBack[i];
    const bBot = bottomBack[i + 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_BACK, baseMat);
    emitFace(aTop, bBot, bTop, OUT_BACK, baseMat);
  }
  // Back edge is closed directly by the vertical wall from top edge down to base back edge
  // (emitted above), avoiding extra overlapping geometry.

  // Removed side faces for the deleted back shelf.
  // Front edge z = z0 + depth → vertical wall using bottom boundary row j=bins-1
  for (let i = 0; i < slices - 1; i++) {
    const aTop = indexTop[i][bins - 1];
    const bTop = indexTop[i + 1][bins - 1];
    const aBot = bottomFront[i];
    const bBot = bottomFront[i + 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_FRONT, baseMat);
    emitFace(aTop, bBot, bTop, OUT_FRONT, baseMat);
  }
  // Side walls (left and right) using boundary bottom edges
  // Left wall
  for (let j = 0; j < bins - 1; j++) {
    const aTop = indexTop[0][j];
    const bTop = indexTop[0][j + 1];
    const aBot = bottomLeft[j];
    const bBot = bottomLeft[j + 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_LEFT, baseMat);
    emitFace(aTop, bBot, bTop, OUT_LEFT, baseMat);
  }
  // Ensure seam closed at top/bottom of left wall
  emitFace(indexTop[0][bins - 2], indexTop[0][bins - 1], bottomLeft[bins - 1], OUT_LEFT, baseMat);
  emitFace(indexTop[0][bins - 2], bottomLeft[bins - 1], bottomLeft[bins - 2], OUT_LEFT, baseMat);
  // Right wall
  for (let j = 0; j < bins - 1; j++) {
    const aTop = indexTop[slices - 1][j];
    const bTop = indexTop[slices - 1][j + 1];
    const aBot = bottomRight[j];
    const bBot = bottomRight[j + 1];
    // Consistent diagonal aTop -> bBot
    emitFace(aTop, aBot, bBot, OUT_RIGHT, baseMat);
    emitFace(aTop, bBot, bTop, OUT_RIGHT, baseMat);
  }
  emitFace(indexTop[slices - 1][bins - 2], bottomRight[bins - 2], indexTop[slices - 1][bins - 1], OUT_RIGHT, baseMat);
  emitFace(indexTop[slices - 1][bins - 1], bottomRight[bins - 2], bottomRight[bins - 1], OUT_RIGHT, baseMat);

  // No extra projection behind the rear plane; clipped at z0

  const header = [
    '# Dial-up spectrogram export',
    'mtllib 3D_spectrogram.mtl',
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

// Build a single-color binary STL from the captured unified shape.
// Uses the exact same geometry/ordering as OBJ and the same export mapping:
// render (x,y,z) -> export (X=x, Y=-z, Z=y+baseThickness)
function buildSTLFromCaptured() {
  let slices = capturedSlices.length;
  if (slices < 2) return new ArrayBuffer(0);
  const activeRowsCount = Math.max(2, Math.floor(pointsPerSlice * activeFrequencyFraction));
  const bins = activeRowsCount + 1;
  const dx = sliceSpacing;
  const x0 = 0;
  const z0 = -depth / 2;
  const baseY = -baseThickness;

  const indexTop = Array.from({ length: slices }, () => new Array(bins));
  const yTop = Array.from({ length: slices }, () => new Array(bins));
  const verts = []; // render-space vertices
  function pushV(x, y, z) { verts.push({ x, y, z }); return verts.length - 1; }

  const zAt = new Array(bins);
  for (let j = 0; j < bins; j++) {
    let z;
    if (j < activeRowsCount) {
      const t = j / (activeRowsCount - 1);
      const tExp = Math.pow(t, frequencyExponent);
      z = z0 + tExp * depth;
    } else {
      z = z0 + depth;
    }
    zAt[j] = z;
  }

  // Top vertices (with near-flat clamp like OBJ exporter)
  const flatEps = 0.02 * depth * heightScale;
  for (let i = 0; i < slices; i++) {
    indexTop[i] = new Array(bins);
    const x = x0 + dx * i;
    const slice = capturedSlices[i];
    for (let j = 0; j < bins; j++) {
      const z = zAt[j];
      const amp = (j < activeRowsCount) ? slice[j] : 0;
      let y = amp * depth * heightScale;
      if (y < flatEps) y = 0; // clamp near-flat to exactly 0 for merge
      indexTop[i][j] = pushV(x, y, z);
      yTop[i][j] = y;
    }
  }

  // Bottom boundary vertices
  const bottomBack = new Array(slices);
  const bottomFront = new Array(slices);
  const bottomLeft = new Array(bins);
  const bottomRight = new Array(bins);
  for (let i = 0; i < slices; i++) {
    const x = x0 + dx * i;
    bottomBack[i] = pushV(x, baseY, z0);
    bottomFront[i] = pushV(x, baseY, z0 + depth);
  }
  for (let j = 0; j < bins; j++) {
    bottomLeft[j] = pushV(x0, baseY, zAt[j]);
    bottomRight[j] = pushV(x0 + dx * (slices - 1), baseY, zAt[j]);
  }

  // Collect oriented triangles in export-space (Z-up, Y flipped)
  const tris = [];
  function mapToExport(p) { return { X: p.x, Y: -p.z, Z: p.y + baseThickness }; }
  function addTriOriented(i1, i2, i3, desired) {
    const A = mapToExport(verts[i1]);
    const B = mapToExport(verts[i2]);
    const C = mapToExport(verts[i3]);
    let abx = B.X - A.X, aby = B.Y - A.Y, abz = B.Z - A.Z;
    let acx = C.X - A.X, acy = C.Y - A.Y, acz = C.Z - A.Z;
    let nx = aby * acz - abz * acy;
    let ny = abz * acx - abx * acz;
    let nz = abx * acy - aby * acx;
    const dot = nx * desired.x + ny * desired.y + nz * desired.z;
    if (dot < 0) {
      // swap B and C to flip
      const tmpX = B.X, tmpY = B.Y, tmpZ = B.Z;
      B.X = C.X; B.Y = C.Y; B.Z = C.Z;
      C.X = tmpX; C.Y = tmpY; C.Z = tmpZ;
      abx = B.X - A.X; aby = B.Y - A.Y; abz = B.Z - A.Z;
      acx = C.X - A.X; acy = C.Y - A.Y; acz = C.Z - A.Z;
      nx = aby * acz - abz * acy;
      ny = abz * acx - abx * acz;
      nz = abx * acy - aby * acx;
    }
    const len = Math.hypot(nx, ny, nz) || 1;
    tris.push([{ x: nx / len, y: ny / len, z: nz / len }, A, B, C]);
  }

  // Outward directions in export space with Y flipped
  const OUT_TOP = { x: 0, y: 0, z: 1 };
  const OUT_BOTTOM = { x: 0, y: 0, z: -1 };
  const OUT_FRONT = { x: 0, y: -1, z: 0 };
  const OUT_BACK = { x: 0, y: 1, z: 0 };
  const OUT_LEFT = { x: -1, y: 0, z: 0 };
  const OUT_RIGHT = { x: 1, y: 0, z: 0 };

  // Triangulate top surface with coplanar merge on near-flat regions (mirror of OBJ)
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
        const a = indexTop[i][j];
        const bIdx = indexTop[i + 1][j];
        const cIdx = indexTop[i + 1][j + 1];
        const d = indexTop[i][j + 1];
        addTriOriented(a, bIdx, cIdx, OUT_TOP);
        addTriOriented(a, cIdx, d, OUT_TOP);
        visited[i][j] = true;
        continue;
      }
      // Flat cell: grow rectangle (i..iW, j..jH)
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
      const a = indexTop[i][j];
      const bIdx = indexTop[iW + 1][j];
      const cIdx = indexTop[iW + 1][jH + 1];
      const d = indexTop[i][jH + 1];
      addTriOriented(a, bIdx, cIdx, OUT_TOP);
      addTriOriented(a, cIdx, d, OUT_TOP);
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
    addTriOriented(a0, a1, b1, OUT_FRONT);
    addTriOriented(a0, b1, b0, OUT_FRONT);
  }
  // Bottom big quad (two triangles)
  const bl = bottomBack[0];
  const br = bottomBack[slices - 1];
  const fr = bottomFront[slices - 1];
  const fl = bottomFront[0];
  addTriOriented(bl, br, fr, OUT_BOTTOM);
  addTriOriented(bl, fr, fl, OUT_BOTTOM);

  // Walls
  for (let i = 0; i < slices - 1; i++) {
    // Back wall at z0
    addTriOriented(indexTop[i][0], bottomBack[i], bottomBack[i + 1], OUT_BACK);
    addTriOriented(indexTop[i][0], bottomBack[i + 1], indexTop[i + 1][0], OUT_BACK);
    // Front wall at z0+depth
    const jF = bins - 1;
    addTriOriented(indexTop[i][jF], bottomFront[i], bottomFront[i + 1], OUT_FRONT);
    addTriOriented(indexTop[i][jF], bottomFront[i + 1], indexTop[i + 1][jF], OUT_FRONT);
  }
  for (let j = 0; j < bins - 1; j++) {
    // Left wall at x0
    addTriOriented(indexTop[0][j], bottomLeft[j], bottomLeft[j + 1], OUT_LEFT);
    addTriOriented(indexTop[0][j], bottomLeft[j + 1], indexTop[0][j + 1], OUT_LEFT);
    // Right wall at x0 + dx*(slices-1)
    const iR = slices - 1;
    addTriOriented(indexTop[iR][j], bottomRight[j], bottomRight[j + 1], OUT_RIGHT);
    addTriOriented(indexTop[iR][j], bottomRight[j + 1], indexTop[iR][j + 1], OUT_RIGHT);
  }
  // Ensure seams closed at top/bottom of left/right walls
  addTriOriented(indexTop[0][bins - 2], indexTop[0][bins - 1], bottomLeft[bins - 1], OUT_LEFT);
  addTriOriented(indexTop[0][bins - 2], bottomLeft[bins - 1], bottomLeft[bins - 2], OUT_LEFT);
  addTriOriented(indexTop[slices - 1][bins - 2], bottomRight[bins - 2], indexTop[slices - 1][bins - 1], OUT_RIGHT);
  addTriOriented(indexTop[slices - 1][bins - 1], bottomRight[bins - 2], bottomRight[bins - 1], OUT_RIGHT);

  // Binary STL encoding
  const triCount = tris.length;
  const buffer = new ArrayBuffer(80 + 4 + triCount * 50);
  const view = new DataView(buffer);
  view.setUint32(80, triCount, true);
  let offset = 84;
  for (const [n, a, b, c] of tris) {
    view.setFloat32(offset + 0, n.x, true);
    view.setFloat32(offset + 4, n.y, true);
    view.setFloat32(offset + 8, n.z, true);
    view.setFloat32(offset + 12, a.X, true);
    view.setFloat32(offset + 16, a.Y, true);
    view.setFloat32(offset + 20, a.Z, true);
    view.setFloat32(offset + 24, b.X, true);
    view.setFloat32(offset + 28, b.Y, true);
    view.setFloat32(offset + 32, b.Z, true);
    view.setFloat32(offset + 36, c.X, true);
    view.setFloat32(offset + 40, c.Y, true);
    view.setFloat32(offset + 44, c.Z, true);
    view.setUint16(offset + 48, 0, true);
    offset += 50;
  }
  return buffer;
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
  const baseColor = { r: 0.0, g: 0.0, b: 0.0 }; // invisible black so grid shows through when hidden
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
  // Keep legacy surface hidden; per-slice modules are the only visible geometry during growth
  surface.visible = false;
  // Ensure grid renders below surface by resetting any modified order
  gridGroup.traverse((obj) => { obj.renderOrder = 0; });
  // Remove any watertight extrusions created previously
  clearExtrusions();
  clearUnifiedMesh();
  clearSliceModules();
  // Clear smoothing history so a new session starts clean
  sliceHistory.length = 0;
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
  buildReferenceGrid(undefined, 0);
  // Reset world-space offset when fully resetting the scene
  worldLeftOffsetX = 0;
  // Make sure all extrusions are deleted on a hard reset
  clearExtrusions();
  // No drawRange updates; legacy surface is not used during growth
  // Clear per-slice modules
  clearSliceModules();
  lastSliceAmps = new Float32Array(pointsPerSlice);
  sliceHistory.length = 0;
}

// Cleanup on hot reload
window.addEventListener('beforeunload', () => {
  isRendering = false;
  try { audioEl.pause(); } catch { }
  try { stopRecordingIfActive(); } catch { }
});


