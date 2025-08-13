Try it out here: https://soundto3d.xyz/

<img width="1290" height="872" alt="Screenshot 2025-08-13 at 12 12 42 PM" src="https://github.com/user-attachments/assets/77ab0f76-6934-4cea-94b0-4f2a6121808d" />

Hit 'Play' to start with the default dial-up sound.

Meshed OBJ with MTL and color data are in a folder and ready to go!

<img width="1039" height="833" alt="Mesh" src="https://github.com/user-attachments/assets/6f4ca077-5cc0-4e26-ba9c-e311351dabfd" />

Inspired to take a crack at this idea from [Joel Telling's Tweet](https://x.com/joeltelling/status/1954712596067885058).

## Usage

1. Load audio
   - Click “Load Audio” and choose a file, or drag & drop a file anywhere on the page.
   - Alternatively, click “Record Audio” to capture from your microphone. Click again to stop.
2. Generate the 3D surface
   - Click “Play” to start. The spectrogram surface grows from left to right while audio plays.
   - The vertical axis is amplitude; the depth axis is frequency (Z‑up in exported files).
3. Inspect
   - Orbit with right mouse, pan with middle mouse, scroll to zoom.
   - “Reset” returns the camera and view to defaults. “Hide Axes” toggles axes/grid.
4. Export
   - “Export Color OBJ” downloads an OBJ+MTL with per‑face colors.
   - “Export STL” downloads a single‑color STL.
   - Exports include a flat base and closed walls for 3D printing.

## Attribution

This project uses the following libraries and services:

- [Three.js](https://threejs.org/) for 3D rendering (via `three` and `three/examples` modules).
- [Tailwind CSS](https://tailwindcss.com/) for UI styling.
- Web Audio API (built into modern browsers) for audio analysis.
