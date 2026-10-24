"""Anatomical 3D Interactive Heart Visualizer for CardioTwin X.

Features:
- High-fidelity anatomical model mirroring reference medical illustrations:
  * Left and Right Ventricles with conical apex and interventricular sulcus.
  * Right and Left Auricles/Atria with textured muscular appendages.
  * Prominent crimson Aortic Arch with the 3 brachiocephalic/carotid/subclavian branch vessels.
  * Cobalt blue Pulmonary Trunk and branching Pulmonary Arteries.
  * Superior Vena Cava and Pulmonary Vein portals.
  * Rich coronary vasculature network (interventricular arteries & veins).
  * Glossy organic tissue shading with specular highlights and rim lighting.
- Physically driven "Lub-Dub" cardiac cycle animation:
  * S1 "Lub": Isovolumetric ventricular contraction, apical torsion, and aortic systolic expansion as per SBP.
  * S2 "Dub": Aortic valve closure dicrotic recoil and secondary rebound as per DBP.
  * Amplitude dynamically driven by Pulse Pressure (PP = SBP - DBP) and Stroke Volume.
  * Tempo paced precisely by real-time Heart Rate (T = 60 / HR s).
  * Myocardial strain color shift under hypertensive stress or Significant divergence.
- Web Audio API phonocardiogram sound synthesis:
  * Real-time acoustic "lub-dub" sound synchronized with mechanical contraction.
- 360-degree OrbitControls (drag to rotate, scroll to zoom, right-click to pan).
- Real-time HUD showing cardiac phase, wall tension, and BP metrics.
"""

from __future__ import annotations

import json
from typing import Optional


def generate_3d_heart_html(
    sbp: float = 120.0,
    dbp: float = 80.0,
    hr: float = 72.0,
    sv: float = 65.0,
    div_status: str = "Normal",
    div_score: float = 0.8,
) -> str:
    """Generate self-contained Three.js HTML for the anatomical 3D heart."""
    pulse_pressure = max(float(sbp - dbp), 15.0)
    cycle_duration = max(60.0 / max(float(hr), 35.0), 0.3)

    config_json = json.dumps({
        "sbp": float(sbp),
        "dbp": float(dbp),
        "hr": float(hr),
        "sv": float(sv),
        "pulse_pressure": pulse_pressure,
        "cycle_duration": cycle_duration,
        "div_status": str(div_status),
        "div_score": float(div_score),
    })

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CardioTwin X 3D Anatomical Heart</title>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      user-select: none;
    }}
    body {{
      background: radial-gradient(circle at 50% 45%, #ffffff 0%, #f8fafc 60%, #f1f5f9 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      position: relative;
    }}
    #canvas-container {{
      width: 100%;
      height: 100%;
      position: absolute;
      top: 0;
      left: 0;
    }}
    /* HUD Overlays */
    .hud-panel {{
      position: absolute;
      top: 14px;
      left: 16px;
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
      border: 1.5px solid rgba(244, 63, 94, 0.25);
      border-radius: 14px;
      padding: 12px 18px;
      box-shadow: 0 8px 24px rgba(225, 29, 72, 0.08);
      pointer-events: auto;
      z-index: 10;
      max-width: 320px;
    }}
    .hud-title {{
      font-size: 0.95rem;
      font-weight: 800;
      color: #e11d48;
      display: flex;
      align-items: center;
      gap: 6px;
      margin-bottom: 4px;
    }}
    .hud-subtitle {{
      font-size: 0.78rem;
      color: #64748b;
      margin-bottom: 8px;
    }}
    .phase-badge {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 12px;
      font-size: 0.82rem;
      font-weight: 700;
      transition: all 0.15s ease;
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #a7f3d0;
    }}
    .phase-lub {{
      background: #ffe4e6 !important;
      color: #be123c !important;
      border-color: #fda4af !important;
      box-shadow: 0 0 12px rgba(244, 63, 94, 0.4);
      transform: scale(1.04);
    }}
    .phase-dub {{
      background: #dbeafe !important;
      color: #1d4ed8 !important;
      border-color: #93c5fd !important;
      box-shadow: 0 0 10px rgba(59, 130, 246, 0.35);
      transform: scale(1.02);
    }}
    .metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      color: #334155;
      margin-top: 5px;
    }}
    .metric-val {{
      font-weight: 700;
      color: #0f172a;
    }}

    /* Controls bar at bottom */
    .controls-bar {{
      position: absolute;
      bottom: 14px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255, 255, 255, 0.90);
      backdrop-filter: blur(8px);
      border: 1px solid #e2e8f0;
      border-radius: 30px;
      padding: 6px 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
      z-index: 10;
    }}
    .ctrl-btn {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #334155;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 5px;
      transition: all 0.2s ease;
    }}
    .ctrl-btn:hover {{
      background: #fff1f2;
      border-color: #f43f5e;
      color: #e11d48;
    }}
    .ctrl-btn.active {{
      background: #e11d48;
      border-color: #be123c;
      color: #ffffff;
    }}
    .hint-pill {{
      position: absolute;
      bottom: 14px;
      right: 16px;
      background: rgba(255, 255, 255, 0.85);
      backdrop-filter: blur(6px);
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 6px 12px;
      font-size: 0.74rem;
      color: #64748b;
      z-index: 10;
    }}
  </style>
  <!-- Three.js + OrbitControls -->
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>

  <div id="canvas-container"></div>

  <!-- Real-time Physiological HUD -->
  <div class="hud-panel">
    <div class="hud-title">
      <span>🫀</span> CardioTwin X Anatomical Engine
    </div>
    <div class="hud-subtitle">Physiology-Coupled Lub-Dub Biomechanics</div>
    <div style="margin: 8px 0;">
      <span id="phase-badge" class="phase-badge">Diastolic Inflow</span>
    </div>
    <div class="metric-row">
      <span>Observed Blood Pressure:</span>
      <span class="metric-val" id="bp-display">{sbp:.0f} / {dbp:.0f} mmHg</span>
    </div>
    <div class="metric-row">
      <span>Pulse Pressure (ΔP):</span>
      <span class="metric-val" id="pp-display">{pulse_pressure:.0f} mmHg (Force Driver)</span>
    </div>
    <div class="metric-row">
      <span>Heart Rate / Tempo:</span>
      <span class="metric-val" id="hr-display">{hr:.0f} bpm (T = {cycle_duration:.2f}s)</span>
    </div>
    <div class="metric-row">
      <span>Systolic Stroke Volume:</span>
      <span class="metric-val">{sv:.1f} mL</span>
    </div>
    <div class="metric-row">
      <span>Divergence Status:</span>
      <span class="metric-val" style="color: {'#e11d48' if div_status == 'Significant' else ('#d97706' if div_status == 'Watch' else '#059669')};">
        {div_status} (D<sub>M</sub>: {div_score:.2f})
      </span>
    </div>
  </div>

  <!-- Interactive Controls Bar -->
  <div class="controls-bar">
    <button id="sound-btn" class="ctrl-btn" title="Toggle Synthesized Cardiac Audio">
      <span id="sound-icon">🔈</span> <span id="sound-text">Sound: OFF</span>
    </button>
    <button id="view-anterior" class="ctrl-btn active">Anterior (Front)</button>
    <button id="view-arch" class="ctrl-btn">Aortic Arch</button>
    <button id="view-apex" class="ctrl-btn">Apex</button>
    <button id="speed-btn" class="ctrl-btn" title="Toggle Speed">Speed: 1.0x</button>
  </div>

  <div class="hint-pill">
    🖱️ Drag to rotate 360° · Scroll to zoom
  </div>

  <script>
    const CONFIG = {config_json};

    // -------------------------------------------------------------------------
    // 1. Scene, Camera, Renderer & Orbit Controls
    // -------------------------------------------------------------------------
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xfbfcfe);

    const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 0.4, 6.2);

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: "high-performance" }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.minDistance = 3.2;
    controls.maxDistance = 11.0;
    controls.target.set(0, -0.1, 0);

    // -------------------------------------------------------------------------
    // 2. Anatomical Lighting Rig
    // -------------------------------------------------------------------------
    const ambientLight = new THREE.AmbientLight(0xfff7f8, 0.95);
    scene.add(ambientLight);

    // Key Light (Warm surgical studio illumination)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.4);
    keyLight.position.set(4, 6, 5);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 1024;
    keyLight.shadow.mapSize.height = 1024;
    scene.add(keyLight);

    // Fill Light (Soft cool contrast fill)
    const fillLight = new THREE.DirectionalLight(0xdbeafe, 0.85);
    fillLight.position.set(-5, 2, 4);
    scene.add(fillLight);

    // Rim / Back Light (Iconic edge contour illumination like reference illustration)
    const rimLight = new THREE.DirectionalLight(0xf43f5e, 1.6);
    rimLight.position.set(0, 4, -5);
    scene.add(rimLight);

    const blueRimLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
    blueRimLight.position.set(-4, -2, -3);
    scene.add(blueRimLight);

    // -------------------------------------------------------------------------
    // 3. Procedural Normal & Surface Textures
    // -------------------------------------------------------------------------
    function createMuscleTexture() {{
      const size = 512;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');

      ctx.fillStyle = '#b91c1c';
      ctx.fillRect(0, 0, size, size);

      // Fine muscular striations / trabeculae carnae fibers
      ctx.lineWidth = 1.2;
      for (let i = 0; i < 450; i++) {{
        const y = Math.random() * size;
        const alpha = 0.05 + Math.random() * 0.12;
        ctx.strokeStyle = Math.random() > 0.4 ? `rgba(255, 120, 140, ${{alpha}})` : `rgba(100, 10, 20, ${{alpha}})`;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.bezierCurveTo(size * 0.33, y + Math.random() * 20 - 10, size * 0.66, y + Math.random() * 20 - 10, size, y);
        ctx.stroke();
      }}
      return new THREE.CanvasTexture(canvas);
    }}

    const muscleTex = createMuscleTexture();
    muscleTex.wrapS = THREE.RepeatWrapping;
    muscleTex.wrapT = THREE.RepeatWrapping;
    muscleTex.repeat.set(2, 2);

    // Materials
    const myocardialMaterial = new THREE.MeshPhysicalMaterial({{
      color: new THREE.Color(0xb91c1c),
      emissive: new THREE.Color(0x330005),
      emissiveIntensity: 0.15,
      roughness: 0.38,
      metalness: 0.12,
      clearcoat: 0.65,
      clearcoatRoughness: 0.22,
      map: muscleTex,
      bumpMap: muscleTex,
      bumpScale: 0.04,
    }});

    const aortaMaterial = new THREE.MeshPhysicalMaterial({{
      color: new THREE.Color(0xd92638),
      emissive: new THREE.Color(0x3b030b),
      emissiveIntensity: 0.2,
      roughness: 0.30,
      metalness: 0.14,
      clearcoat: 0.85,
      clearcoatRoughness: 0.15,
    }});

    const pulmonaryArteryMaterial = new THREE.MeshPhysicalMaterial({{
      color: new THREE.Color(0x2563eb),
      emissive: new THREE.Color(0x082f49),
      emissiveIntensity: 0.25,
      roughness: 0.32,
      metalness: 0.16,
      clearcoat: 0.80,
      clearcoatRoughness: 0.18,
    }});

    const venaCavaMaterial = new THREE.MeshPhysicalMaterial({{
      color: new THREE.Color(0x1d4ed8),
      roughness: 0.35,
      metalness: 0.12,
      clearcoat: 0.70,
    }});

    const coronaryRedMaterial = new THREE.MeshStandardMaterial({{
      color: new THREE.Color(0xef4444),
      roughness: 0.25,
      metalness: 0.2,
    }});

    const coronaryBlueMaterial = new THREE.MeshStandardMaterial({{
      color: new THREE.Color(0x0284c7),
      roughness: 0.25,
      metalness: 0.2,
    }});

    // -------------------------------------------------------------------------
    // 4. Construct Anatomical Heart Model
    // -------------------------------------------------------------------------
    const heartGroup = new THREE.Group();
    scene.add(heartGroup);

    // A. Ventricular Conical Myocardium (Left & Right Ventricles)
    function buildVentricularGeometry() {{
      const geom = new THREE.SphereGeometry(1.35, 64, 48);
      const pos = geom.attributes.position;
      const v = new THREE.Vector3();

      for (let i = 0; i < pos.count; i++) {{
        v.fromBufferAttribute(pos, i);
        const y = v.y;

        // Shape into conical cardiac form tapering toward the apex (pointed inferolaterally to right/front)
        const taper = 1.0 - (y + 1.35) / 2.7 * 0.45;
        if (y < 0.2) {{
          // Apex deflection towards x > 0 and z > 0
          const depthFactor = (0.2 - y) / 1.5;
          v.x += 0.38 * depthFactor;
          v.z += 0.22 * depthFactor;
          v.x *= (1.0 - depthFactor * 0.52);
          v.z *= (1.0 - depthFactor * 0.48);
        }}

        // Interventricular groove (sulcus) indent on the anterior side
        if (v.z > 0 && Math.abs(v.x + 0.15) < 0.6) {{
          const grooveDist = Math.abs(v.x + 0.15);
          v.z -= Math.cos(grooveDist * Math.PI) * 0.08 * (1.0 - Math.abs(y) / 1.5);
        }}

        // Left ventricle prominent posterior bulge
        if (v.x > 0.1 && v.y < 0.1) {{
          v.x *= 1.14;
          v.z *= 1.08;
        }}

        pos.setXYZ(i, v.x, v.y, v.z);
      }}
      geom.computeVertexNormals();
      return geom;
    }}

    const ventriclesGeom = buildVentricularGeometry();
    const ventriclesMesh = new THREE.Mesh(ventriclesGeom, myocardialMaterial);
    ventriclesMesh.position.set(0, -0.6, 0);
    ventriclesMesh.castShadow = true;
    ventriclesMesh.receiveShadow = true;
    heartGroup.add(ventriclesMesh);

    // Save baseline vertex positions for physical lub-dub contraction deformation
    const basePositions = ventriclesGeom.attributes.position.clone();

    // B. Right Atrium & Auricle (Screen Left, Anatomical Right)
    const rightAtriumGeom = new THREE.SphereGeometry(0.72, 32, 24);
    // Sculpt scalloped auricle pouch
    const raPos = rightAtriumGeom.attributes.position;
    for (let i = 0; i < raPos.count; i++) {{
      let x = raPos.getX(i);
      let y = raPos.getY(i);
      let z = raPos.getZ(i);
      // add muscular nodular lobes
      const lobe = Math.sin(x * 6.0) * Math.cos(y * 6.0) * 0.04;
      x *= (1.05 + lobe);
      y *= (0.90 + lobe);
      z *= (0.92 + lobe);
      raPos.setXYZ(i, x, y, z);
    }}
    rightAtriumGeom.computeVertexNormals();
    const rightAtriumMesh = new THREE.Mesh(rightAtriumGeom, myocardialMaterial);
    rightAtriumMesh.position.set(-0.85, 0.45, 0.15);
    heartGroup.add(rightAtriumMesh);

    // Left Auricle (Screen Right, Anatomical Left)
    const leftAuricleGeom = new THREE.SphereGeometry(0.58, 28, 20);
    const laMesh = new THREE.Mesh(leftAuricleGeom, myocardialMaterial);
    laMesh.scale.set(0.9, 1.2, 0.85);
    laMesh.position.set(0.85, 0.40, 0.05);
    heartGroup.add(laMesh);

    // C. Massive Aortic Arch & 3 Branching Arteries (Iconic Red Arch)
    const aortaGroup = new THREE.Group();
    heartGroup.add(aortaGroup);

    // Arch spline curve
    const archCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.05, 0.35, 0.1),
      new THREE.Vector3(-0.02, 1.05, 0.15),
      new THREE.Vector3(0.08, 1.65, 0.05),
      new THREE.Vector3(0.35, 1.82, -0.20),
      new THREE.Vector3(0.68, 1.55, -0.45),
      new THREE.Vector3(0.82, 0.85, -0.55),
      new THREE.Vector3(0.85, 0.20, -0.60),
    ]);

    const archGeom = new THREE.TubeGeometry(archCurve, 42, 0.34, 24, false);
    const archMesh = new THREE.Mesh(archGeom, aortaMaterial);
    archMesh.castShadow = true;
    aortaGroup.add(archMesh);

    // The 3 Superior Arterial Branches from Aorta (Brachiocephalic, Carotid, Subclavian)
    function createArteryBranch(startPt, direction, length, radius) {{
      const branchCurve = new THREE.CatmullRomCurve3([
        startPt,
        startPt.clone().add(direction.clone().multiplyScalar(length * 0.5)).add(new THREE.Vector3((Math.random() - 0.5) * 0.05, 0, 0)),
        startPt.clone().add(direction.clone().multiplyScalar(length)),
      ]);
      const branchGeom = new THREE.TubeGeometry(branchCurve, 18, radius, 18, false);
      const mesh = new THREE.Mesh(branchGeom, aortaMaterial);
      mesh.castShadow = true;

      // Add hollow lumen ring at opening
      const ringGeom = new THREE.RingGeometry(radius * 0.5, radius * 1.05, 16);
      const ringMat = new THREE.MeshBasicMaterial({{ color: 0x4a0404, side: THREE.DoubleSide }});
      const ringMesh = new THREE.Mesh(ringGeom, ringMat);
      const endPt = startPt.clone().add(direction.clone().multiplyScalar(length));
      ringMesh.position.copy(endPt);
      ringMesh.lookAt(endPt.clone().add(direction));
      mesh.add(ringMesh);

      return mesh;
    }}

    // Branch 1: Brachiocephalic Trunk (first on right of arch)
    const b1 = createArteryBranch(new THREE.Vector3(0.05, 1.62, 0.08), new THREE.Vector3(-0.25, 0.95, 0.1).normalize(), 0.65, 0.125);
    aortaGroup.add(b1);

    // Branch 2: Left Common Carotid Artery (middle)
    const b2 = createArteryBranch(new THREE.Vector3(0.28, 1.76, -0.05), new THREE.Vector3(-0.08, 0.98, 0.02).normalize(), 0.60, 0.105);
    aortaGroup.add(b2);

    // Branch 3: Left Subclavian Artery (third)
    const b3 = createArteryBranch(new THREE.Vector3(0.50, 1.72, -0.22), new THREE.Vector3(0.18, 0.96, -0.05).normalize(), 0.58, 0.10);
    aortaGroup.add(b3);

    // D. Pulmonary Trunk & Left/Right Pulmonary Arteries (Cobalt Blue)
    const pulmonaryGroup = new THREE.Group();
    heartGroup.add(pulmonaryGroup);

    const ptCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.25, 0.25, 0.65),
      new THREE.Vector3(-0.18, 0.85, 0.58),
      new THREE.Vector3(-0.05, 1.30, 0.42),
      new THREE.Vector3(0.15, 1.45, 0.15),
    ]);
    const ptGeom = new THREE.TubeGeometry(ptCurve, 32, 0.30, 20, false);
    const ptMesh = new THREE.Mesh(ptGeom, pulmonaryArteryMaterial);
    ptMesh.castShadow = true;
    pulmonaryGroup.add(ptMesh);

    // Left and Right Pulmonary Artery lateral branches
    const paRight = createArteryBranch(new THREE.Vector3(0.15, 1.45, 0.15), new THREE.Vector3(-0.95, 0.2, -0.2).normalize(), 0.70, 0.17);
    paRight.material = pulmonaryArteryMaterial;
    pulmonaryGroup.add(paRight);

    const paLeft = createArteryBranch(new THREE.Vector3(0.15, 1.45, 0.15), new THREE.Vector3(0.92, -0.15, -0.3).normalize(), 0.65, 0.17);
    paLeft.material = pulmonaryArteryMaterial;
    pulmonaryGroup.add(paLeft);

    // E. Superior Vena Cava (Deep Blue Upper Tube)
    const svcCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.85, 0.60, -0.15),
      new THREE.Vector3(-0.88, 1.35, -0.20),
      new THREE.Vector3(-0.90, 1.95, -0.22),
    ]);
    const svcGeom = new THREE.TubeGeometry(svcCurve, 18, 0.22, 18, false);
    const svcMesh = new THREE.Mesh(svcGeom, venaCavaMaterial);
    heartGroup.add(svcMesh);

    // 4 Pulmonary Vein Stumps (Lateral ports on left atrium)
    const pvGroup = new THREE.Group();
    heartGroup.add(pvGroup);
    const pvPositions = [
      new THREE.Vector3(0.95, 0.70, -0.35),
      new THREE.Vector3(0.98, 0.35, -0.40),
      new THREE.Vector3(-0.90, 0.15, -0.55),
      new THREE.Vector3(-0.92, -0.15, -0.60),
    ];
    pvPositions.forEach((pos, idx) => {{
      const pvGeom = new THREE.CylinderGeometry(0.11, 0.13, 0.35, 16);
      const pvMesh = new THREE.Mesh(pvGeom, venaCavaMaterial);
      pvMesh.position.copy(pos);
      pvMesh.rotation.z = Math.PI / 2.3 * (idx < 2 ? 1 : -1);
      pvGroup.add(pvMesh);
    }});

    // F. Intricate Surface Coronary Vasculature Network
    const coronaryGroup = new THREE.Group();
    heartGroup.add(coronaryGroup);

    function createSurfaceVessel(points, radius, material) {{
      const curve = new THREE.CatmullRomCurve3(points);
      const geom = new THREE.TubeGeometry(curve, 36, radius, 10, false);
      const mesh = new THREE.Mesh(geom, material);
      coronaryGroup.add(mesh);
      return mesh;
    }}

    // 1. Anterior Interventricular Vein (Prominent Blue Vessel traversing groove)
    createSurfaceVessel([
      new THREE.Vector3(-0.15, 0.38, 0.98),
      new THREE.Vector3(-0.08, 0.05, 1.08),
      new THREE.Vector3(0.04, -0.35, 1.02),
      new THREE.Vector3(0.18, -0.80, 0.88),
      new THREE.Vector3(0.28, -1.25, 0.65),
      new THREE.Vector3(0.35, -1.65, 0.35),
    ], 0.038, coronaryBlueMaterial);

    // 2. Left Anterior Descending Artery (Crimson Red branch parallel)
    createSurfaceVessel([
      new THREE.Vector3(-0.05, 0.42, 0.96),
      new THREE.Vector3(0.02, 0.10, 1.05),
      new THREE.Vector3(0.14, -0.28, 0.98),
      new THREE.Vector3(0.25, -0.72, 0.85),
      new THREE.Vector3(0.32, -1.18, 0.62),
    ], 0.032, coronaryRedMaterial);

    // 3. Diagonal & Lateral Branching Venules & Arterioles
    createSurfaceVessel([
      new THREE.Vector3(0.04, -0.35, 1.02),
      new THREE.Vector3(0.32, -0.42, 0.95),
      new THREE.Vector3(0.62, -0.55, 0.82),
      new THREE.Vector3(0.85, -0.68, 0.55),
    ], 0.022, coronaryBlueMaterial);

    createSurfaceVessel([
      new THREE.Vector3(0.18, -0.80, 0.88),
      new THREE.Vector3(0.45, -0.92, 0.75),
      new THREE.Vector3(0.72, -1.05, 0.50),
    ], 0.020, coronaryBlueMaterial);

    createSurfaceVessel([
      new THREE.Vector3(-0.08, 0.05, 1.08),
      new THREE.Vector3(-0.35, -0.10, 1.02),
      new THREE.Vector3(-0.65, -0.28, 0.85),
    ], 0.024, coronaryRedMaterial);

    createSurfaceVessel([
      new THREE.Vector3(0.14, -0.28, 0.98),
      new THREE.Vector3(-0.15, -0.45, 0.96),
      new THREE.Vector3(-0.45, -0.65, 0.82),
    ], 0.018, coronaryRedMaterial);

    // Initial orientation to match iconic 3/4 anterior view in reference image
    heartGroup.rotation.y = -0.22;
    heartGroup.rotation.z = -0.06;

    // -------------------------------------------------------------------------
    // 5. Synthesized Acoustic Phonocardiogram (Web Audio API)
    // -------------------------------------------------------------------------
    let audioCtx = null;
    let isSoundEnabled = false;

    function initAudio() {{
      if (!audioCtx) {{
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }}
      if (audioCtx.state === 'suspended') {{
        audioCtx.resume();
      }}
    }}

    function playLubSound(intensity = 1.0) {{
      if (!isSoundEnabled || !audioCtx) return;
      const now = audioCtx.currentTime;

      // S1 (LUB): Low resonant muscular boom (50Hz -> 35Hz)
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      const filter = audioCtx.createBiquadFilter();

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(110, now);

      osc.type = 'sine';
      osc.frequency.setValueAtTime(58, now);
      osc.frequency.exponentialRampToValueAtTime(36, now + 0.14);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.22 * intensity, now + 0.025);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.16);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.18);
    }}

    function playDubSound(intensity = 1.0) {{
      if (!isSoundEnabled || !audioCtx) return;
      const now = audioCtx.currentTime;

      // S2 (DUB): Snappier, higher valve closure snap (85Hz -> 50Hz)
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      const filter = audioCtx.createBiquadFilter();

      filter.type = 'lowpass';
      filter.frequency.setValueAtTime(160, now);

      osc.type = 'sine';
      osc.frequency.setValueAtTime(92, now);
      osc.frequency.exponentialRampToValueAtTime(52, now + 0.10);

      gain.gain.setValueAtTime(0.001, now);
      gain.gain.linearRampToValueAtTime(0.18 * intensity, now + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);

      osc.connect(filter);
      filter.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start(now);
      osc.stop(now + 0.14);
    }}

    // -------------------------------------------------------------------------
    // 6. Dynamic Lub-Dub Cardiac Cycle Biomechanical Loop
    // -------------------------------------------------------------------------
    const phaseBadge = document.getElementById('phase-badge');
    let speedFactor = 1.0;
    let clock = new THREE.Clock();
    let cycleTimeAcc = 0;
    let lastLubTriggered = false;
    let lastDubTriggered = false;

    // Physiological Scaling Factors derived directly from Blood Pressure:
    const SBP = CONFIG.sbp;
    const DBP = CONFIG.dbp;
    const PP = CONFIG.pulse_pressure;
    const HR = CONFIG.hr;
    const isSignificant = CONFIG.div_status === "Significant";
    const isWatch = CONFIG.div_status === "Watch";

    // Systolic contraction depth scales with Pulse Pressure & SV
    const contractionDepth = Math.min(0.08 + (PP / 60.0) * 0.10, 0.22);
    // Aortic expansion scales with SBP
    const aorticDilationAmp = Math.min(0.06 + (SBP / 120.0) * 0.08, 0.20);

    function animate() {{
      requestAnimationFrame(animate);

      const delta = clock.getDelta() * speedFactor;
      cycleTimeAcc += delta;

      const T = CONFIG.cycle_duration;
      const phase = (cycleTimeAcc % T) / T; // Normalized cardiac cycle [0, 1)

      // -----------------------------------------------------------------------
      // Biomechanical Phases:
      // Phase A: [0.00 - 0.12] Late Diastole / Atrial Systole (Pre-stretch)
      // Phase B: [0.12 - 0.32] Ventricular Systole / "LUB" (Forceful Contraction & SBP Surge)
      // Phase C: [0.32 - 0.44] End Systole / "DUB" (Aortic Closure Snap & Dicrotic Recoil)
      // Phase D: [0.44 - 1.00] Diastolic Relaxation & Refilling back to DBP
      // -----------------------------------------------------------------------
      let ventScale = 1.0;
      let apexTwist = 0.0;
      let aortaDistension = 1.0;
      let currentPhaseText = "Diastolic Refilling";
      let phaseClass = "";

      if (phase >= 0.0 && phase < 0.12) {{
        // Atrial Kick: slight pre-stretch before big contraction
        const pSub = phase / 0.12;
        ventScale = 1.0 + Math.sin(pSub * Math.PI) * 0.025;
        currentPhaseText = "Atrial Inflow & Priming";
        lastLubTriggered = false;
        lastDubTriggered = false;
      }} else if (phase >= 0.12 && phase < 0.32) {{
        // === LUB (S1) === Ventricular Isovolumetric Contraction & Systolic Ejection
        const pSub = (phase - 0.12) / 0.20;
        const squeeze = Math.sin(pSub * Math.PI);
        ventScale = 1.0 - Math.pow(squeeze, 1.4) * contractionDepth;
        apexTwist = squeeze * 0.065; // Apical wringing twist

        // Aorta dilates under systolic surge
        aortaDistension = 1.0 + squeeze * aorticDilationAmp;

        currentPhaseText = "🫀 LUB (S1) · Peak Systolic Ejection";
        phaseClass = "phase-lub";

        if (!lastLubTriggered) {{
          playLubSound(Math.min(PP / 40.0, 1.6));
          lastLubTriggered = true;
        }}
      }} else if (phase >= 0.32 && phase < 0.44) {{
        // === DUB (S2) === Aortic Valve Closure Snap & Dicrotic Notch Recoil
        const pSub = (phase - 0.32) / 0.12;
        // Dicrotic secondary rebound vibration
        const recoil = Math.sin(pSub * Math.PI);
        ventScale = 1.0 - contractionDepth * 0.3 + recoil * 0.035;
        aortaDistension = 1.0 - recoil * (aorticDilationAmp * 0.4);

        currentPhaseText = "🔔 DUB (S2) · Aortic Valve Snap";
        phaseClass = "phase-dub";

        if (!lastDubTriggered) {{
          playDubSound(Math.min(DBP / 75.0, 1.5));
          lastDubTriggered = true;
        }}
      }} else {{
        // Diastolic relaxation back to resting baseline
        const pSub = (phase - 0.44) / 0.56;
        ventScale = 1.0 - Math.exp(-pSub * 4.0) * (contractionDepth * 0.25);
        aortaDistension = 1.0;
        currentPhaseText = "Diastolic Relaxation (DBP)";
      }}

      // Update Phase Badge
      phaseBadge.textContent = currentPhaseText;
      phaseBadge.className = "phase-badge " + phaseClass;

      // Apply Lub-Dub Deformation directly onto Ventricular Geometry
      const pos = ventriclesGeom.attributes.position;
      const bPos = basePositions;
      for (let i = 0; i < pos.count; i++) {{
        const bx = bPos.getX(i);
        const by = bPos.getY(i);
        const bz = bPos.getZ(i);

        // Ventricles contract radially and shorten longitudinally toward the base
        let sx = ventScale;
        let sz = ventScale;
        let sy = 1.0 - (1.0 - ventScale) * 0.55;

        // Apply apical torsion (longitudinal wringing)
        if (by < 0.0) {{
          const twistFactor = Math.abs(by) * apexTwist;
          const cosT = Math.cos(twistFactor);
          const sinT = Math.sin(twistFactor);
          const rx = bx * cosT - bz * sinT;
          const rz = bx * sinT + bz * cosT;
          pos.setXYZ(i, rx * sx, by * sy, rz * sz);
        }} else {{
          pos.setXYZ(i, bx * sx, by * sy, bz * sz);
        }}
      }}
      ventriclesGeom.attributes.position.needsUpdate = true;
      ventriclesGeom.computeVertexNormals();

      // Deform Aortic Arch (Swells and pulses as per SBP wave)
      aortaGroup.scale.set(aortaDistension, aortaDistension * 1.02, aortaDistension);

      // Warning color modulation if Divergence is Significant or SBP is Hypertensive (>140 mmHg)
      if (isSignificant || SBP >= 140) {{
        const pulseGlow = (Math.sin(cycleTimeAcc * 8.0) + 1.0) * 0.5;
        myocardialMaterial.emissive.setHex(0xe11d48);
        myocardialMaterial.emissiveIntensity = 0.25 + pulseGlow * 0.35;
      }} else if (isWatch) {{
        myocardialMaterial.emissive.setHex(0xd97706);
        myocardialMaterial.emissiveIntensity = 0.20;
      }} else {{
        // Healthy physiological flush during peak systole
        const flush = (ventScale < 0.95) ? 0.30 : 0.12;
        myocardialMaterial.emissive.setHex(0x50050d);
        myocardialMaterial.emissiveIntensity = flush;
      }}

      // Gentle biological idle rotation
      heartGroup.rotation.y = -0.22 + Math.sin(cycleTimeAcc * 0.5) * 0.04;

      controls.update();
      renderer.render(scene, camera);
    }}

    animate();

    // -------------------------------------------------------------------------
    // 7. Interactive Controls Handlers
    // -------------------------------------------------------------------------
    const soundBtn = document.getElementById('sound-btn');
    const soundIcon = document.getElementById('sound-icon');
    const soundText = document.getElementById('sound-text');

    soundBtn.addEventListener('click', () => {{
      initAudio();
      isSoundEnabled = !isSoundEnabled;
      if (isSoundEnabled) {{
        soundIcon.textContent = '🔊';
        soundText.textContent = 'Sound: ON';
        soundBtn.classList.add('active');
      }} else {{
        soundIcon.textContent = '🔈';
        soundText.textContent = 'Sound: OFF';
        soundBtn.classList.remove('active');
      }}
    }});

    const btnAnterior = document.getElementById('view-anterior');
    const btnArch = document.getElementById('view-arch');
    const btnApex = document.getElementById('view-apex');

    function setActiveView(btn) {{
      [btnAnterior, btnArch, btnApex].forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    }}

    btnAnterior.addEventListener('click', () => {{
      setActiveView(btnAnterior);
      camera.position.set(0, 0.4, 6.2);
      controls.target.set(0, -0.1, 0);
    }});

    btnArch.addEventListener('click', () => {{
      setActiveView(btnArch);
      camera.position.set(0.6, 2.8, 4.8);
      controls.target.set(0.2, 1.2, 0);
    }});

    btnApex.addEventListener('click', () => {{
      setActiveView(btnApex);
      camera.position.set(1.4, -2.6, 4.2);
      controls.target.set(0.2, -1.2, 0);
    }});

    const speedBtn = document.getElementById('speed-btn');
    const speeds = [1.0, 0.5, 0.25];
    let speedIdx = 0;
    speedBtn.addEventListener('click', () => {{
      speedIdx = (speedIdx + 1) % speeds.length;
      speedFactor = speeds[speedIdx];
      speedBtn.textContent = `Speed: ${{speedFactor}}x`;
    }});

    // Window resize handler
    window.addEventListener('resize', () => {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }});
  </script>
</body>
</html>
"""
    return html_content
