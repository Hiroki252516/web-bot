// frontend/room.js（修正版）
// 目的：
// - Three.js で「3Dの部屋」を描画（WebGL）
// - 従来UI(#ui3d)を CSS3DRenderer で「奥の壁一面」に貼る
// - 手前に VRM(ずんだもん) を配置（無ければダミー）
// - busイベント（assistant/user）に連動して身振り手振り
// - Webカメラ + 顔検出（任意）でモーションパララックス（擬似3D）

import * as THREE from "three";
import { CSS3DRenderer, CSS3DObject } from "three/examples/jsm/renderers/CSS3DRenderer.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// three-vrm（three@0.160付近で安定しやすい v2 系）
import { VRMLoaderPlugin, VRMUtils } from "https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.1.0/lib/three-vrm.module.js";

import { on } from "./bus.js";

const AVATAR_URL = "./assets/zundamon.vrm"; // ここに配置したVRMを読みます

// --- ちょっとしたデバッグ表示（致命的エラー時のみ） ---
const debug = (() => {
  const el = document.createElement("pre");
  el.style.cssText = [
    "position:fixed",
    "left:12px",
    "bottom:12px",
    "max-width: min(720px, 96vw)",
    "max-height: 40vh",
    "overflow:auto",
    "padding:10px 12px",
    "border-radius:10px",
    "background: rgba(0,0,0,0.75)",
    "color:#fff",
    "font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
    "z-index:999999",
    "display:none",
  ].join(";");
  document.body.appendChild(el);
  return {
    show(msg) {
      el.textContent = String(msg);
      el.style.display = "block";
    },
    append(msg) {
      el.textContent += `\n${msg}`;
      el.style.display = "block";
    },
    hide() {
      el.style.display = "none";
    },
  };
})();

function requireEl(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: #${id} (index.html を確認してください)`);
  return el;
}

// 状態（busイベント）
let assistantSpeaking = false;
let micRms = 0;

on("assistant:speakingStart", () => (assistantSpeaking = true));
on("assistant:speakingEnd", () => (assistantSpeaking = false));
on("user:micRms", (e) => {
  const v = Number(e?.detail?.rms ?? 0);
  // ちょい平滑化（ガタつき防止）
  micRms = micRms * 0.85 + v * 0.15;
});

// 顔トラッキング（任意）
let detector = null;
let faceVideo = null;
let faceLastMs = 0;

async function tryInitFaceTracking(webcamEl) {
  // index.html で tfjs と face-landmarks-detection を読み込んでいない場合はスキップ
  const tf = window.tf;
  const fld = window.faceLandmarksDetection;
  if (!tf || !fld) {
    console.info("[room] Face tracking disabled (tfjs / face-landmarks-detection not loaded)");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await webcamEl.play();
    faceVideo = webcamEl;

    await tf.setBackend("webgl");
    await tf.ready();

    const model = fld.SupportedModels.MediaPipeFaceMesh;
    detector = await fld.createDetector(model, {
      runtime: "tfjs",
      refineLandmarks: false,
    });

    console.info("[room] Face tracking enabled (runtime=tfjs)");
  } catch (e) {
    console.warn("[room] Face tracking init failed:", e);
  }
}

async function sampleFaceCenter() {
  if (!detector || !faceVideo || faceVideo.readyState < 2) return null;

  const faces = await detector.estimateFaces(faceVideo, { flipHorizontal: true });
  if (!faces?.length) return null;

  const f = faces[0];
  // face-landmarks-detection の box は xMin/xMax/yMin/yMax が基本
  const box = f.box;
  if (!box) return null;

  const xMin = box.xMin ?? box.left ?? 0;
  const xMax = box.xMax ?? (box.left ?? 0) + (box.width ?? 0);
  const yMin = box.yMin ?? box.top ?? 0;
  const yMax = box.yMax ?? (box.top ?? 0) + (box.height ?? 0);

  const cx = (xMin + xMax) * 0.5;
  const cy = (yMin + yMax) * 0.5;
  const w = Math.max(1, xMax - xMin);

  const vw = faceVideo.videoWidth || 640;
  const vh = faceVideo.videoHeight || 480;

  return { cx, cy, w, vw, vh };
}

// 3D初期化
async function init() {
  const stage = requireEl("stage");
  const uiDiv = requireEl("ui3d");
  const webcamEl = requireEl("webcam");

  // stage はフルスクリーン想定（room.cssが無い場合でも最低限効くように）
  stage.style.position = stage.style.position || "fixed";
  stage.style.inset = stage.style.inset || "0";

  // UIの見た目（最低限）
  uiDiv.style.background = uiDiv.style.background || "rgba(15, 15, 15, 0.92)";
  uiDiv.style.color = uiDiv.style.color || "#fff";
  uiDiv.style.borderRadius = uiDiv.style.borderRadius || "14px";
  uiDiv.style.padding = uiDiv.style.padding || "16px";
  uiDiv.style.width = uiDiv.style.width || "920px";
  uiDiv.style.maxWidth = uiDiv.style.maxWidth || "92vw";

  // --- WebGL scene ---
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 200);
  const camBase = new THREE.Vector3(0, 1.65, 6.2);
  const camTarget = camBase.clone();
  camera.position.copy(camBase);
  camera.lookAt(0, 1.55, 0);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.style.position = "absolute";
  renderer.domElement.style.inset = "0";
  // UI操作を優先（必要なら後で切替）
  renderer.domElement.style.pointerEvents = "none";
  stage.appendChild(renderer.domElement);

  // --- CSS3D scene ---
  const cssScene = new THREE.Scene();
  const cssRenderer = new CSS3DRenderer();
  cssRenderer.setSize(window.innerWidth, window.innerHeight);
  cssRenderer.domElement.style.position = "absolute";
  cssRenderer.domElement.style.inset = "0";
  // CSS3DRenderer 自体はポインターを奪わず、オブジェクト要素だけ pointerEvents:auto
  cssRenderer.domElement.style.pointerEvents = "none";
  stage.appendChild(cssRenderer.domElement);

  // --- Lights ---
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(3.5, 4.5, 2.5);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.35);
  rim.position.set(-4, 2.5, -3);
  scene.add(rim);

  // --- Room geometry ---
  const roomW = 9.0;
  const roomH = 4.8;
  const roomD = 12.0;

  const room = new THREE.Mesh(
    new THREE.BoxGeometry(roomW, roomH, roomD),
    new THREE.MeshStandardMaterial({ color: 0x1f1f24, roughness: 0.95, metalness: 0.0, side: THREE.BackSide })
  );
  room.position.set(0, roomH / 2, 0);
  scene.add(room);

  // floor accent
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(roomW, roomD),
    new THREE.MeshStandardMaterial({ color: 0x101012, roughness: 1.0, metalness: 0.0 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0.001, 0);
  scene.add(floor);

  // --- Back wall frame (WebGL plane) ---
  const screenW = roomW * 0.86;
  const screenH = roomH * 0.62;
  const backZ = -roomD / 2 + 0.04;

  const screenFrame = new THREE.Mesh(
    new THREE.PlaneGeometry(screenW + 0.2, screenH + 0.2),
    new THREE.MeshStandardMaterial({ color: 0x09090b, roughness: 0.9, metalness: 0.1 })
  );
  screenFrame.position.set(0, roomH * 0.56, backZ + 0.005);
  scene.add(screenFrame);

  // --- UI as CSS3DObject (back wall) ---
  const uiObj = new CSS3DObject(uiDiv);
  uiObj.position.set(0, roomH * 0.56, backZ);
  uiObj.rotation.y = 0; // 反転させると見えなくなることが多いのでまずは0
  cssScene.add(uiObj);

  function fitUI() {
    // elementがCSS3Dに移されるタイミングによって rect が 0 になる場合があるのでガード
    const rect = uiDiv.getBoundingClientRect();
    const pxW = Math.max(1, rect.width);
    const pxH = Math.max(1, rect.height);

    // UIの横幅を screenW に合わせて縮尺を決める（px → world units）
    const s = screenW / pxW;
    uiObj.scale.set(s, s, 1);

    // UIの縦をスクリーン枠に収める（縦がはみ出る場合はさらに縮める）
    const worldH = pxH * s;
    if (worldH > screenH) {
      const s2 = screenH / pxH;
      uiObj.scale.set(s2, s2, 1);
    }
  }

  // CSS3DRenderer が uiDiv を自分のDOMに移してから計測したいので2フレーム遅らせる
  requestAnimationFrame(() => requestAnimationFrame(fitUI));

  // --- Avatar (VRM) ---
  const avatarGroup = new THREE.Group();
  scene.add(avatarGroup);

  // まずは必ず見えるダミーを置く（VRMロード失敗でも「3Dが動いてる」ことが分かる）
  const dummy = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 1.4, 0.4),
    new THREE.MeshStandardMaterial({ color: 0x2ee59d, roughness: 0.6 })
  );
  dummy.position.set(0, 0.7, 2.8);
  avatarGroup.add(dummy);

  let vrm = null;
  try {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    vrm = await new Promise((resolve, reject) => {
      loader.load(
        AVATAR_URL,
        (gltf) => {
          const v = gltf.userData.vrm;
          if (!v) return reject(new Error("VRM parse failed (gltf.userData.vrm is null)"));
          resolve(v);
        },
        undefined,
        reject
      );
    });

    VRMUtils.removeUnnecessaryJoints(vrm.scene);

    // ダミーを消してVRMを置く
    avatarGroup.remove(dummy);

    // VRMは多くが -Z 向きなので、カメラ(+Z)に向くように回す
    vrm.scene.rotation.y = Math.PI;
    vrm.scene.position.set(0, 0, 2.8);

    // ちょいスケール調整（モデルによっては大き過ぎ/小さ過ぎる）
    vrm.scene.scale.setScalar(1.0);

    avatarGroup.add(vrm.scene);

    console.info("[room] VRM loaded:", AVATAR_URL);
  } catch (e) {
    console.warn("[room] VRM load failed (fallback to dummy):", e);
    debug.append("VRM load failed. Using dummy avatar.\n" + String(e));
  }

  // --- Face tracking (optional) ---
  tryInitFaceTracking(webcamEl);

  // --- Animation loop ---
  const clock = new THREE.Clock();

  function applyAvatarMotion(t, dt) {
    // ざっくり状態
    const speak = assistantSpeaking ? 1 : 0;
    const listen = Math.min(1, micRms * 10);

    // ダミー用（VRMがない時）
    if (!vrm) {
      dummy.position.y = 0.7 + 0.05 * listen + 0.04 * speak * Math.sin(t * 16);
      dummy.rotation.y = 0.15 * Math.sin(t * 0.8);
      return;
    }

    // VRMボーンで軽い身振り手振り
    const head = vrm.humanoid?.getNormalizedBoneNode("head");
    const jaw = vrm.humanoid?.getNormalizedBoneNode("jaw");
    const lArm = vrm.humanoid?.getNormalizedBoneNode("leftUpperArm");
    const rArm = vrm.humanoid?.getNormalizedBoneNode("rightUpperArm");

    if (head) {
      head.rotation.x = -0.10 * listen + 0.04 * speak * Math.sin(t * 10);
      head.rotation.y = 0.08 * Math.sin(t * 0.6);
    }

    // 口パク（超簡易）：喋ってる間だけ顎を動かす
    if (jaw) {
      const mouth = speak * (0.25 + 0.15 * Math.sin(t * 22));
      jaw.rotation.x = -mouth;
    }

    if (lArm) lArm.rotation.z = 0.35 * speak * Math.sin(t * 5.5);
    if (rArm) rArm.rotation.z = -0.35 * speak * Math.sin(t * 5.5);

    // VRM内部更新
    vrm.update(dt);
  }

  async function updateCameraByFace() {
    // 10〜15fps程度に間引き
    const now = performance.now();
    if (now - faceLastMs < 80) return;
    faceLastMs = now;

    const sample = await sampleFaceCenter();
    if (!sample) return;

    const { cx, cy, w, vw, vh } = sample;

    // -1..+1
    const nx = (cx / vw - 0.5) * 2;
    const ny = (0.5 - cy / vh) * 2;

    // 顔が大きい＝近い として Z も少し動かす
    const z = THREE.MathUtils.clamp((220 / w) - 1.0, -0.6, 0.8);

    camTarget.set(
      camBase.x + nx * 0.85,
      camBase.y + ny * 0.45,
      camBase.z + z * 1.2
    );
  }

  function animate() {
    requestAnimationFrame(animate);

    const dt = clock.getDelta();
    const t = clock.elapsedTime;

    // アバター
    applyAvatarMotion(t, dt);

    // 顔追従カメラ（任意）
    if (detector) {
      // 非同期（重いので await しない）
      updateCameraByFace().catch(() => {});
    }

    // なめらかに追従
    camera.position.lerp(camTarget, 0.12);
    camera.lookAt(0, 1.55, 0);

    renderer.render(scene, camera);
    cssRenderer.render(cssScene, camera);
  }

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    cssRenderer.setSize(window.innerWidth, window.innerHeight);
    fitUI();
  }

  window.addEventListener("resize", onResize);

  animate();
}

// 初期化タイミング：DOM構築後
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    init().catch((e) => {
      console.error("[room] init failed:", e);
      debug.show("room.js init failed:\n" + String(e));
    });
  });
} else {
  init().catch((e) => {
    console.error("[room] init failed:", e);
    debug.show("room.js init failed:\n" + String(e));
  });
}
