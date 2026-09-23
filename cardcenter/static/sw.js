// Bakugo service worker: the second vote on card identification.
//
// OCR (server) names the card. This worker embeds the same crop with an ONNX
// model and compares it with the cards THIS device has confirmed before
// (IndexedDB, never uploaded). The server accepts an identification only when
// both agree and the frame passes the shot-noise gate (confidence.py); the
// vote alone can never name a card.
//
// Served by cardcenter/serve.py at /sw.js as a module worker. The import line
// below is filled in only when CARDCENTER_EMBED_DIR holds a valid model and a
// self-hosted onnxruntime-web WASM bundle; otherwise every vote answers
// {available: false} with the reason.
/*__ORT_IMPORT__*/
const EMBED_STATUS = /*__STATUS__*/;

const DB_NAME = 'bakugo-embed';
const STORE = 'cards';
const PENDING_MAX = 16;
let session = null;
let manifest = null;
let loading = null;
const pending = new Map();

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));

async function loadModel() {
  if (typeof ort === 'undefined' || !ort || !EMBED_STATUS.available) {
    throw new Error(EMBED_STATUS.reason || 'embedding model not installed');
  }
  if (session) return session;
  if (!loading) {
    loading = (async () => {
      const m = await (await fetch('/embed/model.json', { cache: 'no-cache' })).json();
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      const cache = await caches.open('bakugo-embed-' + m.model_id);
      let resp = await cache.match('/embed/model.onnx');
      if (!resp) {
        resp = await fetch('/embed/model.onnx');
        if (!resp.ok) throw new Error('model download failed: ' + resp.status);
        await cache.put('/embed/model.onnx', resp.clone());
      }
      const bytes = new Uint8Array(await resp.arrayBuffer());
      const s = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
      manifest = m;
      session = s;
      return s;
    })().catch((err) => { loading = null; throw err; });
  }
  return loading;
}

function toTensor(image, m) {
  const H = m.input_size[0];
  const W = m.input_size[1];
  if (image.width !== W || image.height !== H) {
    throw new Error('crop must be ' + W + 'x' + H + ', got ' + image.width + 'x' + image.height);
  }
  const d = image.data;
  const out = new Float32Array(3 * H * W);
  const mean = m.mean || [0, 0, 0];
  const std = m.std || [1, 1, 1];
  const scale = m.scale || 255;
  const order = (m.channels || 'RGB') === 'BGR' ? [2, 1, 0] : [0, 1, 2];
  const nhwc = m.layout === 'NHWC';
  for (let i = 0; i < H * W; i++) {
    for (let c = 0; c < 3; c++) {
      const v = d[i * 4 + order[c]] / scale;
      out[nhwc ? i * 3 + c : c * H * W + i] = (v - mean[c]) / std[c];
    }
  }
  return new ort.Tensor('float32', out, nhwc ? [1, H, W, 3] : [1, 3, H, W]);
}

async function embed(image) {
  const s = await loadModel();
  const inputName = manifest.input_name || s.inputNames[0];
  const outputName = manifest.output_name || s.outputNames[0];
  const out = await s.run({ [inputName]: toTensor(image, manifest) });
  const v = Float32Array.from(out[outputName].data);
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < v.length; i++) v[i] /= norm;
  return v;
}

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const st = req.result.createObjectStore(STORE, { keyPath: 'key', autoIncrement: true });
      st.createIndex('model_id', 'model_id');
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function entriesFor(modelId) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const req = db.transaction(STORE).objectStore(STORE).index('model_id').getAll(modelId);
    req.onsuccess = () => resolve(req.result || []);
    req.onerror = () => reject(req.error);
  });
}

async function addEntry(entry) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).add(entry);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

async function clearModel(modelId) {
  const entries = await entriesFor(modelId);
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    for (const e of entries) tx.objectStore(STORE).delete(e.key);
    tx.oncomplete = () => resolve(entries.length);
    tx.onerror = () => reject(tx.error);
  });
}

function nearest(vec, entries) {
  // best similarity per label, then best vs runner-up label
  const byLabel = new Map();
  for (const e of entries) {
    let dot = 0;
    const w = e.vector;
    const n = Math.min(w.length, vec.length);
    for (let i = 0; i < n; i++) dot += w[i] * vec[i];
    const label = String(e.name).toLowerCase();
    const cur = byLabel.get(label);
    if (!cur || dot > cur.similarity) byLabel.set(label, { name: e.name, number: e.number, similarity: dot });
  }
  const ranked = [...byLabel.values()].sort((a, b) => b.similarity - a.similarity);
  if (!ranked.length) return null;
  const second = ranked.length > 1 ? ranked[1].similarity : 0;
  return { name: ranked[0].name, number: ranked[0].number, similarity: ranked[0].similarity,
           margin: ranked[0].similarity - second, labels: ranked.length };
}

async function handle(msg) {
  const kind = msg && msg.type;
  if (kind === 'status') {
    if (!EMBED_STATUS.available) return { available: false, reason: EMBED_STATUS.reason };
    const entries = await entriesFor(EMBED_STATUS.model_id);
    return { available: true, model_id: EMBED_STATUS.model_id,
             input_size: EMBED_STATUS.input_size, index_size: entries.length };
  }
  if (kind === 'vote') {
    const t0 = Date.now();
    const vec = await embed(msg.image);
    pending.set(msg.id, vec);
    while (pending.size > PENDING_MAX) pending.delete(pending.keys().next().value);
    const entries = await entriesFor(manifest.model_id);
    const best = nearest(vec, entries);
    const base = { available: true, model_id: manifest.model_id, index_size: entries.length,
                   embed_ms: Date.now() - t0 };
    if (!best) return Object.assign(base, { name: null, similarity: 0, margin: 0,
                                            reason: 'no confirmed cards on this device yet' });
    return Object.assign(base, best);
  }
  if (kind === 'learn') {
    const vec = pending.get(msg.id);
    if (!vec) return { ok: false, reason: 'no embedding held for this identification' };
    if (!msg.name) return { ok: false, reason: 'name required' };
    await addEntry({ model_id: manifest.model_id, name: String(msg.name),
                     number: msg.number == null ? null : Number(msg.number),
                     vector: Array.from(vec), created: Date.now() });
    pending.delete(msg.id);
    const entries = await entriesFor(manifest.model_id);
    return { ok: true, index_size: entries.length };
  }
  if (kind === 'forget') {
    if (!EMBED_STATUS.available) return { ok: true, removed: 0 };
    return { ok: true, removed: await clearModel(EMBED_STATUS.model_id) };
  }
  return { ok: false, reason: 'unknown message' };
}

self.addEventListener('message', (e) => {
  const port = e.ports && e.ports[0];
  if (!port) return;
  handle(e.data).then(
    (r) => port.postMessage(r),
    (err) => port.postMessage({ available: false, ok: false, reason: String(err && err.message || err) }),
  );
});
