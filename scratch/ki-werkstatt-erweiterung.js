// KI-Werkstatt für Scratch (TurboWarp) · MINT2Go · MINT erleben
//
// Diese Erweiterung nimmt ein Modell, das du in der KI-Werkstatt trainiert hast, und erkennt damit live per Kamera
// Bilder oder Körperhaltungen. Die Ergebnisse stehen als Scratch-Blöcke zur Verfügung, zum Beispiel
// „Wenn KI [Klasse 1] erkennt“.
//
// Wichtig: Die Erweiterung braucht Kamera und Fenster und läuft deshalb nur OHNE Sandbox.
// Beim Laden in TurboWarp den Haken „Run extension without sandbox“ (Erweiterung ohne Sandbox ausführen) setzen.
// Der Code wird als Text eingefügt oder als Datei geladen. Projekte mit dieser Erweiterung laufen nur in TurboWarp,
// nicht auf scratch.mit.edu.
//
// Alles läuft im Browser. Es werden keine Kamerabilder verschickt. Nur die fertigen Erkennungsmodelle
// (MobileNet oder PoseNet) werden einmal aus dem Internet geladen.

(function (Scratch) {
  'use strict';

  if (!Scratch.extensions.unsandboxed) {
    throw new Error('Die KI-Werkstatt-Erweiterung muss ohne Sandbox laufen. Beim Laden den Haken „Run extension without sandbox“ setzen.');
  }

  const APP_URL = window.KIW_APP_URL || 'https://makingvs.github.io/lukas-rennbahn/ki-werkstatt.html';
  const APP_ORIGIN = new URL(APP_URL).origin;
  const TF_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
  const MN_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js';
  const PN_URL = 'https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet@2.2.2/dist/posenet.min.js';
  const STORE = 'kiwerkstatt-modell-v1';
  const LIMBS = [[5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [0, 1], [0, 2], [1, 3], [2, 4]];
  const PANEL = 220;

  /* ---------- Hilfsfunktionen ---------- */
  const scripts = {};
  const loadScript = (src) => scripts[src] || (scripts[src] = new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src; s.onload = res;
    s.onerror = () => { delete scripts[src]; rej(new Error('Laden fehlgeschlagen: ' + src)); };
    document.head.appendChild(s);
  }));

  const b64ToF32 = (s) => {
    const bin = atob(s), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Float32Array(u.buffer);
  };

  // Pose: Mitte und Größe herausrechnen (genau wie in der KI-Werkstatt)
  function poseFeat(pose) {
    const k = pose.keypoints, good = k.filter((a) => a.score >= 0.3);
    if (good.length < 5) return null;
    let cx = 0, cy = 0;
    good.forEach((a) => { cx += a.position.x; cy += a.position.y; });
    cx /= good.length; cy /= good.length;
    let sc = 0;
    good.forEach((a) => { sc = Math.max(sc, Math.hypot(a.position.x - cx, a.position.y - cy)); });
    sc = sc || 1;
    const f = new Float32Array(34);
    for (let i = 0; i < k.length; i++) {
      if (k[i].score >= 0.3) { f[2 * i] = (k[i].position.x - cx) / sc; f[2 * i + 1] = (k[i].position.y - cy) / sc; }
    }
    return f;
  }

  function pixelFeat(cv, ctx) {
    ctx.drawImage(cv, 0, 0, 16, 16);
    const d = ctx.getImageData(0, 0, 16, 16).data, f = new Float32Array(768);
    let k = 0, m = 0;
    for (let i = 0; i < d.length; i += 4) { f[k++] = d[i]; f[k++] = d[i + 1]; f[k++] = d[i + 2]; }
    for (let i = 0; i < f.length; i++) m += f[i];
    m /= f.length;
    for (let i = 0; i < f.length; i++) f[i] = (f[i] - m) / 255;
    return f;
  }

  // Das kleine Netz aus der KI-Werkstatt: Eingabe -> 100 Neuronen (ReLU) -> Klassen (Softmax)
  function headPredict(m, f) {
    const D = m.dim, U = m.units, C = m.klassen.length;
    const x = new Float32Array(D);
    for (let i = 0; i < D; i++) x[i] = (f[i] - m.mu[i]) / m.sigma;
    const h = new Float32Array(U);
    for (let j = 0; j < U; j++) h[j] = m.b1[j];
    for (let i = 0; i < D; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * U;
      for (let j = 0; j < U; j++) h[j] += xi * m.w1[row + j];
    }
    const z = new Float32Array(C);
    for (let k = 0; k < C; k++) z[k] = m.b2[k];
    for (let j = 0; j < U; j++) {
      const hj = h[j] > 0 ? h[j] : 0;
      if (hj === 0) continue;
      const row = j * C;
      for (let k = 0; k < C; k++) z[k] += hj * m.w2[row + k];
    }
    let mx = -Infinity;
    for (let k = 0; k < C; k++) if (z[k] > mx) mx = z[k];
    let sum = 0;
    for (let k = 0; k < C; k++) { z[k] = Math.exp(z[k] - mx); sum += z[k]; }
    for (let k = 0; k < C; k++) z[k] /= sum;
    return z;
  }

  /* ---------- Die Erweiterung ---------- */
  class KIWerkstatt {
    constructor() {
      this.model = null;
      this.probs = [];
      this.best = -1;
      this.conf = 0;
      this.minConf = 0.6;
      this.cameraOn = false;
      this.showPanel = true;
      this.pose = null;
      this._busy = false;
      this._timer = 0;
      this._net = null;
      this._netKey = '';
      this._netPromise = null;
      this._cap = document.createElement('canvas');
      this._px = document.createElement('canvas'); this._px.width = this._px.height = 16;
      this._pxCtx = this._px.getContext('2d', { willReadFrequently: true });
      window.addEventListener('message', (e) => this._onMessage(e));
      try {
        const s = localStorage.getItem(STORE);
        if (s) this._setModel(JSON.parse(s), false);
      } catch (e) { /* kein gespeichertes Modell */ }
    }

    getInfo() {
      return {
        id: 'kiwerkstatt',
        name: 'KI-Werkstatt',
        color1: '#99c328',
        color2: '#7ea51b',
        color3: '#5e8014',
        blocks: [
          { opcode: 'oeffnen', blockType: Scratch.BlockType.COMMAND, text: 'KI-Werkstatt öffnen (Modell holen)' },
          { opcode: 'modellEinfuegen', blockType: Scratch.BlockType.COMMAND, text: 'Modell aus Zwischenablage einfügen' },
          { opcode: 'modellDatei', blockType: Scratch.BlockType.COMMAND, text: 'Modell aus Datei laden' },
          '---',
          {
            opcode: 'kamera', blockType: Scratch.BlockType.COMMAND, text: 'Kamera [AN_AUS]',
            arguments: { AN_AUS: { type: Scratch.ArgumentType.STRING, menu: 'anAus', defaultValue: 'an' } }
          },
          {
            opcode: 'bild', blockType: Scratch.BlockType.COMMAND, text: 'Kamerabild [ZEIGEN]',
            arguments: { ZEIGEN: { type: Scratch.ArgumentType.STRING, menu: 'zeigen', defaultValue: 'zeigen' } }
          },
          { opcode: 'modellDa', blockType: Scratch.BlockType.BOOLEAN, text: 'Modell geladen?' },
          '---',
          {
            opcode: 'wennErkannt', blockType: Scratch.BlockType.HAT, text: 'Wenn KI [KLASSE] erkennt', isEdgeActivated: true,
            arguments: { KLASSE: { type: Scratch.ArgumentType.STRING, menu: 'klassen' } }
          },
          {
            opcode: 'erkennt', blockType: Scratch.BlockType.BOOLEAN, text: 'KI erkennt [KLASSE]?',
            arguments: { KLASSE: { type: Scratch.ArgumentType.STRING, menu: 'klassen' } }
          },
          { opcode: 'klasse', blockType: Scratch.BlockType.REPORTER, text: 'erkannte Klasse' },
          { opcode: 'sicherheit', blockType: Scratch.BlockType.REPORTER, text: 'Sicherheit der erkannten Klasse (%)' },
          {
            opcode: 'sicherheitVon', blockType: Scratch.BlockType.REPORTER, text: 'Sicherheit von [KLASSE] (%)',
            arguments: { KLASSE: { type: Scratch.ArgumentType.STRING, menu: 'klassen' } }
          },
          {
            opcode: 'minSicherheit', blockType: Scratch.BlockType.COMMAND, text: 'Mindest-Sicherheit auf [PROZENT] % setzen',
            arguments: { PROZENT: { type: Scratch.ArgumentType.NUMBER, defaultValue: 60 } }
          },
          '---',
          { opcode: 'anzahlKlassen', blockType: Scratch.BlockType.REPORTER, text: 'Anzahl der Klassen' },
          {
            opcode: 'klassenName', blockType: Scratch.BlockType.REPORTER, text: 'Name der Klasse [NR]',
            arguments: { NR: { type: Scratch.ArgumentType.NUMBER, defaultValue: 1 } }
          }
        ],
        menus: {
          anAus: { acceptReporters: false, items: ['an', 'aus'] },
          zeigen: { acceptReporters: false, items: ['zeigen', 'verstecken'] },
          klassen: { acceptReporters: true, items: 'klassenListe' }
        }
      };
    }

    /* ----- Blöcke ----- */
    oeffnen() {
      const w = window.open(APP_URL + (APP_URL.includes('?') ? '&' : '?') + 'scratch=1', 'kiwerkstatt');
      if (!w) this._say('Das Fenster wurde blockiert. Erlaube Pop-ups für diese Seite und klicke den Block noch einmal.');
    }
    async modellEinfuegen() {
      try {
        const t = await navigator.clipboard.readText();
        this._setModel(JSON.parse(t), true);
      } catch (e) {
        this._say('Kein Modell in der Zwischenablage gefunden. Klicke in der KI-Werkstatt auf „Modell kopieren“ und versuche es noch einmal.');
      }
    }
    modellDatei() {
      const inp = document.createElement('input');
      inp.type = 'file'; inp.accept = '.json,application/json';
      inp.onchange = async () => {
        try { this._setModel(JSON.parse(await inp.files[0].text()), true); }
        catch (e) { this._say('Diese Datei ist kein Modell aus der KI-Werkstatt.'); }
      };
      inp.click();
    }
    async kamera(args) {
      if (String(args.AN_AUS) === 'an') await this._start(); else this._stop();
    }
    bild(args) {
      this.showPanel = String(args.ZEIGEN) === 'zeigen';
      this._updatePanel();
    }
    modellDa() { return !!this.model; }
    klasse() { return this.best >= 0 && this.conf >= this.minConf ? this.model.klassen[this.best] : ''; }
    sicherheit() { return this.best >= 0 ? Math.round(this.conf * 100) : 0; }
    sicherheitVon(args) {
      const i = this._index(args.KLASSE);
      return i < 0 ? 0 : Math.round((this.probs[i] || 0) * 100);
    }
    erkennt(args) {
      const i = this._index(args.KLASSE);
      return i >= 0 && i === this.best && this.conf >= this.minConf;
    }
    wennErkannt(args) { return this.erkennt(args); }
    minSicherheit(args) {
      this.minConf = Math.max(0, Math.min(100, Scratch.Cast.toNumber(args.PROZENT))) / 100;
    }
    anzahlKlassen() { return this.model ? this.model.klassen.length : 0; }
    klassenName(args) {
      const n = Math.round(Scratch.Cast.toNumber(args.NR));
      return this.model && n >= 1 && n <= this.model.klassen.length ? this.model.klassen[n - 1] : '';
    }
    klassenListe() {
      return this.model && this.model.klassen.length ? this.model.klassen : [{ text: '(noch kein Modell)', value: '' }];
    }

    /* ----- Modell ----- */
    _index(name) {
      if (!this.model) return -1;
      const n = String(name).trim().toLowerCase();
      return this.model.klassen.findIndex((k) => String(k).trim().toLowerCase() === n);
    }
    _setModel(m, announce) {
      if (!m || m.kind !== 'ki-werkstatt-modell' || !m.klassen || !m.w1 || !m.w2) throw new Error('kein Modell');
      this.model = {
        art: m.art === 'pose' ? 'pose' : 'image', mode: m.mode === 'pixel' ? 'pixel' : 'model', alpha: m.alpha || 0.25,
        dim: m.dim, units: m.units || 100, klassen: m.klassen.map(String),
        mu: b64ToF32(m.mu), sigma: m.sigma || 1, w1: b64ToF32(m.w1), b1: b64ToF32(m.b1), w2: b64ToF32(m.w2), b2: b64ToF32(m.b2)
      };
      this.probs = []; this.best = -1; this.conf = 0;
      try { localStorage.setItem(STORE, JSON.stringify(m)); } catch (e) { /* zu groß: egal */ }
      if (announce) this._say('✓ Modell geladen: ' + this.model.klassen.join(', '));
      this._updatePanel();
      if (this.cameraOn) this._ensureNet().catch(() => {});
    }
    _onMessage(e) {
      const d = e.data;
      if (e.origin !== APP_ORIGIN || !d || d.kind !== 'ki-werkstatt-modell') return;
      try { this._setModel(d, true); } catch (err) { this._say('Das Modell konnte nicht gelesen werden.'); }
    }

    /* ----- Kamera und Erkennung ----- */
    async _ensureNet() {
      const m = this.model; if (!m) return;
      if (m.mode === 'pixel' && m.art !== 'pose') return;
      if (Date.now() - (this._netFailAt || 0) < 10000) throw new Error('Das KI-Modell konnte nicht geladen werden. Neuer Versuch in wenigen Sekunden.');
      const key = m.art + ':' + m.alpha;
      if (this._net && this._netKey === key) return;
      if (this._netPromise && this._netKey === key) return this._netPromise;
      this._netKey = key;
      this._netPromise = (async () => {
        this._say('KI-Modell wird geladen …', 2500);
        await loadScript(TF_URL);
        await tf.ready();
        if (m.art === 'pose') {
          await loadScript(PN_URL);
          this._net = await posenet.load({ architecture: 'MobileNetV1', outputStride: 16, inputResolution: { width: 257, height: 257 }, multiplier: 0.5 });
        } else {
          await loadScript(MN_URL);
          this._net = await mobilenet.load({ version: 1, alpha: m.alpha });
        }
        this._say('KI bereit.', 1500);
      })();
      try { await this._netPromise; } catch (e) { this._net = null; this._netKey = ''; this._netFailAt = Date.now(); this._say('Das KI-Modell konnte nicht geladen werden. Prüfe das Internet.', 6000); throw e; }
    }
    async _start() {
      if (this.cameraOn) return;
      if (!this.model) this._say('Lade zuerst ein Modell: Block „Modell aus Zwischenablage einfügen“.', 5000);
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
      } catch (e) {
        this._say('Die Kamera konnte nicht gestartet werden. Erlaube die Kamera für diese Seite.', 6000);
        return;
      }
      this.cameraOn = true;
      this._buildPanel();
      this.video.srcObject = this.stream;
      try { await this.video.play(); } catch (e) { /* ok */ }
      if (this.model) this._ensureNet().catch(() => {});
      this._loop();
    }
    _stop() {
      this.cameraOn = false;
      clearTimeout(this._timer);
      if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
      this.probs = []; this.best = -1; this.conf = 0; this.pose = null;
      if (this.panel) { this.panel.remove(); this.panel = null; this.video = null; }
    }
    _loop() {
      if (!this.cameraOn) return;
      const next = () => { if (this.cameraOn) this._timer = setTimeout(() => this._loop(), 90); };
      if (this._busy || !this.model || document.hidden) { next(); return; }
      this._busy = true;
      this._frame().catch((e) => console.warn('KI-Werkstatt:', e)).finally(() => { this._busy = false; next(); });
    }
    _grab(v, size) {
      const w = v.videoWidth, h = v.videoHeight, s = Math.min(w, h), c = this._cap;
      if (c.width !== size) c.width = c.height = size;
      c.getContext('2d', { willReadFrequently: true }).drawImage(v, (w - s) / 2, (h - s) / 2, s, s, 0, 0, size, size);
      return c;
    }
    async _frame() {
      const m = this.model, v = this.video;
      if (!v || !v.videoWidth) return;
      await this._ensureNet();
      const cv = this._grab(v, m.art === 'pose' ? 257 : 224);
      let f = null, pose = null;
      if (m.art === 'pose') {
        pose = await this._net.estimateSinglePose(cv, { flipHorizontal: false });
        f = poseFeat(pose);
      } else if (m.mode === 'model') {
        const t = tf.tidy(() => this._net.infer(cv, true).flatten());
        f = await t.data(); t.dispose();
      } else {
        f = pixelFeat(cv, this._pxCtx);
      }
      this.pose = pose;
      if (!f || f.length !== m.dim) { this.probs = []; this.best = -1; this.conf = 0; this._draw(); return; }
      this.probs = Array.from(headPredict(m, f));
      let best = 0;
      this.probs.forEach((p, i) => { if (p > this.probs[best]) best = i; });
      this.best = best; this.conf = this.probs[best];
      this._draw();
    }

    /* ----- Anzeige ----- */
    _buildPanel() {
      const p = document.createElement('div');
      p.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:100000;width:' + PANEL + 'px;background:#fff;border-radius:14px;' +
        'box-shadow:0 4px 18px rgba(0,0,0,.35);font:700 14px "Avenir Next",Avenir,Futura,Nunito,system-ui,sans-serif;color:#27304d;overflow:hidden';
      p.innerHTML = '<div style="position:relative;width:' + PANEL + 'px;height:' + PANEL + 'px;background:#222">' +
        '<video playsinline muted autoplay style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)"></video>' +
        '<canvas width="' + PANEL + '" height="' + PANEL + '" style="position:absolute;inset:0;width:100%;height:100%;transform:scaleX(-1)"></canvas></div>' +
        '<div style="padding:8px 12px;border-top:5px solid #99c328"><div style="font-size:11px;letter-spacing:.1em;color:#5b6482">MINT2GO · KI-WERKSTATT</div>' +
        '<div class="kiw-t" style="font-size:18px;min-height:24px"></div></div>';
      document.body.appendChild(p);
      this.panel = p; this.video = p.querySelector('video'); this.canvas = p.querySelector('canvas'); this.label = p.querySelector('.kiw-t');
      this._updatePanel();
    }
    _updatePanel() {
      if (!this.panel) return;
      this.panel.style.display = this.showPanel ? '' : 'none';
      this._draw();
    }
    _draw() {
      if (!this.panel) return;
      const g = this.canvas.getContext('2d');
      g.clearRect(0, 0, PANEL, PANEL);
      if (this.pose) {
        const k = this.pose.keypoints, s = PANEL / 257;
        g.lineWidth = 3; g.lineCap = 'round'; g.strokeStyle = 'rgba(58,187,234,.98)';
        LIMBS.forEach(([a, b]) => {
          if (k[a].score >= 0.3 && k[b].score >= 0.3) { g.beginPath(); g.moveTo(k[a].position.x * s, k[a].position.y * s); g.lineTo(k[b].position.x * s, k[b].position.y * s); g.stroke(); }
        });
        g.fillStyle = '#fff';
        k.forEach((q) => { if (q.score >= 0.3) { g.beginPath(); g.arc(q.position.x * s, q.position.y * s, 4.5, 0, 6.3); g.fill(); } });
      }
      let t;
      if (!this.model) t = 'Kein Modell geladen';
      else if (this.best < 0) t = this.model.art === 'pose' ? 'Keine Person erkannt' : '…';
      else t = (this.conf >= this.minConf ? this.model.klassen[this.best] : 'nicht sicher') + ' · ' + Math.round(this.conf * 100) + ' %';
      if (this.label && this.label.textContent !== t) this.label.textContent = t;
    }
    _say(text, ms) {
      console.log('KI-Werkstatt:', text);
      let el = document.getElementById('kiw-toast');
      if (!el) {
        el = document.createElement('div'); el.id = 'kiw-toast';
        el.style.cssText = 'position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:100001;background:#27304d;color:#fff;' +
          'border-radius:10px;padding:12px 18px;max-width:min(92vw,560px);font:600 14px system-ui,sans-serif;box-shadow:0 4px 14px rgba(0,0,0,.4)';
        document.body.appendChild(el);
      }
      el.textContent = text; el.style.display = '';
      clearTimeout(this._toastTimer);
      this._toastTimer = setTimeout(() => { el.style.display = 'none'; }, ms || 5000);
    }
  }

  Scratch.extensions.register(new KIWerkstatt());
})(Scratch);
