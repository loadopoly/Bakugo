# Bakugo Mobile: Native App & Progressive Web App (PWA)

This directory provides the cross-platform **Mobile Native** wrapper (Android / iOS) and **PWA** configuration for Bakugo. Both the native app and web client communicate directly with the **same central Bakugo container backend** (`cardcenter.serve`).

---

## 🏗️ Architecture: Dual Client, Single Container

```
+------------------------------------------------------------------+
|                    Bakugo Container Hub                          |
|         (Docker / Python: cardcenter --serve :8765)              |
|                                                                  |
|   • Snell Ray-Tracing Metrology Engine                           |
|   • Sequential Probability Ratio Test (SPRT / Chi-sq Fusion)     |
|   • SQLite WAL Ingest & DuckDB Analytical Layer                  |
|   • QUIPU Mesh / Supabase Cloud Sync                             |
|   • REST & Streaming AR API: /measure, /ar/push, /my-scans       |
+------------------------------------------------------------------+
               ▲                                    ▲
               │ (LAN / WiFi / Tunnel)              │ (Browser HTTP / HTTPS)
               │                                    │
+-----------------------------+      +-----------------------------+
|   Native Mobile App         |      |   Progressive Web App       |
|   (Capacitor / Android/iOS) |      |   (Chrome / Safari / PWA)   |
|                             |      |                             |
| • Native Camera Hardware    |      | • WebAR Camera Stream       |
| • Real-time AR Laser HUD    |      | • Web Audio Synthesizer     |
| • Native Haptic Feedback    |      | • Zero-Install Home Screen  |
+-----------------------------+      +-----------------------------+
```

---

## 📱 Quick Setup: Two Ways to Run on Mobile

### Option A: Instant PWA (Zero Build, Zero Install)

1. Start your Bakugo container or desktop server:
   ```bash
   docker compose up -d
   # or
   cardcenter --serve --lan --port 8765
   ```
2. On your phone (connected to the same WiFi), open Chrome or Safari:
   ```
   http://<YOUR_COMPUTER_IP>:8765
   ```
3. Tap the browser menu:
   * **Android Chrome**: Tap `...` -> **Install App** or **Add to Home screen**.
   * **iOS Safari**: Tap **Share** -> **Add to Home Screen**.
4. Launch Bakugo directly from your home screen as a full-screen, hardware-accelerated AR app.

---

### Option B: Build Native Android APK (Capacitor)

1. Ensure Node.js and Android Studio / JDK are installed.
2. Install mobile dependencies:
   ```bash
   cd mobile
   npm install
   ```
3. Bundle assets and sync with native Android project:
   ```bash
   pwsh Build-Mobile.ps1
   npx cap add android
   npx cap sync
   ```
4. Open in Android Studio or build debug APK:
   ```bash
   npx cap open android
   ```
5. In the app on your phone, tap the **Container** pill in the top header to enter your host container IP (e.g. `http://192.168.1.150:8765` or your Cloudflare Tunnel URL).

---

## ⚡ Live AR Features

* **Real-time Quad Tracking**: Continuous 4-corner perspective tracking with subpixel line-fitting.
* **Laser Caliper HUD**: Visual neon caliper beams with mm tick marks.
* **Sequential Probability Ratio Test (SPRT)**: Dials indicating grade boundary convergence (e.g. PSA 10 55/45).
* **Web Audio API Synth**: Sci-fi acoustic lock-on and settlement cues without external audio files.
* **Haptics**: Tactile vibration pulses on edge acquisition and grade lock.
