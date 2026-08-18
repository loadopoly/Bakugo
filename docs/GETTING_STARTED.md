# Getting Started with Bakugo (CardCenter)

This guide walks you through setting up Bakugo on mobile devices (Android Termux) and desktop environments (Linux, macOS, Windows).

---

## 1. Android Installation (Termux)

No app compilation or native build toolchain required. The backend runs on Python, NumPy, and OpenCV inside Termux, while your phone's browser serves as the touch and AR front end.

### Step 1: Install Termux
Download and install Termux from **F-Droid** (do *not* use the outdated Google Play Store build):
👉 [Termux on F-Droid](https://f-droid.org/packages/com.termux/)

### Step 2: Clone & Install Dependencies
Open Termux and run:
```bash
pkg update && pkg install -y git python python-numpy opencv-python
git clone https://github.com/PoodlesOfWar/Bakugo.git
cd Bakugo
pip install -e .
```

### Step 3: Launch Local Web UI
```bash
cardcenter --serve
```
Open `http://127.0.0.1:8765` in Chrome on your phone. Tap the capture button to measure cards directly with your rear camera.

---

## 2. Desktop Installation (Linux, macOS, Windows)

### Prerequisites
* Python 3.10+
* Git

### Installation
```bash
git clone https://github.com/PoodlesOfWar/Bakugo.git
cd Bakugo
pip install -e ".[dev]"
```

### Quick Test
```bash
# Check version & system capabilities
cardcenter --info
cardcenter --capabilities

# Run Monte Carlo self-test
cardcenter --self-test 20
```

---

## 3. Remote Sync Setup

To synchronize scans from your phone (Termux) to your central desktop database:
```bash
# On Desktop:
cardcenter --serve --lan --port 8765 --db central_vault.db

# On Phone (Termux):
cardcenter --sync-url http://<DESKTOP_LAN_IP>:8765 --db mobile_scans.db
```
