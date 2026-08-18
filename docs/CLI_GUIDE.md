# CLI User & Developer Guide

The `cardcenter` command-line tool provides full access to centering measurement, batch scanning, AR serving, versioning, and sync.

---

## 1. Single Card Centering Measurement

### Basic Scan
```bash
cardcenter card.jpg
```

### Measuring Through Slabs & Display Cases
Specify `--slab` or `--holder` to apply optical refraction corrections:
```bash
# PSA slabbed card
cardcenter card.jpg --slab psa --fov 68

# BGS slabbed card viewed through display case glass
cardcenter card.jpg --holder case_bgs --fov 68
```

### Visual Overlays & Machine-Readable Output
```bash
cardcenter card.jpg --overlay annotated.png --json results.json
```

---

## 2. Batch Scanning & Video Sweeps

### Multi-Card Display Case Photo
Scan and rank every card visible in a single photo:
```bash
cardcenter case_photo.jpg --scan --holder case_psa --db shop_inventory.db
```

### Continuous Video Pan
Measure cards continuously from a phone video pan:
```bash
cardcenter --video pan.mp4 --stride 6 --holder case_bgs --db shop_inventory.db
```

---

## 3. Web UI & Live Augmented Reality (AR)

### Phone Capture Server (Offline Web Front-End)
Starts the lightweight camera server:
```bash
# On local device
cardcenter --serve

# Listen on LAN for cross-device capture
cardcenter --serve --lan --port 8765
```

### Perceptopoly (AR Live Guidance Mode)
Starts the real-time AR triage interface with instant feedback on glare, standoff, and centering:
```bash
cardcenter --ar --lan --port 8766
```

---

## 4. Versioning, Capabilities & Remote Sync

### Comprehensive Build Information
```bash
cardcenter --info
```

### Engine Capability Matrix
```bash
cardcenter --capabilities
```

### Check Upstream GitHub Updates
```bash
cardcenter --check-updates
```

### Database Schema Migration
```bash
cardcenter --migrate-db shop_inventory.db
```

### Endpoint Health & Synchronization
```bash
# Check remote hub health
cardcenter --check-health http://192.168.1.50:8765

# Bidirectional sync (rebuilds the grade-outcome model from certified labels)
cardcenter --sync-url http://192.168.1.50:8765 --db shop_inventory.db

# Push unsynced local scans to the Loadopoly-OCR Supabase project (metadata only)
cardcenter --sync-cloud --db shop_inventory.db

# Rebuild the grade-outcome model from certified labels already in the store
cardcenter --ingest-grades --db shop_inventory.db
```

Single-image and `--serve` paths also write `--db` / `CARDCENTER_DB` after a successful measure, then best-effort upsert to `bakugo_scans` when the anon-key env vars are set. Photos are never uploaded.

`--serve` and `--ar` load the same model from `--db` or the `CARDCENTER_DB` environment variable. Measure and JSON output report `used_learned` / `n_observations` when certified labels have expanded the heuristic.

---

## 5. Provenance & Self-Test Verification

### Circularity & Provenance Audit
```bash
cardcenter --circularity --db shop_inventory.db
```

### Synthetic Ground-Truth Self-Test
Runs Monte Carlo synthetic validation against known mathematical ground truth:
```bash
cardcenter --self-test 40
```
