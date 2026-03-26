#!/usr/bin/env bash
# ============================================================================
# deploy.sh — One-shot deployment script for the Jetson Orin Nano
#
# Run on the Jetson after copying the project files over:
#   sudo bash setup/deploy.sh
#
# What it does:
#   1. Sets up the WiFi access point
#   2. Creates a Python venv & installs Flask
#   3. Installs systemd services for the Flask app and bridge
#   4. Enables & starts everything
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== SSS Deployment ==="
echo "  Project dir: $PROJECT_DIR"
echo ""

# ── 1. WiFi AP ──────────────────────────────────────────────────────────────
echo "--- Step 1: WiFi Access Point ---"
bash "$SCRIPT_DIR/setup-wifi-ap.sh"
echo ""

# ── 2. Python venv ──────────────────────────────────────────────────────────
echo "--- Step 2: Python virtual environment ---"
if [ ! -d "$PROJECT_DIR/venv" ]; then
  python3 -m venv "$PROJECT_DIR/venv"
fi
"$PROJECT_DIR/venv/bin/pip" install --upgrade pip
"$PROJECT_DIR/venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
"$PROJECT_DIR/venv/bin/pip" install requests   # needed by bridge.py
echo ""

# ── 3. Data directory ──────────────────────────────────────────────────────
echo "--- Step 3: Data directory ---"
mkdir -p /home/sss/data
chown -R sss:sss /home/sss/data 2>/dev/null || true
echo ""

# ── 4. Systemd services ────────────────────────────────────────────────────
echo "--- Step 4: Systemd services ---"
cp "$SCRIPT_DIR/sss-flask.service" /etc/systemd/system/
cp "$SCRIPT_DIR/sss-bridge.service" /etc/systemd/system/

# Update paths in service files to match actual project location
sed -i "s|/home/sss/user-deliverables|$PROJECT_DIR|g" /etc/systemd/system/sss-flask.service
sed -i "s|/home/sss/user-deliverables|$PROJECT_DIR|g" /etc/systemd/system/sss-bridge.service

systemctl daemon-reload
systemctl enable sss-flask sss-bridge
systemctl restart sss-flask sss-bridge
echo ""

# ── Done ────────────────────────────────────────────────────────────────────
echo "=== Deployment complete ==="
echo ""
echo "  WiFi SSID : SSS-Field"
echo "  Dashboard : http://10.42.0.1:5000"
echo "  Login     : admin / sss"
echo ""
echo "  Services:"
echo "    systemctl status sss-flask"
echo "    systemctl status sss-bridge"
echo "    journalctl -u sss-flask -f"
echo "    journalctl -u sss-bridge -f"
