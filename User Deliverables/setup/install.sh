#!/usr/bin/env bash
# ============================================================================
# install.sh — Install (or update) the SSS Flask systemd service on the Jetson
#
# Run on the Jetson from anywhere:
#   sudo bash ~/SSS/User\ Deliverables/setup/install.sh
#
# Prerequisites (one-time, manual):
#   1. Hotspot mode is enabled in the Jetson's GNOME network settings.
#   2. Project is cloned to ~/SSS/ and the venv lives at ~/venvs/sss/.
#   3. Requirements are installed inside that venv.
#
# What this script does:
#   1. Upgrades pip + reinstalls requirements.txt into the existing venv.
#   2. Installs sss-flask.service into /etc/systemd/system/.
#   3. Reloads systemd, enables the service for boot, and starts it now.
# ============================================================================

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "This script must be run with sudo." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="/home/SubstationSquirrelSleuths/venvs/sss"

echo "=== SSS Flask Service Install ==="
echo "  Project dir: $PROJECT_DIR"
echo "  Venv:        $VENV_DIR"
echo ""

# ── 1. Refresh the venv ─────────────────────────────────────────────────────
echo "--- Step 1: Refreshing Python environment ---"
sudo -u SubstationSquirrelSleuths "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u SubstationSquirrelSleuths "$VENV_DIR/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
echo ""

# ── 2. Install the systemd unit ─────────────────────────────────────────────
echo "--- Step 2: Installing systemd unit ---"
cp "$SCRIPT_DIR/sss-flask.service" /etc/systemd/system/sss-flask.service
chmod 644 /etc/systemd/system/sss-flask.service
systemctl daemon-reload
systemctl enable sss-flask.service
systemctl restart sss-flask.service
echo ""

# ── Done ────────────────────────────────────────────────────────────────────
echo "=== Done ==="
echo ""
echo "  Dashboard : http://10.42.0.1:5000"
echo "  Login     : admin / sss"
echo ""
echo "  Useful commands:"
echo "    systemctl status sss-flask"
echo "    journalctl -u sss-flask -f"
echo "    sudo systemctl restart sss-flask"
