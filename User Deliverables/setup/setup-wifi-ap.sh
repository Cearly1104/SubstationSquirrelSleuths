#!/usr/bin/env bash
# ============================================================================
# setup-wifi-ap.sh — Configure the Jetson Orin Nano as a WiFi Access Point
#
# This script uses NetworkManager (nmcli) which is the default network manager
# on JetPack / Ubuntu for Jetson.  It creates a hotspot on the built-in or
# USB WiFi adapter so field users can connect and reach the Flask dashboard.
#
# Usage:
#   sudo bash setup-wifi-ap.sh [OPTIONS]
#
# Options (override via environment or args):
#   WIFI_IFACE   — wireless interface name        (default: wlan0)
#   AP_SSID      — network name                   (default: SSS-Field)
#   AP_PASSWORD  — WPA2 passphrase (>=8 chars)    (default: squirrel2024)
#   AP_BAND      — wifi band: bg (2.4 GHz) or a   (default: bg)
#   AP_CHANNEL   — channel number                  (default: 6)
#   AP_IP        — static IP for the Jetson        (default: 10.42.0.1)
#   AP_GATEWAY   — subnet gateway                  (default: 10.42.0.1)
#   AP_PREFIX    — CIDR prefix length              (default: 24)
# ============================================================================

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
WIFI_IFACE="${WIFI_IFACE:-wlan0}"
AP_SSID="${AP_SSID:-SSS-Field}"
AP_PASSWORD="${AP_PASSWORD:-squirrel2024}"
AP_BAND="${AP_BAND:-bg}"
AP_CHANNEL="${AP_CHANNEL:-6}"
AP_IP="${AP_IP:-10.42.0.1}"
AP_GATEWAY="${AP_GATEWAY:-10.42.0.1}"
AP_PREFIX="${AP_PREFIX:-24}"
CON_NAME="sss-hotspot"

echo "=== Substation Squirrel Sleuths — WiFi AP Setup ==="
echo "  Interface : $WIFI_IFACE"
echo "  SSID      : $AP_SSID"
echo "  Band      : $AP_BAND  Channel: $AP_CHANNEL"
echo "  IP        : $AP_IP/$AP_PREFIX"
echo ""

# ── Preflight checks ───────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  echo "ERROR: This script must be run as root (sudo)." >&2
  exit 1
fi

if ! command -v nmcli &>/dev/null; then
  echo "ERROR: nmcli not found. Install NetworkManager." >&2
  exit 1
fi

if ! nmcli device status | grep -q "$WIFI_IFACE"; then
  echo "ERROR: Wireless interface '$WIFI_IFACE' not found." >&2
  echo "Available interfaces:"
  nmcli device status
  exit 1
fi

# ── Remove any existing SSS hotspot connection ─────────────────────────────
if nmcli connection show "$CON_NAME" &>/dev/null; then
  echo "Removing existing '$CON_NAME' connection..."
  nmcli connection delete "$CON_NAME"
fi

# ── Create the access point ────────────────────────────────────────────────
echo "Creating WiFi hotspot '$AP_SSID' on $WIFI_IFACE..."

nmcli connection add \
  type wifi \
  ifname "$WIFI_IFACE" \
  con-name "$CON_NAME" \
  autoconnect yes \
  ssid "$AP_SSID" \
  -- \
  wifi.mode ap \
  wifi.band "$AP_BAND" \
  wifi.channel "$AP_CHANNEL" \
  wifi-sec.key-mgmt wpa-psk \
  wifi-sec.psk "$AP_PASSWORD" \
  ipv4.method shared \
  ipv4.addresses "${AP_IP}/${AP_PREFIX}" \
  ipv4.gateway "$AP_GATEWAY" \
  ipv6.method disabled

# ── Bring it up ────────────────────────────────────────────────────────────
echo "Activating hotspot..."
nmcli connection up "$CON_NAME"

echo ""
echo "=== WiFi AP is live ==="
echo "  SSID     : $AP_SSID"
echo "  Password : $AP_PASSWORD"
echo "  Jetson IP: $AP_IP"
echo "  Dashboard: http://$AP_IP:5000"
echo ""
echo "Users can now connect to '$AP_SSID' and open http://$AP_IP:5000 in a browser."
