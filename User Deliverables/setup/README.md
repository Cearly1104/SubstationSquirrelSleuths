# SSS Jetson Setup

How the user-facing dashboard is deployed on the Jetson Orin Nano.

## One-time manual setup

These steps only need to be done once, when the Jetson is first imaged.

1. **Enable hotspot mode** — open *Settings → Wi-Fi → ⋮ menu → Turn On Wi-Fi Hotspot*.
   NetworkManager remembers the setting and will bring the AP back automatically
   on every boot. The AP comes up at **10.42.0.1**.

2. **Clone the project** to `~/SSS/` so the layout is:
   ```
   /home/SubstationSquirrelSleuths/SSS/
     Codebase/            # detection / recording pipeline
     User Deliverables/   # this Flask dashboard
   ```

3. **Create the venv** at `~/venvs/sss` and install requirements:
   ```bash
   python3 -m venv ~/venvs/sss
   ~/venvs/sss/bin/pip install --upgrade pip
   ~/venvs/sss/bin/pip install -r ~/SSS/User\ Deliverables/requirements.txt
   ```

## Installing / updating the Flask service

After the project is on disk, run:

```bash
sudo bash ~/SSS/User\ Deliverables/setup/install.sh
```

This will:
- Refresh pip and reinstall `requirements.txt` into the existing venv.
- Copy `sss-flask.service` into `/etc/systemd/system/`.
- Enable the service so it auto-starts on boot.
- Restart the service immediately.

Re-run `install.sh` any time you pull new changes or edit the unit file.

## Service environment

`sss-flask.service` sets these environment variables for `app.py`:

| Variable                | Value                                                              |
| ----------------------- | ------------------------------------------------------------------ |
| `SSS_DATA_DIR`          | `~/SSS/User Deliverables/data`                                     |
| `SSS_ACCOUNTS_FILE`     | `~/SSS/User Deliverables/accounts.json`                            |
| `SSS_SETTINGS_FILE`     | `~/SSS/Codebase/settings.json` (read/written by the Settings page) |
| `SSS_SETTINGS_DEFAULTS` | `~/SSS/Codebase/settings.default.json`                             |
| `SSS_SECRET_KEY`        | Flask session signing key — **change before deploying for real**   |

## Useful commands

```bash
systemctl status sss-flask           # current state
journalctl -u sss-flask -f           # live logs
sudo systemctl restart sss-flask     # bounce after a config change
sudo systemctl stop sss-flask        # stop without disabling
sudo systemctl disable sss-flask     # don't start on next boot
```

## Accessing the dashboard

1. Connect a phone or laptop to the Jetson's hotspot SSID.
2. Browse to **http://10.42.0.1:5000**.
   The captive-portal probes built into `app.py` will also try to redirect
   most devices automatically.
3. Log in with an account from `accounts.json`.
