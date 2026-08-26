# Elevator Runner (standalone local tool)

Lives in the Dex_Elevator project but is **fully standalone** — it does NOT import
the Dex_Elevator code or the CloudPlatform app. Run it on a laptop that can reach
both the AutoXing cloud and the AGX.

**What it does:** pick navigation points, mark ONE as the elevator point, set how
many times to run (a loop), and dispatch the robot via AutoXing. When the robot
finishes a task and stops at the elevator point **within tolerance**, it SSHes into
the AGX and runs the validated button press (`press_buttons.py <floors> --go --lift`).

## Setup
    cd Dex_Elevator/elevator_runner
    cp config.example.json config.json          # optional non-secret overrides
    # put AutoXing creds in .env (gitignored):
    #   AUTOXING_APP_ID="..."  AUTOXING_APP_SECRET="..."  AUTOXING_APP_CODE="APPCODE ..."
    #   ROBOT_SERIAL="<chassis serial>"   AGX_SSH="dex-agx"
    python3 -m pip install flask requests
    python3 server.py                            # http://127.0.0.1:8765/

Restart the server after editing `.env` (it is read at startup).

## Reaching the AGX for the press
`AGX_SSH` is an ssh target/alias. On this Mac the working route is the USB-ethernet
cable, via an alias in `~/.ssh/config`:

    Host dex-agx
        HostName fe80::9b69:ea6e:4c17:294a%%en5   # IPv6 link-local over the cable
        User jetson
        AddressFamily inet6
        StrictHostKeyChecking no

If the cable is unplugged there is no route and the press cannot run. Alternatives:
plug the cable back in (the adapter re-appears as `en5`), or use Tailscale
(`AGX_SSH="jetson@100.122.187.11"` once Tailscale is up on both ends).

## Security
- Credentials live ONLY in `.env` (gitignored, chmod 600). Never committed.
- The tool is LOCKED to a single robot (`ROBOT_SERIAL`); any other serial → HTTP 403.
- **Dry run is ON by default** — it walks the whole flow (load points, plan, simulate
  arrival) without dispatching the robot or pressing. Untick only when the area is
  clear and you are watching.

## Retry policy (the arm can't reach past ~10 cm)
The arm re-detects the panel live and absorbs a few cm of docking error, but past its
reach envelope it cannot reach the buttons. So after each task finishes we read the
robot's real `x,y[,yaw]` and compare to the elevator point:
- within tolerance (default 8 cm / 12°) → press.
- didn't finish (obstacle / auto-cancel / timeout) → back off, retry (spot may clear).
- finished but off → bounded retries, then stop and surface it (a systematic docking
  offset won't fix by repeating — tighten that point's docking instead).
- retries are capped; it never loops forever.

## AutoXing endpoints used (mirror CloudPlatform robot_api.py / task_poc.py)
- `POST /auth/v1.1/token` (header Authorization: appCode, body sign=md5(appId+ts+appSecret)) → X-Token
- `POST /map/v1.1/poi/list` — points (elevator waiting point = type 6/28)
- `POST /task/v3/create` — dispatch (taskPts: areaId,x,y,yaw,stopRadius,stepActs)
- `GET  /task/v3/{id}` — status (isFinish / isCancel)
- `POST /task/v3/{id}/cancel`
- `GET  /robot/v2.0/{serial}/state` — live x,y,yaw,moveState

## Running ON the AGX (on-robot deployment)
Runs on the robot itself, so it does not depend on a laptop being connected and the
press is a **local** call (no SSH).

    # on the AGX (ssh in), in the repo:
    cd ~/Dex_Elevator/elevator_runner
    printf 'AUTOXING_APP_ID="..."\nAUTOXING_APP_SECRET="..."\nAUTOXING_APP_CODE="APPCODE ..."\nROBOT_SERIAL="1352512k06960Og"\nAGX_SSH="local"\n' > .env
    chmod 600 .env
    /usr/bin/python3 -m pip install --user flask requests
    /usr/bin/python3 server.py            # binds 127.0.0.1:8765 (localhost only)

- Set **`AGX_SSH="local"`** so the press runs `press_buttons.py` directly (no SSH).
- The page binds to **localhost only**. To open it from your laptop, SSH-tunnel:
      ssh -L 8765:127.0.0.1:8765 dex-agx      # then browse http://127.0.0.1:8765/
  (Do NOT bind 0.0.0.0 — the page has no auth and this is a shared machine.)
- **Shared-machine caveat:** the AGX hosts other people's projects. Keep `.env` at
  `chmod 600`, never commit it (it is gitignored), and remove it if the tool is retired.
