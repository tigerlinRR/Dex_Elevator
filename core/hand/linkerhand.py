"""LinkerHand O6 driven through the RealMan arm's tool-side RS485 port.

The hand is a Modbus RTU slave hanging off the arm's tool connector, so every
command is a Modbus write forwarded by the arm controller. Two prerequisites,
both configured through the arm:

    rm_set_tool_voltage(3)              -> 24 V on the tool connector (the hand
                                           is dead without it)
    rm_set_modbus_mode(1, 115200, tmo)  -> tool-board RS485 as an RTU *master*

**This DEX has a GEN-3 controller** (``rm_get_robot_info()`` reports
``robot_controller_version: 3``), so the gen-4 API family — ``rm_set_tool_rs485_mode``,
``rm_write_modbus_rtu_registers``, ``rm_read_modbus_rtu_holding_registers`` — all
return ``-4`` ("三代控制器不支持该接口"). Use the gen-3 calls instead:
``rm_set_modbus_mode`` / ``rm_write_registers`` / ``rm_read_multiple_holding_registers``,
which take ``rm_peripheral_read_write_params_t``.

Register map (reverse-engineered from the working ADAM script ``~/hand.py`` on
the robot; addresses are the same, only the transport differs):

    read  addr 0..5    -> CURRENT joint positions
    write addr 0..5    -> TARGET joint positions   (same addresses, different meaning)
    write addr 12..17  -> per-joint speed

The same-address-read/write-different-meaning bit is the easy thing to get wrong:
``hand.py``'s ``__init__`` also pokes address 6, but that is some other parameter —
the joint targets go to address 0 (see its ``write_all``). Writing 6 lands fine and
reads back, but the hand never moves.

Joint order, 0 = closed .. 255 = open:
    0 thumb curl | 1 thumb rotation | 2 index | 3 middle | 4 ring | 5 pinky

NOTE: the ADAM robot reaches its hands over an **xArm** Modbus bridge
(``~/hand.py``, ``XArmAPI.getset_tgpio_modbus_data``). That code does NOT work
here — the DEX has RealMan arms. The protocol is reused; the transport is not.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from core.robot.realman import RealmanArm

try:  # SDK is optional at import time, like everywhere else in this repo
    from Robotic_Arm.rm_ctypes_wrap import (  # type: ignore
        rm_peripheral_read_write_params_t,
    )

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - laptop / no SDK
    _SDK_AVAILABLE = False


# Modbus register addresses on the hand. Reading 0..5 gives the CURRENT positions;
# writing the SAME addresses sets the TARGET positions.
REG_POS = 0             # read: current position | write: target position
REG_SPEED = 12          # per-joint speed
NUM_JOINTS = 6

# Modbus ``port``: 0 = controller RS485, 1 = tool-board RS485 (where the hand is).
TOOL_PORT = 1
CONTROLLER_PORT = 0

# Slave address of the hand. 0x27/0x28 (right/left) on the ADAM robot — treat as
# a starting guess here and confirm with probe().
DEFAULT_SLAVE_RIGHT = 0x27
DEFAULT_SLAVE_LEFT = 0x28

DEFAULT_SPEED = 200

# Presets copied verbatim from the robot's ~/hand.py (tuned on real hardware).
# "point" is the one that matters here: index finger extended, everything else
# curled — the pressing posture.
POSES: dict[str, list[int]] = {
    "open": [255, 60, 255, 255, 255, 255],
    "fist": [0, 60, 0, 0, 0, 0],
    "point": [0, 60, 255, 0, 0, 0],
    "peace": [0, 60, 255, 255, 0, 0],
    "thumbs_up": [255, 60, 0, 0, 0, 0],
    "ok": [0, 60, 0, 255, 255, 255],
}


class LinkerHandError(RuntimeError):
    """A hand command was rejected by the arm controller or the hand."""


class LinkerHand:
    """LinkerHand O6 on the tool port of a :class:`~core.robot.realman.RealmanArm`.

    Shares the arm's existing connection — the tool RS485 lives on the arm
    controller, so opening a second connection just to talk to the hand would
    contend with motion commands.

    Typical use::

        arm = RealmanArm(side="right"); arm.connect()
        hand = LinkerHand(arm)
        hand.power_on()          # 24 V + RS485; hand does not move
        print(hand.read_joints())
        hand.pose("point")       # hand MOVES
    """

    def __init__(
        self,
        arm: "RealmanArm",
        slave: int = DEFAULT_SLAVE_RIGHT,
        port: int = TOOL_PORT,
    ):
        if not _SDK_AVAILABLE:
            raise ImportError(
                "RealMan SDK not importable (`Robotic_Arm`); run this on the robot."
            )
        self.arm = arm
        self.slave = slave
        self.port = port

    # -- setup --------------------------------------------------------------
    def power_on(self, baudrate: int = 115200, timeout_100ms: int = 5,
                 settle: float = 0.5) -> None:
        """Enable 24 V on the tool connector and put its RS485 into RTU-master mode.

        Does NOT move the hand. Safe to call repeatedly. ``timeout_100ms`` is the
        controller's Modbus response timeout in units of 100 ms (must be > 0).
        """
        sdk = self.arm._require()
        code = sdk.rm_set_tool_voltage(3)  # 0:0V 1:5V 2:12V 3:24V
        if code != 0:
            raise LinkerHandError(f"rm_set_tool_voltage(24V) failed (code={code}).")
        # Gen-3 API. The gen-4 rm_set_tool_rs485_mode() returns -4 on this controller.
        code = sdk.rm_set_modbus_mode(self.port, baudrate, timeout_100ms)
        if code != 0:
            raise LinkerHandError(
                f"rm_set_modbus_mode(port={self.port}, {baudrate}) failed (code={code})."
            )
        time.sleep(settle)  # the hand's MCU needs a moment after power-up

    def power_off(self) -> None:
        """Cut tool-connector power (the hand goes limp)."""
        self.arm._require().rm_set_tool_voltage(0)

    # -- raw modbus ---------------------------------------------------------
    def _params(self, address: int, num: int):
        return rm_peripheral_read_write_params_t(
            port=self.port, address=address, device=self.slave, num=num
        )

    def _read(self, address: int, num: int) -> Optional[list[int]]:
        """Read ``num`` holding registers, or ``None`` if the device didn't answer.

        The SDK hands back the raw payload as BYTES (``num * 2``, big-endian per
        register), NOT as register values — verified on hardware: asking for 6
        registers returns 12 bytes ``[0,254, 0,254, ...]`` meaning ``[254] * 6``.
        Gen-3 API caps this at 12 registers (2 < num < 13).
        """
        code, data = self.arm._require().rm_read_multiple_holding_registers(
            self._params(address, num)
        )
        if code != 0 or data is None:
            return None
        raw = [int(v) for v in list(data)]
        if len(raw) >= num * 2:
            return [(raw[2 * i] << 8) | raw[2 * i + 1] for i in range(num)]
        return raw[:num]  # fall back if a future SDK returns register values directly

    def _write(self, address: int, values: list[int]) -> None:
        """Write ``values`` one register at a time, starting at ``address``.

        Deliberately NOT ``rm_write_registers`` (the multi-register call): on this
        gen-3 controller it returns 0 but the values never reach the device — read
        the registers back and they are unchanged. ``rm_write_single_register``
        does land, verified on hardware, so we loop. Six writes at ~30 ms is
        plenty fast for posing a hand.
        """
        sdk = self.arm._require()
        for i, v in enumerate(values):
            v = max(0, min(255, int(v)))
            code = sdk.rm_write_single_register(self._params(address + i, 1), v)
            if code != 0:
                raise LinkerHandError(
                    f"register write to addr {address + i} failed (code={code}); "
                    f"slave=0x{self.slave:02x} port={self.port}. "
                    "Run probe() to find the right slave/port."
                )
            time.sleep(0.03)

    # -- read-only ----------------------------------------------------------
    def read_joints(self) -> Optional[list[int]]:
        """Current joint positions (6 x 0..255), or ``None`` if the hand didn't answer."""
        return self._read(REG_POS, NUM_JOINTS)

    def is_responding(self) -> bool:
        return self.read_joints() is not None

    # -- motion -------------------------------------------------------------
    def set_speed(self, speed: int = DEFAULT_SPEED) -> None:
        self._write(REG_SPEED, [speed] * NUM_JOINTS)

    def write_joints(self, joints: list[int]) -> None:
        """Command all six joints (0 = closed, 255 = open). The hand MOVES."""
        if len(joints) != NUM_JOINTS:
            raise ValueError(f"expected {NUM_JOINTS} joint values, got {len(joints)}")
        self._write(REG_POS, joints)

    def pose(self, name: str) -> None:
        """Move to a named preset (see :data:`POSES`). The hand MOVES."""
        if name not in POSES:
            raise ValueError(f"unknown pose {name!r}; have {sorted(POSES)}")
        self.write_joints(POSES[name])

    # -- discovery ----------------------------------------------------------
    def probe(self, slaves: Optional[list[int]] = None,
              ports: Optional[list[int]] = None) -> list[tuple[int, int, list[int]]]:
        """Find which (port, slave) actually answers. READ-ONLY — nothing moves.

        The hand's Modbus slave address on this robot isn't documented anywhere we
        control (0x27/0x28 are the ADAM robot's values), so sweep the likely ones on
        both RS485 ports. Returns ``(port, slave, joints)`` for every combination
        that answered, and latches onto the first hit.
        """
        slaves = slaves or [DEFAULT_SLAVE_RIGHT, DEFAULT_SLAVE_LEFT, 0x01, 0x02]
        ports = ports or [TOOL_PORT, CONTROLLER_PORT]
        found: list[tuple[int, int, list[int]]] = []
        saved = (self.slave, self.port)
        try:
            for p in ports:
                # Each RS485 port needs its own RTU-master setup before it will talk.
                self.port = p
                try:
                    self.power_on(settle=0.2)
                except LinkerHandError as exc:
                    print(f"  [probe] port {p} unusable: {exc}")
                    continue
                for sl in slaves:
                    self.slave = sl
                    joints = self.read_joints()
                    if joints and any(joints):  # all-zero is usually a phantom reply
                        found.append((p, sl, joints))
        finally:
            self.slave, self.port = saved
        if found:
            self.port, self.slave, _ = found[0]
        return found
