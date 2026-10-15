"""
QR hunt for the Viam rover - "measure, turn once, drive straight":
- snake-pattern search (drive + scan, turn + scan)
- when a QR code is seen: stop, take a few sharp frames, compute
  bearing (angle) and distance, turn once, drive straight
- drives ~70% of the distance, re-measures, then a final committed drive to touch it
- optional distance sensor for obstacles

Setup:
  pip install viam-sdk opencv-python-headless numpy

Calibrate speed and turn rate first:  python jinbey.py calibrate
"""
import asyncio
import math
import sys
import time

import cv2
import numpy as np
from viam.components.base import Base
from viam.components.camera import Camera
from viam.components.sensor import Sensor
from viam.proto.common import Vector3
from viam.robot.client import RobotClient

# ---------------- CONFIG ----------------
ADDRESS = ""
API_KEY = ""
API_KEY_ID = ""

CAMERA_NAME = "camera-1"
BASE_NAME = "viam_base"

NUM_TAGS = 3               # how many QR codes to touch (set 4 if you have 4)
TIMEOUT_SECONDS = 500
COOLDOWN_SECONDS = 10      # after a touch, ignore codes whose text can't be read

# ---- Direction ----
FORWARD_SIGN = 1           # 1 = drive same way as W in Viam; -1 = camera at the BACK
TURN_SIGN = 1              # set to -1 if it turns away from the code
ROTATE_IMAGE_180 = False   # set True if the camera image is upside-down

# ---- Power ----
CRUISE_POWER = 0.35        # driving while searching
APPROACH_POWER = 0.40      # driving toward the code (speed below is measured at THIS power)
SPIN_POWER = 0.35          # turning (turn rate below is measured at THIS power)

# ---- Geometry & calibration (measure these, they decide how accurate it is) ----
QR_WIDTH_M = 0.12          # real width of the printed QR code, in meters (measure it!)
HFOV_DEG = 65.0            # camera horizontal field of view (webcams: 60-78)
SPEED_M_PER_S = 0.15       # rover speed at APPROACH_POWER  (run "calibrate")
TURN_RATE_DEG_PER_S = 60.0 # rover turn speed at SPIN_POWER (run "calibrate")

# ---- Approach behavior ----
SETTLE_SECONDS = 0.30      # wait after stopping so the frame is sharp
MEASURE_TRIES = 8          # frames tried when measuring
MEASURE_NEED = 3           # detections needed (median is used)
LEG_FRACTION = 0.70        # drive this fraction of the distance, then re-measure
COMMIT_DISTANCE_M = 0.60   # closer than this: final straight drive to touch
PUSH_EXTRA_SECONDS = 0.7   # extra driving at the end to make contact (raise if it stops short)
MIN_TURN_DEG = 3.0         # ignore turns smaller than this
MAX_LEG_SECONDS = 8.0
MAX_LEGS = 5
TOUCH_SIZE = 0.60          # code this big in the image = basically touching

# ---- Touching / leaving ----
BACKUP_SECONDS = 0.8
BACKUP_POWER = 0.3

# ---- Optional distance sensor (e.g. ultrasonic). Leave None if you don't have one. ----
SENSOR_NAME = None
SENSOR_KEY = "distance"
OBSTACLE_DISTANCE_M = 0.25
AVOID_BACKUP_SECONDS = 0.8

# ---- Snake search pattern (calibrate the turn times!) ----
LEG_SECONDS = 3.0
SHIFT_SECONDS = 1.5
TURN_90_SECONDS = 1.5
TURN_180_SECONDS = 3.0
SCAN_STEP_SECONDS = 0.25
# ----------------------------------------


class Ctx:
    def __init__(self, base, camera, sensor):
        self.base = base
        self.camera = camera
        self.sensor = sensor
        self.detectors = []
        if hasattr(cv2, "QRCodeDetectorAruco"):
            self.detectors.append(cv2.QRCodeDetectorAruco())
        self.detectors.append(cv2.QRCodeDetector())
        self.visited = set()
        self.touched = 0
        self.ignore_until = 0.0


def power(base, linear, angular):
    """linear > 0 = toward what the camera sees."""
    return base.set_power(
        linear=Vector3(x=0, y=linear * FORWARD_SIGN, z=0),
        angular=Vector3(x=0, y=0, z=angular),
    )


async def halt(ctx):
    await ctx.base.stop()


async def look(ctx):
    """Return (text, error, size) for the biggest usable QR code, or None.
    error: -1 (left edge) .. +1 (right edge); size: code width / frame width."""
    result = await ctx.camera.get_images()
    images = result[0] if isinstance(result, tuple) else result
    if not images:
        return None
    frame = cv2.imdecode(np.frombuffer(images[0].data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    if ROTATE_IMAGE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)

    w = frame.shape[1]
    text, pts = "", None
    for det in ctx.detectors:
        try:
            text, pts, _ = det.detectAndDecode(frame)
        except cv2.error:
            continue
        if pts is not None:
            break
    if pts is None:
        return None

    pts = np.array(pts).reshape(-1, 2)
    size = float(pts[:, 0].max() - pts[:, 0].min()) / w
    error = (float(pts[:, 0].mean()) - w / 2) / (w / 2)
    if size < 0.02:
        return None

    text = (text or "").strip()
    if text and text in ctx.visited:
        return None
    if not text and time.time() < ctx.ignore_until:
        return None
    return text, error, size


def bearing_deg(error):
    """Angle from the camera's center line to the code (positive = right)."""
    return math.degrees(math.atan(error * math.tan(math.radians(HFOV_DEG) / 2)))


def distance_m(size):
    """Distance from the apparent width of the code (pinhole camera model)."""
    return QR_WIDTH_M / (2 * math.tan(math.radians(HFOV_DEG) / 2) * size)


async def too_close(ctx):
    if ctx.sensor is None:
        return False
    try:
        r = await ctx.sensor.get_readings()
        val = r.get(SENSOR_KEY)
        if val is None:
            val = next(v for v in r.values() if isinstance(v, (int, float)))
        return float(val) < OBSTACLE_DISTANCE_M
    except Exception:
        return False


async def avoid(ctx, direction):
    print("Obstacle! backing up and turning away")
    await halt(ctx)
    await power(ctx.base, -BACKUP_POWER, 0)
    await asyncio.sleep(AVOID_BACKUP_SECONDS)
    await ctx.base.stop()
    await power(ctx.base, 0, direction * TURN_SIGN * SPIN_POWER)
    await asyncio.sleep(TURN_90_SECONDS)
    await ctx.base.stop()


async def turn_by(ctx, deg_right):
    """Turn by an angle (positive = right/clockwise, negative = left)."""
    if abs(deg_right) < MIN_TURN_DEG:
        return
    direction = -1 if deg_right > 0 else +1
    seconds = abs(deg_right) / TURN_RATE_DEG_PER_S
    print(f"  turning {deg_right:+.0f} deg ({seconds:.2f}s)")
    await power(ctx.base, 0, direction * TURN_SIGN * SPIN_POWER)
    await asyncio.sleep(seconds)
    await ctx.base.stop()
    await asyncio.sleep(0.2)


async def drive_for(ctx, seconds, power_val):
    """Drive straight for a time. Returns True if the sensor says we're blocked."""
    seconds = min(seconds, MAX_LEG_SECONDS)
    print(f"  driving straight {seconds:.2f}s")
    await power(ctx.base, power_val, 0)
    end = time.time() + seconds
    blocked = False
    while time.time() < end:
        await asyncio.sleep(0.05)
        if ctx.sensor is not None and await too_close(ctx):
            blocked = True
            break
    await ctx.base.stop()
    await asyncio.sleep(0.2)
    return blocked


async def rotate_scan(ctx, direction, seconds):
    end = time.time() + seconds
    while time.time() < end:
        await power(ctx.base, 0, direction * TURN_SIGN * SPIN_POWER)
        await asyncio.sleep(SCAN_STEP_SECONDS)
        await ctx.base.stop()
        await asyncio.sleep(0.15)
        det = await look(ctx)
        if det:
            return det
    return None


async def drive_scan(ctx, seconds):
    """Drive straight while scanning. Returns (detection, blocked)."""
    await power(ctx.base, CRUISE_POWER, 0)
    end = time.time() + seconds
    while time.time() < end:
        det = await look(ctx)
        if det:
            await halt(ctx)
            return det, False
        if await too_close(ctx):
            await halt(ctx)
            return None, True
    await halt(ctx)
    return None, False


async def measure(ctx):
    """Standing still: take several frames, return median (text, error, size) or None."""
    await asyncio.sleep(SETTLE_SECONDS)
    hits, text = [], ""
    for _ in range(MEASURE_TRIES):
        det = await look(ctx)
        if det:
            hits.append(det)
            text = text or det[0]
            if len(hits) >= MEASURE_NEED:
                break
    if not hits:
        return None
    errs = sorted(h[1] for h in hits)
    sizes = sorted(h[2] for h in hits)
    mid = len(hits) // 2
    return text, errs[mid], sizes[mid]


async def finish_touch(ctx, text):
    name = text or "unknown"
    print(f"TOUCHED code '{name}'")
    if text:
        ctx.visited.add(text)
    ctx.touched += 1
    ctx.ignore_until = time.time() + COOLDOWN_SECONDS
    await asyncio.sleep(0.3)
    await power(ctx.base, -BACKUP_POWER, 0)
    await asyncio.sleep(BACKUP_SECONDS)
    await ctx.base.stop()
    await power(ctx.base, 0, TURN_SIGN * SPIN_POWER)
    await asyncio.sleep(TURN_180_SECONDS)
    await ctx.base.stop()
    return True


async def approach(ctx):
    """Measure -> turn once -> drive straight. Returns True if touched."""
    await halt(ctx)
    fails = 0
    last_dir = +1
    for leg in range(1, MAX_LEGS + 1):
        m = await measure(ctx)
        if m is None:
            fails += 1
            print("  can't see the code, sweeping a little...")
            if fails >= 3:
                return False
            if await rotate_scan(ctx, last_dir, 0.6) is None:
                await rotate_scan(ctx, -last_dir, 1.2)
            continue

        text, error, size = m
        dist = distance_m(size)
        ang = bearing_deg(error)
        last_dir = -1 if error > 0 else +1
        print(f"Leg {leg}: QR '{text or '?'}' angle={ang:+.0f} deg, distance={dist:.2f} m (size={size:.2f})")

        await turn_by(ctx, ang)

        if dist <= COMMIT_DISTANCE_M or size >= TOUCH_SIZE:
            # final committed drive: the remaining distance plus a little extra to make contact
            blocked = await drive_for(ctx, dist / SPEED_M_PER_S + PUSH_EXTRA_SECONDS, APPROACH_POWER)
            return await finish_touch(ctx, text)

        blocked = await drive_for(ctx, LEG_FRACTION * dist / SPEED_M_PER_S, APPROACH_POWER)
        if blocked:
            await avoid(ctx, +1)
            return False
    return False


async def calibrate(ctx):
    print("CALIBRATION - put the rover on the floor with free space ahead.")
    print("1) Driving straight for 2 seconds at APPROACH_POWER... measure how far it goes (meters).")
    await asyncio.sleep(2)
    await power(ctx.base, APPROACH_POWER, 0)
    await asyncio.sleep(2.0)
    await ctx.base.stop()
    print("   -> SPEED_M_PER_S = distance_in_meters / 2")
    await asyncio.sleep(3)
    print("2) Turning left for 2 seconds at SPIN_POWER... measure the angle it turned (degrees).")
    await power(ctx.base, 0, TURN_SIGN * SPIN_POWER)
    await asyncio.sleep(2.0)
    await ctx.base.stop()
    print("   -> TURN_RATE_DEG_PER_S = angle_in_degrees / 2")


async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    robot = await RobotClient.at_address(ADDRESS, opts)
    camera = Camera.from_robot(robot, CAMERA_NAME)
    base = Base.from_robot(robot, BASE_NAME)
    sensor = Sensor.from_robot(robot, SENSOR_NAME) if SENSOR_NAME else None
    ctx = Ctx(base, camera, sensor)

    try:
        if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
            await calibrate(ctx)
            return

        start = time.time()
        turn_dir = +1
        phases = [("drive", LEG_SECONDS), ("turn", TURN_90_SECONDS),
                  ("drive", SHIFT_SECONDS), ("turn", TURN_90_SECONDS)]

        while ctx.touched < NUM_TAGS and time.time() - start < TIMEOUT_SECONDS:
            for kind, secs in phases:
                if kind == "drive":
                    print("Driving straight and scanning...")
                    det, blocked = await drive_scan(ctx, secs)
                else:
                    print("Turning and scanning...")
                    det, blocked = await rotate_scan(ctx, turn_dir, secs), False

                if det:
                    await halt(ctx)
                    if await approach(ctx):
                        print(f"Touched {ctx.touched}/{NUM_TAGS}: {sorted(ctx.visited)}")
                    break
                if blocked:
                    await avoid(ctx, turn_dir)
                    turn_dir *= -1
                    break
            else:
                turn_dir *= -1  # snake: alternate left/right each lane

        if ctx.touched >= NUM_TAGS:
            print(f"DONE. Touched {ctx.touched} codes: {sorted(ctx.visited)}")
        else:
            print(f"Stopped. Touched {ctx.touched}: {sorted(ctx.visited)}")
    finally:
        await base.stop()
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
