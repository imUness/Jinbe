"""
Tag hunt for the Viam rover: snake-pattern search, fast approach, touch the tag,
then go find the next one until all tags are found.

Setup:
  pip install viam-sdk opencv-python-headless numpy pillow
"""
import asyncio
import os
import time

import cv2
import numpy as np
from viam.components.base import Base
from viam.components.camera import Camera
from viam.media.utils.pil import viam_to_pil_image
from viam.proto.common import Vector3
from viam.robot.client import RobotClient

# ---------------- CONFIG ----------------
ADDRESS = os.getenv("VIAM_ADDRESS", "")
API_KEY = os.getenv("VIAM_API_KEY", "PUT_YOUR_KEY_HERE")
API_KEY_ID = os.getenv("VIAM_API_KEY_ID", "PUT_YOUR_KEY_ID_HERE")

CAMERA_NAME = "camera-1"
BASE_NAME = "viam_base"

NUM_TAGS = 4              # stop after this many different tags were touched
TARGET_IDS = None         # None = any tag id, or e.g. [1, 2, 3, 4]
TIMEOUT_SECONDS = 400

# ---- Speed (all 0..1) ----
CRUISE_POWER = 0.75       # driving straight while searching
APPROACH_POWER = 0.85     # driving toward a tag when it's far
MIN_APPROACH_POWER = 0.45 # never slower than this while approaching
SPIN_POWER = 0.6          # turning power (search turns)
STEER_GAIN = 0.9          # how hard it steers toward the tag while driving
MAX_STEER = 0.6

# ---- Reaching / touching the tag ----
CENTER_TOLERANCE = 0.12   # error below this = "centered", drive straight
SPIN_ONLY_ERROR = 0.45    # tag this far off-center: spin in place first
TOUCH_SIZE = 0.65         # tag width / frame width at which we ram forward
LOST_PUSH_SIZE = 0.30     # if tag disappears (too close) after being this big, keep going
PUSH_SECONDS = 1.2        # final push forward to make contact (increase if it stops short)
PUSH_POWER = 0.6
BACKUP_SECONDS = 1.2      # reverse after touching
LOST_FRAMES_GIVE_UP = 6   # frames without the tag (when still far) before resuming search

# ---- Snake search pattern ----
LEG_SECONDS = 4.0         # how long to drive straight per leg (tune to your room size)
SHIFT_SECONDS = 1.0       # sideways shift between legs (about one camera field of view)
TURN_90_SECONDS = 0.9     # time to turn ~90 degrees at SPIN_POWER (calibrate!)
TURN_180_SECONDS = 1.8    # time to turn ~180 degrees
SCAN_STEP_SECONDS = 0.15  # turn in small steps and look between each

# If the rover turns the wrong way toward the code, change this to -1.
TURN_SIGN = 1
# ----------------------------------------


def power(base: Base, linear: float, angular: float):
    return base.set_power(
        linear=Vector3(x=0, y=linear, z=0),
        angular=Vector3(x=0, y=0, z=angular),
    )


async def look(camera: Camera, detector, visited: set):
    """Return (tag_id, error, size) of the biggest unvisited tag, or None."""
    img = await camera.get_image(mime_type="image/jpeg")
    frame = cv2.cvtColor(np.array(viam_to_pil_image(img)), cv2.COLOR_RGB2GRAY)
    w = frame.shape[1]
    corners, ids, _ = detector.detectMarkers(frame)
    if ids is None:
        return None
    best = None
    for c, i in zip(corners, ids.flatten()):
        i = int(i)
        if i in visited:
            continue
        if TARGET_IDS is not None and i not in TARGET_IDS:
            continue
        pts = c.reshape(-1, 2)
        size = float(pts[:, 0].max() - pts[:, 0].min()) / w
        error = (float(pts[:, 0].mean()) - w / 2) / (w / 2)  # -1 left .. +1 right
        if best is None or size > best[2]:
            best = (i, error, size)
    return best


async def rotate_scan(base, camera, detector, visited, direction, seconds):
    """Turn in small steps, looking after each one. Returns a detection or None."""
    end = time.time() + seconds
    while time.time() < end:
        await power(base, 0, direction * TURN_SIGN * SPIN_POWER)
        await asyncio.sleep(SCAN_STEP_SECONDS)
        await base.stop()
        det = await look(camera, detector, visited)
        if det:
            return det
    return None


async def drive_scan(base, camera, detector, visited, seconds):
    """Drive straight while continuously looking. Returns a detection or None."""
    await power(base, CRUISE_POWER, 0)
    end = time.time() + seconds
    while time.time() < end:
        det = await look(camera, detector, visited)
        if det:
            return det
    await base.stop()
    return None


async def approach(base, camera, detector, visited):
    """Drive to the tag and touch it. Returns the tag id when touched, else None."""
    lost = 0
    last = None
    while True:
        det = await look(camera, detector, visited)

        if det is None:
            lost += 1
            # Tag vanished while big/close -> it's right in front of us: finish the push.
            if last and last[2] >= LOST_PUSH_SIZE and abs(last[1]) < 0.4:
                return await touch(base, last[0])
            if lost >= LOST_FRAMES_GIVE_UP:
                await base.stop()
                return None
            continue

        lost = 0
        last = det
        tag_id, error, size = det
        steer = -TURN_SIGN * max(-MAX_STEER, min(MAX_STEER, STEER_GAIN * error))
        print(f"Tag {tag_id}: error={error:+.2f} size={size:.2f}")

        if size >= TOUCH_SIZE and abs(error) < 0.3:
            return await touch(base, tag_id)

        if abs(error) > SPIN_ONLY_ERROR:
            # far off-center: rotate in place so we don't lose it
            await power(base, 0, steer * 1.2)
        else:
            # drive and steer at the same time; slow down gradually as it gets close
            speed = APPROACH_POWER * (1.0 - 0.5 * min(size / TOUCH_SIZE, 1.0))
            speed = max(MIN_APPROACH_POWER, speed)
            if abs(error) < CENTER_TOLERANCE:
                steer = 0.0
            await power(base, speed, steer)


async def touch(base, tag_id):
    print(f"Pushing forward to touch tag {tag_id}")
    await power(base, PUSH_POWER, 0)
    await asyncio.sleep(PUSH_SECONDS)
    await base.stop()
    print(f"TOUCHED tag {tag_id}")
    await asyncio.sleep(0.3)
    # back away and turn around so we don't stare at the same tag
    await power(base, -0.5, 0)
    await asyncio.sleep(BACKUP_SECONDS)
    await base.stop()
    await power(base, 0, TURN_SIGN * SPIN_POWER)
    await asyncio.sleep(TURN_180_SECONDS)
    await base.stop()
    return tag_id


async def main():
    opts = RobotClient.Options.with_api_key(api_key=API_KEY, api_key_id=API_KEY_ID)
    robot = await RobotClient.at_address(ADDRESS, opts)
    camera = Camera.from_robot(robot, CAMERA_NAME)
    base = Base.from_robot(robot, BASE_NAME)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )

    visited: set = set()
    start = time.time()
    turn_dir = +1  # alternates each leg: snake left, then right, ...

    async def handle(det):
        """Go touch a detected tag. Returns True if touched."""
        tag_id = await approach(base, camera, detector, visited)
        if tag_id is not None:
            visited.add(tag_id)
            print(f"Found {len(visited)}/{NUM_TAGS}: {sorted(visited)}")
            return True
        return False

    try:
        while len(visited) < NUM_TAGS and time.time() - start < TIMEOUT_SECONDS:
            # 1) straight leg, scanning the whole time
            print("Driving straight and scanning...")
            det = await drive_scan(base, camera, detector, visited, LEG_SECONDS)
            if det:
                await handle(det)
                continue

            # 2) turn 90, scanning as we turn
            print("Turning and scanning...")
            det = await rotate_scan(base, camera, detector, visited, turn_dir, TURN_90_SECONDS)
            if det:
                await handle(det)
                continue

            # 3) short shift sideways to the next lane
            det = await drive_scan(base, camera, detector, visited, SHIFT_SECONDS)
            if det:
                await handle(det)
                continue

            # 4) second 90 turn -> now facing back down the next lane
            det = await rotate_scan(base, camera, detector, visited, turn_dir, TURN_90_SECONDS)
            if det:
                await handle(det)
                continue

            turn_dir *= -1  # zig-zag like a snake

        if len(visited) >= NUM_TAGS:
            print(f"DONE. Touched all tags: {sorted(visited)}")
        else:
            print(f"Timed out. Touched {sorted(visited)}")
    finally:
        await base.stop()
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
