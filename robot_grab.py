import os
# Force OSMesa for both MuJoCo and PyOpenGL BEFORE importing robosuite
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"

import cv2
import mediapipe as mp
import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config

# 1. Initialize Fast MediaPipe Hands (model_complexity=0 for maximum FPS)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)

# 2. Initialize Robosuite at low 160x120 internal resolution for high-FPS OSMesa rendering
config = load_controller_config(default_controller="OSC_POSE")
env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=config,
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="frontview",
    camera_heights=120,
    camera_widths=160,
    control_freq=20,
    ignore_done=True, # Prevents the episode timeout crash
)
obs = env.reset()
init_cube_pos = obs["cube_pos"].copy()

# 3. Open WSL Webcam Bridge with MJPG
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

smooth_tx, smooth_ty = 320.0, 240.0
alpha = 0.55

# Invisible reference point on screen corresponding to the cube's position
CUBE_SCREEN_X = 320
CUBE_SCREEN_Y = 300

is_grasping = False

print("Clean Low-Res Full-Color Teleop Ready! Press 'R' to reset, 'ESC' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(image_rgb)

    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]

    # Check if the cube is currently lifted off the table
    cube_lifted = cube_pos[2] > (init_cube_pos[2] + 0.015)

    # 7-DoF Action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
    action = np.zeros(7)

    if hand_results.multi_hand_landmarks:
        hand_landmarks = hand_results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Smooth hand coordinates
        raw_tx, raw_ty = index_mcp.x * w, index_mcp.y * h
        smooth_tx = alpha * raw_tx + (1.0 - alpha) * smooth_tx
        smooth_ty = alpha * raw_ty + (1.0 - alpha) * smooth_ty
        tx, ty = int(smooth_tx), int(smooth_ty)

        # Pinch detection with hysteresis so gripper stays closed while moving
        px1, py1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        px2, py2 = int(index_tip.x * w), int(index_tip.y * h)
        pinch_dist = int(np.hypot(px2 - px1, py2 - py1))
        threshold = 58 if is_grasping else 45
        is_grasping = pinch_dist < threshold

        # Draw Hand Pinch Line & Knuckle Tracking Dot Only (No Cube Box)
        pinch_color = (0, 255, 0) if is_grasping else (0, 165, 255)
        cv2.line(frame, (px1, py1), (px2, py2), pinch_color, 3)
        cv2.circle(frame, (tx, ty), 9, (255, 255, 0), -1)

        # --- DIRECT 1:1 POSITION MAPPING (UP / DOWN / LEFT / RIGHT) ---
        offset_y = (tx - CUBE_SCREEN_X) * 0.0012
        offset_z = (CUBE_SCREEN_Y - ty) * 0.0014

        in_cube_zone = abs(tx - CUBE_SCREEN_X) < 35 and abs(ty - CUBE_SCREEN_Y) < 35

        # Keep depth aligned with the cube
        target_x = cube_pos[0] if not cube_lifted else init_cube_pos[0]

        # Snap Y onto the cube when close before lifting, then allow full Left/Right movement
        if in_cube_zone and not cube_lifted:
            target_y = cube_pos[1]
        else:
            target_y = init_cube_pos[1] + offset_y

        # Lowered Z-floor (-0.020m) so the gripper fingers go around the cube
        base_z = init_cube_pos[2] - 0.020
        target_z = np.clip(base_z + offset_z, init_cube_pos[2] - 0.024, init_cube_pos[2] + 0.35)

        # High-gain P-Controller for fast 1:1 tracking while keeping wrist upright
        Kp_pos = 10.0
        action[0] = np.clip((target_x - eef_pos[0]) * Kp_pos, -1.0, 1.0)
        action[1] = np.clip((target_y - eef_pos[1]) * Kp_pos, -1.0, 1.0)
        action[2] = np.clip((target_z - eef_pos[2]) * Kp_pos, -1.0, 1.0)
        action[6] = 1.0 if is_grasping else -1.0

        grip_text = "CLOSED (GRABBING)" if is_grasping else "OPEN"
        cv2.putText(frame, f"GRIPPER: {grip_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, pinch_color, 2)
    else:
        # Keep gripper closed if MediaPipe loses hand for a frame during a pinch
        action[6] = 1.0 if is_grasping else -1.0
        cv2.putText(frame, "NO HAND - HOLDING POSITION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # Step MuJoCo Physics every single frame (fast at 160x120)
    obs, reward, done, info = env.step(action)

    # Upscale 160x120 render to 640x480 to match the webcam window
    sim_img = cv2.flip(obs["frontview_image"], 0)
    sim_img = cv2.cvtColor(sim_img, cv2.COLOR_RGB2BGR)
    sim_img = cv2.resize(sim_img, (640, 480), interpolation=cv2.INTER_NEAREST)
    cv2.putText(sim_img, "1:1 FAST TELEOP (R: Reset)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    dashboard = np.hstack((frame, sim_img))
    cv2.imshow("Vision Teleoperation Dashboard", dashboard)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r'):
        obs = env.reset()
        init_cube_pos = obs["cube_pos"].copy()
        is_grasping = False

cap.release()
cv2.destroyAllWindows()
env.close()
