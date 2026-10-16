# Monocular Hand-to-Robot Teleoperation Challenge (MuJoCo / Robosuite)

## My Approach & Iterative Journey

When I first started this challenge, I wanted to explore different ways of bridging human motion to a robotic manipulator. Since this was my first time building a real-time vision-based teleoperation pipeline, it took a few iterations to get right:

1. **The Phone Video Attempt:** At first, I tried recording a manipulation video on my phone to use as an offline dataset. However, synchronizing offline video frames with a live simulation loop proved too rigid for a responsive control interface.
2. **The IP Webcam Network Hurdle:** Next, I tried setting up my phone as a live IP webcam stream over Wi-Fi to get a clean first-person POV. Due to local router firewalls and network security/routing restrictions (`No route to host`), the stream kept dropping. 
3. **The Mouse-Tracking Experiment:** I even experimented with tracking my computer mouse as a proxy object to sync with the simulation cube. While it worked technically, it lacked the natural intuitiveness of hand manipulation.
4. **The Final Solution:** I finally settled on using the laptop webcam combined with MediaPipe hand tracking and direct task-space retargeting. It provides a reliable, low-latency setup where the user's hand drives the robot's end-effector directly.

---

## Mathematical Formulation & Kinematics

Studying robotics this year has given me a deeper appreciation for how mathematical models govern robot motion, stability, and joint control. With the help of my professor, I learned how to apply operational space control equations to map Cartesian hand movements to the Franka Emika Panda arm.

Instead of trying to manually compute joint angles for a 7-DoF arm (which involves complex non-linear inverse kinematics), the system uses an **Operational Space Control (`OSC_POSE`)** formulation:

1. **Task-Space Mapping:** The user's hand coordinates $(tx, ty)$ from the webcam frame are scaled into 3D operational space offsets:
   $$\Delta y = (tx - X_{center}) \cdot k_y$$
   $$\Delta z = (Y_{center} - ty) \cdot k_z$$

2. **Proportional Control Law:** The target end-effector velocity/action vector is calculated using a proportional controller acting on the position error between the current end-effector position $X_{eef}$ and the target position $X_{target}$:
   $$a_{pos} = K_p \cdot (X_{target} - X_{eef})$$
   where $K_p = 10.0$ provides fast, stable convergence without overshoot.

3. **Jacobian Integration:** Under the hood, Robosuite's controller uses the manipulator Jacobian matrix $J(q)$ to map these task-space forces into joint torques/velocities:
   $$\tau = J(q)^T F_{desired} + \text{compensations}$$
   This ensures that the robot maintains its balance, respects joint limits, and keeps its wrist upright while following the user's hand.

---

## What Worked & What Didn't

* **What Didn't Work:** 
  * Relying on 2D depth estimation from palm size caused the robot to lunge forward unpredictably due to perspective changes when tilting the hand.
  * Using unconstrained position controllers (`OSC_POSITION`) allowed the wrist to buckle under high gains against the simulation table.
  * Brief self-occlusions during finger pinching caused MediaPipe to drop tracking for a single frame, instantly opening the gripper.

* **What Worked:**
  * Anchoring the depth plane to the table and shifting to 1:1 operational space position control (`OSC_POSE`).
  * Implementing dual-threshold pinch hysteresis (45px to close, 58px to release) and state latching so the gripper securely holds the object even if hand tracking flickers.
  * Optimizing performance by running MuJoCo's offscreen renderer at a low internal resolution (`160x120`) via `OSMesa`, enabling smooth 30 FPS execution on consumer CPU hardware.

---

## How to Run

### Prerequisites
```bash
sudo chmod 666 /dev/video*
conda activate libero_env
pip install -r requirements.txt
