import pandas as pd
import numpy as np
import os

# =============================
# 1. LOAD DATA
# =============================
base = r"C:\Users\arunbalaji.madasamy\Downloads\DIY_EKF"

acc_df = pd.read_csv(base + r"\accel_clean.csv")
gyro_df = pd.read_csv(base + r"\gyro_clean.csv")
gps_df = pd.read_csv(base + r"\gps_clean.csv")
gpsvel_df = pd.read_csv(base + r"\gpsvel_clean.csv")

# Convert accel → m/s²
acc = acc_df[['ax','ay','az']].values
gyro = gyro_df[['gx','gy','gz']].values

lat = gps_df['lat'].values
lon = gps_df['lon'].values
alt = gps_df['alt'].values

gps_vel = gpsvel_df[['vn','ve','vd']].values

gps_vel_enu = np.zeros_like(gps_vel)

gps_vel_enu[:,0] = gps_vel[:,1]      # East = Ve
gps_vel_enu[:,1] = gps_vel[:,0]      # North = Vn
gps_vel_enu[:,2] = -gps_vel[:,2]     # Up = -Vd


# =============================
# 2. GPS → ENU CONVERSION
# =============================
R_EARTH = 6378137.0

lat0 = np.deg2rad(lat[0])
lon0 = np.deg2rad(lon[0])
alt0 = alt[0]

gps_enu = []

for i in range(len(lat)):
    dlat = np.deg2rad(lat[i]) - lat0
    dlon = np.deg2rad(lon[i]) - lon0
    dalt = alt[i] - alt0

    x_e = dlon * R_EARTH * np.cos(lat0)
    y_n = dlat * R_EARTH
    z_u = dalt

    gps_enu.append([x_e, y_n, z_u])

gps_enu = np.array(gps_enu)


# =============================
# 3. CONSTANTS
# =============================
dt = 1/50

# IMPORTANT: try ZERO first (most IMUs already include gravity)
g = np.array([0, 0, 0])

print(np.mean(acc, axis=0))

# Increased noise (to prevent drift explosion)
SIG_W_A = 1.0
SIG_W_G = 0.02
SIG_B_A = 0.0001
SIG_B_G = 0.00001


# =============================
# 4. INITIALIZATION
# =============================
x = np.zeros(15)

# Initialize from GPS
x[0:3] = gps_enu[0]
x[3:6] = gps_vel[0]

P = np.eye(15)
P[0:3,0:3] *= 10.0
P[3:6,3:6] *= 1.0

R_meas = np.eye(6)
R_meas[0:3,0:3] *= 5.0
R_meas[3:6,3:6] *= 1.0


# =============================
# 5. ROTATION MATRIX
# =============================
def rotation_matrix(theta):
    roll, pitch, yaw = theta

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


# =============================
# 6. Q MATRIX
# =============================
def compute_Q(dt):
    var_a  = SIG_W_A**2
    var_g  = SIG_W_G**2
    var_ba = SIG_B_A**2
    var_bg = SIG_B_G**2

    Qc = np.diag([
        var_a, var_a, var_a,
        var_g, var_g, var_g,
        var_ba, var_ba, var_ba,
        var_bg, var_bg, var_bg
    ])

    G = np.zeros((15,12))
    G[3:6,0:3] = np.eye(3)
    G[6:9,3:6] = np.eye(3)
    G[9:12,6:9] = np.eye(3)
    G[12:15,9:12] = np.eye(3)

    return G @ Qc @ G.T * dt


# =============================
# 7. F MATRIX
# =============================
def compute_F(dt):
    F = np.eye(15)
    F[0:3,3:6] = np.eye(3) * dt
    F[3:6,9:12] = -np.eye(3) * dt
    F[6:9,12:15] = -np.eye(3) * dt
    return F


H = np.zeros((6,15))
H[0:6,0:6] = np.eye(6)


# =============================
# 8. EKF LOOP
# =============================
N = len(acc)
gps_index = 0

estimates = []

for k in range(N):

    # ---- Prediction ----
    p = x[0:3]
    v = x[3:6]
    theta = x[6:9]
    b_a = x[9:12]
    b_g = x[12:15]

    a_meas = acc[k]
    w_meas = gyro[k]

    R_mat = rotation_matrix(theta)

    p = p + v * dt
    v = v + (R_mat @ (a_meas - b_a) + g) * dt
    theta = theta + (w_meas - b_g) * dt

    x[0:3] = p
    x[3:6] = v
    x[6:9] = theta

    F = compute_F(dt)
    Q = compute_Q(dt)
    P = F @ P @ F.T + Q

    # ---- GPS UPDATE ----
    
    #gps_step = int(50/30)  # ≈ 1.66 → use 2
    gps_step = 2   # 50 Hz IMU / 30 Hz GPS ≈ 2

    if (k % gps_step == 0) and (gps_index < len(gps_enu)):

        z = np.hstack((gps_enu[gps_index], gps_vel_enu[gps_index]))

        y = z - H @ x
        S = H @ P @ H.T + R_meas
        K = P @ H.T @ np.linalg.inv(S)

        x = x + K @ y
        P = (np.eye(15) - K @ H) @ P

        gps_index += 1

    estimates.append(x.copy())

estimates = np.array(estimates)

# =============================
# 9. RESULT
# =============================
print("Final position:", estimates[-1,0:3])