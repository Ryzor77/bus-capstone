"""
Bus cabin door heat-loss model WITH an under-floor PCM air heater.

The PCM store sits (insulated) under the cabin. Starting PCM_PREHEAT_S seconds
before each door opens, a fan blows cabin air through the PCM and the warmed air
enters the cabin at a vent zone just inside that door. It keeps running until
the door closes. The same hour is simulated with and without the PCM so the
two can be compared.

Edit the PARAMETERS section, then run:  python3 bus_door_heatmap_pcm.py
Needs: numpy, matplotlib.  Output: printed comparison + bus_door_heatmap_pcm.png

All numbers are illustrative placeholders. Replace them with your own data.
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================== PARAMETERS ===============================
# --- door pattern ---
DOOR_OPEN_TIME_S = 20
FRONT_OPENINGS_PER_HR = 24
REAR_OPENINGS_PER_HR = 14

# --- weather and comfort ---
T_OUT = -5.0
T_SET = 21.0

# --- bus and heating system ---
HEATER_MAX_KW = 20.0
UA_KW_PER_K = 0.25
C_AIR_KJ_PER_K = 80.0
C_MASS_KJ_PER_K = 800.0
HA_KW_PER_K = 4.0
HEATER_GAIN_KW_PER_K = 10.0

# --- PCM under the cabin ---
PCM_MATERIAL = "erythritol"      # "erythritol", "erythritol-urea eutectic", "sodium acetate trihydrate"
PCM_CAPACITY_KWH = 6.0           # usable latent heat when fully charged at the depot
PCM_PREHEAT_S = 10               # fan starts this many seconds before a door opens
PCM_AIRFLOW_M3S = 0.5            # air through the PCM (split if both doors are active)
PCM_MAX_KW = 15.0                # max heat the PCM heat exchanger can deliver
PCM_APPROACH_K = 10.0            # supply air stays this far below the PCM melting point
PCM_MAX_SUPPLY_C = 60.0          # supply air cap (mixed down for safety/comfort)
PCM_VENT_DEPTH_M = 0.75          # vent zone depth inside the door-side wall
PCM_FAN_KW = 0.3                 # electrical power of the fan
PCM_LOSS_KW_PER_K = 0.003        # insulation loss while the PCM is hot
PCM_SYSTEM_MASS_FACTOR = 1.4     # container + insulation + heat exchanger vs PCM mass
PCM_RECHARGE_EFF = 0.98          # depot heater efficiency

# --- geometry ---
LENGTH_M, WIDTH_M = 12.0, 2.5
DOOR_WIDTH_M = 1.25
DOOR_CENTERS_M = (1.875, 9.125)
DOOR_DRAFT_MS = 0.5
MIXING_M2_PER_S = 0.04

# --- simulation and figure ---
SIM_HOURS = 1.0
DT = 0.1
SNAPSHOT_OPENING = 3                 # which front-door opening to photograph
SNAPSHOT_OFFSETS_S = (-5, 15, 35)    # seconds relative to that opening's start
SHOW_WINDOW = False
# ==========================================================================

PCM_LIBRARY = {  # latent heat only; approximate values, replace with sourced data
    "erythritol": (118.0, 340.0),
    "erythritol-urea eutectic": (82.0, 227.0),
    "sodium acetate trihydrate": (58.0, 250.0),
}
T_MELT, L_KJ_KG = PCM_LIBRARY[PCM_MATERIAL]
RHO_CP = 1.2                                   # kJ/(m3 K)
DX = 0.25
NX, NY = int(LENGTH_M / DX), int(WIDTH_M / DX)
N = NX * NY
CA, CM, HA, UAC = C_AIR_KJ_PER_K / N, C_MASS_KJ_PER_K / N, HA_KW_PER_K / N, UA_KW_PER_K / N
CELL_VOL = C_AIR_KJ_PER_K / RHO_CP / N         # m3 of air per cell
TEND = SIM_HOURS * 3600
COUNTS = (FRONT_OPENINGS_PER_HR * SIM_HOURS, REAR_OPENINGS_PER_HR * SIM_HOURS)
OFFSETS = (0.5, 0.25)
PCM_MASS = PCM_CAPACITY_KWH * 3600 / L_KJ_KG
T_SUPPLY = min(T_MELT - PCM_APPROACH_K, PCM_MAX_SUPPLY_C)


def is_open(k, t):
    n = int(COUNTS[k])
    if n == 0:
        return False
    s = TEND / n
    o = OFFSETS[k] * s
    if t < o:
        return False
    i = int((t - o) // s)
    return i < n and (t - o) - i * s < DOOR_OPEN_TIME_S


def opening_start(k, m):
    s = TEND / int(COUNTS[k])
    return OFFSETS[k] * s + (m - 1) * s


jj, ii = np.mgrid[0:NY, 0:NX]
depth = (NY - 1 - jj + 0.5) * DX
x_cell = (ii + 0.5) * DX
W_JET, IN_DOOR, ZONE = [], [], []
for xc in DOOR_CENTERS_M:
    xr = x_cell - xc
    v = DOOR_DRAFT_MS * np.exp(-depth / 2.5) * np.exp(-(xr / (0.6 * DOOR_WIDTH_M)) ** 2)
    W_JET.append(v * DT / DX)
    IN_DOOR.append(np.abs(xr[0]) < DOOR_WIDTH_M / 2 + 0.01)
    ZONE.append((np.abs(xr) < DOOR_WIDTH_M / 2 + 0.01) & (depth < PCM_VENT_DEPTH_M))


def diffuse(T):
    P = np.pad(T, 1, mode="edge")
    lap = P[1:-1, :-2] + P[1:-1, 2:] + P[:-2, 1:-1] + P[2:, 1:-1] - 4 * T
    return T + MIXING_M2_PER_S * DT / DX ** 2 * lap


def door_draft(T, k):
    outside_row = np.where(IN_DOOR[k], T_OUT, T[-1, :])
    Tn = (1 - W_JET[k]) * T + W_JET[k] * np.vstack([T, outside_row])[1:]
    return Tn, CA * (T.sum() - Tn.sum())


def pcm_blow(Ta, E_kwh, t):
    """Blow PCM-warmed air into the vent zone of every door that is about to open or open.
    Returns new field, heat delivered (kJ), fan on (bool)."""
    active = [k for k in (0, 1) if is_open(k, t) or is_open(k, t + PCM_PREHEAT_S)]
    if not active or E_kwh <= 0:
        return Ta, 0.0, False
    flow = PCM_AIRFLOW_M3S / len(active)
    deltas, q = [], 0.0
    for k in active:
        z = ZONE[k]
        f = flow * DT / (z.sum() * CELL_VOL)
        d = np.where(z, f * (T_SUPPLY - Ta), 0.0)
        deltas.append(d)
        q += CA * d.sum()
    cap = min(PCM_MAX_KW * DT, E_kwh * 3600)       # kJ available this step
    scale = min(1.0, cap / q) if q > 0 else 0.0
    for d in deltas:
        Ta = Ta + scale * d
    return Ta, q * scale, True


def run(use_pcm, snap_steps):
    Ta = np.full((NY, NX), T_SET)
    Tm = np.full((NY, NX), (HA_KW_PER_K * T_SET + UA_KW_PER_K * T_OUT) / (HA_KW_PER_K + UA_KW_PER_K))

    def thermo(Ta, Tm):
        P = np.clip(HEATER_GAIN_KW_PER_K * (T_SET - Ta.mean()), 0, HEATER_MAX_KW)
        q = HA * (Tm - Ta)
        return Ta + DT * (q + P / N) / CA, Tm + DT * (-q - UAC * (Tm - T_OUT)) / CM, P

    for _ in range(3000):
        Ta, Tm, _ = thermo(diffuse(Ta), Tm)

    E = PCM_CAPACITY_KWH if use_pcm else 0.0
    E0 = E
    r = dict(heater=0.0, env=0.0, door=0.0, pcm=0.0, pcm_loss=0.0, fan=0.0, snaps={}, ts=[], tmean=[])
    for s in range(int(TEND / DT)):
        t = s * DT
        Ta = diffuse(Ta)
        Ta, Tm, P = thermo(Ta, Tm)
        r["heater"] += P * DT / 3600
        r["env"] += UA_KW_PER_K * (Tm.mean() - T_OUT) * DT / 3600
        if use_pcm:
            Ta, q, fan = pcm_blow(Ta, E, t)
            E -= q / 3600
            r["pcm"] += q / 3600
            if fan:
                r["fan"] += PCM_FAN_KW * DT / 3600
            if E > 0:
                loss = min(PCM_LOSS_KW_PER_K * (T_MELT - T_OUT) * DT / 3600, E)
                E -= loss
                r["pcm_loss"] += loss
        for k in (0, 1):
            if is_open(k, t):
                Ta, lost = door_draft(Ta, k)
                r["door"] += lost / 3600
        if s in snap_steps:
            r["snaps"][s] = Ta.copy()
        if s % 50 == 0:
            r["ts"].append(t / 60)
            r["tmean"].append(Ta.mean())
    r["tmean"] = np.array(r["tmean"])
    r["ts"] = np.array(r["ts"])
    r["E_end"] = E
    r["recharge"] = (E0 - E) / PCM_RECHARGE_EFF
    return r


def main():
    if int(COUNTS[0]) >= SNAPSHOT_OPENING:
        t0 = opening_start(0, SNAPSHOT_OPENING)
    else:
        t0 = 600.0
    snap_steps = [int((t0 + o) / DT) for o in SNAPSHOT_OFFSETS_S]
    A = run(False, snap_steps)
    B = run(True, snap_steps)

    heater_saved = A["heater"] - B["heater"]
    net_saved = heater_saved - B["fan"]
    weight = PCM_MASS * PCM_SYSTEM_MASS_FACTOR
    print(f"\n=== {PCM_MATERIAL}, {PCM_CAPACITY_KWH:g} kWh PCM, {PCM_PREHEAT_S}s pre-blow ===")
    print(f"Door {DOOR_OPEN_TIME_S}s | front {FRONT_OPENINGS_PER_HR}/h, rear {REAR_OPENINGS_PER_HR}/h | outside {T_OUT} C\n")
    print(f"{'':<26}{'no PCM':>10}{'with PCM':>10}")
    rows = [("Heater energy (kWh)", A["heater"], B["heater"]), ("PCM heat delivered (kWh)", 0, B["pcm"]),
            ("Door heat loss (kWh)", A["door"], B["door"]), ("Envelope loss (kWh)", A["env"], B["env"]),
            ("Lowest mean air (C)", A["tmean"].min(), B["tmean"].min()), ("Average mean air (C)", A["tmean"].mean(), B["tmean"].mean())]
    for name, a, b in rows:
        print(f"{name:<26}{a:>10.2f}{b:>10.2f}")
    print(f"\nHeater energy saved:      {heater_saved:.2f} kWh (fan uses {B['fan']:.2f} kWh) -> net {net_saved:.2f} kWh")
    print(f"PCM heat lost in storage: {B['pcm_loss']:.2f} kWh | PCM left at end: {100 * B['E_end'] / PCM_CAPACITY_KWH:.0f}%")
    print(f"Depot recharge needed:    {B['recharge']:.2f} kWh")
    print(f"PCM mass {PCM_MASS:.0f} kg, about {weight:.0f} kg with container, insulation and heat exchanger")
    if B["pcm"] > 0:
        print(f"Net battery energy saved per 100 kg carried: {100 * net_saved / weight:.2f} kWh")

    lo, hi = max(T_OUT, T_SET - 15), T_SET + 1
    fig = plt.figure(figsize=(12, 2.4 * len(snap_steps) + 3))
    gs = fig.add_gridspec(len(snap_steps) + 1, 2, height_ratios=[1] * len(snap_steps) + [1.5])
    im = None
    for r_i, (s, off) in enumerate(zip(snap_steps, SNAPSHOT_OFFSETS_S)):
        for c_i, (res, label) in enumerate(((A, "no PCM"), (B, "with PCM"))):
            ax = fig.add_subplot(gs[r_i, c_i])
            im = ax.imshow(res["snaps"][s], extent=[0, LENGTH_M, 0, WIDTH_M], cmap="turbo", vmin=lo, vmax=hi)
            for k, xc in enumerate(DOOR_CENTERS_M):
                o = is_open(k, s * DT)
                ax.plot([xc - DOOR_WIDTH_M / 2, xc + DOOR_WIDTH_M / 2], [0, 0], lw=7,
                        color="orange" if o else "gray", solid_capstyle="butt", clip_on=False)
            ax.set_title(f"{label}: {off:+d} s from opening {SNAPSHOT_OPENING}", fontsize=9, loc="left")
            ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes, label="air temperature (degC)", shrink=0.8, pad=0.02)
    ax = fig.add_subplot(gs[-1, :])
    ax.plot(A["ts"], A["tmean"], label="mean cabin air, no PCM")
    ax.plot(B["ts"], B["tmean"], label="mean cabin air, with PCM")
    ax.axhline(T_SET, ls="--", color="gray", label="setpoint")
    ax.set_xlabel("time (min)")
    ax.set_ylabel("degC")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig("bus_door_heatmap_pcm.png", dpi=130, bbox_inches="tight")
    print("\nSaved figure: bus_door_heatmap_pcm.png")
    if SHOW_WINDOW:
        plt.show()


if __name__ == "__main__":
    main()
