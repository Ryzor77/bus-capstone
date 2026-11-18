"""
3D bus cabin door heat-loss model WITH an under-floor PCM air heater.

The cabin is split into 0.25 m cubes (length x width x height). Every cube holds
an air temperature. Cubes that touch the bus shell (walls, floor, roof) also hold
a shell / interior-mass temperature. Each time step applies, in this order:

  1. Air mixing          turbulent mixing treated as diffusion (horizontal + vertical)
  2. Buoyancy            warm air sitting under colder air overturns (convective adjustment)
  3. Heater + shell      heater power, air <-> shell convection, shell -> outside conduction (UA)
  4. PCM fan             PCM-warmed supply air into a vent zone just inside each door
  5. Door exchange flow  while a door is open: cold air in low, warm cabin air out high

Starting PCM preheat seconds before each door opens, a fan blows cabin air through
the insulated under-floor PCM and returns it at a vent zone just inside that door.
The same period is simulated with and without the PCM so the two can be compared,
and an energy balance is checked for each run.

Streamlit app.  Run locally:  streamlit run bus_door_pcm_3d.py
Needs: streamlit, numpy, matplotlib, plotly (plotly is only for the interactive 3D view).

All numbers are illustrative placeholders. Replace them with your own data.
"""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
except ImportError:          # the app still runs, just without the 3D view
    go = None

# =============================== CONSTANTS ===============================
DX = 0.25            # m, size of each cube (all geometry sliders step in 0.25 m)
RHO_CP = 1.2         # kJ/(m3 K), volumetric heat capacity of air (1.2 kg/m3 x 1.0 kJ/kg K)
G = 9.81             # m/s2
T_ABS = 273.15       # K offset
DT_MAX = 0.1         # s, largest time step allowed (smaller if stability needs it)
WARMUP_S = 600.0     # s, doors-closed settling period before the timed run
SAMPLE_S = 5.0       # s, interval for the time-series plots
OFFSETS = (0.5, 0.25)  # fraction of the opening spacing before the first front / rear opening

PCM_LIBRARY = {  # (melting point degC, latent heat kJ/kg); approximate, replace with sourced data
    "erythritol": (118.0, 340.0),
    "erythritol-urea eutectic": (82.0, 227.0),
    "sodium acetate trihydrate": (58.0, 250.0),
}
HEATER_LOCATIONS = ["Floor, along side walls", "Overhead, along side walls", "Spread evenly"]

DEFAULTS = dict(
    # door pattern
    door_open_s=20, front_per_hr=24, rear_per_hr=14,
    # weather and comfort
    t_out=-5.0, t_set=21.0, head_z=1.1,
    # PCM
    pcm_material="erythritol", pcm_kwh=6.0, pcm_preheat_s=10, pcm_airflow=0.5, pcm_max_kw=15.0,
    pcm_approach_k=10.0, pcm_max_supply_c=60.0, pcm_vent_depth=0.75, pcm_vent_height=0.5,
    pcm_fan_kw=0.3, pcm_loss_kw_per_k=0.003, pcm_mass_factor=1.4, pcm_recharge_eff=0.98,
    # heating system and shell
    heater_max_kw=20.0, heater_kp=10.0, heater_ki_per_min=0.0, heater_location=HEATER_LOCATIONS[0],
    ua=0.25, c_mass=800.0, ha=4.0,
    # geometry
    length=12.0, width=2.5, height=2.25, door_width=1.25, door_height=2.0, front_x=1.875, rear_x=2.875,
    # air movement
    door_cd=0.6, door_flow_mult=1.0, door_penetration=1.5, mix_h=0.04, mix_v_factor=0.5, buoy_tau=2.0,
    # simulation and figures
    sim_hours=1.0, snap_opening=3, snap_offsets=(-5, 15, 35),
)


# =============================== MODEL SETUP ===============================
def door_flow(m, t_in):
    """Two-way buoyancy-driven flow through a large opening (m3/s each way):
       Q = M * Cd/3 * W * H * sqrt(g * H * |T_in - T_out| / T_mean_K)"""
    p = m["p"]
    t_mean_k = 0.5 * (t_in + p["t_out"]) + T_ABS
    q = (p["door_flow_mult"] * p["door_cd"] / 3 * p["door_width"] * m["door_h"]
         * np.sqrt(G * m["door_h"] * abs(t_in - p["t_out"]) / t_mean_k))
    return q


def build_model(p):
    """Grid, shell weights, heater cells, door flow fields, PCM vent zones and time step."""
    m = dict(p=p)
    NX, NY, NZ = (int(round(p[k] / DX)) for k in ("length", "width", "height"))
    m.update(NX=NX, NY=NY, NZ=NZ, N=NX * NY * NZ)
    m["Vc"] = DX ** 3                       # m3 per cube
    m["CA"] = RHO_CP * m["Vc"]              # kJ/K of air per cube
    m["C_air_total"] = m["CA"] * m["N"]     # kJ/K of all cabin air
    xc = (np.arange(NX) + 0.5) * DX          # cube centres: x along the bus, front = 0
    yc = (np.arange(NY) + 0.5) * DX          # y = depth from the door-side wall
    zc = (np.arange(NZ) + 0.5) * DX          # z = height above the floor
    m.update(xc=xc, yc=yc, zc=zc)
    K, J, I = np.meshgrid(np.arange(NZ), np.arange(NY), np.arange(NX), indexing="ij")  # arrays are [z, y, x]

    # --- shell: every cube face on the boundary carries an equal share of hA, UA and C_mass ---
    faces = ((I == 0).astype(int) + (I == NX - 1) + (J == 0) + (J == NY - 1) + (K == 0) + (K == NZ - 1))
    S = faces > 0
    w = faces[S] / faces.sum()
    m.update(S=S, HA_s=p["ha"] * w, UA_s=p["ua"] * w, CM_s=p["c_mass"] * w, shell_area=faces.sum() * DX ** 2)

    # --- heater cells (heater power split evenly over them) ---
    loc = p["heater_location"]
    if loc.startswith("Floor"):
        H = (K == 0) & ((J == 0) | (J == NY - 1))
    elif loc.startswith("Overhead"):
        H = (K == NZ - 1) & ((J == 0) | (J == NY - 1))
    else:
        H = np.ones((NZ, NY, NX), bool)
    m["heat_w"] = H / H.sum()

    # --- door exchange flow from a streamfunction psi(y, z) on cube corners ---
    m["door_h"] = min(p["door_height"], NZ * DX)
    m["door_centres"] = (p["front_x"], p["length"] - p["rear_x"])
    z_c = np.arange(NZ + 1) * DX
    g = np.where(z_c < m["door_h"], np.sin(np.pi * z_c / m["door_h"]), 0.0)
    g /= g.max()                                         # so the inflow equals Q exactly
    y_c = np.arange(NY + 1) * DX
    h = np.exp(-y_c / p["door_penetration"]) * (1 - y_c / (NY * DX))   # zero at the far wall
    m["doors"] = []
    for xd in m["door_centres"]:
        cols = np.where(np.abs(xc - xd) < p["door_width"] / 2 + 0.01)[0]
        psi = np.outer(g, h) / len(cols)                 # m3/s per column for Q = 1 m3/s
        Fy = psi[1:, :] - psi[:-1, :]                    # flow through y-faces, + = away from door  (NZ, NY+1)
        Fz = -(psi[:, 1:] - psi[:, :-1])                 # flow through z-faces, + = upward          (NZ+1, NY)
        out = (np.maximum(-Fy[:, :-1], 0) + np.maximum(Fy[:, 1:], 0)
               + np.maximum(-Fz[:-1, :], 0) + np.maximum(Fz[1:, :], 0))
        colmask = np.zeros(NX, bool)
        colmask[cols] = True
        zone = colmask[None, None, :] & (yc[None, :, None] < p["pcm_vent_depth"]) & (zc[:, None, None] < p["pcm_vent_height"])
        m["doors"].append(dict(cols=slice(cols[0], cols[-1] + 1), Fy=Fy[:, :, None], Fz=Fz[:, :, None],
                               out_max=out.max(), zone=zone, zone_vol=zone.sum() * m["Vc"],
                               zone_dims=(zone.any(axis=(0, 2)).sum() * DX, zone.any(axis=(1, 2)).sum() * DX)))

    # --- PCM ---
    m["T_melt"], m["L_kj_kg"] = PCM_LIBRARY[p["pcm_material"]]
    m["pcm_mass"] = p["pcm_kwh"] * 3600 / m["L_kj_kg"]
    m["T_supply"] = min(m["T_melt"] - p["pcm_approach_k"], p["pcm_max_supply_c"])

    # --- door schedule ---
    m["TEND"] = p["sim_hours"] * 3600
    m["counts"] = (int(p["front_per_hr"] * p["sim_hours"]), int(p["rear_per_hr"] * p["sim_hours"]))

    # --- time step from the stability limits ---
    a_h, a_v = p["mix_h"], p["mix_h"] * p["mix_v_factor"]
    m["Q_worst"] = door_flow(m, p["t_set"] + 10)        # door flow is capped at this value
    dt_mix = 0.8 * DX ** 2 / (2 * (2 * a_h + a_v))      # explicit diffusion limit
    dt_door = min(0.8 * m["Vc"] / (m["Q_worst"] * d["out_max"]) for d in m["doors"])  # upwind Courant limit
    dt_pcm = min(0.5 * d["zone_vol"] / p["pcm_airflow"] for d in m["doors"])         # vent air replaced <= 50 %/step
    m["DT"] = min(DT_MAX, dt_mix, dt_door, dt_pcm)
    m["k_ankle"] = 0
    m["k_head"] = min(int(p["head_z"] / DX), NZ - 1)
    return m


def is_open(m, k, t):
    n = m["counts"][k]
    if n == 0:
        return False
    s = m["TEND"] / n
    o = OFFSETS[k] * s
    if t < o:
        return False
    i = int((t - o) // s)
    return i < n and (t - o) - i * s < m["p"]["door_open_s"]


def opening_start(m, k, nth):
    s = m["TEND"] / m["counts"][k]
    return OFFSETS[k] * s + (nth - 1) * s


# =============================== PHYSICS STEPS ===============================
def run(m, use_pcm, snap_steps):
    p = m["p"]
    DT, CA, Vc, S = m["DT"], m["CA"], m["Vc"], m["S"]
    NZ, NY, NX = m["NZ"], m["NY"], m["NX"]
    t_out, t_set = p["t_out"], p["t_set"]
    mix_h = p["mix_h"] * DT / DX ** 2
    mix_v = mix_h * p["mix_v_factor"]
    kp, ki = p["heater_kp"], p["heater_ki_per_min"] / 60.0
    pmax = p["heater_max_kw"]
    HA_s, UA_s, CM_s, heat_w = m["HA_s"], m["UA_s"], m["CM_s"], m["heat_w"]
    ctrl = dict(I=0.0)

    def mixing(T):
        """T += dt * [a_h (Txx + Tyy) + a_v Tzz]; edge padding = zero flux at the shell."""
        out = T.copy()
        for axis, coef in ((2, mix_h), (1, mix_h), (0, mix_v)):           # x, y, z
            d = coef * np.diff(T, axis=axis)                               # flow of heat between neighbours
            hi = [slice(None)] * 3; lo = [slice(None)] * 3
            hi[axis] = slice(None, -1); lo[axis] = slice(1, None)
            out[tuple(hi)] += d                                            # walls: no neighbour = no flux
            out[tuple(lo)] -= d
        return out

    tri_lo = np.tril(np.ones((NZ, NZ), bool), -1)[:, :, None, None]       # j < i  (not a segment)
    seg_len = np.maximum(np.arange(NZ)[None, :] - np.arange(NZ)[:, None] + 1, 1).astype(float)[:, :, None, None]
    n_buoy = max(1, int(p["buoy_tau"] / (4 * DT)))      # apply every n_buoy steps (n_buoy * dt <= tau_b / 4)
    r_buoy = min(1.0, n_buoy * DT / p["buoy_tau"])
    bcount = [0]

    def buoyancy(T):
        """Convective adjustment. In every vertical column the stable profile T* is the closest
        profile (least squares) that never gets colder going up, with the same total heat:
        T*(k) = max_{i<=k} min_{j>=k} mean(T[i..j]).  Each step: T += r (T* - T), r = min(1, dt/tau_b).
        Warm air under cold air therefore rises; stable layers are left alone. Energy is conserved."""
        bcount[0] += 1
        if bcount[0] % n_buoy:
            return T
        T2 = T.reshape(NZ, -1)                                             # one column per (y, x)
        cols = np.nonzero(((T2[:-1] - T2[1:]) > 0.01).any(axis=0))[0]       # columns unstable by > 0.01 K
        if cols.size == 0:
            return T
        Tc = T2[:, cols]
        C = np.concatenate([np.zeros_like(Tc[:1]), np.cumsum(Tc, axis=0)], axis=0)
        M = (C[None, 1:] - C[:-1, None]) / seg_len[..., 0]                  # M[i, j] = mean(T[i..j])
        for k in range(NZ - 2, -1, -1):                                    # min over j >= k
            np.minimum(M[:, k], M[:, k + 1], out=M[:, k])
        Tstar = np.where(tri_lo[..., 0], -np.inf, M).max(axis=0)           # max over i <= k
        T2 = T2.copy()
        T2[:, cols] = Tc + r_buoy * (Tstar - Tc)
        return T2.reshape(T.shape)

    def heater_and_shell(Ta, Tm):
        """PI heater on mean cabin air; air <-> shell (hA); shell -> outside (UA)."""
        e = t_set - Ta.mean()
        if ki > 0:
            ctrl["I"] = min(max(ctrl["I"] + e * DT, 0.0), pmax / ki)   # integrator clamped (anti-windup)
        P = min(max(kp * e + ki * ctrl["I"], 0.0), pmax)
        q = HA_s * (Tm - Ta[S])                 # kW from shell to air, per shell cube
        env = UA_s * (Tm - t_out)               # kW from shell to outside
        Ta = Ta + DT * P * heat_w / CA
        Ta[S] += DT * q / CA
        Tm = Tm + DT * (-q - env) / CM_s
        return Ta, Tm, P, env.sum()

    def pcm_blow(Ta, E_kwh, t):
        """Blow PCM-warmed air into the vent zone of every door that is open or about to open.
        dT = f (T_sup - T_air), f = V_dot dt / V_zone. Returns new field, heat (kJ), fan on."""
        active = [k for k in (0, 1) if is_open(m, k, t) or is_open(m, k, t + p["pcm_preheat_s"])]
        if not active or E_kwh <= 0:
            return Ta, 0.0, False
        flow = p["pcm_airflow"] / len(active)
        deltas, q = [], 0.0
        for k in active:
            d = m["doors"][k]
            f = flow * DT / d["zone_vol"]
            dT = np.where(d["zone"], f * (m["T_supply"] - Ta), 0.0)
            deltas.append(dT)
            q += CA * dT.sum()
        cap = min(p["pcm_max_kw"] * DT, E_kwh * 3600)    # kJ available this step
        scale = min(1.0, cap / q) if q > 0 else 0.0
        for dT in deltas:
            Ta = Ta + scale * dT
        return Ta, q * scale, True

    def door_exchange(Ta, d, Q):
        """Upwind finite-volume advection with the divergence-free door flow (Q m3/s each way).
        V dT/dt = -sum(F * T_upwind) over the cube faces. Returns new field, heat lost (kJ)."""
        sub = Ta[:, :, d["cols"]]
        Fy, Fz = Q * d["Fy"], Q * d["Fz"]
        Py = np.concatenate([np.full_like(sub[:, :1], t_out), sub, sub[:, -1:]], axis=1)
        Hy = Fy * np.where(Fy > 0, Py[:, :-1], Py[:, 1:])
        Pz = np.concatenate([sub[:1], sub, sub[-1:]], axis=0)
        Hz = Fz * np.where(Fz > 0, Pz[:-1], Pz[1:])
        Ta[:, :, d["cols"]] = sub - DT / Vc * (Hy[:, 1:] - Hy[:, :-1] + Hz[1:] - Hz[:-1])
        return Ta, -RHO_CP * DT * Hy[:, 0].sum()

    def stored(Ta, Tm):
        return CA * Ta.sum() + (CM_s * Tm).sum()       # kJ, referenced to 0 degC

    # --- initial state and doors-closed warm-up ---
    Ta = np.full((NZ, NY, NX), t_set)
    Tm = np.full(S.sum(), (p["ha"] * t_set + p["ua"] * t_out) / (p["ha"] + p["ua"]))
    for _ in range(int(round(WARMUP_S / DT))):
        Ta, Tm, _, _ = heater_and_shell(buoyancy(mixing(Ta)), Tm)
    U0 = stored(Ta, Tm)

    E = p["pcm_kwh"] if use_pcm else 0.0
    E0 = E
    r = dict(heater=0.0, env=0.0, door=0.0, pcm=0.0, pcm_loss=0.0, fan=0.0, ran_out=None, snaps={})
    ser = {k: [] for k in ("t", "mean", "ankle", "head", "door_ankle", "P_heater", "P_pcm", "P_door")}
    n_samp = max(1, int(round(SAMPLE_S / DT)))
    acc = dict(h=0.0, pcm=0.0, door=0.0)
    near = np.zeros((NY, NX), bool)                     # floor-level cubes within 1 m of the front door
    near[m["yc"] < 1.0, m["doors"][0]["cols"]] = True
    occ = np.zeros((NY, NX), bool)                      # occupied zone: not against a wall or heater outlet
    occ[1:-1, 1:-1] = True
    snap_set = set(snap_steps)

    for s in range(int(round(m["TEND"] / DT))):
        t = s * DT
        Ta = buoyancy(mixing(Ta))
        Ta, Tm, P, env = heater_and_shell(Ta, Tm)
        r["heater"] += P * DT / 3600
        r["env"] += env * DT / 3600
        acc["h"] += P * DT
        if use_pcm:
            Ta, q, fan = pcm_blow(Ta, E, t)
            E -= q / 3600
            r["pcm"] += q / 3600
            acc["pcm"] += q
            if fan:
                r["fan"] += p["pcm_fan_kw"] * DT / 3600
            if E > 0:
                loss = min(p["pcm_loss_kw_per_k"] * (m["T_melt"] - t_out) * DT / 3600, E)
                E -= loss
                r["pcm_loss"] += loss
            if E0 > 0 and E <= 1e-9 and r["ran_out"] is None:
                r["ran_out"] = t
        for k, d in enumerate(m["doors"]):
            if is_open(m, k, t):
                Q = min(door_flow(m, Ta.mean()), m["Q_worst"])
                Ta, lost = door_exchange(Ta, d, Q)
                r["door"] += lost / 3600
                acc["door"] += lost
        if s in snap_set:
            r["snaps"][s] = Ta.copy()
        if (s + 1) % n_samp == 0:
            span = n_samp * DT
            ser["t"].append((t + DT) / 60)
            ser["mean"].append(Ta.mean())
            ser["ankle"].append(Ta[m["k_ankle"]][occ].mean())
            ser["head"].append(Ta[m["k_head"]][occ].mean())
            ser["door_ankle"].append(Ta[m["k_ankle"]][near].mean())
            ser["P_heater"].append(acc["h"] / span)
            ser["P_pcm"].append(acc["pcm"] / span)
            ser["P_door"].append(acc["door"] / span)
            acc = dict(h=0.0, pcm=0.0, door=0.0)

    r.update({k: np.array(v) for k, v in ser.items()})
    r["E_end"] = E
    r["recharge"] = (E0 - E) / p["pcm_recharge_eff"]
    r["dU"] = (stored(Ta, Tm) - U0) / 3600
    r["residual"] = r["heater"] + r["pcm"] - r["door"] - r["env"] - r["dU"]
    return r


# =============================== RESULTS ===============================
def summarise(m, A, B):
    p = m["p"]
    lines = []
    out = lines.append
    heater_saved = A["heater"] - B["heater"]
    net_saved = heater_saved - B["fan"]
    weight = m["pcm_mass"] * p["pcm_mass_factor"]
    q0 = door_flow(m, p["t_set"])
    out(f"=== {p['pcm_material']}, {p['pcm_kwh']:g} kWh PCM, {p['pcm_preheat_s']}s pre-blow ===")
    out(f"Door {p['door_open_s']}s | front {p['front_per_hr']}/h, rear {p['rear_per_hr']}/h | outside {p['t_out']} C")
    out(f"Grid {m['NX']} x {m['NY']} x {m['NZ']} = {m['N']} cubes | time step {m['DT']:.3f} s | "
        f"cabin air {m['C_air_total']:.0f} kJ/K | heater: {p['heater_location']}")
    out(f"Door exchange flow at setpoint: {q0:.2f} m3/s each way, about "
        f"{RHO_CP * q0 * (p['t_set'] - p['t_out']):.1f} kW while open | PCM supply air {m['T_supply']:.0f} C\n")
    out(f"{'':<34}{'no PCM':>10}{'with PCM':>10}")
    rows = [("Heater energy (kWh)", A["heater"], B["heater"]),
            ("PCM heat delivered (kWh)", 0, B["pcm"]),
            ("Door heat loss (kWh)", A["door"], B["door"]),
            ("Envelope loss (kWh)", A["env"], B["env"]),
            ("Change in stored heat (kWh)", A["dU"], B["dU"]),
            ("Energy balance residual (kWh)", A["residual"], B["residual"]),
            ("Lowest mean air (C)", A["mean"].min(), B["mean"].min()),
            ("Average mean air (C)", A["mean"].mean(), B["mean"].mean()),
            ("Lowest floor air by front door (C)", A["door_ankle"].min(), B["door_ankle"].min()),
            (f"Avg head({p['head_z']:g} m)-ankle diff (K)", (A["head"] - A["ankle"]).mean(), (B["head"] - B["ankle"]).mean()),
            ("Max head-ankle diff (K)", (A["head"] - A["ankle"]).max(), (B["head"] - B["ankle"]).max())]
    for name, a, b in rows:
        out(f"{name:<34}{a:>10.2f}{b:>10.2f}")
    out(f"\nHeater energy saved:      {heater_saved:.2f} kWh (fan uses {B['fan']:.2f} kWh) -> net {net_saved:.2f} kWh")
    out(f"PCM heat lost in storage: {B['pcm_loss']:.2f} kWh | PCM left at end: {100 * B['E_end'] / p['pcm_kwh']:.0f}%")
    out("PCM ran out at:           " + (f"{B['ran_out'] / 60:.1f} min" if B["ran_out"] is not None else "did not run out"))
    out(f"Depot recharge needed:    {B['recharge']:.2f} kWh")
    out(f"PCM mass {m['pcm_mass']:.0f} kg, about {weight:.0f} kg with container, insulation and heat exchanger")
    if B["pcm"] > 0:
        out(f"Net battery energy saved per 100 kg carried: {100 * net_saved / weight:.2f} kWh")
    return "\n".join(lines)


def colour_limits(p):
    return max(p["t_out"], p["t_set"] - 15), p["t_set"] + 3


def draw_figures(m, A, B, snap_steps):
    """Returns PNG bytes for: floor-level plans, door cross-sections, time series."""
    p = m["p"]
    lo, hi = colour_limits(p)
    offs = p["snap_offsets"]
    pngs = []

    def save(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        pngs.append(buf.getvalue())

    # plan view at ankle height
    fig, axes = plt.subplots(len(snap_steps), 2, figsize=(12, 2.0 * len(snap_steps) + 0.6), squeeze=False, layout="constrained")
    for ri, (s, off) in enumerate(zip(snap_steps, offs)):
        for ci, (res, label) in enumerate(((A, "no PCM"), (B, "with PCM"))):
            ax = axes[ri, ci]
            im = ax.imshow(res["snaps"][s][m["k_ankle"]], origin="lower", extent=[0, p["length"], 0, p["width"]],
                           cmap="turbo", vmin=lo, vmax=hi)
            for k, xc in enumerate(m["door_centres"]):
                ax.plot([xc - p["door_width"] / 2, xc + p["door_width"] / 2], [0, 0], lw=7, solid_capstyle="butt",
                        color="orange" if is_open(m, k, s * m["DT"]) else "gray", clip_on=False)
                if ci == 1:
                    dd, _ = m["doors"][k]["zone_dims"]
                    ax.add_patch(plt.Rectangle((xc - p["door_width"] / 2, 0), p["door_width"], dd,
                                               fill=False, ls=":", ec="white", lw=1))
            ax.set_title(f"{label}: {off:+d} s from opening {p['snap_opening']} (floor level, 0-0.25 m)", fontsize=8, loc="left")
            ax.set_yticks([])
    fig.colorbar(im, ax=axes, label="air temperature (degC)", shrink=0.8)
    save(fig)

    # cross-section through the front door centre (depth x height)
    i_front = min(int(m["door_centres"][0] / DX), m["NX"] - 1)
    fig, axes = plt.subplots(len(snap_steps), 2, figsize=(8, 2.9 * len(snap_steps)), squeeze=False, layout="constrained")
    for ri, (s, off) in enumerate(zip(snap_steps, offs)):
        for ci, (res, label) in enumerate(((A, "no PCM"), (B, "with PCM"))):
            ax = axes[ri, ci]
            im = ax.imshow(res["snaps"][s][:, :, i_front], origin="lower", extent=[0, p["width"], 0, p["height"]],
                           cmap="turbo", vmin=lo, vmax=hi)
            ax.plot([0, 0], [0, m["door_h"]], lw=7, solid_capstyle="butt", clip_on=False,
                    color="orange" if is_open(m, 0, s * m["DT"]) else "gray")
            if ci == 1:
                dd, hh = m["doors"][0]["zone_dims"]
                ax.add_patch(plt.Rectangle((0, 0), dd, hh, fill=False, ls=":", ec="white", lw=1))
            ax.set_title(f"{label}: {off:+d} s", fontsize=8, loc="left")
            ax.set_xlabel("depth from door (m)", fontsize=8)
            ax.set_ylabel("height (m)", fontsize=8)
    fig.colorbar(im, ax=axes, label="air temperature (degC)", shrink=0.8)
    fig.suptitle("Vertical section through the front door centre", fontsize=9)
    save(fig)

    # time series
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for res, label, ls in ((A, "no PCM", "-"), (B, "with PCM", "--")):   # solid = no PCM, dashed = with PCM
        axs[0].plot(res["t"], res["mean"], ls, color="tab:blue", label=f"mean cabin air, {label}")
        axs[0].plot(res["t"], res["door_ankle"], ls, color="tab:orange", lw=1, label=f"floor air by front door, {label}")
        axs[1].plot(res["t"], res["head"], ls, color="tab:green", label=f"head height ({p['head_z']:g} m), {label}")
        axs[1].plot(res["t"], res["ankle"], ls, color="tab:purple", label=f"floor level, {label}")
        axs[2].plot(res["t"], res["P_heater"], ls, color="tab:blue", label=f"heater, {label}")
        axs[2].plot(res["t"], res["P_door"], ls, color="tab:gray", lw=1, label=f"door loss, {label}")
    axs[2].plot(B["t"], B["P_pcm"], color="tab:red", lw=1, label="PCM heat")
    for ax in axs[:2]:
        ax.axhline(p["t_set"], ls=":", color="gray", label="setpoint")
        ax.set_ylabel("degC")
    axs[0].set_title("Cabin air", fontsize=9, loc="left")
    axs[1].set_title("Stratification (occupied-zone layer averages)", fontsize=9, loc="left")
    axs[2].set_title(f"Heat flows ({SAMPLE_S:g} s averages)", fontsize=9, loc="left")
    axs[2].set_ylabel("kW")
    axs[2].set_xlabel("time (min)")
    for ax in axs:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    save(fig)
    return pngs


def figure_3d(m, T, t, title):
    """Interactive 3D view: cold air drawn more opaque, comfortable air nearly clear."""
    p = m["p"]
    lo, hi = colour_limits(p)
    Z, Y, X = np.meshgrid(m["zc"], m["yc"], m["xc"], indexing="ij")
    fig = go.Figure(go.Volume(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=np.clip(T, lo, hi).ravel(),
                              isomin=lo, isomax=hi, surface_count=14, colorscale="Turbo",
                              opacityscale=[[0, 0.9], [0.5, 0.25], [0.8, 0.03], [1, 0.5]],
                              colorbar=dict(title="°C")))
    L, W, H = p["length"], p["width"], p["height"]
    ex, ey, ez = [], [], []
    for a, b in [((0, 0, 0), (L, 0, 0)), ((0, W, 0), (L, W, 0)), ((0, 0, H), (L, 0, H)), ((0, W, H), (L, W, H)),
                 ((0, 0, 0), (0, W, 0)), ((L, 0, 0), (L, W, 0)), ((0, 0, H), (0, W, H)), ((L, 0, H), (L, W, H)),
                 ((0, 0, 0), (0, 0, H)), ((L, 0, 0), (L, 0, H)), ((0, W, 0), (0, W, H)), ((L, W, 0), (L, W, H))]:
        ex += [a[0], b[0], None]; ey += [a[1], b[1], None]; ez += [a[2], b[2], None]
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="gray", width=2), showlegend=False))
    for k, xc in enumerate(m["door_centres"]):
        x0, x1 = xc - p["door_width"] / 2, xc + p["door_width"] / 2
        fig.add_trace(go.Scatter3d(x=[x0, x1, x1, x0, x0], y=[0] * 5, z=[0, 0, m["door_h"], m["door_h"], 0], mode="lines",
                                   line=dict(color="orange" if is_open(m, k, t) else "black", width=8), showlegend=False))
    fig.update_layout(title=title, height=550, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(aspectmode="data", xaxis_title="length from front (m)",
                                 yaxis_title="depth from door side (m)", zaxis_title="height (m)"))
    return fig


def simulate_all(p):
    m = build_model(p)
    if m["counts"][0] >= p["snap_opening"]:
        t0 = opening_start(m, 0, p["snap_opening"])
    else:
        t0 = 600.0
    last = int(round(m["TEND"] / m["DT"])) - 1
    snap_steps = [min(max(int(round((t0 + o) / m["DT"])), 0), last) for o in p["snap_offsets"]]
    A = run(m, False, snap_steps)
    B = run(m, True, snap_steps)
    return m, A, B, snap_steps, summarise(m, A, B), draw_figures(m, A, B, snap_steps)


METHODS = r"""
**Grid and time stepping.** The cabin is split into cubes of $\Delta x = 0.25$ m, indexed $[z, y, x]$
($y$ = depth from the door-side wall). Each cube has air heat capacity $C_a = \rho c_p \Delta x^3$ with
$\rho c_p = 1.2$ kJ/m³K. Cubes touching the shell also have a shell (interior mass) node. Explicit Euler steps
are used with $\Delta t = \min(0.1\,\text{s},\ \Delta t_{mix},\ \Delta t_{door},\ \Delta t_{PCM})$, and the five
processes below are applied one after another each step (operator splitting). A 600 s doors-closed warm-up runs first.

**1. Air mixing** (turbulent mixing as diffusion, zero flux at the shell):
$$T \leftarrow T + \Delta t\left[\alpha_h\left(\frac{\partial^2 T}{\partial x^2}+\frac{\partial^2 T}{\partial y^2}\right)+\alpha_v\frac{\partial^2 T}{\partial z^2}\right],\quad \alpha_v = f_v\,\alpha_h,\quad \Delta t_{mix}=0.8\,\frac{\Delta x^2}{2(2\alpha_h+\alpha_v)}$$

**2. Buoyancy** (convective adjustment). Warm air cannot stay under colder air. In each vertical column the
stable profile $T^*$ is the least-squares closest profile that never gets colder going up, with the same total heat
(isotonic regression), and the column relaxes toward it with the overturning time $\tau_b$:
$$T^*_k=\max_{i\le k}\ \min_{j\ge k}\ \overline{T}_{i..j},\qquad T \leftarrow T + \min\!\left(1,\frac{n\,\Delta t}{\tau_b}\right)(T^*-T)$$
applied every $n$ steps, with $n\,\Delta t \le \tau_b/4$ (saves run time; same result because $\tau_b \gg \Delta t$).
Stable layers (cold below warm) are left alone, so cold door air pools at the floor.

**3. Heater and shell.** PI control on mean cabin air, split evenly over the heater cubes:
$$P=\text{clip}\!\left(K_p e + K_i\!\int e\,dt,\ 0,\ P_{max}\right),\qquad e = T_{set}-\bar T_{air}$$
Shell cube $i$ gets a share of $hA$, $UA$ and $C_m$ in proportion to its exposed face area:
$$q_i = hA_i\,(T_{m,i}-T_{a,i}),\qquad C_a\frac{dT_{a,i}}{dt}=q_i+P_i,\qquad C_{m,i}\frac{dT_{m,i}}{dt}=-q_i-UA_i\,(T_{m,i}-T_{out})$$

**4. PCM fan.** On while a door is open or will open within $t_{pre}$. Airflow is shared between active doors.
$$T_{sup}=\min(T_{melt}-\Delta T_{approach},\ T_{cap}),\qquad f=\frac{\dot V\,\Delta t}{V_{zone}},\qquad \Delta T_i=f\,(T_{sup}-T_{a,i})$$
$$Q_{PCM}=C_a\sum\Delta T_i \le \min(P_{PCM,max}\Delta t,\ E),\qquad E \leftarrow E - Q_{PCM} - UA_{PCM}(T_{melt}-T_{out})\Delta t$$
$$m_{PCM}=E_0/L,\qquad \text{recharge}=(E_0-E_{end})/\eta,\qquad \Delta t_{PCM}=0.5\,V_{zone}/\dot V$$

**5. Door exchange flow** (two-way buoyancy-driven flow through a large opening):
$$Q_{door}=M\,\frac{C_d}{3}\,W H\sqrt{\frac{g H\,|\bar T_{air}-T_{out}|}{\bar T_K}}$$
The flow pattern comes from a streamfunction on the cube corners of each door column,
$$\psi(y,z)=\frac{Q_{door}}{n_{col}}\,g(z)\,h(y),\qquad g=\sin(\pi z/H)\ \text{(normalised)},\qquad h=e^{-y/L_p}\left(1-\frac{y}{W_{cab}}\right)$$
Face flows $F_y=\Delta_z\psi$, $F_z=-\Delta_y\psi$ are exactly divergence-free: cold air enters below the neutral plane,
travels in along the floor, rises, and warm air leaves through the top of the door. Temperatures are moved with
first-order upwind finite volumes, $V\,dT/dt=-\sum_{faces}F\,T_{upwind}$, with outside air at $T_{out}$ upwind of the door face.
$$Q_{door\ loss}=-\rho c_p\,\Delta t\sum_{door\ face}F\,T_{upwind},\qquad \Delta t_{door}=0.8\,V/(\text{max outflow of any cube})$$

**Comfort outputs:** floor-level and head-height temperatures are averaged over the occupied zone (cubes not
against a wall). "Floor air by front door" = floor-level cubes within 1 m of the front door.

**Energy balance check:** $\;E_{heater}+E_{PCM}-E_{door}-E_{envelope}=\Delta U$, with
$U=\sum C_a T_a+\sum C_m T_m$. The residual is printed and should be ≈ 0.

**Savings:** heater saved = heater(no PCM) − heater(with PCM); net = heater saved − fan energy;
per 100 kg = net / (m_PCM × system mass factor) × 100.
"""


# =============================== STREAMLIT APP ===============================
def sidebar_inputs(st):
    d = DEFAULTS
    form = st.sidebar.form("params")
    form.header("Inputs")
    p = {}
    with form.expander("Door pattern", expanded=True):
        p["door_open_s"] = st.slider("Door open time (s)", 5, 60, d["door_open_s"])
        p["front_per_hr"] = st.slider("Front door openings per hour", 0, 60, d["front_per_hr"])
        p["rear_per_hr"] = st.slider("Rear door openings per hour", 0, 60, d["rear_per_hr"])
    with form.expander("Weather and comfort", expanded=True):
        p["t_out"] = st.slider("Outside temperature (°C)", -35.0, 10.0, d["t_out"], 0.5)
        p["t_set"] = st.slider("Cabin setpoint (°C)", 15.0, 25.0, d["t_set"], 0.5)
        p["head_z"] = st.slider("Head height for comfort check (m)", 0.5, 1.75, d["head_z"], 0.05,
                                help="1.1 m seated, 1.7 m standing")
    with form.expander("PCM under the cabin", expanded=True):
        p["pcm_material"] = st.selectbox("PCM material", list(PCM_LIBRARY))
        p["pcm_kwh"] = st.slider("Usable PCM capacity (kWh)", 0.5, 20.0, d["pcm_kwh"], 0.5)
        p["pcm_preheat_s"] = st.slider("Fan starts before door opens (s)", 0, 30, d["pcm_preheat_s"])
        p["pcm_airflow"] = st.slider("Airflow through PCM (m³/s)", 0.1, 1.5, d["pcm_airflow"], 0.05)
        p["pcm_max_kw"] = st.slider("Max PCM heat output (kW)", 2.0, 40.0, d["pcm_max_kw"], 1.0)
        p["pcm_approach_k"] = st.slider("Supply air below melting point (K)", 2.0, 30.0, d["pcm_approach_k"], 1.0)
        p["pcm_max_supply_c"] = st.slider("Supply air cap (°C)", 30.0, 90.0, d["pcm_max_supply_c"], 1.0)
        p["pcm_vent_depth"] = st.slider("Vent zone depth (m)", 0.25, 1.5, d["pcm_vent_depth"], 0.25)
        p["pcm_vent_height"] = st.slider("Vent zone height above floor (m)", 0.25, 1.0, d["pcm_vent_height"], 0.25)
        p["pcm_fan_kw"] = st.slider("Fan power (kW)", 0.05, 1.0, d["pcm_fan_kw"], 0.05)
        p["pcm_loss_kw_per_k"] = st.slider("PCM insulation loss (kW/K)", 0.0, 0.02, d["pcm_loss_kw_per_k"], 0.001, format="%.3f")
        p["pcm_mass_factor"] = st.slider("System mass / PCM mass", 1.0, 2.5, d["pcm_mass_factor"], 0.05)
        p["pcm_recharge_eff"] = st.slider("Depot recharge efficiency", 0.5, 1.0, d["pcm_recharge_eff"], 0.01)
    with form.expander("Bus and heating system"):
        p["heater_location"] = st.selectbox("Heater outlets", HEATER_LOCATIONS)
        p["heater_max_kw"] = st.slider("Max heater power (kW)", 5.0, 50.0, d["heater_max_kw"], 1.0)
        p["heater_kp"] = st.slider("Heater proportional gain (kW/K)", 2.0, 30.0, d["heater_kp"], 1.0)
        p["heater_ki_per_min"] = st.slider("Heater integral gain (kW/K per min)", 0.0, 10.0, d["heater_ki_per_min"], 0.5,
                                           help="0 = proportional only (same as the 2D model)")
        p["ua"] = st.slider("Envelope UA (kW/K)", 0.05, 1.0, d["ua"], 0.05)
        p["c_mass"] = st.slider("Interior mass heat capacity (kJ/K)", 200.0, 2000.0, d["c_mass"], 50.0)
        p["ha"] = st.slider("Air-to-mass transfer (kW/K)", 1.0, 10.0, d["ha"], 0.5)
    with form.expander("Geometry"):
        p["length"] = st.slider("Cabin length (m)", 8.0, 18.0, d["length"], 0.25)
        p["width"] = st.slider("Cabin width (m)", 2.0, 3.0, d["width"], 0.25)
        p["height"] = st.slider("Cabin height (m)", 2.0, 2.75, d["height"], 0.25)
        p["door_width"] = st.slider("Door width (m)", 0.75, 2.0, d["door_width"], 0.25)
        p["door_height"] = st.slider("Door height (m)", 1.5, 2.5, d["door_height"], 0.25)
        p["front_x"] = st.slider("Front door centre from front (m)", 0.75, 5.0, d["front_x"], 0.125)
        p["rear_x"] = st.slider("Rear door centre from rear (m)", 0.75, 5.0, d["rear_x"], 0.125)
    with form.expander("Air movement"):
        p["door_cd"] = st.slider("Door discharge coefficient Cd", 0.3, 0.9, d["door_cd"], 0.05)
        p["door_flow_mult"] = st.slider("Door flow multiplier (wind, traffic)", 0.5, 3.0, d["door_flow_mult"], 0.1)
        p["door_penetration"] = st.slider("Door flow penetration length (m)", 0.5, 3.0, d["door_penetration"], 0.25)
        p["mix_h"] = st.slider("Horizontal air mixing (m²/s)", 0.01, 0.15, d["mix_h"], 0.01)
        p["mix_v_factor"] = st.slider("Vertical / horizontal mixing", 0.1, 1.0, d["mix_v_factor"], 0.1)
        p["buoy_tau"] = st.slider("Buoyant overturning time (s)", 0.5, 20.0, d["buoy_tau"], 0.5,
                                  help="Time for warm air under cold air to rise; smaller = stronger buoyancy")
    with form.expander("Simulation and figure"):
        p["sim_hours"] = st.slider("Simulated time (h)", 0.25, 2.0, d["sim_hours"], 0.25, help="Longer runs take longer.")
        p["snap_opening"] = st.number_input("Front door opening to show", 1, 120, d["snap_opening"])
        p["snap_offsets"] = (st.slider("Snapshot 1 (s from opening)", -30, 120, d["snap_offsets"][0]),
                             st.slider("Snapshot 2 (s from opening)", -30, 120, d["snap_offsets"][1]),
                             st.slider("Snapshot 3 (s from opening)", -30, 120, d["snap_offsets"][2]))
    form.form_submit_button("Run simulation", type="primary", width="stretch")
    return p


def app():
    import streamlit as st
    st.set_page_config(page_title="3D bus door heat loss with PCM", layout="wide")
    p = sidebar_inputs(st)

    @st.cache_data(show_spinner=False)
    def cached(params):
        return simulate_all(params)

    st.title("3D bus door heat loss with PCM heater")
    st.caption("Change the inputs in the sidebar, then press Run simulation. All defaults are illustrative placeholders.")
    with st.spinner("Running both 3D simulations (this can take a minute)..."):
        m, A, B, snap_steps, summary, pngs = cached(p)
    st.code(summary)
    with st.expander("Model equations and method"):
        st.markdown(METHODS)

    t1, t2, t3, t4 = st.tabs(["Floor-level plan", "Cross-section at front door", "Time series", "Interactive 3D"])
    t1.image(pngs[0], width="stretch")
    t2.image(pngs[1], width="stretch")
    t3.image(pngs[2], width="stretch")
    with t4:
        if go is None:
            st.info("Install plotly (pip install plotly) to see the interactive 3D view.")
        else:
            c1, c2 = st.columns(2)
            which = c1.radio("Run", ["no PCM", "with PCM"], horizontal=True)
            i = c2.radio("Snapshot (s from opening)", list(range(len(snap_steps))), horizontal=True,
                         format_func=lambda j: f"{p['snap_offsets'][j]:+d} s")
            res = A if which == "no PCM" else B
            s = snap_steps[i]
            st.plotly_chart(figure_3d(m, res["snaps"][s], s * m["DT"],
                                      f"{which}: {p['snap_offsets'][i]:+d} s from front opening {p['snap_opening']}"))
            st.caption("Cold air is drawn more solid, air near the setpoint is almost clear. "
                       "Orange door frame = open. Drag to rotate.")


if __name__ == "__main__":
    app()
