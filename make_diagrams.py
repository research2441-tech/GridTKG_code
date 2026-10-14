"""Architecture, ontology and analytics-layer diagrams (drawn from scratch with matplotlib)."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

C_ING, C_SEM, C_STO, C_QRY, C_APP, C_BASE = "#E8F1FB", "#FFF4E0", "#E9F6EC", "#F3ECF9", "#FDECEC", "#F2F2F2"


def box(ax, x, y, w, h, title, lines=(), fc="#FFFFFF", ec="#333333", fs=9, tfs=9.5, bold=True, ls=0.032, t0=0.022, l0=0.06):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h - t0, title, ha="center", va="top", fontsize=tfs, fontweight="bold" if bold else "normal")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - l0 - i * ls, ln, ha="center", va="top", fontsize=fs)


def arrow(ax, x0, y0, x1, y1, txt=None, both=False, color="#444444"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if both else "-|>", mutation_scale=10,
                                 lw=1.0, color=color))
    if txt:
        ax.text((x0 + x1) / 2 + 0.01, (y0 + y1) / 2, txt, fontsize=7.5, color="#333333", va="center")


def fig_architecture():
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.01, 0.83), 0.98, 0.16, boxstyle="round,pad=0.005,rounding_size=0.015", fc=C_ING, ec="#333333"))
    ax.text(0.5, 0.975, "Data sources and clients", ha="center", va="top", fontsize=9.5, fontweight="bold")
    for i, (t, s_) in enumerate([("SCADA / RTU", "branch flows, 15 min"), ("PMU", "bus angles"),
                                 ("Smart meters", "load (MW)"), ("Condition monitor", "daily indices"),
                                 ("Event logs", "outages, maintenance"), ("Operators", "OLTP / OLAP calls")]):
        box(ax, 0.02 + i * 0.163, 0.845, 0.150, 0.095, t, [s_], fc="#FFFFFF", fs=7.5, tfs=8)
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.68, 0.78, boxstyle="round,pad=0.003,rounding_size=0.02",
                                fc="none", ec="#1F4E79", lw=1.8))
    ax.text(0.03, 0.775, "GridTKG engine (one process, Python/NumPy, no external DBMS)", ha="left", fontsize=9.5,
            fontweight="bold", color="#1F4E79")
    box(ax, 0.03, 0.60, 0.64, 0.15, "Ingestion and transaction manager",
        ["measurement ingest (O1, S1) | outage transactions (O4) | topology growth (S2)",
         "single writer, write-ahead log, 10 ms group commit, atomic publication"], fc=C_ING, fs=8)
    box(ax, 0.03, 0.43, 0.64, 0.15, "Semantic layer (Section 4)",
        ["CIM-aligned ontology: 19 classes, 14 predicates; typing, cardinality,",
         "voltage and validity checks; rules for busNeighbor, adjacentTo, substationLink"], fc=C_SEM, fs=8)
    box(ax, 0.03, 0.20, 0.31, 0.21, "Fact store",
        ["per-predicate CSR (forward, reverse)", "write-optimised delta (merge 4096)",
         "validity arrays aligned with CSR", "typed entity columns"], fc=C_STO, fs=7.8)
    box(ax, 0.36, 0.20, 0.31, 0.21, "Measurement store",
        ["slot-aligned dense chunks (S x 96)", "current-state vector", "NaN for missing samples",
         "commit watermark for reads"], fc=C_STO, fs=7.8)
    box(ax, 0.03, 0.04, 0.64, 0.14, "Query layer (14 operations, Table 5)",
        ["operational: state reads, one-hop lookup | analytical: aggregation, k-hop, hop distance",
         "maintenance: history windows, overload screening, adjacency context"], fc=C_QRY, fs=8)
    arrow(ax, 0.62, 0.83, 0.62, 0.75)
    arrow(ax, 0.35, 0.60, 0.35, 0.58)
    arrow(ax, 0.185, 0.43, 0.185, 0.41); arrow(ax, 0.515, 0.43, 0.515, 0.41)
    arrow(ax, 0.185, 0.20, 0.185, 0.18); arrow(ax, 0.515, 0.20, 0.515, 0.18)
    box(ax, 0.72, 0.36, 0.27, 0.44, "Application layer (Section 6)", fc=C_APP)
    box(ax, 0.74, 0.585, 0.23, 0.15, "GTN", ["outage detection (hourly)", "7-day failure risk (daily)"], fc="#FFFFFF", fs=7.8)
    box(ax, 0.74, 0.43, 0.23, 0.13, "PPO agent", ["maintenance-crew allocation", "(daily), uses GTN risk"], fc="#FFFFFF", fs=7.8)
    arrow(ax, 0.855, 0.585, 0.855, 0.56)
    ax.text(0.855, 0.395, "inputs read through the query layer", ha="center", fontsize=7.5, style="italic")
    ax.plot([0.67, 0.705, 0.705], [0.11, 0.11, 0.50], color="#444444", lw=1.0)
    arrow(ax, 0.705, 0.50, 0.72, 0.50)
    box(ax, 0.72, 0.02, 0.27, 0.31, "Baselines (Section 7.3)",
        ["PostgreSQL 16: row store,", "B-tree indexes, recursive CTE", "DuckDB 1.5.6: column store",
         "Kuzu 0.11.3 + DuckDB:", "graph DBMS + analytics", "identical facts and queries"], fc=C_BASE, fs=7.8)
    fig.savefig(os.path.join(OUT, "fig01_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_ontology():
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.06); ax.axis("off")
    N = {}
    BW, BH = 0.19, 0.095

    def node(key, x, y, title, cim, fc):
        ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0.004,rounding_size=0.012",
                                    fc=fc, ec="#333333", lw=1))
        ax.text(x, y + 0.015, title, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x, y - 0.022, "cim:" + cim, ha="center", va="center", fontsize=7, color="#444444")
        N[key] = (x, y)

    def clip(x, y, dx, dy):
        tx = (BW / 2 + 0.006) / abs(dx) if dx else 1e9
        ty = (BH / 2 + 0.008) / abs(dy) if dy else 1e9
        return min(tx, ty)

    def edge(a, b, label, color="#333333", dashed=False, rad=0.0, lab=None, both=False):
        (x0, y0), (x1, y1) = N[a], N[b]
        dx, dy = x1 - x0, y1 - y0
        ta, tb = clip(x0, y0, dx, dy), clip(x1, y1, dx, dy)
        p0 = (x0 + ta * dx, y0 + ta * dy); p1 = (x1 - tb * dx, y1 - tb * dy)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>" if both else "-|>", mutation_scale=10, lw=1.1,
                                     color=color, ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
        lx, ly = lab if lab else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
        ax.text(lx, ly, label, fontsize=7.5, color=color, ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))

    node("reg", 0.11, 0.86, "Region", "SubGeographicalRegion", C_ING)
    node("sub", 0.42, 0.86, "Substation", "Substation", C_ING)
    node("brk", 0.76, 0.86, "Breaker", "Breaker", C_SEM)
    node("gen", 0.11, 0.58, "Generator", "GeneratingUnit", C_STO)
    node("bus", 0.42, 0.58, "Bus", "BusbarSection", C_STO)
    node("sen", 0.89, 0.58, "Sensor", "Analog", C_QRY)
    node("line", 0.11, 0.30, "Line", "ACLineSegment", C_SEM)
    node("load", 0.42, 0.30, "Load", "EnergyConsumer", C_STO)
    node("tr", 0.76, 0.30, "Transformer", "PowerTransformer", C_SEM)
    node("fe", 0.60, 0.07, "FaultEvent", "Outage", C_APP)
    node("mr", 0.89, 0.07, "MaintenanceRecord", "WorkTask", C_APP)
    edge("sub", "reg", "locatedIn")
    edge("bus", "sub", "containedIn")
    edge("brk", "sub", "containedIn")
    edge("gen", "bus", "connectedTo")
    edge("load", "bus", "connectedTo", lab=(0.475, 0.405))
    edge("line", "bus", "fromBus / toBus")
    edge("tr", "bus", "hvBus / lvBus")
    edge("brk", "tr", "protects (Branch)", lab=(0.76, 0.66))
    edge("sen", "tr", "monitors (Bus, Branch,\nGenerator, Load)", lab=(0.86, 0.43))
    edge("fe", "tr", "affects [vf, vt)")
    edge("mr", "tr", "maintains")
    red = "#C00000"
    ax.add_patch(FancyArrowPatch((0.37, 0.535), (0.47, 0.535), arrowstyle="<|-|>", mutation_scale=10, lw=1.2,
                                 color=red, ls="--", connectionstyle="arc3,rad=1.3"))
    ax.text(0.545, 0.48, "busNeighbor (derived;\nvalidity from FaultEvent)", fontsize=7.5, color=red, ha="left")
    ax.add_patch(FancyArrowPatch((0.37, 0.905), (0.47, 0.905), arrowstyle="<|-|>", mutation_scale=10, lw=1.2,
                                 color=red, ls="--", connectionstyle="arc3,rad=-1.3"))
    ax.text(0.50, 1.0, "substationLink (derived)", fontsize=7.5, color=red, ha="left")
    ax.add_patch(FancyArrowPatch((0.11, 0.25), (0.70, 0.25), arrowstyle="<|-|>", mutation_scale=10, lw=1.2,
                                 color=red, ls="--", connectionstyle="arc3,rad=0.28"))
    ax.text(0.30, 0.135, "adjacentTo (derived, Branch-Branch)", fontsize=7.5, color=red, ha="center")
    ax.text(0.0, -0.02, "Solid arrows: explicit predicates stored as temporal quadruples. Dashed red: relations materialised "
            "by rules.\nSubclasses: Line, Transformer < Branch;  Thermal/Wind/SolarUnit < Generator;  PMU/SCADA/SmartMeter < Sensor.",
            fontsize=7.5, color="#333333", va="top")
    fig.savefig(os.path.join(OUT, "fig02_ontology.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_analytics():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 1); ax.set_ylim(-0.12, 1); ax.axis("off")
    kw = dict(fs=8, tfs=9, ls=0.075, t0=0.04, l0=0.15)
    box(ax, 0.01, 0.30, 0.17, 0.62, "GridTKG queries",
        ["history(sensors, window)", "adjacentTo partition", "connectedTo, monitors", "maintains, affects"], fc=C_STO, **kw)
    box(ax, 0.21, 0.30, 0.17, 0.62, "Feature tensor",
        ["X: W x N x F", "FD: W = 4 h, F = 9", "PM: W = 14 d, F = 11", "N = 1,991 branches"], fc=C_SEM, **kw)
    box(ax, 0.41, 0.30, 0.22, 0.62, "Graph temporal network",
        ["input projection (48)", "2 graph convolutions", "GRU over the window (48)", "graph readout + MLP",
         "one logit per branch"], fc=C_QRY, **kw)
    box(ax, 0.66, 0.62, 0.15, 0.30, "Outputs", ["outage alarm (FD)", "7-day failure prob."], fc=C_APP, **kw)
    box(ax, 0.66, 0.05, 0.15, 0.48, "PPO agent", ["per-branch scores", "+ C skip tokens", "Plackett-Luce",
                                                   "C = 5 crews/day"], fc=C_APP, **kw)
    box(ax, 0.84, 0.05, 0.15, 0.87, "Maintenance", ["environment", "", "degradation", "storms", "failures",
                                                     "condition readings", "costs, Eq. (24)"], fc=C_BASE,
        fs=8, tfs=9, ls=0.075, t0=0.04, l0=0.12)
    arrow(ax, 0.18, 0.61, 0.21, 0.61)
    arrow(ax, 0.38, 0.61, 0.41, 0.61)
    arrow(ax, 0.63, 0.75, 0.66, 0.77)
    arrow(ax, 0.735, 0.62, 0.735, 0.53)
    arrow(ax, 0.81, 0.29, 0.84, 0.29)
    ax.add_patch(FancyArrowPatch((0.915, 0.05), (0.295, 0.30), arrowstyle="-|>", mutation_scale=10, lw=1.0,
                                 color="#444444", connectionstyle="arc3,rad=-0.22"))
    ax.text(0.47, -0.10, "simulated next-day loading, condition readings and failures form the next input window",
            fontsize=8, ha="center")
    fig.savefig(os.path.join(OUT, "fig_analytics.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture()
    fig_ontology()
    fig_analytics()
