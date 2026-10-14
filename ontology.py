"""
CIM-aligned ontology for the power-grid temporal knowledge graph (GridTKG).

The schema follows the class names of IEC 61970-301 (Common Information Model).
Every fact is a temporal quadruple (subject, predicate, object, [valid_from, valid_to)).
Semantic constraints (typing, cardinality, voltage consistency, temporal validity)
are checked on every write, and derived relations are materialised by
forward-chaining rules.
"""
from dataclasses import dataclass, field

CIM_NS = "http://iec.ch/TC57/CIM100#"
GRID_NS = "http://example.org/gridtkg#"
INF_T = 2**62  # open validity interval

# ---------------------------------------------------------------------------
# Entity classes (id, name, CIM class, parent)
# ---------------------------------------------------------------------------
CLASSES = {
    "Region":            dict(cim="SubGeographicalRegion", parent=None),
    "Substation":        dict(cim="Substation", parent=None),
    "Bus":               dict(cim="BusbarSection", parent="Equipment"),
    "Equipment":         dict(cim="ConductingEquipment", parent=None),
    "Branch":            dict(cim="ConductingEquipment", parent="Equipment"),
    "Line":              dict(cim="ACLineSegment", parent="Branch"),
    "Transformer":       dict(cim="PowerTransformer", parent="Branch"),
    "Breaker":           dict(cim="Breaker", parent="Equipment"),
    "Generator":         dict(cim="GeneratingUnit", parent="Equipment"),
    "ThermalUnit":       dict(cim="ThermalGeneratingUnit", parent="Generator"),
    "WindUnit":          dict(cim="WindGeneratingUnit", parent="Generator"),
    "SolarUnit":         dict(cim="SolarGeneratingUnit", parent="Generator"),
    "Load":              dict(cim="EnergyConsumer", parent="Equipment"),
    "Sensor":            dict(cim="Analog", parent=None),
    "PMU":               dict(cim="Analog", parent="Sensor"),
    "SCADA":             dict(cim="Analog", parent="Sensor"),
    "SmartMeter":        dict(cim="Analog", parent="Sensor"),
    "FaultEvent":        dict(cim="Outage", parent=None),
    "MaintenanceRecord": dict(cim="WorkTask", parent=None),
}
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
ID_CLASS = {i: c for c, i in CLASS_ID.items()}


def is_a(cls, ancestor):
    while cls is not None:
        if cls == ancestor:
            return True
        cls = CLASSES[cls]["parent"]
    return False


# ---------------------------------------------------------------------------
# Predicates: domain, range, functional (max one valid object per subject)
# ---------------------------------------------------------------------------
@dataclass
class Predicate:
    name: str
    domain: tuple
    range: tuple
    functional: bool
    cim: str
    derived: bool = False
    symmetric: bool = False


PREDICATES = [
    Predicate("locatedIn", ("Substation",), ("Region",), True, "Substation.Region"),
    Predicate("containedIn", ("Bus", "Transformer", "Breaker"), ("Substation",), True,
              "Equipment.EquipmentContainer"),
    Predicate("fromBus", ("Line",), ("Bus",), True, "Terminal.ConnectivityNode"),
    Predicate("toBus", ("Line",), ("Bus",), True, "Terminal.ConnectivityNode"),
    Predicate("hvBus", ("Transformer",), ("Bus",), True, "PowerTransformerEnd.Terminal"),
    Predicate("lvBus", ("Transformer",), ("Bus",), True, "PowerTransformerEnd.Terminal"),
    Predicate("connectedTo", ("Generator", "Load"), ("Bus",), True, "Terminal.ConnectivityNode"),
    Predicate("protects", ("Breaker",), ("Branch",), True, "ProtectedSwitch.ProtectionEquipment"),
    Predicate("monitors", ("Sensor",), ("Bus", "Branch", "Generator", "Load"), True,
              "Measurement.PowerSystemResource"),
    Predicate("affects", ("FaultEvent",), ("Branch",), True, "Outage.Equipments"),
    Predicate("maintains", ("MaintenanceRecord",), ("Branch",), True, "WorkTask.Assets"),
    # derived by rules (materialised)
    Predicate("adjacentTo", ("Branch",), ("Branch",), False, "-", derived=True, symmetric=True),
    Predicate("busNeighbor", ("Bus",), ("Bus",), False, "-", derived=True, symmetric=True),
    Predicate("substationLink", ("Substation",), ("Substation",), False, "-", derived=True,
              symmetric=True),
]
PRED = {p.name: p for p in PREDICATES}
PRED_ID = {p.name: i for i, p in enumerate(PREDICATES)}
ID_PRED = {i: n for n, i in PRED_ID.items()}

ENDPOINT_PREDS = ("fromBus", "toBus", "hvBus", "lvBus")

# Rules (Datalog notation, materialised by gridtkg_store.materialise)
RULES = [
    "busNeighbor(x,y) :- fromBus(l,x), toBus(l,y).",
    "busNeighbor(x,y) :- hvBus(t,x), lvBus(t,y).",
    "adjacentTo(a,b)  :- endpoint(a,x), endpoint(b,x), a != b.",
    "substationLink(s,u) :- busNeighbor(x,y), containedIn(x,s), containedIn(y,u), s != u.",
]


class ConstraintViolation(Exception):
    pass


def check_typing(pred, s_cls, o_cls):
    p = PRED[pred]
    if not any(is_a(s_cls, d) for d in p.domain):
        raise ConstraintViolation(f"domain violation: {pred}({s_cls}, .)")
    if not any(is_a(o_cls, r) for r in p.range):
        raise ConstraintViolation(f"range violation: {pred}(., {o_cls})")


def check_interval(vf, vt):
    if not (vf < vt):
        raise ConstraintViolation(f"temporal violation: valid_from {vf} >= valid_to {vt}")


def check_voltage(pred, kv_a, kv_b):
    """Lines join equal nominal voltages (5 % tolerance); transformers step down hv -> lv."""
    if pred == "line" and abs(kv_a - kv_b) > 0.05 * max(kv_a, kv_b):
        raise ConstraintViolation(f"voltage mismatch on line: {kv_a} kV vs {kv_b} kV")
    if pred == "trafo" and not kv_a >= kv_b:
        raise ConstraintViolation(f"transformer hv {kv_a} kV below lv {kv_b} kV")


def rdf_triples(store, t):
    """Snapshot of the KG at time t as RDF triples (CIM namespace)."""
    from rdflib import Graph, Namespace, URIRef, Literal, RDF
    cim, g_ns = Namespace(CIM_NS), Namespace(GRID_NS)
    g = Graph()
    g.bind("cim", cim)
    g.bind("grid", g_ns)
    for eid in range(store.n_entities):
        cls = ID_CLASS[int(store.cols["etype"][eid])]
        g.add((URIRef(g_ns[f"e{eid}"]), RDF.type, cim[CLASSES[cls]["cim"]]))
    for pid, part in store.parts.items():
        name = ID_PRED[pid]
        src = list(part.src) + part.d_src
        dst = list(part.dst) + part.d_dst
        vf = list(part.vf) + part.d_vf
        vt = list(part.vt) + part.d_vt
        for s, o, a, b in zip(src, dst, vf, vt):
            if a <= t < b:
                g.add((URIRef(g_ns[f"e{s}"]), g_ns[name], URIRef(g_ns[f"e{o}"])))
    return g
