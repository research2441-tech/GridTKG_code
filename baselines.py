"""
Baseline systems that receive the same explicit and derived facts and the same
measurements as GridTKG:

* PgSystem      - PostgreSQL 16 (row store, B-tree indexes, recursive CTEs)
* DuckSystem    - DuckDB (embedded column store, recursive CTEs)
* PolyglotSystem- Kuzu (embedded native property-graph DBMS, Cypher) for the
                  topology plus DuckDB for measurements; the application joins
                  the two result sets, as in a typical graph-DB + analytics stack.
"""
import os
import re
import shutil
import subprocess
import time

import numpy as np

from ontology import CLASS_ID, PRED_ID, INF_T

P = PRED_ID
C = CLASS_ID
INF = INF_T
ENT_COLS = ("etype", "e_sub", "e_reg", "e_kv", "e_rating", "e_age")


def _fact_frame(d, derived):
    rows = []
    for p in PRED_ID:
        arr = derived.get(p) if p in derived else d.get(f"fact_{p}")
        if arr is None or len(arr) == 0:
            continue
        via = arr[:, 4] if arr.shape[1] > 4 else np.full(len(arr), -1)
        rows.append(np.c_[np.full(len(arr), PRED_ID[p]), arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], via])
    return np.vstack(rows).astype(np.int64)


def _meas_frame(d):
    s0 = d["meta_blocks"]["Sensor"][0]
    V = d["V"]
    ns, T = V.shape
    sensor = np.repeat(np.arange(ns, dtype=np.int64) + s0, T)
    ts = np.tile(np.arange(T, dtype=np.int64), ns)
    return sensor, ts, V.reshape(-1)


# ---------------------------------------------------------------------------
# SQL text shared by PostgreSQL and DuckDB
# ---------------------------------------------------------------------------
SQL = dict(
    o2=f"""
WITH c AS (SELECT s FROM fact WHERE pred={P['containedIn']} AND o=$sub AND vt>={INF}),
b AS (SELECT c.s FROM c JOIN entity e ON e.id=c.s WHERE e.etype={C['Bus']}),
eq AS (SELECT s FROM c
       UNION SELECT f.s FROM fact f JOIN b ON f.o=b.s WHERE f.pred={P['connectedTo']} AND f.vt>={INF}
       UNION SELECT f.s FROM fact f JOIN b ON f.o=b.s WHERE f.pred={P['fromBus']} AND f.vt>={INF})
SELECT m.s, cur.val FROM fact m JOIN eq ON m.o=eq.s JOIN cur ON cur.sensor=m.s
WHERE m.pred={P['monitors']} AND m.vt>={INF} ORDER BY m.s""",
    o3=f"""
SELECT f.o, f.via, c.val FROM fact f
LEFT JOIN fact m ON m.pred={P['monitors']} AND m.o=f.via
LEFT JOIN cur c ON c.sensor=m.s
WHERE f.pred={P['busNeighbor']} AND f.s=$bus AND f.vf<=$t AND $t<f.vt ORDER BY f.o, f.via""",
    a1=f"""
WITH b AS (SELECT f.s FROM fact f JOIN entity e ON e.id=f.s
           WHERE f.pred={P['containedIn']} AND f.o=$sub AND e.etype={C['Bus']} AND f.vt>={INF}),
l AS (SELECT f.s FROM fact f JOIN b ON f.o=b.s JOIN entity e ON e.id=f.s
      WHERE f.pred={P['connectedTo']} AND e.etype={C['Load']} AND f.vt>={INF}),
sn AS (SELECT m.s FROM fact m JOIN l ON m.o=l.s WHERE m.pred={P['monitors']} AND m.vt>={INF}),
tot AS (SELECT ts, SUM(val) AS v FROM meas WHERE sensor IN (SELECT s FROM sn) AND ts>=$t0 AND ts<$t1 GROUP BY ts)
SELECT ((ts % $spd)*24) IDIV $spd AS h, AVG(v) FROM tot GROUP BY h ORDER BY h""",
    a2=f"""
SELECT e.reg, (m.ts-$t0) IDIV $spd AS day, SUM(m.val)*$dt FROM meas m JOIN entity e ON e.id=m.sensor
WHERE e.etype={C['SmartMeter']} AND m.ts>=$t0 AND m.ts<$t1 GROUP BY e.reg, day ORDER BY e.reg, day""",
    a3=f"""
WITH RECURSIVE r(n, d) AS (SELECT CAST($bus AS BIGINT), 0
    UNION SELECT f.o, r.d+1 FROM r JOIN fact f ON f.pred={P['busNeighbor']} AND f.s=r.n AND f.vf<=$t AND $t<f.vt
    WHERE r.d < $k),
nodes AS (SELECT DISTINCT n FROM r),
l AS (SELECT f.s FROM fact f JOIN nodes ON f.o=nodes.n JOIN entity e ON e.id=f.s
      WHERE f.pred={P['connectedTo']} AND e.etype={C['Load']} AND f.vt>={INF}),
sn AS (SELECT m.s FROM fact m JOIN l ON m.o=l.s WHERE m.pred={P['monitors']} AND m.vt>={INF})
SELECT (SELECT COUNT(*) FROM nodes), (SELECT COALESCE(SUM(c.val),0) FROM cur c JOIN sn ON c.sensor=sn.s)""",
    a4=f"""
SELECT s, COUNT(*) AS deg FROM fact WHERE pred={P['substationLink']} GROUP BY s ORDER BY deg DESC, s ASC LIMIT $k""",
    a5=f"""
WITH RECURSIVE r(n, d) AS (SELECT CAST($a AS BIGINT), 0
    UNION SELECT f.o, r.d+1 FROM r JOIN fact f ON f.pred={P['busNeighbor']} AND f.s=r.n AND f.vf<=$t AND $t<f.vt
    WHERE r.d < $maxd)
SELECT MIN(d) FROM r WHERE n=$b""",
    m1="SELECT ts, val FROM meas WHERE sensor=$s AND ts>=$t0 AND ts<$t1 ORDER BY ts",
    m2=f"""
SELECT e.id, AVG(ABS(m.val))/e.rating AS ld FROM entity e
JOIN fact mo ON mo.pred={P['monitors']} AND mo.o=e.id
JOIN meas m ON m.sensor=mo.s
WHERE e.etype={C['Transformer']} AND m.ts>=$t0 AND m.ts<$t1
GROUP BY e.id, e.rating HAVING AVG(ABS(m.val))/e.rating > $thr ORDER BY e.id""",
    m3=f"""
WITH o AS (
  SELECT e.id FROM entity e JOIN fact mo ON mo.pred={P['monitors']} AND mo.o=e.id
  JOIN meas m ON m.sensor=mo.s
  WHERE e.etype={C['Transformer']} AND m.ts>=$t0 AND m.ts<$t1
  GROUP BY e.id, e.rating HAVING AVG(ABS(m.val))/e.rating > $thr)
SELECT o.id, a.o, MAX(ABS(m.val))/e2.rating AS ld FROM o
JOIN fact a ON a.pred={P['adjacentTo']} AND a.s=o.id
JOIN entity e2 ON e2.id=a.o
JOIN fact mo ON mo.pred={P['monitors']} AND mo.o=a.o
JOIN meas m ON m.sensor=mo.s AND m.ts>=$t0 AND m.ts<$t1
GROUP BY o.id, a.o, e2.rating ORDER BY o.id, a.o""",
    o1_ins="INSERT INTO meas VALUES ($s, $ts, $v)",
    o1_cur="""INSERT INTO cur VALUES ($s, $ts, $v) ON CONFLICT (sensor) DO UPDATE
SET ts=excluded.ts, val=excluded.val WHERE excluded.ts >= cur.ts""",
    o4_ends=f"SELECT o FROM fact WHERE pred IN ({P['fromBus']},{P['toBus']},{P['hvBus']},{P['lvBus']}) AND s=$br ORDER BY pred",
    o4_ent="INSERT INTO entity VALUES ($id, " + str(C['FaultEvent']) + ", $sub, $reg, 0, 0, 0)",
    o4_aff=f"INSERT INTO fact VALUES ({P['affects']}, $id, $br, $t0, $t1, -1)",
    o4_split=f"""INSERT INTO fact SELECT pred, s, o, $t1, vt, via FROM fact
WHERE pred={P['busNeighbor']} AND s IN ($x, $y) AND via=$br AND vf<=$t0 AND $t0<vt AND $t1<vt""",
    o4_close=f"""UPDATE fact SET vt=$t0 WHERE pred={P['busNeighbor']} AND s IN ($x, $y) AND via=$br AND vf<=$t0 AND $t0<vt""",
    o4_attr="SELECT sub, reg FROM entity WHERE id=$br",
    s2_kv="SELECT kv, sub FROM entity WHERE id=$bus",
)


def _sub_named(sql, style):
    if style == "pg":
        return re.sub(r"\$([a-z_0-9]+)", r"%(\1)s", sql.replace("%", "%%").replace(" IDIV ", " / "))
    return sql.replace(" IDIV ", " // ")


class _SqlSystem:
    """Common logic for PostgreSQL and DuckDB."""
    style = "duck"

    def q(self, key):
        return self._sql[key]

    def _prep(self):
        self._sql = {k: _sub_named(v, self.style) for k, v in SQL.items()}

    # ---- operations ---------------------------------------------------
    def ingest(self, sensor, slot, value):
        v = None if np.isnan(value) else float(value)
        self.begin()
        self.ex(self.q("o1_ins"), dict(s=int(sensor), ts=int(slot), v=v))
        self.ex(self.q("o1_cur"), dict(s=int(sensor), ts=int(slot), v=v))
        self.commit()

    def substation_status(self, sub):
        r = self.fetch(self.q("o2"), dict(sub=int(sub)))
        return np.array([x[0] for x in r], np.int64), np.array([np.nan if x[1] is None else x[1] for x in r], np.float32)

    def bus_neighbours_now(self, bus, t):
        r = self.fetch(self.q("o3"), dict(bus=int(bus), t=int(t)))
        return [(int(a), int(b), None if c is None else float(c)) for a, b, c in r]

    def switch_out(self, branch, t0, t1):
        self.begin()
        ends = [x[0] for x in self.fetch(self.q("o4_ends"), dict(br=int(branch)))]
        sub, reg = self.fetch(self.q("o4_attr"), dict(br=int(branch)))[0]
        fe = self.next_id
        self.next_id += 1
        self.ex(self.q("o4_ent"), dict(id=fe, sub=int(sub), reg=int(reg)))
        self.ex(self.q("o4_aff"), dict(id=fe, br=int(branch), t0=int(t0), t1=int(t1)))
        prm = dict(x=int(ends[0]), y=int(ends[1]), br=int(branch), t0=int(t0), t1=int(t1))
        self.ex(self.q("o4_split"), prm)
        self.ex(self.q("o4_close"), {k: v for k, v in prm.items() if k != "t1"})
        self.commit()
        return fe

    def add_substation_bay(self, region, bus_attach, kv_hv, kv_lv, rating, t0):
        self.begin()
        kv, sub_attach = self.fetch(self.q("s2_kv"), dict(bus=int(bus_attach)))[0]
        if abs(kv - kv_hv) > 0.05 * kv_hv or not kv_hv >= kv_lv:
            self.rollback()
            raise ValueError("constraint")
        i = self.next_id
        self.next_id += 7
        sub, hv, lv, tr, ln, s1, s2 = range(i, i + 7)
        ents = [(sub, C["Substation"], sub, region, 0, 0), (hv, C["Bus"], sub, region, kv_hv, 0),
                (lv, C["Bus"], sub, region, kv_lv, 0), (tr, C["Transformer"], sub, region, kv_hv, rating),
                (ln, C["Line"], sub, region, kv_hv, rating), (s1, C["SCADA"], sub, region, 0, rating),
                (s2, C["SCADA"], sub, region, 0, rating)]
        self.insert_entities(ents)
        f = [("locatedIn", sub, region, -1), ("containedIn", hv, sub, -1), ("containedIn", lv, sub, -1),
             ("containedIn", tr, sub, -1), ("hvBus", tr, hv, -1), ("lvBus", tr, lv, -1),
             ("fromBus", ln, hv, -1), ("toBus", ln, bus_attach, -1), ("monitors", s1, tr, -1),
             ("monitors", s2, ln, -1), ("busNeighbor", hv, lv, tr), ("busNeighbor", lv, hv, tr),
             ("busNeighbor", hv, bus_attach, ln), ("busNeighbor", bus_attach, hv, ln),
             ("substationLink", sub, sub_attach, -1), ("substationLink", sub_attach, sub, -1)]
        self.insert_facts([(P[p], s, o, t0, INF, v) for p, s, o, v in f])
        self.commit()
        return sub

    def load_profile(self, sub, t0, t1, spd):
        r = self.fetch(self.q("a1"), dict(sub=int(sub), t0=int(t0), t1=int(t1), spd=int(spd)))
        out = np.full(24, np.nan)
        for h, v in r:
            out[int(h)] = np.nan if v is None else v
        return out

    def region_daily_energy(self, t0, t1, spd, dt_h):
        nd = (t1 - t0) // spd
        r = self.fetch(self.q("a2"), dict(t0=int(t0), t1=int(t0 + nd * spd), spd=int(spd), dt=float(dt_h)))
        nreg = max(int(x[0]) for x in r) + 1
        out = np.zeros((nreg, nd))
        for reg, day, v in r:
            out[int(reg), int(day)] = 0.0 if v is None else v
        return out

    def khop(self, bus, k, t):
        n, s = self.fetch(self.q("a3"), dict(bus=int(bus), t=int(t), k=int(k)))[0]
        return int(n), float(s or 0.0)

    def top_substations(self, k=20):
        return [(int(a), int(b)) for a, b in self.fetch(self.q("a4"), dict(k=int(k)))]

    def shortest_hops(self, a, b, t, max_depth=30):
        r = self.fetch(self.q("a5"), dict(a=int(a), b=int(b), t=int(t), maxd=int(max_depth)))[0][0]
        return -1 if r is None else int(r)

    def history(self, sensors, t0, t1):
        r = self.fetch(self.q("m1"), dict(s=int(sensors[0]), t0=int(t0), t1=int(t1)))
        return np.array([[np.nan if v is None else v for _, v in r]], np.float32)

    def overloaded_transformers(self, t, window, thr):
        r = self.fetch(self.q("m2"), dict(t0=int(t - window), t1=int(t), thr=float(thr)))
        return np.array([x[0] for x in r], np.int64), np.array([x[1] for x in r])

    def overload_context(self, t, window, thr):
        r = self.fetch(self.q("m3"), dict(t0=int(t - window), t1=int(t), thr=float(thr)))
        best = {}
        for tr, nb, ld in r:
            if ld is None:
                continue
            if tr not in best or ld > best[tr][1]:
                best[tr] = (int(nb), float(ld))
        return [(int(tr), nb, ld) for tr, (nb, ld) in sorted(best.items())]


# ---------------------------------------------------------------------------
class PgSystem(_SqlSystem):
    style = "pg"

    def __init__(self, workdir, port=54329):
        self.dir = os.path.join(workdir, "pgdata")
        self.port = port
        self.sock = os.path.join(workdir, "pgsock")
        self._prep()

    def start(self, fresh=True):
        import psycopg
        bindir = "/usr/lib/postgresql/16/bin"
        if fresh:
            subprocess.run([f"{bindir}/pg_ctl", "-D", self.dir, "-m", "immediate", "stop"], capture_output=True,
                           user="postgres" if os.geteuid() == 0 else None)
            shutil.rmtree(self.dir, ignore_errors=True)
            os.makedirs(self.sock, exist_ok=True)
            os.makedirs(self.dir, exist_ok=True)
            if os.geteuid() == 0:
                import pwd
                pw = pwd.getpwnam("postgres")
                for p in (self.dir, self.sock):
                    os.chown(p, pw.pw_uid, pw.pw_gid)
                os.chmod(self.dir, 0o700)
            env = dict(os.environ)
            subprocess.run([f"{bindir}/initdb", "-D", self.dir, "-U", "bench", "--auth=trust", "-E", "UTF8"],
                           check=True, capture_output=True, env=env, user="postgres" if os.geteuid() == 0 else None)
            with open(os.path.join(self.dir, "postgresql.conf"), "a") as fh:
                fh.write(f"""
listen_addresses = ''
unix_socket_directories = '{self.sock}'
port = {self.port}
shared_buffers = 1GB
work_mem = 64MB
maintenance_work_mem = 512MB
effective_cache_size = 4GB
synchronous_commit = off
wal_writer_delay = 10ms
max_wal_size = 4GB
jit = off
max_parallel_workers_per_gather = 2
random_page_cost = 1.1
""")
        subprocess.run([f"{bindir}/pg_ctl", "-D", self.dir, "-l", os.path.join(self.dir, "log.txt"), "-w", "start"],
                       check=True, capture_output=True, user="postgres" if os.geteuid() == 0 else None)
        self.conn = psycopg.connect(host=self.sock, port=self.port, user="bench", dbname="postgres",
                                    autocommit=True, prepare_threshold=1)
        self.cur = self.conn.cursor()
        self.version = self.fetch("SHOW server_version", {})[0][0]

    def stop(self):
        try:
            self.conn.close()
        except Exception:
            pass
        subprocess.run(["/usr/lib/postgresql/16/bin/pg_ctl", "-D", self.dir, "-m", "fast", "stop"],
                       capture_output=True, user="postgres" if os.geteuid() == 0 else None)

    def ex(self, sql, prm):
        self.cur.execute(sql, prm)

    def fetch(self, sql, prm):
        self.cur.execute(sql, prm)
        return self.cur.fetchall()

    def begin(self):
        self.cur.execute("BEGIN")

    def commit(self):
        self.cur.execute("COMMIT")

    def rollback(self):
        self.cur.execute("ROLLBACK")

    def load(self, d, derived):
        cur = self.cur
        cur.execute("""CREATE TABLE entity(id INT PRIMARY KEY, etype SMALLINT, sub INT, reg INT, kv REAL, rating REAL, age REAL);
CREATE TABLE fact(pred SMALLINT, s INT, o INT, vf BIGINT, vt BIGINT, via INT);
CREATE TABLE meas(sensor INT, ts INT, val REAL);
CREATE TABLE cur(sensor INT PRIMARY KEY, ts INT, val REAL);""")
        n = len(d["etype"])
        with cur.copy("COPY entity FROM STDIN") as cp:
            for i in range(n):
                cp.write_row((i, int(d["etype"][i]), int(d["e_sub"][i]), int(d["e_reg"][i]),
                              float(d["e_kv"][i]), float(d["e_rating"][i]), float(d["e_age"][i])))
        fr = _fact_frame(d, derived)
        with cur.copy("COPY fact FROM STDIN") as cp:
            for r in fr.tolist():
                cp.write_row(r)
        sensor, ts, val = _meas_frame(d)
        import io
        buf = io.StringIO()
        nanm = np.isnan(val)
        chunk = 2_000_000
        for a in range(0, len(sensor), chunk):
            b = a + chunk
            buf = io.StringIO()
            arr = np.c_[sensor[a:b], ts[a:b]]
            lines = [f"{s}\t{t}\t{'\\N' if nn else repr(v)}\n"
                     for (s, t), v, nn in zip(arr.tolist(), val[a:b].tolist(), nanm[a:b].tolist())]
            buf.write("".join(lines))
            with cur.copy("COPY meas FROM STDIN") as cp:
                cp.write(buf.getvalue())
        s0 = d["meta_blocks"]["Sensor"][0]
        V = d["V"]
        with cur.copy("COPY cur FROM STDIN") as cp:
            for k in range(V.shape[0]):
                v = V[k, -1]
                cp.write_row((s0 + k, V.shape[1] - 1, None if np.isnan(v) else float(v)))
        cur.execute("""CREATE INDEX fact_ps ON fact(pred, s); CREATE INDEX fact_po ON fact(pred, o);
CREATE INDEX fact_pv ON fact(pred, via); CREATE INDEX meas_st ON meas(sensor, ts);
ANALYZE; CHECKPOINT;""")
        self.next_id = n

    def insert_entities(self, rows):
        self.cur.executemany("INSERT INTO entity VALUES (%s,%s,%s,%s,%s,%s,0)", rows)

    def insert_facts(self, rows):
        self.cur.executemany("INSERT INTO fact VALUES (%s,%s,%s,%s,%s,%s)", rows)

    def ingest_slot(self, slot, values, sensors):
        v = [None if np.isnan(x) else float(x) for x in values.tolist()]
        s = sensors.tolist()
        self.begin()
        self.cur.execute("INSERT INTO meas SELECT unnest(%s::int[]), %s, unnest(%s::real[])", (s, int(slot), v))
        self.cur.execute("""UPDATE cur SET ts=%s, val=x.v FROM (SELECT unnest(%s::int[]) AS s, unnest(%s::real[]) AS v) x
WHERE cur.sensor=x.s""", (int(slot), s, v))
        self.commit()

    def footprint_bytes(self):
        tot = self.fetch("SELECT pg_database_size('postgres')", {})[0][0]
        wal = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(self.dir, "pg_wal")) for f in fs)
        return int(tot), int(wal)


# ---------------------------------------------------------------------------
class DuckSystem(_SqlSystem):
    style = "duck"

    def __init__(self, workdir, threads=2):
        self.path = os.path.join(workdir, "duck.db")
        self.threads = threads
        self._prep()

    def start(self, fresh=True):
        import duckdb
        if fresh:
            for p in (self.path, self.path + ".wal"):
                if os.path.exists(p):
                    os.remove(p)
        self.con = duckdb.connect(self.path)
        self.con.execute(f"SET threads={self.threads}; SET memory_limit='3GB'; SET preserve_insertion_order=false")
        self.version = duckdb.__version__

    def stop(self):
        self.con.close()

    def ex(self, sql, prm):
        self.con.execute(sql, prm)

    def fetch(self, sql, prm):
        return self.con.execute(sql, prm).fetchall()

    def begin(self):
        self.con.execute("BEGIN TRANSACTION")

    def commit(self):
        self.con.execute("COMMIT")

    def rollback(self):
        self.con.execute("ROLLBACK")

    def load(self, d, derived, with_facts=True):
        import pandas as pd
        con = self.con
        n = len(d["etype"])
        ent = pd.DataFrame(dict(id=np.arange(n, dtype=np.int32), etype=d["etype"].astype(np.int16),
                                sub=d["e_sub"], reg=d["e_reg"], kv=d["e_kv"], rating=d["e_rating"], age=d["e_age"]))
        con.execute("CREATE TABLE entity(id INTEGER PRIMARY KEY, etype SMALLINT, sub INTEGER, reg INTEGER, kv REAL, rating REAL, age REAL)")
        con.execute("INSERT INTO entity SELECT * FROM ent")
        if with_facts:
            fr = _fact_frame(d, derived)
            fdf = pd.DataFrame(dict(pred=fr[:, 0].astype(np.int16), s=fr[:, 1].astype(np.int32), o=fr[:, 2].astype(np.int32),
                                    vf=fr[:, 3], vt=fr[:, 4], via=fr[:, 5].astype(np.int32)))
            con.execute("CREATE TABLE fact(pred SMALLINT, s INTEGER, o INTEGER, vf BIGINT, vt BIGINT, via INTEGER)")
            con.execute("INSERT INTO fact SELECT * FROM fdf ORDER BY pred, s")
            con.execute("CREATE INDEX fact_ps ON fact(pred, s); CREATE INDEX fact_po ON fact(pred, o); CREATE INDEX fact_pv ON fact(pred, via)")
        sensor, ts, val = _meas_frame(d)
        mdf = pd.DataFrame(dict(sensor=sensor.astype(np.int32), ts=ts.astype(np.int32), val=val))
        con.execute("CREATE TABLE meas(sensor INTEGER, ts INTEGER, val REAL)")
        con.execute("INSERT INTO meas SELECT sensor, ts, CASE WHEN isnan(val) THEN NULL ELSE val END FROM mdf")
        s0 = d["meta_blocks"]["Sensor"][0]
        V = d["V"]
        cdf = pd.DataFrame(dict(sensor=np.arange(V.shape[0], dtype=np.int32) + s0, ts=np.full(V.shape[0], V.shape[1] - 1, np.int32),
                                val=V[:, -1]))
        con.execute("CREATE TABLE cur(sensor INTEGER PRIMARY KEY, ts INTEGER, val REAL)")
        con.execute("INSERT INTO cur SELECT sensor, ts, CASE WHEN isnan(val) THEN NULL ELSE val END FROM cdf")
        con.execute("CHECKPOINT")
        self.next_id = n

    def insert_entities(self, rows):
        self.con.executemany("INSERT INTO entity VALUES (?,?,?,?,?,?,0)", rows)

    def insert_facts(self, rows):
        self.con.executemany("INSERT INTO fact VALUES (?,?,?,?,?,?)", rows)

    def ingest_slot(self, slot, values, sensors):
        import pandas as pd
        x = pd.DataFrame(dict(s=sensors.astype(np.int32), v=values.astype(np.float32)))
        self.con.register("xslot", x)
        self.begin()
        self.con.execute(f"INSERT INTO meas SELECT s, {int(slot)}, CASE WHEN isnan(v) THEN NULL ELSE v END FROM xslot")
        self.con.execute(f"UPDATE cur SET ts={int(slot)}, val=CASE WHEN isnan(xslot.v) THEN NULL ELSE xslot.v END FROM xslot WHERE cur.sensor=xslot.s")
        self.commit()
        self.con.unregister("xslot")

    def footprint_bytes(self):
        self.con.execute("CHECKPOINT")
        wal = os.path.getsize(self.path + ".wal") if os.path.exists(self.path + ".wal") else 0
        return int(os.path.getsize(self.path)), int(wal)


# ---------------------------------------------------------------------------
class PolyglotSystem:
    """Kuzu (topology, Cypher) + DuckDB (measurements, sensor dimension)."""

    def __init__(self, workdir, threads=2):
        self.kpath = os.path.join(workdir, "kuzu_db")
        self.duck = DuckSystem(os.path.join(workdir), threads)
        self.duck.path = os.path.join(workdir, "poly_duck.db")
        self.threads = threads

    def start(self, fresh=True):
        import kuzu
        if fresh:
            shutil.rmtree(self.kpath, ignore_errors=True)
            if os.path.exists(self.kpath):
                os.remove(self.kpath)
        self.db = kuzu.Database(self.kpath, buffer_pool_size=2 * 1024 ** 3, max_num_threads=self.threads)
        self.k = kuzu.Connection(self.db, num_threads=self.threads)
        self.duck.start(fresh)
        self.version = f"Kuzu {kuzu.__version__} + DuckDB {self.duck.version}"

    def stop(self):
        self.duck.stop()
        del self.k
        del self.db

    def kq(self, q, prm=None):
        return self.k.execute(q, prm or {}).get_all()

    def load(self, d, derived):
        import pandas as pd
        n = len(d["etype"])
        ent = pd.DataFrame(dict(id=np.arange(n, dtype=np.int64), etype=d["etype"].astype(np.int16),
                                sub=d["e_sub"].astype(np.int32), reg=d["e_reg"].astype(np.int32),
                                kv=d["e_kv"].astype(np.float32), rating=d["e_rating"].astype(np.float32)))
        self.k.execute("CREATE NODE TABLE Ent(id INT64, etype INT16, sub INT32, reg INT32, kv FLOAT, rating FLOAT, PRIMARY KEY(id))")
        self.k.execute("CREATE REL TABLE F(FROM Ent TO Ent, pred INT16, vf INT64, vt INT64, via INT64)")
        self.k.execute("COPY Ent FROM ent")
        fr = _fact_frame(d, derived)
        fdf = pd.DataFrame({"from": fr[:, 1], "to": fr[:, 2], "pred": fr[:, 0].astype(np.int16), "vf": fr[:, 3],
                            "vt": fr[:, 4], "via": fr[:, 5]})
        self.k.execute("COPY F FROM fdf")
        # DuckDB side: measurements + entity/sensor dimension (replicated metadata)
        self.duck.load(d, derived, with_facts=False)
        s0 = d["meta_blocks"]["Sensor"][0]
        smap = pd.DataFrame(dict(sensor=np.arange(len(d["sens_target"]), dtype=np.int32) + s0,
                                 target=d["sens_target"].astype(np.int32)))
        self.duck.con.execute("CREATE TABLE smap AS SELECT * FROM smap")
        self.duck.con.execute("CHECKPOINT")
        self.next_id = n

    # -- helpers
    def _cur(self, sensors):
        if len(sensors) == 0:
            return []
        return self.duck.fetch("SELECT sensor, val FROM cur WHERE sensor IN (SELECT unnest($s)) ORDER BY sensor",
                               dict(s=[int(x) for x in sensors]))

    def ingest(self, sensor, slot, value):
        self.duck.ingest(sensor, slot, value)

    def ingest_slot(self, slot, values, sensors):
        self.duck.ingest_slot(slot, values, sensors)

    def substation_status(self, sub):
        r = self.kq(f"""
MATCH (e:Ent)-[c:F]->(s:Ent {{id:$sub}}) WHERE c.pred={P['containedIn']} AND c.vt>={INF}
OPTIONAL MATCH (x:Ent)-[f:F]->(e) WHERE e.etype={C['Bus']} AND f.pred IN [{P['connectedTo']},{P['fromBus']}] AND f.vt>={INF}
WITH collect(DISTINCT e.id) + collect(DISTINCT x.id) AS ids
UNWIND ids AS q
MATCH (sn:Ent)-[m:F]->(t:Ent {{id:q}}) WHERE m.pred={P['monitors']} AND m.vt>={INF}
RETURN DISTINCT sn.id""", dict(sub=int(sub)))
        sens = sorted(x[0] for x in r)
        vals = dict(self._cur(sens))
        return np.array(sens, np.int64), np.array([np.nan if vals.get(s) is None else vals[s] for s in sens], np.float32)

    def bus_neighbours_now(self, bus, t):
        r = self.kq(f"""
MATCH (a:Ent {{id:$bus}})-[f:F]->(b:Ent) WHERE f.pred={P['busNeighbor']} AND f.vf<=$t AND f.vt>$t
OPTIONAL MATCH (sn:Ent)-[m:F]->(br:Ent {{id:f.via}}) WHERE m.pred={P['monitors']}
RETURN b.id, f.via, sn.id ORDER BY b.id, f.via""", dict(bus=int(bus), t=int(t)))
        vals = dict(self._cur([x[2] for x in r if x[2] is not None]))
        return [(int(a), int(b), None if s is None or vals.get(s) is None else float(vals[s])) for a, b, s in r]

    def switch_out(self, branch, t0, t1):
        fe = self.next_id
        self.next_id += 1
        self.k.execute("BEGIN TRANSACTION")
        ends = [x[0] for x in self.kq(f"""MATCH (b:Ent {{id:$br}})-[f:F]->(x:Ent)
WHERE f.pred IN [{P['fromBus']},{P['toBus']},{P['hvBus']},{P['lvBus']}] RETURN x.id ORDER BY f.pred""", dict(br=int(branch)))]
        sub, reg = self.kq("MATCH (b:Ent {id:$br}) RETURN b.sub, b.reg", dict(br=int(branch)))[0]
        self.k.execute(f"CREATE (:Ent {{id:$id, etype:{C['FaultEvent']}, sub:$sub, reg:$reg, kv:0.0, rating:0.0}})",
                       dict(id=fe, sub=int(sub), reg=int(reg)))
        self.k.execute(f"MATCH (e:Ent {{id:$id}}), (b:Ent {{id:$br}}) CREATE (e)-[:F {{pred:{P['affects']}, vf:$t0, vt:$t1, via:-1}}]->(b)",
                       dict(id=fe, br=int(branch), t0=int(t0), t1=int(t1)))
        prm = dict(x=int(ends[0]), y=int(ends[1]), br=int(branch), t0=int(t0), t1=int(t1))
        self.k.execute(f"""MATCH (a:Ent)-[f:F]->(b:Ent) WHERE a.id IN [$x,$y] AND b.id IN [$x,$y] AND f.pred={P['busNeighbor']}
AND f.via=$br AND f.vf<=$t0 AND $t0<f.vt AND $t1<f.vt
CREATE (a)-[:F {{pred:{P['busNeighbor']}, vf:$t1, vt:f.vt, via:$br}}]->(b)""", prm)
        self.k.execute(f"""MATCH (a:Ent)-[f:F]->(b:Ent) WHERE a.id IN [$x,$y] AND b.id IN [$x,$y] AND f.pred={P['busNeighbor']}
AND f.via=$br AND f.vf<=$t0 AND $t0<f.vt SET f.vt=$t0""", {k: v for k, v in prm.items() if k != "t1"})
        self.k.execute("COMMIT")
        return fe

    def add_substation_bay(self, region, bus_attach, kv_hv, kv_lv, rating, t0):
        self.k.execute("BEGIN TRANSACTION")
        kv, sub_attach = self.kq("MATCH (b:Ent {id:$b}) RETURN b.kv, b.sub", dict(b=int(bus_attach)))[0]
        if abs(kv - kv_hv) > 0.05 * kv_hv or not kv_hv >= kv_lv:
            self.k.execute("ROLLBACK")
            raise ValueError("constraint")
        i = self.next_id
        self.next_id += 7
        sub, hv, lv, tr, ln, s1, s2 = range(i, i + 7)
        ents = [(sub, C["Substation"], sub, region, 0, 0), (hv, C["Bus"], sub, region, kv_hv, 0),
                (lv, C["Bus"], sub, region, kv_lv, 0), (tr, C["Transformer"], sub, region, kv_hv, rating),
                (ln, C["Line"], sub, region, kv_hv, rating), (s1, C["SCADA"], sub, region, 0, rating),
                (s2, C["SCADA"], sub, region, 0, rating)]
        for e in ents:
            self.k.execute("CREATE (:Ent {id:$a, etype:$b, sub:$c, reg:$d, kv:$e, rating:$f})",
                           dict(a=e[0], b=e[1], c=e[2], d=int(e[3]), e=float(e[4]), f=float(e[5])))
        f = [("locatedIn", sub, region, -1), ("containedIn", hv, sub, -1), ("containedIn", lv, sub, -1),
             ("containedIn", tr, sub, -1), ("hvBus", tr, hv, -1), ("lvBus", tr, lv, -1),
             ("fromBus", ln, hv, -1), ("toBus", ln, bus_attach, -1), ("monitors", s1, tr, -1),
             ("monitors", s2, ln, -1), ("busNeighbor", hv, lv, tr), ("busNeighbor", lv, hv, tr),
             ("busNeighbor", hv, bus_attach, ln), ("busNeighbor", bus_attach, hv, ln),
             ("substationLink", sub, int(sub_attach), -1), ("substationLink", int(sub_attach), sub, -1)]
        for p, s, o, v in f:
            self.k.execute("MATCH (a:Ent {id:$s}), (b:Ent {id:$o}) CREATE (a)-[:F {pred:$p, vf:$vf, vt:$vt, via:$v}]->(b)",
                           dict(s=int(s), o=int(o), p=P[p], vf=int(t0), vt=INF, v=int(v)))
        self.k.execute("COMMIT")
        # replicate the sensor dimension to the analytical store
        self.duck.begin()
        self.duck.insert_entities(ents)
        self.duck.con.executemany("INSERT INTO smap VALUES (?, ?)", [(s1, tr), (s2, ln)])
        self.duck.commit()
        return sub

    def _load_sensors_of_buses(self, buses_cypher, prm):
        return self.kq(buses_cypher, prm)

    def load_profile(self, sub, t0, t1, spd):
        r = self.kq(f"""
MATCH (b:Ent)-[c:F]->(s:Ent {{id:$sub}}) WHERE c.pred={P['containedIn']} AND b.etype={C['Bus']} AND c.vt>={INF}
MATCH (l:Ent)-[f:F]->(b) WHERE f.pred={P['connectedTo']} AND l.etype={C['Load']} AND f.vt>={INF}
MATCH (sn:Ent)-[m:F]->(l) WHERE m.pred={P['monitors']} AND m.vt>={INF}
RETURN DISTINCT sn.id""", dict(sub=int(sub)))
        sens = [x[0] for x in r]
        out = np.full(24, np.nan)
        if not sens:
            return out
        rr = self.duck.fetch("""WITH tot AS (SELECT ts, SUM(val) AS v FROM meas WHERE sensor IN (SELECT unnest($s)) AND ts>=$t0 AND ts<$t1 GROUP BY ts)
SELECT ((ts % $spd)*24) // $spd AS h, AVG(v) FROM tot GROUP BY h ORDER BY h""", dict(s=sens, t0=int(t0), t1=int(t1), spd=int(spd)))
        for h, v in rr:
            out[int(h)] = np.nan if v is None else v
        return out

    def region_daily_energy(self, t0, t1, spd, dt_h):
        return self.duck.region_daily_energy(t0, t1, spd, dt_h)

    def khop(self, bus, k, t):
        r = self.kq(f"""
MATCH (a:Ent {{id:$bus}})-[:F*1..{int(k)} (e, n | WHERE e.pred={P['busNeighbor']} AND e.vf<=$t AND e.vt>$t)]->(b:Ent)
WITH collect(DISTINCT b.id) + [CAST($bus AS INT64)] AS ids
UNWIND ids AS q
WITH DISTINCT q
MATCH (x:Ent {{id:q}})
OPTIONAL MATCH (l:Ent)-[c:F]->(x) WHERE c.pred={P['connectedTo']} AND l.etype={C['Load']} AND c.vt>={INF}
OPTIONAL MATCH (sn:Ent)-[m:F]->(l) WHERE m.pred={P['monitors']} AND m.vt>={INF}
RETURN count(DISTINCT x.id), collect(DISTINCT sn.id)""", dict(bus=int(bus), t=int(t)))
        n, sens = r[0]
        sens = [s for s in (sens or []) if s is not None]
        tot = 0.0
        if sens:
            v = self.duck.fetch("SELECT COALESCE(SUM(val),0) FROM cur WHERE sensor IN (SELECT unnest($s))", dict(s=sens))[0][0]
            tot = float(v)
        return int(n), tot

    def top_substations(self, k=20):
        r = self.kq(f"""MATCH (a:Ent)-[f:F]->(b:Ent) WHERE f.pred={P['substationLink']}
RETURN a.id AS s, count(*) AS deg ORDER BY deg DESC, s ASC LIMIT {int(k)}""")
        return [(int(a), int(b)) for a, b in r]

    def shortest_hops(self, a, b, t, max_depth=30):
        if a == b:
            return 0
        r = self.kq(f"""MATCH (x:Ent {{id:$a}})-[r:F* SHORTEST 1..{int(max_depth)} (e, n | WHERE e.pred={P['busNeighbor']} AND e.vf<=$t AND e.vt>$t)]->(y:Ent {{id:$b}})
RETURN length(r)""", dict(a=int(a), b=int(b), t=int(t)))
        return int(r[0][0]) if r else -1

    def history(self, sensors, t0, t1):
        return self.duck.history(sensors, t0, t1)

    def overloaded_transformers(self, t, window, thr):
        r = self.duck.fetch(f"""SELECT e.id, AVG(ABS(m.val))/e.rating FROM entity e JOIN smap ON smap.target=e.id
JOIN meas m ON m.sensor=smap.sensor WHERE e.etype={C['Transformer']} AND m.ts>=$t0 AND m.ts<$t1
GROUP BY e.id, e.rating HAVING AVG(ABS(m.val))/e.rating > $thr ORDER BY e.id""", dict(t0=int(t - window), t1=int(t), thr=float(thr)))
        return np.array([x[0] for x in r], np.int64), np.array([x[1] for x in r])

    def overload_context(self, t, window, thr):
        trs, _ = self.overloaded_transformers(t, window, thr)
        if len(trs) == 0:
            return []
        adj = self.kq(f"""MATCH (a:Ent)-[f:F]->(b:Ent) WHERE a.id IN $ids AND f.pred={P['adjacentTo']} RETURN a.id, b.id""",
                      dict(ids=[int(x) for x in trs]))
        if not adj:
            return []
        import pandas as pd
        pairs = pd.DataFrame(dict(tr=[a for a, _ in adj], nb=[b for _, b in adj]))
        self.duck.con.register("pairs", pairs)
        r = self.duck.fetch("""SELECT p.tr, p.nb, MAX(ABS(m.val))/e.rating FROM pairs p JOIN entity e ON e.id=p.nb
JOIN smap ON smap.target=p.nb JOIN meas m ON m.sensor=smap.sensor AND m.ts>=$t0 AND m.ts<$t1
GROUP BY p.tr, p.nb, e.rating""", dict(t0=int(t - window), t1=int(t)))
        self.duck.con.unregister("pairs")
        best = {}
        for tr, nb, ld in r:
            if ld is None:
                continue
            if tr not in best or ld > best[tr][1] or (ld == best[tr][1] and nb < best[tr][0]):
                best[tr] = (int(nb), float(ld))
        return [(int(tr), nb, ld) for tr, (nb, ld) in sorted(best.items())]

    def footprint_bytes(self):
        k = 0
        if os.path.isdir(self.kpath):
            k = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(self.kpath) for f in fs)
        else:
            k = os.path.getsize(self.kpath)
            for ext in (".wal",):
                if os.path.exists(self.kpath + ext):
                    k += os.path.getsize(self.kpath + ext)
        dsz, dwal = self.duck.footprint_bytes()
        return int(k + dsz), int(dwal)
