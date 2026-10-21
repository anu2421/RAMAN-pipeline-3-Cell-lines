import os
import re
import numpy as np
import pandas as pd

from .config import (
    AXIS_OFFSETS_CM1,
    CONTROL_PATH_OVERRIDE,
    CONTROL_ROOT,
    EXPOSURE,
    FILENAME_REPLICATE_REGEX,
    GROUP_DISPLAY,
    GROUP_KEYWORDS,
    GROUP_PATH_OVERRIDES,
    QUARTZ_FIT,
    REPLICATE_DIR_REGEX,
    REPLICATE_MERGE,
    SET_DIR_REGEX,
    SET_GROUP_DIR_REGEX,
    SET_GROUP_LETTERS,
)
from .preprocessing import preprocess_until_baseline, remove_quartz_800


def read_raman_csv(path):
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            lines = f.readlines()
        header_idx = next((i for i, l in enumerate(lines) if "RamanShift" in l and "Intensity" in l), None)
        if header_idx is None:
            print(f"    [WARN] Could not find data header in {path}, skipping.")
            return None
        df = pd.read_csv(path, skiprows=header_idx, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        shift_col = [c for c in df.columns if "RamanShift" in c][0]
        intens_col = [c for c in df.columns if "Intensity" in c][0]
        shift = pd.to_numeric(df[shift_col], errors="coerce").to_numpy()
        intens = pd.to_numeric(df[intens_col], errors="coerce").to_numpy()
        mask = ~np.isnan(shift) & ~np.isnan(intens)
        shift, intens = shift[mask], intens[mask]
        order = np.argsort(shift)
        return shift[order], intens[order].astype(float)
    except Exception as e:
        print(f"    [WARN] Failed to read {path}: {e}")
        return None


SONIC_RX = re.compile(r'soni|sono|(?<![a-z])son(?![a-z])|son$')


HEAT_RX = re.compile(r'heat|inactiv|kill|(?<![a-z])h[ik](?![a-z])')


SKIP_RX = re.compile(r'analysis|output|result')


def _normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _excluded(name):
    l = name.lower()
    if "soni" not in EXPOSURE.lower() and SONIC_RX.search(l):
        return True
    return bool(SKIP_RX.search(l))


def _exposure_rx():
    e = EXPOSURE.lower()
    if "heat" in e:
        return HEAT_RX, SONIC_RX
    if "soni" in e:
        return SONIC_RX, HEAT_RX
    return None, None


def _group_of(name):
    n = _normalize_name(name)
    hits = [g for g, kws in GROUP_KEYWORDS.items() if any(k in n for k in kws)]
    return hits[0] if len(hits) == 1 else None


def _csvs_under(d):
    out = []
    for dp, dns, fns in os.walk(d):
        dns[:] = sorted(x for x in dns if not _excluded(x))
        out += [os.path.join(dp, f) for f in sorted(fns) if f.lower().endswith(".csv")]
    return out


def discover_groups(ds, root, only=None):
    prefer, avoid = _exposure_rx()
    cands = {g: [] for g in GROUP_KEYWORDS}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if not _excluded(x))
        if os.path.normpath(dp) == os.path.normpath(root):
            continue
        g = _group_of(os.path.basename(dp))
        if g is None:
            continue
        rel = os.path.relpath(dp, root)
        n = len(_csvs_under(dp))
        if n:
            cands[g].append({"path": dp, "rel": rel, "depth": rel.count(os.sep),
                             "pref": bool(prefer) and any(prefer.search(c.lower()) for c in rel.split(os.sep)),
                             "avoid": bool(avoid) and any(avoid.search(c.lower()) for c in rel.split(os.sep)),
                             "n": n})
    chosen = {}
    for g in (only or GROUP_KEYWORDS):
        if (ds, g) in GROUP_PATH_OVERRIDES:
            chosen[g] = GROUP_PATH_OVERRIDES[(ds, g)]
            print(f"  {GROUP_DISPLAY[g]:<26} OVERRIDE -> {chosen[g]}")
            continue
        own = _normalize_name(ds)
        if len(cands[g]) > 1:
            print(f"  {GROUP_DISPLAY[g]:<26} candidates: {[d['rel'] for d in cands[g]]}")
        c = sorted(cands[g], key=lambda d: (not d["pref"], d["avoid"],
                                            own not in _normalize_name(d["rel"]) and len(cands[g]) > 1,
                                            d["depth"], d["rel"]))
        if not c:
            print(f"  {GROUP_DISPLAY[g]:<26} [WARN] no folder found")
            continue
        chosen[g] = c[0]["path"]
        print(f"  {GROUP_DISPLAY[g]:<26} -> {c[0]['rel']}  ({c[0]['n']} CSVs)")
        ties = [d for d in c[1:] if (d["pref"], d["avoid"], d["depth"]) == (c[0]["pref"], c[0]["avoid"], c[0]["depth"])]
        if ties:
            print(f"      [CHECK] other equally likely folders: {[d['rel'] for d in ties]} "
                  f"-> set GROUP_PATH_OVERRIDES if the choice is wrong")
    return chosen


def assign_replicates(gdir, files):
    dir_rx = re.compile(REPLICATE_DIR_REGEX, re.I)
    file_rx = re.compile(FILENAME_REPLICATE_REGEX, re.I)
    reps, unassigned = {}, []
    for p in files:
        parts = os.path.relpath(p, gdir).split(os.sep)
        rep = None
        for comp in parts[:-1]:
            m = dir_rx.match(comp.strip())
            if m:
                rep = f"R{int(m.group(1))}"
        if rep is None:
            m = file_rx.match(parts[-1])
            if m:
                rep = f"R{int(m.group(1))}"
        if rep is None:
            unassigned.append(p)
        else:
            reps.setdefault(rep, []).append(p)
    if unassigned and not reps:
        print(f"    [WARN] no R#/S# replicate naming found -> all {len(unassigned)} CSVs treated as R1")
        reps["R1"] = unassigned
    elif unassigned:
        print(f"    [WARN] {len(unassigned)} CSV(s) without replicate prefix ignored, "
              f"e.g. {[os.path.basename(u) for u in unassigned[:3]]}")
    return reps


def rep_sort(reps):
    return sorted(reps, key=lambda r: int(r[1:]))


def discover_set_groups(ds, root):
    g_rx, set_rx = re.compile(SET_GROUP_DIR_REGEX, re.I), re.compile(SET_DIR_REGEX, re.I)
    found = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if not _excluded(x))
        m = g_rx.match(os.path.basename(dp).strip())
        if not m:
            continue
        g = SET_GROUP_LETTERS[m.group(1).lower()]
        rep_no = int(m.group(2))
        parent = os.path.basename(os.path.dirname(dp)).strip()
        ms = set_rx.match(parent)
        if ms and int(ms.group(1)) != rep_no:
            print(f"  [CHECK] {os.path.relpath(dp, root)}: folder number {rep_no} differs from {parent}; using {parent}")
            rep_no = int(ms.group(1))
        files = _csvs_under(dp)
        dns[:] = []
        if files:
            found.setdefault(g, {}).setdefault(f"R{rep_no}", []).extend(files)
    for g in GROUP_KEYWORDS:
        reps = found.get(g, {})
        desc = ", ".join(f"{r}: {len(reps[r])} CSVs" for r in rep_sort(reps)) or "[WARN] nothing found"
        print(f"  {GROUP_DISPLAY[g]:<10} {desc}")
    return found


def find_group_folders(ds, root):
    croot = CONTROL_ROOT.get(ds)
    if not croot and ds not in CONTROL_PATH_OVERRIDE:
        return discover_groups(ds, root)
    found = discover_groups(ds, root, only=[g for g in GROUP_KEYWORDS if g != "control"])
    if ds in CONTROL_PATH_OVERRIDE:
        found["control"] = CONTROL_PATH_OVERRIDE[ds]
        print(f"  {'Control':<26} OVERRIDE -> {found['control']}")
    elif os.path.isdir(croot):
        print(f"  Control taken from: {croot}")
        cdir = discover_groups(ds, croot, only=("control",)).get("control")
        if cdir:
            found["control"] = cdir
    else:
        print(f"[ERROR] control folder not found: {croot}")
    return dict(sorted(found.items(), key=lambda kv: list(GROUP_KEYWORDS).index(kv[0])))


def load_group(ds, g, gdir, paths_by_rep=None):
    if paths_by_rep is None:
        paths_by_rep = assign_replicates(gdir, _csvs_under(gdir))
    for src, dst in REPLICATE_MERGE.get((ds, g), {}).items():
        if src in paths_by_rep:
            print(f"    {src}: relabelled as {dst}")
            paths_by_rep.setdefault(dst, []).extend(paths_by_rep.pop(src))
    out = {}
    for rep in rep_sort(paths_by_rep):
        cells = []
        for p in paths_by_rep[rep]:
            res = read_raman_csv(p)
            if res is None:
                continue
            x_raw = res[0] + AXIS_OFFSETS_CM1.get(ds, 0.0)
            pre = preprocess_until_baseline(x_raw, res[1], f"{rep}/{os.path.basename(p)}")
            if pre is not None:
                yq, qamp = remove_quartz_800(pre[0], pre[1]) if QUARTZ_FIT else (pre[1], np.nan)
                cells.append({"x": pre[0], "y": yq, "file": p, "quartz_amp": qamp, "x_start": float(res[0][0])})
        if cells:
            out[rep] = cells
            print(f"    {rep}: {len(cells)} usable cell(s)")
    return out
