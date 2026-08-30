"""Per-tracer cohort constants for the store's PET z-score inversion.

Channel 1 of the store is z = (SUV - mu_full) / sd_full (build_store.py restores the
full-volume statistics after the body crop), so the inversion is SUV = z*sd + mu and
the constants to use are the cohort medians of mu_full / sd_full. networks_re.py
currently uses ONE pair for both tracers; this splits them.
"""
import glob, os, pickle, statistics as st

files = sorted(glob.glob("/content/nnUNet/prep_local/Dataset998_AutoPETV/nnUNetPlans_3d_fullres/*.pkl"))
groups = {"fdg": [], "psma": [], "other": []}
skipped = 0
for f in files:
    case = os.path.basename(f)[:-4]
    with open(f, "rb") as fh:
        p = pickle.load(fh)
    c = p.get("pet_norm_correction")
    if not isinstance(c, dict):
        skipped += 1
        continue
    g = "fdg" if case.startswith("fdg") else "psma" if case.startswith("psma") else "other"
    groups[g].append((c["mu_full"], c["sd_full"]))

print(f"cases with a usable correction: {sum(len(v) for v in groups.values())} "
      f"of {len(files)} ({skipped} skipped)")
allv = [x for v in groups.values() for x in v]
def rep(name, v):
    if not v: return
    mu = [a for a, _ in v]; sd = [b for _, b in v]
    q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
    print(f"{name:<8} n={len(v):>5}  mu median={st.median(mu):.4f} "
          f"[p10 {q(mu,.1):.4f}, p90 {q(mu,.9):.4f}]   "
          f"sd median={st.median(sd):.4f} [p10 {q(sd,.1):.4f}, p90 {q(sd,.9):.4f}]")
rep("ALL", allv)
for g in ("fdg", "psma", "other"):
    rep(g.upper(), groups[g])
# where the SUV clip floor 1.0433 lands in z for each tracer
for g in ("fdg", "psma"):
    v = groups[g]
    if not v: continue
    mu, sd = st.median([a for a, _ in v]), st.median([b for _, b in v])
    print(f"{g.upper()}: SUV clip floor 1.0433 sits at z = {(1.0433 - mu)/sd:+.3f}; "
          f"ceiling 51.211 at z = {(51.211 - mu)/sd:+.1f}")
