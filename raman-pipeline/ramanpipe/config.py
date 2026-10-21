import numpy as np

DATASETS = {}
CONTROL_ROOT = {}
CONTROL_PATH_OVERRIDE = {}
GROUP_PATH_OVERRIDES = {}
REPLICATE_MERGE = {}
OUTPUT_DIR = "output"
EXPOSURE = ""
RUN_TAG = ""
LAYOUT = "folders"
LABEL_GROUPS_WITH_EXPOSURE = True

GROUP_KEYWORDS = {
    "control": ["control", "ctrl", "untreated"],
    "ecoli": ["ecoli", "coli"],
    "saureus": ["aureus", "saur"],
}
GROUP_DISPLAY = {"control": "Control", "ecoli": "E.coli", "saureus": "S.aureus"}
SET_GROUP_LETTERS = {"c": "control", "e": "ecoli", "s": "saureus"}
SET_GROUP_DIR_REGEX = r'^([ces])[\s\-_]?(\d+)$'
SET_DIR_REGEX = r'^set[\s\-_]?(\d+)$'

PASTEL_LINE = {"control": "#8C8C8C", "ecoli": "#D27C78", "saureus": "#6F9BCF"}
PASTEL_FILL = {"#8C8C8C": "#D9D9D9", "#D27C78": "#E8A9A6", "#6F9BCF": "#A9C4E2"}
BAR_EDGE = "#3A3A3A"
COLORS = {}

AXIS_OFFSETS_CM1 = {}
METHODS = [("area", "global")]
NORM_LABELS = {"area": "AREA", "l2": "L2"}
SHIFT_LABELS = {"global": "GLOBAL"}

ARTIFACT_REGIONS_CM1 = [
    (600.0, 625.0),
    (792.0, 845.0),
    (915.0, 995.0),
]

QUARTZ_FIT = True
QUARTZ_FIT_WINDOW_CM1 = (755.0, 875.0)
QUARTZ_CENTER_RANGE_CM1 = (792.0, 818.0)
QUARTZ_SIGMA_RANGE_CM1 = (12.0, 40.0)

SPIKE_WINDOW = 5
SPIKE_THRESHOLD = 7.0
ROI_MIN_CM1 = 600
ROI_MAX_CM1 = 1800
SG_WINDOW = 11
SG_POLYORDER = 3
ALS_LAMBDA = 1e5
ALS_P = 0.01
ALS_NITER = 10
COMMON_GRID_POINTS = 1201

REPLICATE_DIR_REGEX = r'^(?:rep(?:licate)?|[rsec])[\s\-_]?(\d+)$'
FILENAME_REPLICATE_REGEX = r'^(?:rep(?:licate)?|[rsec])[\s\-_]?(\d+)(?=[\s\-_.])'

MAX_ALLOWED_SHIFT_CM1 = 15.0
GLOBAL_SHIFT_STEP_CM1 = 0.5
GLOBAL_SHIFT_MAX_CM1 = 15
GLOBAL_SHIFT_MATCH_MIN_CM1 = 620.0
GLOBAL_SHIFT_MATCH_MAX_CM1 = 1780.0
GLOBAL_SHIFT_LEVEL = "replicate"
GLOBAL_SHIFT_ITERATIONS = 2
GLOBAL_SHIFT_REFERENCE = "all_controls"

PHE_ANCHOR = True
ANCHOR_BANDS_CM1 = [747.0, 1004.0, 1130.0, 1455.0, 1593.0, 1669.0]
ANCHOR_SEARCH_HALF_WIDTH_CM1 = 8.0
PHE_SEARCH_MIN_CM1 = 996.0
PHE_SEARCH_MAX_CM1 = 1014.0

CONSENSUS_MATCH_TOL_CM1 = 6.0
REFERENCE_BANDS_CM1 = [640, 668, 718, 747, 782, 851, 879, 1004, 1035, 1089, 1100, 1130, 1175,
                       1213, 1237, 1248, 1315, 1344, 1367, 1404, 1455, 1593, 1669, 1786]
SNAP_TOL_CM1 = 3.0
CONSENSUS_MAX_SPAN_CM1 = 12.0
PEAK_INTENSITY_HALF_WINDOW_CM1 = 2.0
N_MARKERS = 8
LOCAL_BASELINE_NEAR_CUT_CM1 = 15.0
TAIL_FIT_REGIONS_CM1 = [(915.0, 995.0)]
LOCAL_BASELINE_SPAN_CM1 = 20.0

PEAK_PROMINENCE = 0.03
PEAK_DISTANCE = 15
PEAK_SHIFT_MATCH_TOLERANCE_CM1 = 15.0
PEAK_SHIFT_BOOTSTRAP_N = 500
PEAK_SHIFT_BOOTSTRAP_SEED = 4242
PEAK_SHIFT_REFINE_WINDOW = 8

MIN_ETA_SQUARED = 0.06
N_PERMUTATIONS = 1000
CLUSTER_ALPHA = 0.05
CLUSTER_FORMING_ALPHA = 0.05
RANDOM_SEED = 42
PEAK_SEARCH_WINDOW = 5
POSTHOC_ALPHA = 0.05

CLASSIFICATION_CV_FOLDS = 5

SIG_SHADE = "#F6E7A8"


def configure(**settings):
    unknown = [k for k in settings if k not in globals()]
    if unknown:
        raise KeyError(f"unknown setting(s): {', '.join(unknown)}")
    globals().update(settings)
    g = globals()
    sfx = f" {g['EXPOSURE']}" if g["EXPOSURE"] and g["LABEL_GROUPS_WITH_EXPOSURE"] else ""
    g["GROUP_DISPLAY"] = {"control": "Control", "ecoli": f"E.coli{sfx}", "saureus": f"S.aureus{sfx}"}
    g["COLORS"] = {(cl, k): PASTEL_LINE[k] for cl in g["DATASETS"] for k in PASTEL_LINE}
