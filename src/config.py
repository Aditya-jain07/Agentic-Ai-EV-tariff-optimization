from pathlib import Path

# ==============================================================================
# Project root and core directories
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ==============================================================================
# Dynamic search utilities
# ==============================================================================
def _first_existing_dir(*candidates: str) -> Path:
    """
    Scan the raw data folder for candidate directory names or wildcards so the
    project can tolerate minor folder-name differences across machines.
    """
    for candidate in candidates:
        path = RAW_DATA_DIR / candidate
        if path.exists():
            return path
        matches = sorted(RAW_DATA_DIR.glob(candidate))
        if matches:
            return matches[0]
    return RAW_DATA_DIR / candidates[0]


def _first_matching_file(folder: Path, *patterns: str) -> Path:
    """Return the first matching file for one of the provided glob patterns."""
    for pattern in patterns:
        matches = sorted(folder.rglob(pattern))
        if matches:
            return matches[0]
    return folder / patterns[0].replace("*", "")


# ==============================================================================
# Data discovery and ingestion targets (Case Study Directories)
# ==============================================================================
ACN_DATA_DIR = _first_existing_dir("acn_data", "acn*", "ACN*")
URBAN_DATA_DIR = _first_existing_dir("urban_data", "urban*", "URBAN*")

# Ingestion file discovery mappings
ACN_RAW_PATH = _first_matching_file(ACN_DATA_DIR, "*sessions*.xlsx", "*.xlsx", "*.json")
URBAN_RAW_DIR = URBAN_DATA_DIR

# Retain legacy references for safety across historical model files
ACN_SESSIONS_PATH = ACN_RAW_PATH
URBAN_SESSIONS_PATH = _first_matching_file(URBAN_DATA_DIR, "*sessions*.csv", "*.csv")


# ==============================================================================
# Intermediate preprocessed states
# ==============================================================================
CLEAN_ACN_PATH = PROCESSED_DATA_DIR / "acn_cleaned_sessions.csv"
CLEAN_URBAN_PATH = PROCESSED_DATA_DIR / "urban_cleaned_sessions.csv"

ACN_HOURLY_PATH = PROCESSED_DATA_DIR / "acn_hourly_demand.csv"
URBAN_HOURLY_PATH = PROCESSED_DATA_DIR / "urban_hourly_demand.csv"
UNIFIED_HOURLY_PATH = PROCESSED_DATA_DIR / "unified_hourly_base.csv"
FORECAST_CACHE_PATH = PROCESSED_DATA_DIR / "forecasted_demand_cache.csv"
FORECASTED_DEMAND_CACHE_PATH = FORECAST_CACHE_PATH


# ==============================================================================
# Evaluation and submission outputs
# ==============================================================================
MODEL_METRICS_PATH = OUTPUTS_DIR / "model_accuracy_scores.csv"
DYNAMIC_TARIFF_PATH = OUTPUTS_DIR / "dynamic_tariff_schedule.csv"
BUSINESS_OUTCOMES_PATH = OUTPUTS_DIR / "business_outcomes.csv"
CONTEXT_METRICS_PATH = OUTPUTS_DIR / "model_accuracy_by_context.csv"
TOP_ERROR_PATH = OUTPUTS_DIR / "demand_forecast_top_errors.csv"


# ==============================================================================
# Business and operational assumptions
# ==============================================================================
BASE_TARIFF_INR_PER_KWH = 15.0
BASE_TARIFF_RS_PER_KWH = BASE_TARIFF_INR_PER_KWH

ACN_ENERGY_COST_FACTOR = 0.55
URBAN_VOLUME_TO_KWH_FACTOR = 1.0
DEFAULT_PILE_CAPACITY = 12.0

# Optimized policy thresholds (Expanded off-peak window to capture more shifted volume)
SURGE_THRESHOLD_UPPER = 0.75
DISCOUNT_THRESHOLD_LOWER = 0.40

# Optimized tariff multipliers
DEFAULT_SURGE_MULTIPLIER = 1.3333      # FIXED: Yields exactly INR 20.00 Peak Price
DEFAULT_DISCOUNT_MULTIPLIER = 0.8000   # PERFECT: Yields exactly INR 12.00 Off-Peak Price

DEMAND_ELASTICITY = -0.25
TARGET_REVENUE_GROWTH_PERCENT = 7.0


# ==============================================================================
# Multi-Objective Balanced Optimization Parameters
# ==============================================================================
REVENUE_WEIGHT = 0.40       # ADJUSTED: De-prioritizes short-term aggressive harvesting
CONGESTION_WEIGHT = 0.60    # ADJUSTED: Prioritizes demand flattening & grid safety

MIN_SURGE_MULTIPLIER = 1.15
MAX_SURGE_MULTIPLIER = 1.3333  # FIXED: Hard ceiling ensures agent never surges past INR 20.00

MIN_DISCOUNT_MULTIPLIER = 0.70
MAX_DISCOUNT_MULTIPLIER = 0.8333 # FIXED: Prevents discounts from bleeding into base rates too high

# ==============================================================================
# Machine Learning & Demand Agent Tuning Configuration
# ==============================================================================
RANDOM_SEED = 42
TEST_FRACTION = 0.20  # Safe chronological 80/20 train/test partition split

# ==============================================================================
# Pricing Agent and Congestion Threshold Definitions
# ==============================================================================
CONGESTION_PROBABILITY_THRESHOLD = 0.65  # 65% probability triggers congestion rules