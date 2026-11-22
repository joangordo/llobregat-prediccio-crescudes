"""Per-station configuration, validated with Pydantic.

Each target station has a YAML file in configs/stations/. Adding a station means adding a
file: the code is the same for all of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

TargetTransform = Literal["level", "delta", "log_ratio", "log_level"]


class SensorRef(BaseModel):
    """Reference to a sensor of the ACA Sentilo platform."""

    provider: str
    sensor: str
    # codi_variable of the ACA open data portal (5 min; same code since 1996)
    opendata: str | None = None
    # Daily series (cabals_diaris: codi_variable; embassaments_diaris: station name).
    # If there are several, they are chained in order and the later one wins where they overlap.
    opendata_daily: list[str] = Field(default_factory=list)


class Thresholds(BaseModel):
    """Warning thresholds: statistical quantiles and, if available, official values (m³/s)."""

    warning_quantile: float = Field(0.99, gt=0, lt=1)
    alert_quantile: float = Field(0.999, gt=0, lt=1)
    warning_m3s: float | None = None
    alert_m3s: float | None = None

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if self.alert_quantile <= self.warning_quantile:
            raise ValueError("alert_quantile must be larger than warning_quantile")
        return self


class LoggedSensor(SensorRef):
    """Sensor logged in real time that is not (yet) a model input."""

    variable: str  # e.g. "flow", "level", "rain_intensity", "canal_flow"
    note: str | None = None


class UpstreamGauge(BaseModel):
    name: str
    aca: SensorRef
    code: str | None = None  # ACA order code (e.g. LB-25)
    area_km2: float | None = None
    note: str | None = None
    # Lag window (hours) around the travel time; if None, it is estimated by
    # cross-correlation (llobregat.features.estimate_travel_time).
    lag_window_h: tuple[float, float] | None = None


class Reservoir(BaseModel):
    name: str
    outflow: SensorRef | None = None  # virtual gauging station of the dam
    volume: SensorRef | None = None
    level: SensorRef | None = None
    capacity_hm3: float | None = None


class RainConfig(BaseModel):
    xema_stations: list[str] = Field(default_factory=list)
    # Sub-basin -> list of gauge codes (for sub-basin means)
    subbasins: dict[str, list[str]] = Field(default_factory=dict)
    windows_h: list[float] = Field(default_factory=lambda: [0.5, 1, 3, 6, 12, 24, 72])
    per_station_features: bool = True  # representation A (each gauge as a feature)
    per_subbasin_features: bool = True  # representation B (sub-basin mean)
    # Splices: new station -> old station it replaces (same site)
    splices: dict[str, str] = Field(default_factory=dict)
    # How to treat the hourly (HO) XEMA readings before 2014:
    # "split" spreads them over two half hours; "drop" discards them.
    hourly_policy: Literal["split", "drop"] = "split"
    # ACA rain gauges with records since 1996 (codes of the per-basin Excel files)
    aca_gauges: list[str] = Field(default_factory=list)
    # Real-time (Sentilo) sensor of each ACA rain gauge code, for the live forecast
    aca_realtime: dict[str, SensorRef] = Field(default_factory=dict)


class FlowQCConfig(BaseModel):
    """Quality rules for the 5-min flow series (llobregat qc-apply)."""

    spike_factor: float = 4.0  # spike: > 4 x the median of the 3 values before and after...
    spike_min_m3s: float = 5.0  # ... by more than 5 m³/s, and above the station p99
    dropout_factor: float = (
        10.0  # dropout: < 1/10 of the neighbours, with neighbours > dropout_min_m3s
    )
    dropout_min_m3s: float = 1.0
    zero_block_max_m3s: float = 0.02  # run of zeros in the middle of flowing water...
    zero_block_return_hours: float = 6.0  # ... that recovers within 6 h
    excursion_factor: float = 10.0  # excursion: > x10 the level of the previous hour...
    excursion_return_hours: float = 3.0  # ... and back to < 2x that level within 3 h
    oscillation_ratio: float = 20.0  # jump between consecutive readings larger than x20...
    oscillation_min_jumps: int = 3  # ... 3 or more times within an hour
    stuck_days: float = 7.0  # same value for 7 days or more: stuck sensor...
    stuck_min_m3s: float = 0.5  # ... if the flow is > 0.5 m³/s (a dry river is not flagged)
    high_flat_hours: float = 6.0  # same value for 6 h or more at high flow: saturated sensor
    high_flat_quantile: float = 0.95
    unit_step_ratio: float = 100.0  # x100 jump where the unit factor changes
    daily_ratio: float = 3.0  # balance: daily mean vs official daily flow (x3 or /3)
    daily_min_m3s: float = 0.05
    daily_min_coverage: float = 0.8


class Exclusion(BaseModel):
    """Stretch discarded by hand after review (qc-event)."""

    series: str  # codi_variable (ACA series code)
    start: str
    end: str
    reason: str


class QCConfig(BaseModel):
    min_value: float = 0.0
    max_value: float | None = None  # plausible maximum for the station (m³/s)
    flatline_intervals: int = 12  # same value for 6 h (at 30 min) = suspicious
    flatline_min_value: float = 0.5  # do not flag flat series at flows ~0
    max_jump: float | None = None  # maximum plausible change between intervals
    short_gap_intervals: int = 6  # gaps ≤ 3 h are interpolated (predictors only)
    flow: FlowQCConfig = Field(default_factory=FlowQCConfig)
    exclusions: list[Exclusion] = Field(default_factory=list)


class ValidationConfig(BaseModel):
    hydro_year_start_month: int = Field(10, ge=1, le=12)
    n_outer_folds: int = Field(5, ge=1)
    holdout_years: int = Field(2, ge=0)
    inner_val_years: int = Field(2, ge=1)
    calibration_years: int = Field(1, ge=1)
    gap_days: float = Field(7, ge=0)


class IntervalConfig(BaseModel):
    alpha: float = Field(0.2, gt=0, lt=1)  # 80 % interval
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    aci_gamma: float = Field(0.005, gt=0)
    mondrian_min_n: int = 100

    @model_validator(mode="after")
    def _consistent(self) -> IntervalConfig:
        lo, mid, hi = self.quantiles
        if not (0 < lo < mid < hi < 1):
            raise ValueError("The quantiles must be ordered within (0, 1)")
        if abs((hi - lo) - (1 - self.alpha)) > 1e-9:
            raise ValueError("hi - lo must equal 1 - alpha")
        return self


class ModelVariant(BaseModel):
    """Model variant: training period and input selection rule.

    Inputs are not chosen by hand: every series (flow, reservoir, rain gauge) with at least
    `min_coverage` of data in the variant's period is an input. The selection is therefore
    reproducible and can be justified with the coverage table.
    """

    name: str
    description: str | None = None
    train_start: str  # ISO date (start of a hydrological year, e.g. 2000-10-01)
    min_coverage: float = Field(0.85, gt=0, le=1)
    # ValidationConfig fields that change in this variant (e.g. n_outer_folds)
    validation: dict[str, float | int] = Field(default_factory=dict)


class StationConfig(BaseModel):
    station_id: str
    name: str
    target: SensorRef
    resolution_min: int = 30
    horizons_h: list[float] = Field(default_factory=lambda: [1, 3, 6, 12, 24])
    thresholds: Thresholds = Field(default_factory=Thresholds)
    upstream: list[UpstreamGauge] = Field(default_factory=list)
    reservoirs: list[Reservoir] = Field(default_factory=list)
    log_only: list[LoggedSensor] = Field(default_factory=list)
    rain: RainConfig = Field(default_factory=RainConfig)
    antecedent_kd: list[float] = Field(default_factory=lambda: [0.85, 0.90, 0.95])
    target_transforms: list[TargetTransform] = Field(
        default_factory=lambda: ["level", "delta", "log_ratio", "log_level"]
    )
    qc: QCConfig = Field(default_factory=QCConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    intervals: IntervalConfig = Field(default_factory=IntervalConfig)
    model_release: str | None = None  # GitHub Release tag of the production model
    variants: list[ModelVariant] = Field(default_factory=list)

    def validation_for(self, variant: ModelVariant) -> ValidationConfig:
        return self.validation.model_copy(update=variant.validation)

    def variant(self, name: str) -> ModelVariant:
        for v in self.variants:
            if v.name == name:
                return v
        raise KeyError(f"Unknown variant: {name} ({[v.name for v in self.variants]})")

    @field_validator("horizons_h")
    @classmethod
    def _horizons(cls, v: list[float]) -> list[float]:
        if not v or any(h <= 0 for h in v):
            raise ValueError("Horizons must be positive")
        return sorted(v)

    @model_validator(mode="after")
    def _horizons_on_grid(self) -> StationConfig:
        for h in self.horizons_h:
            steps = h * 60 / self.resolution_min
            if abs(steps - round(steps)) > 1e-9:
                raise ValueError(
                    f"The {h} h horizon is not a multiple of {self.resolution_min} min"
                )
        return self

    def steps(self, hours: float) -> int:
        """Convert hours to a number of grid steps."""
        return round(hours * 60 / self.resolution_min)


def load_station_config(path: str | Path) -> StationConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return StationConfig.model_validate(data)


def load_all_stations(directory: str | Path = "configs/stations") -> dict[str, StationConfig]:
    configs = {}
    for p in sorted(Path(directory).glob("*.yaml")):
        cfg = load_station_config(p)
        configs[cfg.station_id] = cfg
    return configs
