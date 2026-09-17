from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# An empty value in .env means "not set". The template ships every optional
# setting blank, so a blank value must not fail validation.
BLANK_AS_NONE = (
    "llama_pid",
    "idle_baseline_w",
    "vrm_token",
    "vrm_installation_id",
    "latitude",
    "longitude",
    "dashboard_token",
    "llama_api_key",
    "llama_api_key_file",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://compute:compute@127.0.0.1:5432/compute"
    demo: bool = False
    serve_scheduler: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)
    dashboard_token: SecretStr | None = None
    llama_url: str = "http://127.0.0.1:8082"
    llama_model: str = "qwen3.5-9b"
    llama_api_key: SecretStr | None = None
    llama_api_key_file: Path | None = None
    llama_pid: int | None = None
    idle_baseline_w: float | None = Field(default=None, ge=0)
    energy_measure_max_seconds: float = Field(default=3600, gt=0)
    vrm_token: SecretStr | None = None
    vrm_installation_id: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    briefing_hour_utc: int = Field(default=7, ge=0, le=23)
    briefing_lead_minutes: int = Field(default=30, ge=1, le=360)
    battery_floor: float = Field(default=40, ge=0, le=100)
    high_soc: float = Field(default=85, ge=0, le=100)
    surplus_minimum_watts: float = Field(default=250, ge=0)
    surplus_duration_seconds: float = Field(default=300, ge=0)
    energy_max_age_seconds: float = Field(default=300, gt=0)
    poll_seconds: float = Field(default=30, gt=0)
    workload_plugins: list[str] = Field(default_factory=list)

    @field_validator(*BLANK_AS_NONE, mode="before")
    @classmethod
    def blank_is_unset(cls, value):
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def validate_configuration(self):
        if self.host not in {"127.0.0.1", "::1", "localhost"} and not self.dashboard_token:
            raise ValueError("Set DASHBOARD_TOKEN before a LAN bind")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("Set both LATITUDE and LONGITUDE")
        if self.high_soc < self.battery_floor:
            raise ValueError("HIGH_SOC must be at least BATTERY_FLOOR")
        return self

    def inference_key(self) -> str | None:
        if self.llama_api_key:
            return self.llama_api_key.get_secret_value()
        if self.llama_api_key_file:
            return self.llama_api_key_file.read_text().strip()
        return None
