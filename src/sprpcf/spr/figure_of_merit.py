"""Figure of merit (FOM), sensor resolution, and limit of detection (LOD) metrics."""

from typing import Dict, Any, Optional


class SensorPerformanceMetrics:
    """Consolidated figure-of-merit and resolution metrics under explicit instrumentation assumptions."""

    def __init__(
        self,
        wavelength_sensitivity_nm_per_riu: float,
        fwhm_nm: float,
        instrument_resolution_nm: Optional[float] = 0.1,  # Optical Spectrum Analyzer (OSA) resolution
        noise_std_nm: Optional[float] = None,  # Measured spectral noise standard deviation
    ):
        self.sensitivity = float(wavelength_sensitivity_nm_per_riu)
        self.fwhm_nm = float(fwhm_nm)
        self.instrument_resolution_nm = float(instrument_resolution_nm) if instrument_resolution_nm is not None else None
        self.noise_std_nm = float(noise_std_nm) if noise_std_nm is not None else None

        # Figure of Merit: FOM = S_lambda / FWHM (in RIU^-1)
        self.fom_riu_inv = self.sensitivity / max(1e-12, self.fwhm_nm)

        # Sensor Resolution: R = Delta_lambda_inst / S_lambda
        # Strict requirement: calculate ONLY when instrument resolution is explicitly specified
        if self.instrument_resolution_nm is not None and self.instrument_resolution_nm > 0:
            self.sensor_resolution_riu = self.instrument_resolution_nm / max(1e-12, self.sensitivity)
            self.resolution_riu = self.sensor_resolution_riu
            self.resolution_statement = (
                f"{self.sensor_resolution_riu:.2e} RIU (at specified OSA resolution Delta_lambda_inst = {self.instrument_resolution_nm:.2f} nm)"
            )
        else:
            self.sensor_resolution_riu = None
            self.resolution_riu = None
            self.resolution_statement = "Not evaluated: requires explicitly specified instrument resolution (Delta_lambda_inst)."

        # Limit of Detection (LOD): LOD = 3 * sigma_noise / S_lambda
        # Strict requirement: calculate ONLY when noise assumptions are available
        if self.noise_std_nm is not None and self.noise_std_nm > 0:
            self.detection_limit_riu = 3.0 * self.noise_std_nm / max(1e-12, self.sensitivity)
            self.lod_statement = f"{self.detection_limit_riu:.2e} RIU (at calibrated noise sigma = {self.noise_std_nm:.3f} nm)"
        else:
            self.detection_limit_riu = None
            self.lod_statement = "Not evaluated: requires calibrated baseline noise floor standard deviation (sigma_noise)."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sensitivity_nm_per_riu": self.sensitivity,
            "fwhm_nm": self.fwhm_nm,
            "figure_of_merit_riu_inv": self.fom_riu_inv,
            "sensor_resolution_riu": self.sensor_resolution_riu,
            "resolution_statement": self.resolution_statement,
            "assumed_instrument_resolution_nm": self.instrument_resolution_nm,
            "detection_limit_riu": self.detection_limit_riu,
            "lod_statement": self.lod_statement,
        }

    def __repr__(self) -> str:
        res_str = f"{self.sensor_resolution_riu:.2e} RIU" if self.sensor_resolution_riu else "N/A"
        lod_str = f"{self.detection_limit_riu:.2e} RIU" if self.detection_limit_riu else "N/A"
        return (
            f"SensorPerformanceMetrics(FOM={self.fom_riu_inv:.1f} RIU^-1, "
            f"Resolution={res_str}, LOD={lod_str})"
        )
