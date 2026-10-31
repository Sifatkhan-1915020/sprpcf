"""High-level SPR analyzer for resonance extraction, sensitivity calculation, and performance evaluation."""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from .resonance import extract_resonance_peaks, ResonancePeak
from .sensitivity import SensitivityReport, calculate_amplitude_sensitivity_spectrum
from .figure_of_merit import SensorPerformanceMetrics
from .loss import LossSpectrum
from ..solvers.base import SolverResult
from ..solvers.mode_tracking import ModeTracker


class SPRAnalyzer:
    """Orchestrates comprehensive SPR analysis on simulation results."""

    def __init__(self, sweep_results: Union[List[SolverResult], SolverResult]):
        if isinstance(sweep_results, SolverResult):
            self.results = [sweep_results]
        else:
            self.results = list(sweep_results)

    def extract_loss_spectrum(self, mode_idx: int = 0) -> LossSpectrum:
        """Extract spectral loss curve for mode index across wavelength sweep."""
        wls = []
        neffs = []
        for res in self.results:
            wls.append(res.wavelength_m)
            if mode_idx < len(res.modes):
                neffs.append(res.modes[mode_idx].n_eff)
            else:
                neffs.append(complex(1.0, 0.0))

        return LossSpectrum(np.array(wls), np.array(neffs))

    def find_resonance(self, mode_idx: int = 0) -> Optional[ResonancePeak]:
        """Detect the dominant SPR resonance peak for a mode across wavelengths."""
        spectrum = self.extract_loss_spectrum(mode_idx)
        peaks = extract_resonance_peaks(spectrum.wavelengths_m, spectrum.loss_db_per_cm)
        return peaks[0] if peaks else None

    def calculate_sensitivity_from_analyte_sweep(
        self,
        analyte_results: Dict[float, List[SolverResult]],
        instrument_resolution_nm: float = 0.1,
    ) -> Dict[str, Any]:
        """Calculate wavelength and amplitude sensitivity across multiple analyte RI sweeps.
        
        Parameters:
            analyte_results: Dict mapping analyte_ri (e.g. 1.33, 1.34, 1.35) -> List[SolverResult]
            instrument_resolution_nm: Assumed detector wavelength resolution (e.g. 0.1 nm)
        """
        analyte_indices = sorted(list(analyte_results.keys()))
        res_wavelengths_nm = []
        fwhms_nm = []
        loss_spectra_db_cm = []
        wavelengths_m = None

        for n_a in analyte_indices:
            results_for_na = analyte_results[n_a]
            analyzer = SPRAnalyzer(results_for_na)
            peak = analyzer.find_resonance(mode_idx=0)
            if peak is not None:
                res_wavelengths_nm.append(peak.wavelength_nm)
                fwhms_nm.append(peak.fwhm_nm)
            else:
                # Approximate peak from maximum loss
                spectrum = analyzer.extract_loss_spectrum(0)
                max_i = int(np.argmax(spectrum.loss_db_per_cm))
                res_wavelengths_nm.append(spectrum.wavelengths_m[max_i] * 1e9)
                fwhms_nm.append(50.0)

            spec = analyzer.extract_loss_spectrum(0)
            loss_spectra_db_cm.append(spec.loss_db_per_cm)
            if wavelengths_m is None:
                wavelengths_m = spec.wavelengths_m

        # Compute amplitude sensitivities between consecutive RI steps
        amp_sensitivities = []
        if len(analyte_indices) >= 2 and wavelengths_m is not None:
            for i in range(len(analyte_indices) - 1):
                dn = analyte_indices[i + 1] - analyte_indices[i]
                _, _, s_a_max = calculate_amplitude_sensitivity_spectrum(
                    wavelengths_m,
                    loss_spectra_db_cm[i],
                    loss_spectra_db_cm[i + 1],
                    dn,
                )
                amp_sensitivities.append(s_a_max)

        sens_report = SensitivityReport(
            analyte_indices=analyte_indices,
            resonance_wavelengths_nm=res_wavelengths_nm,
            amplitude_sensitivities=amp_sensitivities,
        )

        mean_fwhm = float(np.mean(fwhms_nm)) if fwhms_nm else 50.0
        perf = SensorPerformanceMetrics(
            wavelength_sensitivity_nm_per_riu=sens_report.linear_sensitivity_nm_per_riu,
            fwhm_nm=mean_fwhm,
            instrument_resolution_nm=instrument_resolution_nm,
        )

        return {
            "sensitivity": sens_report.to_dict(),
            "performance": perf.to_dict(),
            "fwhm_nm_mean": mean_fwhm,
        }

