# Theoretical Foundation of SPR-PCF Biosensing

---

## 1. Introduction to Surface Plasmon Resonance in Photonic Crystal Fibers

Surface Plasmon Resonance (SPR) is an optical phenomenon in which incident electromagnetic radiation couples resonantly to coherent electron oscillations (Surface Plasmon Polaritons, SPPs) propagating along a metal-dielectric interface. 

In conventional prism-based Kretschmann configurations, bulky optical setups limit miniaturization and microfluidic integration. Photonic Crystal Fibers (PCFs) overcome these limitations by combining microstructured air-hole cladding with integrated nanoscale plasmonic coatings (such as Gold, Silver, Copper, or Aluminum) and microfluidic channels.

---

## 2. Maxwell Governing Equations

Assuming time-harmonic fields with angular frequency $\omega$ under the $e^{i \omega t}$ convention:

$$\nabla \times \mathbf{E} = -i \omega \mu_0 \mu_r \mathbf{H}$$
$$\nabla \times \mathbf{H} = i \omega \varepsilon_0 \varepsilon_r \mathbf{E}$$

For a longitudinally invariant fiber propagating along the $z$-axis with complex propagation constant $\beta$:

$$\mathbf{E}(x, y, z) = \mathbf{E}(x, y) e^{i \beta z}$$
$$\mathbf{H}(x, y, z) = \mathbf{H}(x, y) e^{i \beta z}$$

The complex propagation constant is partitioned as:

$$\beta = \beta_{r} + i \beta_{i}$$

where $\beta_{r} > 0$ denotes forward phase propagation, and $\beta_{i} \ge 0$ represents attenuation due to material absorption (ohmic damping in metals) and open-boundary radiation/leakage.

---

## 3. Full-Vector Transverse Formulation

Eliminating the longitudinal components $E_z$ and $H_z$ via Gauss's divergence law $\nabla \cdot (\varepsilon_r \mathbf{E}) = 0$ yields the coupled transverse vector wave equation:

$$\begin{bmatrix} P_{xx} & P_{xy} \\ P_{yx} & P_{yy} \end{bmatrix} \begin{bmatrix} E_x \\ E_y \end{bmatrix} = \beta^2 \begin{bmatrix} E_x \\ E_y \end{bmatrix}$$

where the differential operators are:

$$P_{xx} E_x = \frac{\partial}{\partial x}\left( \frac{1}{\varepsilon_r} \frac{\partial (\varepsilon_r E_x)}{\partial x} \right) + \frac{\partial^2 E_x}{\partial y^2} + k_0^2 \varepsilon_r E_x$$

$$P_{xy} E_y = \frac{\partial}{\partial x}\left( \frac{1}{\varepsilon_r} \frac{\partial (\varepsilon_r E_y)}{\partial y} \right) - \frac{\partial^2 E_y}{\partial x \partial y}$$

$$P_{yx} E_x = \frac{\partial}{\partial y}\left( \frac{1}{\varepsilon_r} \frac{\partial (\varepsilon_r E_x)}{\partial x} \right) - \frac{\partial^2 E_x}{\partial y \partial x}$$

$$P_{yy} E_y = \frac{\partial}{\partial y}\left( \frac{1}{\varepsilon_r} \frac{\partial (\varepsilon_y E_y)}{\partial y} \right) + \frac{\partial^2 E_y}{\partial x^2} + k_0^2 \varepsilon_r E_y$$

The cross-coupling terms $P_{xy}$ and $P_{yx}$ are non-zero at dielectric-metal interfaces where $\nabla \varepsilon_r \ne 0$. They account for hybrid polarization, modal birefringence, and polarization-selective SPR excitation.

---

## 4. Complex Coordinate-Stretched PML

To truncate the open transverse computational domain without artificial reflections, an absorbing Perfectly Matched Layer (PML) is applied using complex coordinate stretching:

$$\tilde{x} = \int_0^x s_x(x') dx', \quad \tilde{y} = \int_0^y s_y(y') dy'$$

with stretching factor:

$$s_x(x) = 1 - i \alpha \left( \frac{|x| - x_0}{d_{\text{PML}}} \right)^m$$

where $d_{\text{PML}}$ is the layer thickness, $m$ is the polynomial absorption profile exponent (typically $m=2$), and $\alpha$ is the scaling factor.

---

## 5. Modal Confinement Loss

The complex effective refractive index is defined as:

$$n_{\text{eff}} = \frac{\beta}{k_0} = n_{\text{real}} + i n_{\text{imag}}$$

The modal attenuation (confinement and absorption loss) is:

$$\alpha_{\text{dB/m}} = \frac{20}{\ln 10} \text{Im}(\beta) = \frac{20}{\ln 10} k_0 \text{Im}(n_{\text{eff}}) \approx 8.6858896 \cdot k_0 \cdot \text{Im}(n_{\text{eff}})$$

and in practical units of $\text{dB/cm}$:

$$\alpha_{\text{dB/cm}} = 10^{-2} \cdot \alpha_{\text{dB/m}}$$

---

## 6. Biosensor Performance Metrics

### 6.1 Wavelength Sensitivity ($S_\lambda$)
Wavelength interrogation monitors the shift in resonance peak wavelength $\Delta \lambda_{\text{res}}$ as a function of analyte refractive index change $\Delta n_a$:

$$S_\lambda = \frac{\Delta \lambda_{\text{res}}}{\Delta n_a} \quad [\text{nm/RIU}]$$

### 6.2 Amplitude Sensitivity ($S_A$)
Amplitude interrogation monitors the relative change in transmission loss at a fixed wavelength:

$$S_A(\lambda) = -\frac{1}{\alpha(\lambda, n_a)} \frac{\partial \alpha(\lambda, n_a)}{\partial n_a} \approx \frac{|\alpha(\lambda, n_a + \Delta n_a) - \alpha(\lambda, n_a)|}{\alpha(\lambda, n_a) \cdot \Delta n_a} \quad [\text{RIU}^{-1}]$$

### 6.3 Figure of Merit (FOM)
$$FOM = \frac{S_\lambda}{\text{FWHM}} \quad [\text{RIU}^{-1}]$$

### 6.4 Sensor Resolution ($R$)
Given an optical spectrum analyzer with spectral resolution $\Delta \lambda_{\text{min}}$ (e.g. 0.1 nm):

$$R = \frac{\Delta \lambda_{\text{min}}}{S_\lambda} \quad [\text{RIU}]$$

