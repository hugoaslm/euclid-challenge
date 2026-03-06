# Data

---

## 1) What is one row?

Each row corresponds to **one galaxy** (one object in the sky), identified by:

- `object_id` (unique identifier)

You are given **observables** (what the telescope measures), not “intrinsic” physical quantities.

---

## 2) Feature groups

<div style="border:1px solid #eee;border-radius:12px;padding:12px;background:#fafafa;">
  <div style="font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono','Courier New', monospace;white-space:pre;overflow:auto;">
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT FEATURES (X)                           │
├───────────────────────────────┬─────────────────────────────────────┤
│ A) Euclid photometry          │ fluxes + flux errors in VIS/Y/J/H    │
│                               │ measured with apertures (1FWHM,2FWHM)│
├───────────────────────────────┼─────────────────────────────────────┤
│ B) External optical photometry│ u/g/r/i/z (unified from LSST or DECam)│
│                               │ often missing (NaNs)                 │
├───────────────────────────────┼─────────────────────────────────────┤
│ C) Morphology / shape         │ size, ellipticity, concentration,    │
│                               │ Gini, asymmetry, smoothness, etc.    │
├───────────────────────────────┼─────────────────────────────────────┤
│ D) Foreground dust proxy      │ gal_ebv (Milky Way extinction)        │
└───────────────────────────────┴─────────────────────────────────────┘

TARGET (y): y_quenched ∈ {0,1}  (NOT included in features)
  </div>
</div>

---

## 3) Photometry (Euclid fluxes) :

### What it is
Photometry measures the **brightness** of the galaxy through filters (wavelength bands).  
In this challenge you get Euclid bands:

- VIS (optical-ish)
- Y, J, H (near-infrared)

For each band you typically have:

- `flux_*` (signal)
- `fluxerr_*` (uncertainty)

Some fluxes are measured in different apertures (`1fwhm`, `2fwhm`), which roughly capture light in a smaller vs larger region.

### ML interpretation
- The fluxes are raw, positive-valued and strongly skewed.
- A classic transformation is to work in **magnitudes** or **log flux**, or create **colors** (differences of magnitudes / ratios of fluxes).

---

## 4) External optical photometry (u/g/r/i/z) : missingness is expected

### What it is
Some galaxies have optical measurements from external surveys (e.g., LSST-like or DECam-like).  
These bands are **not guaranteed** for all objects, so NaNs are normal.

We provide a unified set:
- `flux_u_opt, ..., flux_z_opt`
- `fluxerr_u_opt, ..., fluxerr_z_opt`
- `n_opt` (how many optical bands are present)
- `opt_source` (which external survey was used)

### ML interpretation
- Missingness is informative (not random), especially across redshift and depth.
- You can treat missingness as a signal:
  - keep `n_opt`
  - add “is_missing” indicators per band
  - use models that handle NaNs well (CatBoost / LightGBM can be strong here)

---

## 5) Morphology :

### What it is
Morphological features summarize the **shape and structure** of galaxies. Examples:

- size (`semimajor_axis`, `kron_radius`)
- ellipticity (`ellipticity`)
- non-parametric morphology (e.g., `concentration`, `gini`, `asymmetry`, `smoothness`)

### ML interpretation
- Morphology often correlates with star formation state:
  - quenched galaxies are frequently more compact and centrally concentrated
  - star-forming galaxies are often more extended / disk-like
- These features are typically already engineered, so tree models often exploit them well.

---

## 6) Foreground dust proxy (`gal_ebv`)

This approximates the amount of Milky Way dust in the line of sight.  
It affects observed fluxes (reddening/extinction).

### ML interpretation
- It can help models correct systematic color shifts caused by the foreground.

---

## 7) Target definition

The label is based on **specific star formation rate**:

$$
\mathrm{sSFR} = \frac{\mathrm{SFR}}{M_*}
$$

A galaxy is labeled quenched if:

$$
\mathrm{sSFR} < \frac{0.3}{t_H(z)} \quad\text{with}\quad t_H(z)=\frac{1}{H(z)}.
$$

**Important:** participants do **not** receive SFR, stellar mass, or redshift.  
They must infer quenching from observables (photometry + morphology).


---

## 8) Practical notes for participants

- Expect **NaNs** (especially in optical bands).
- Use **probability outputs** (the primary metric is log loss).
- Consider feature engineering carefully
- Strong baselines: gradient boosted trees + calibration.

See "Starting Kit & Baselines" for a runnable baseline notebook.