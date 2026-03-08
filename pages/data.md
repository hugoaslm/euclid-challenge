# Data

---

## 1) What is one row?

Each row corresponds to **one galaxy** (one object in the sky), identified by:

- `object_id` (unique identifier)

You are given **observables** (what the telescope measures), not “intrinsic” physical quantities.

---

## 2) Feature groups

<div style="overflow-x:auto; margin: 20px 0; border-radius: 12px; border: 1px solid #e1e4e8; box-shadow: 0 4px 6px rgba(0,0,0,0.05); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;">
<table style="width:100%; border-collapse: collapse; background-color: white;">
<thead>
<tr style="background-color: #0b1020; color: white;">
<th style="padding: 15px; text-align: left; border-bottom: 2px solid #2d333b; width: 30%;">Feature Group</th>
<th style="padding: 15px; text-align: left; border-bottom: 2px solid #2d333b;">Description &amp; Specifics</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid #eee;">
<td style="padding: 15px; vertical-align: top;"><strong>A) Euclid Photometry</strong></td>
<td style="padding: 15px; line-height: 1.5;">
Fluxes + flux errors in <strong>VIS, Y, J, H</strong> bands.


<span style="font-size: 0.9em; color: #555;">Measured with various apertures (1FWHM, 2FWHM) to capture light distribution across the galaxy profile.</span>
</td>
</tr>
<tr style="border-bottom: 1px solid #eee; background-color: #fcfcfc;">
<td style="padding: 15px; vertical-align: top;"><strong>B) External Optical</strong></td>
<td style="padding: 15px; line-height: 1.5;">
<strong>u, g, r, i, z</strong> bands (unified from LSST or DECam).


<span style="font-weight: bold; color: #d73a49;">Note:</span> High missingness (NaNs) is expected.
</td>
</tr>
<tr style="border-bottom: 1px solid #eee;">
<td style="padding: 15px; vertical-align: top;"><strong>C) Morphology &amp; Shape</strong></td>
<td style="padding: 15px; line-height: 1.5;">
Structural markers: <strong>Size, Ellipticity, and Concentration.</strong>


<span style="font-size: 0.9em; color: #555;">Includes non-parametric indices such as Gini, Asymmetry, and Smoothness.</span>
</td>
</tr>
<tr style="background-color: #fcfcfc;">
<td style="padding: 15px; vertical-align: top;"><strong>D) Foreground Dust</strong></td>
<td style="padding: 15px; line-height: 1.5;">
<code>gal_ebv</code> (Milky Way extinction proxy).


<span style="font-size: 0.9em; color: #555;">Used to correct for reddening caused by dust within our own galaxy along the line of sight.</span>
</td>
</tr>
</tbody>
<tfoot>
<tr style="background-color: #f6f8fa; border-top: 2px solid #e1e4e8;">
<td colspan="2" style="padding: 12px 15px; font-size: 0.85em; color: #24292f; text-align: center;">
<strong>Target (y):</strong> <code>y_quenched</code> (0: Star-forming, 1: Quenched). <em>Not included in input features.</em>
</td>
</tr>
</tfoot>
</table>
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