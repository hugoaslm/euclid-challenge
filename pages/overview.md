<div style="display:flex;align-items:flex-start;gap:14px;flex-wrap:wrap;margin-bottom:14px;">
  <div style="flex:1;min-width:280px;">
    <h1 style="margin:0;">Euclid QuenchBench (Q1)</h1>
    <p style="margin:6px 0 0 0;font-size:1.05em;line-height:1.35;">
      A probabilistic ML challenge on <b>galaxy quenching</b> using <b>Euclid Q1-style</b> observables:
      <b>photometry</b> + <b>morphology</b>.
    </p>
  </div>
</div>

<hr style="border:none;border-top:1px solid #eee;margin:16px 0;"/>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;">
  <div style="border:1px solid #eee;border-radius:12px;padding:12px;background:white;">
    <h3 style="margin-top:0;">Why this challenge?</h3>
    <ul style="margin:0;padding-left:18px;line-height:1.35;">
      <li><b>Euclid mission context:</b> wide + deep imaging to map the dark Universe.</li>
      <li><b>Realistic observables:</b> fluxes, errors, and morphology.</li>
      <li><b>Imbalanced labels:</b> quenched galaxies are rarer in training.</li>
      <li><b>Missingness:</b> external optical bands are not always available.</li>
      <li><b>Domain shift:</b> redshift changes the data distribution and the quenched fraction.</li>
    </ul>
  </div>

  <div style="border:1px solid #eee;border-radius:12px;padding:12px;background:white;">
    <h3 style="margin-top:0;">What you will practice</h3>
    <ul style="margin:0;padding-left:18px;line-height:1.35;">
      <li>Tabular ML with NaNs and heterogeneous features.</li>
      <li>Imbalanced learning + probability calibration.</li>
      <li>Robust evaluation (macro across redshift bins).</li>
      <li>Science-aware features (colors, flux ratios, morphology summaries).</li>
    </ul>
  </div>

  <div style="border:1px solid #eee;border-radius:12px;padding:12px;background:white;">
    <h3 style="margin-top:0;">What you submit</h3>
    <p style="margin:0 0 8px 0;">A CSV with two columns:</p>
    <pre style="margin:0;background:#0b1020;color:#e8e8e8;padding:10px;border-radius:10px;overflow:auto;"><code>object_id,p_quenched
12345,0.083
12346,0.912
12347,0.301</code></pre>
    <p style="margin:8px 0 0 0;font-size:0.95em;opacity:0.85;line-height:1.35;">
      <b>Important:</b> <code>p_quenched</code> must be a probability in <code>[0, 1]</code>.
    </p>
  </div>
</div>

<hr style="border:none;border-top:1px solid #eee;margin:16px 0;"/>

### The Astrophysical Quest
Galaxies in our Universe fundamentally fall into two categories: actively forming stars (the "Blue Cloud") or passively aging (the "Red Sequence" or **Quenched** galaxies). Understanding how, when, and why massive galaxies stop forming stars—a process known as *quenching*—is one of the most critical questions in modern cosmology. 

The ESA **Euclid Space Telescope** is currently mapping the dark universe, looking back in time to observe galaxies across different cosmic epochs. However, identifying quenched galaxies at high redshift is exceptionally difficult because they emit very little UV/blue light so robust classification from survey observables is challenging.

---

## The ML Challenge

This is not a standard classification task — it is deliberately built to test:

### 1) Imbalanced learning
Quenched galaxies are a minority class, especially at low redshift.  
A model that predicts “always star-forming” can look deceptively good if you only use accuracy.

### 2) Domain adaptation: the "Redshift Drift"
We split data by **redshift** :

- your model is trained on **nearer galaxies**
- and evaluated on **more distant galaxies** where observables and class balance shift

This introduces covariate shift (feature distributions change) and mild concept shift (the quenched population evolves).

---

## Data: what you get

You are given **survey observables** only:

- **Photometry**: fluxes and flux errors in Euclid VIS + NISP (and sometimes external optical bands)
- **Morphology**: size, ellipticity, concentration, Gini/M20, etc.
- External optical bands can be missing → you must handle NaNs robustly.

You must predict a probability:

- `y_quenched = 1` (quenched)
- `y_quenched = 0` (star-forming)
- submission is **probabilistic**: `p_quenched = P(y_quenched=1)`

---

## Labels: simple explanation + rigorous definition

We define the target from the **specific star-formation rate**:

$$
\mathrm{sSFR} = \frac{\mathrm{SFR}}{M_*}
$$

A galaxy is labeled **quenched** if:

$$
\mathrm{sSFR} < \frac{0.3}{t_H(z)}
\quad\text{with}\quad
t_H(z) = \frac{1}{H(z)}.
$$

**Intuition:** the “quenching threshold” evolves with redshift because typical star formation activity is higher earlier in the Universe.

---

## Train / Public test / Private test splits

<div style="border:1px solid #eee;border-radius:12px;padding:12px;background:#fafafa;">
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr>
        <th style="text-align:left;padding:6px;border-bottom:1px solid #e6e6e6;">Split</th>
        <th style="text-align:left;padding:6px;border-bottom:1px solid #e6e6e6;">Redshift range</th>
        <th style="text-align:left;padding:6px;border-bottom:1px solid #e6e6e6;">Purpose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:6px;"><b>Train</b></td>
        <td style="padding:6px;"><code>z &lt; 1.0</code></td>
        <td style="padding:6px;">fit models</td>
      </tr>
      <tr>
        <td style="padding:6px;"><b>Public test</b></td>
        <td style="padding:6px;"><code>1.0 ≤ z &lt; 1.5</code></td>
        <td style="padding:6px;">leaderboard feedback</td>
      </tr>
      <tr>
        <td style="padding:6px;"><b>Private test</b></td>
        <td style="padding:6px;"><code>1.5 ≤ z &lt; 2.5</code></td>
        <td style="padding:6px;">final ranking</td>
      </tr>
    </tbody>
  </table>
</div>

As you go to higher redshift, the **quenched fraction increases** and observables become noisier, so your model must generalize under distribution shift.

---

## Evaluation

We use three metrics:

1. **Primary: macro redshift weighted log loss** (lower is better)  
   - evaluates probability quality  
   - computed per redshift bin, then averaged (macro over bins)

2. **Secondary: AUPRC** (higher is better)  
   - emphasizes performance on the minority class

3. **Tertiary: Recall at 90% precision (purity)** (higher is better)  
   - “How many quenched galaxies can you recover at high purity?”

See the **Evaluation Metrics** page for details.

---

## Acknowledgement

This work has made use of the Euclid Q1 data from the Euclid mission of the European Space Agency (ESA), 2025, https://doi.org/10.57780/esa-2853f3b.