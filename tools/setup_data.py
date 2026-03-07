import argparse
import shutil
from pathlib import Path

from astroquery.esa.euclid import Euclid
import pandas as pd
import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


PHASE = "dev_phase"
PRIVATE_PHASE = "final_phase"


QUERY = """
SELECT TOP 2000000
    mer.object_id,

    mer.flux_vis_1fwhm_aper,
    mer.flux_y_1fwhm_aper,
    mer.flux_j_1fwhm_aper,
    mer.flux_h_1fwhm_aper,

    mer.flux_h_templfit, mer.fluxerr_h_templfit,

    mer.flux_vis_2fwhm_aper,
    mer.flux_y_2fwhm_aper,
    mer.flux_j_2fwhm_aper,
    mer.flux_h_2fwhm_aper,

    mer.fluxerr_vis_1fwhm_aper,
    mer.fluxerr_y_1fwhm_aper,
    mer.fluxerr_j_1fwhm_aper,
    mer.fluxerr_h_1fwhm_aper,

    mer.fluxerr_vis_2fwhm_aper,
    mer.fluxerr_y_2fwhm_aper,
    mer.fluxerr_j_2fwhm_aper,
    mer.fluxerr_h_2fwhm_aper,

    mer.flux_u_ext_lsst_1fwhm_aper,
    mer.flux_g_ext_lsst_1fwhm_aper,
    mer.flux_r_ext_lsst_1fwhm_aper,
    mer.flux_i_ext_lsst_1fwhm_aper,
    mer.flux_z_ext_lsst_1fwhm_aper,

    mer.fluxerr_u_ext_lsst_1fwhm_aper,
    mer.fluxerr_g_ext_lsst_1fwhm_aper,
    mer.fluxerr_r_ext_lsst_1fwhm_aper,
    mer.fluxerr_i_ext_lsst_1fwhm_aper,
    mer.fluxerr_z_ext_lsst_1fwhm_aper,

    mer.flux_u_ext_decam_1fwhm_aper,
    mer.flux_g_ext_decam_1fwhm_aper,
    mer.flux_r_ext_decam_1fwhm_aper,
    mer.flux_i_ext_decam_1fwhm_aper,
    mer.flux_z_ext_decam_1fwhm_aper,

    mer.fluxerr_u_ext_decam_1fwhm_aper,
    mer.fluxerr_g_ext_decam_1fwhm_aper,
    mer.fluxerr_r_ext_decam_1fwhm_aper,
    mer.fluxerr_i_ext_decam_1fwhm_aper,
    mer.fluxerr_z_ext_decam_1fwhm_aper,

    mer.semimajor_axis,
    mer.semimajor_axis_err,
    mer.ellipticity,
    mer.ellipticity_err,
    mer.kron_radius,
    mer.kron_radius_err,
    mer.fwhm,
    mer.mu_max,
    mer.mumax_minus_mag,
    mer.sersic_fract_vis_disk_sersic,
    mer.sersic_fract_vis_disk_sersic_err,

    mer.det_quality_flag,
    mer.spurious_flag,
    mer.spurious_prob,
    mer.blended_prob,
    mer.point_like_flag,
    mer.point_like_prob,
    mer.extended_flag,
    mer.extended_prob,
    mer.deblended_flag,

    mer.gal_ebv,
    mer.gal_ebv_err,

    morph.concentration,
    morph.gini,
    morph.moment_20,
    morph.asymmetry,
    morph.smoothness,

    cl.phz_gal_prob,

    pp.quality_flag      AS pp_quality_flag,
    pp.phys_param_flags  AS pp_phys_param_flags,

    pp.phz_pp_median_redshift    AS redshift_z,
    pp.phz_pp_median_sfr         AS sfr,
    pp.phz_pp_median_stellarmass AS log_mass

FROM catalogue.mer_catalogue AS mer
LEFT JOIN catalogue.phz_physical_parameters AS pp
    ON mer.object_id = pp.object_id
LEFT JOIN catalogue.phz_classification AS cl
    ON mer.object_id = cl.object_id
LEFT JOIN catalogue.mer_morphology AS morph
    ON mer.object_id = morph.object_id

WHERE
    cl.phz_gal_prob >= 0.5
    AND mer.spurious_flag = 0
    AND mer.det_quality_flag >= 0
    AND pp.phz_pp_median_sfr IS NOT NULL
    AND pp.phz_pp_median_stellarmass IS NOT NULL
    AND pp.phz_pp_median_redshift IS NOT NULL
    AND mer.flux_vis_1fwhm_aper IS NOT NULL
    AND mer.flux_y_1fwhm_aper IS NOT NULL
    AND mer.flux_j_1fwhm_aper IS NOT NULL
    AND mer.flux_h_1fwhm_aper IS NOT NULL
    AND mer.flux_h_templfit > 0
"""


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()

    dev_data_dir = Path(PHASE) / "input_data"
    dev_ref_dir = Path(PHASE) / "reference_data"
    final_data_dir = Path(PRIVATE_PHASE) / "input_data"
    final_ref_dir = Path(PRIVATE_PHASE) / "reference_data"

    train_dir = dev_data_dir / "train"
    test_dir = dev_data_dir / "test"
    private_test_dir = final_data_dir

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    private_test_dir.mkdir(parents=True, exist_ok=True)
    dev_ref_dir.mkdir(parents=True, exist_ok=True)
    final_ref_dir.mkdir(parents=True, exist_ok=True)

    cosmo = FlatLambdaCDM(
        H0=67.74 * u.km / u.s / u.Mpc,
        Om0=0.3089,
        Tcmb0=2.7255 * u.K
    )

    job = Euclid.launch_job_async(QUERY)
    results = job.get_results()
    df_euclid = results.to_pandas()

    print("Raw Euclid table shape:", df_euclid.shape)

    df = df_euclid.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    need = ["object_id", "sfr", "log_mass", "redshift_z", "flux_h_templfit"]
    df = df.dropna(subset=need).copy()

    df["object_id"] = df["object_id"].astype(str).str.strip()

    df = df[df["sfr"] > 0]
    df = df[(df["redshift_z"] > 0) & (df["redshift_z"] < 6)]
    df = df[df["flux_h_templfit"] > 0]

    sort_cols = [c for c in ["det_quality_flag", "spurious_prob", "blended_prob"] if c in df.columns]
    if sort_cols:
        asc = [False] + [True] * (len(sort_cols) - 1)
        df = df.sort_values(sort_cols, ascending=asc, kind="mergesort")

    df = df.drop_duplicates(subset=["object_id"], keep="first").reset_index(drop=True)

    core = [
        "flux_vis_1fwhm_aper", "flux_y_1fwhm_aper", "flux_j_1fwhm_aper", "flux_h_1fwhm_aper",
        "fluxerr_vis_1fwhm_aper", "fluxerr_y_1fwhm_aper", "fluxerr_j_1fwhm_aper", "fluxerr_h_1fwhm_aper",
    ]
    df = df.dropna(subset=core).copy()

    z = df["redshift_z"].to_numpy()
    mstar = 10.0 ** df["log_mass"].to_numpy()
    sfr = np.clip(df["sfr"].to_numpy(), 1e-12, None)

    ssfr = sfr / mstar
    hz = cosmo.H(z).to(1 / u.yr).value
    t_h_yr = 1.0 / hz
    ssfr_thresh = 0.3 / t_h_yr

    df["y_quenched"] = (ssfr < ssfr_thresh).astype(np.int8)

    print("Quenched fraction (0.3/t_H(z)):", df["y_quenched"].mean())
    print("N after core+label:", len(df))

    df["m_ref"] = -2.5 * np.log10(df["flux_h_templfit"].clip(lower=1e-6)) + 23.9
    m_lim = 24.0
    df["logM_lim"] = df["log_mass"] + 0.4 * (df["m_ref"] - m_lim)

    z_bins = [0.0, 0.5, 1.0, 1.2, 1.5, 2.0, 2.5, 6.0]
    df["z_bin"] = pd.cut(df["redshift_z"], bins=z_bins, include_lowest=True)

    q = 0.85
    faint_frac = 0.10

    def comp_limit_pozzetti(g):
        g = g.dropna(subset=["m_ref", "logM_lim"])
        if len(g) < 50:
            return np.nan

        m_cut = g["m_ref"].quantile(1 - faint_frac)
        faint = g[g["m_ref"] >= m_cut]

        if len(faint) < 20:
            faint = g

        return faint["logM_lim"].quantile(q)

    comp_by_bin = df.groupby("z_bin", observed=True).apply(comp_limit_pozzetti)
    fallback = df.groupby("z_bin", observed=True)["logM_lim"].quantile(q)
    comp_by_bin = comp_by_bin.fillna(fallback)

    df["logM_comp"] = df["z_bin"].map(comp_by_bin).astype(float)
    df["log_mass"] = pd.to_numeric(df["log_mass"], errors="coerce")
    df["logM_comp"] = pd.to_numeric(df["z_bin"].map(comp_by_bin), errors="coerce")

    df = df.dropna(subset=["log_mass", "logM_comp"]).copy()
    df = df[df["log_mass"].to_numpy() >= df["logM_comp"].to_numpy()].copy()

    print("After mass completeness cut:", df.shape)
    print("Quenched fraction after completeness:", df["y_quenched"].mean())
    print("M_comp per z-bin:\n", comp_by_bin)

    opt_bands = ["u", "g", "r", "i", "z"]
    lsst_flux = [f"flux_{b}_ext_lsst_1fwhm_aper" for b in opt_bands]
    decam_flux = [f"flux_{b}_ext_decam_1fwhm_aper" for b in opt_bands]

    for c in lsst_flux + decam_flux:
        if c not in df.columns:
            df[c] = np.nan

    df["n_lsst"] = df[lsst_flux].notna().sum(axis=1)
    df["n_decam"] = df[decam_flux].notna().sum(axis=1)

    for b in opt_bands:
        df[f"flux_{b}_opt"] = np.where(
            df["n_lsst"] >= df["n_decam"],
            df[f"flux_{b}_ext_lsst_1fwhm_aper"],
            df[f"flux_{b}_ext_decam_1fwhm_aper"],
        )
        df[f"fluxerr_{b}_opt"] = np.where(
            df["n_lsst"] >= df["n_decam"],
            df.get(f"fluxerr_{b}_ext_lsst_1fwhm_aper", np.nan),
            df.get(f"fluxerr_{b}_ext_decam_1fwhm_aper", np.nan),
        )

    df["opt_source"] = (df["n_decam"] > df["n_lsst"]).astype(np.int8)
    df["n_opt"] = df[[f"flux_{b}_opt" for b in opt_bands]].notna().sum(axis=1)

    feature_cols = [
        "flux_vis_1fwhm_aper", "flux_y_1fwhm_aper", "flux_j_1fwhm_aper", "flux_h_1fwhm_aper",
        "fluxerr_vis_1fwhm_aper", "fluxerr_y_1fwhm_aper", "fluxerr_j_1fwhm_aper", "fluxerr_h_1fwhm_aper",

        "flux_vis_2fwhm_aper", "flux_y_2fwhm_aper", "flux_j_2fwhm_aper", "flux_h_2fwhm_aper",
        "fluxerr_vis_2fwhm_aper", "fluxerr_y_2fwhm_aper", "fluxerr_j_2fwhm_aper", "fluxerr_h_2fwhm_aper",

        "flux_u_opt", "flux_g_opt", "flux_r_opt", "flux_i_opt", "flux_z_opt",
        "fluxerr_u_opt", "fluxerr_g_opt", "fluxerr_r_opt", "fluxerr_i_opt", "fluxerr_z_opt",
        "opt_source", "n_opt",

        "semimajor_axis", "ellipticity", "kron_radius", "fwhm", "mu_max",
        "mumax_minus_mag", "sersic_fract_vis_disk_sersic",

        "concentration", "gini", "moment_20", "asymmetry", "smoothness",

        "gal_ebv",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    release_cols = ["object_id"] + feature_cols

    train_pool = df[df["redshift_z"] < 1.2].copy()
    public_pool = df[(df["redshift_z"] >= 1.2) & (df["redshift_z"] < 1.5)].copy()
    private_pool = df[(df["redshift_z"] >= 1.5) & (df["redshift_z"] < 2.5)].copy()

    print("Pools after completeness (z split):", len(train_pool), len(public_pool), len(private_pool))
    print(
        "Quenched fractions train/public/private:",
        train_pool["y_quenched"].mean(),
        public_pool["y_quenched"].mean(),
        private_pool["y_quenched"].mean(),
    )

    desired_train, desired_public, desired_private = 80_000, 15_000, 30_000
    n_train = min(desired_train, len(train_pool))
    n_public = min(desired_public, len(public_pool))
    n_private = min(desired_private, len(private_pool))

    print("Requested sizes:", n_train, n_public, n_private)

    train_df = train_pool.sample(n=n_train, random_state=0).drop_duplicates("object_id")
    train_ids = set(train_df["object_id"])

    public_df = public_pool.sample(n=n_public, random_state=43).drop_duplicates("object_id")
    public_df = public_df[~public_df["object_id"].isin(train_ids)].copy()
    if len(public_df) > n_public:
        public_df = public_df.sample(n=n_public, random_state=143)

    public_ids = set(public_df["object_id"])

    private_df = private_pool.sample(n=n_private, random_state=44).drop_duplicates("object_id")
    private_df = private_df[~private_df["object_id"].isin(train_ids | public_ids)].copy()
    if len(private_df) > n_private:
        private_df = private_df.sample(n=n_private, random_state=144)

    print("Final sizes:", len(train_df), len(public_df), len(private_df))
    print(
        "Final quenched fractions:",
        train_df["y_quenched"].mean(),
        public_df["y_quenched"].mean(),
        private_df["y_quenched"].mean(),
    )

    x_train = train_df[release_cols].copy()
    x_test = public_df[release_cols].copy()
    x_private = private_df[release_cols].copy()

    y_train = train_df[["object_id", "y_quenched"]].copy()
    y_test = public_df[["object_id", "y_quenched", "z_bin"]].copy()
    y_private = private_df[["object_id", "y_quenched", "z_bin"]].copy()

    x_train.to_csv(train_dir / "train_features.csv", index=False)
    y_train.to_csv(train_dir / "train_labels.csv", index=False)

    x_test.to_csv(test_dir / "test_features.csv", index=False)
    x_private.to_csv(private_test_dir / "private_test_features.csv", index=False)

    y_test.to_csv(dev_ref_dir / "test_labels.csv", index=False)
    y_private.to_csv(final_ref_dir / "private_test_labels.csv", index=False)

    required = [
        dev_data_dir / "train" / "train_features.csv",
        dev_data_dir / "train" / "train_labels.csv",
        dev_data_dir / "test" / "test_features.csv",
        final_data_dir / "private_test_features.csv",
        dev_ref_dir / "test_labels.csv",
        final_ref_dir / "private_test_labels.csv",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required competition files:\n  - " + "\n  - ".join(missing)
        )

    print("OK: dev_phase and final_phase folders look ready.")


if __name__ == "__main__":
    main()