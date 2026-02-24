import numpy as np
import pandas as pd
import h5py

def to_array_or_nan(x, n=4):
    if isinstance(x, np.ndarray):
        return x
    else:
        return np.full(n, np.nan)

def prepare_data(data_file):
    ata = pd.read_csv(data_file)
    ata['TOTCORR2'] = ata['TOTCORR'].str.split().apply(lambda x: np.array(x, dtype=float))
    ata['TOTCORR2_fixed'] = ata['TOTCORR2'].apply(to_array_or_nan)
    new_cols = ['TOTCORR_1', 'TOTCORR_2', 'TOTCORR_3', 'TOTCORR_4']
    ata[new_cols] = pd.DataFrame(ata['TOTCORR2_fixed'].tolist(), index=ata.index)
    # add a "sum" column
    ata['TOTCORR_SUM'] = ata[new_cols].sum(axis=1, min_count=4)

    # apply masks 
    date_mask = ata['MJD-OBS'] > max(ata['MJD-OBS']) - 765
    no_short_exp_mask = ata['EXPTIME'] > 60
    magmask = (ata['GAIAMAG'] > 4.) & (ata['GAIAMAG'] < 15.)
    magmask2 = (ata['GAIAMAG'] > 9.2) & (ata['GAIAMAG'] < 9.5) #because of missing labeling in OBs, this was a default value, so remove all stars with this value as the data is not usable.
    count_range = (ata['TOTCORR_SUM'] > 0) & (ata['TOTCORR_SUM'] < 2*10**9)

    ata = ata[date_mask&no_short_exp_mask&magmask&~magmask2&count_range]
    return ata

def save_fit_results_h5(filename, fit_results):
    """
    Save fit_results (from fit_teff_bins) into an HDF5 file.
    """
    with h5py.File(filename, "w") as f:
        for i, r in enumerate(fit_results):
            grp = f.create_group(f"bin_{i}")
            grp.attrs["tmin"] = r["tmin"]
            grp.attrs["tmax"] = r["tmax"]
            
            # params might be None
            if r["params"] is not None:
                grp.create_dataset("params", data=np.array(r["params"]))
            else:
                grp.create_dataset("params", data=np.array([np.nan]*4))
            
            # x, y arrays
            grp.create_dataset("x", data=np.array(r["x"]))
            grp.create_dataset("y", data=np.array(r["y"]))
            
            # color as RGBA
            grp.create_dataset("color", data=np.array(r["color"]))
            
def load_fit_results_h5(filename):
    """
    Load fit_results from an HDF5 file saved by save_fit_results_h5.
    """
    fit_results = []
    with h5py.File(filename, "r") as f:
        for key in sorted(f.keys(), key=lambda x: int(x.split('_')[1])):
            grp = f[key]
            params = np.array(grp["params"])
            # convert nan back to None
            if np.isnan(params).all():
                params = None
            else:
                params = params.tolist()
            
            fit_results.append({
                "tmin": grp.attrs["tmin"],
                "tmax": grp.attrs["tmax"],
                "x": np.array(grp["x"]),
                "y": np.array(grp["y"]),
                "color": np.array(grp["color"]),
                "params": params
            })
    return fit_results
