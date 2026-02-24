import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
from scipy.stats import binned_statistic
import math
from scipy.optimize import curve_fit

def get_mag_to_EMcountrate_slope_intercept(ata):

    x = np.array(ata['GAIAMAG'])
    y = np.array(np.log10(ata['TOTCORR_SUM'] / ata['EXPTIME']))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    bin_half_width = 0.1
    bin_width = 2 * bin_half_width
    bins = np.arange(x.min(), x.max() + bin_width, bin_width)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    counts, _, _ = binned_statistic(
        x, y, statistic='count', bins=bins
    )
    p75, _, _ = binned_statistic(
        x, y,
        statistic=lambda v: np.percentile(v, 75),
        bins=bins
    )
    valid = (counts >= 20) & np.isfinite(p75)
    x_fit = bin_centers[valid]
    y_fit = p75[valid]
    coeff = np.polyfit(x_fit, y_fit, 1)
    relation1_slope, relation1_intercept = coeff
    return relation1_slope, relation1_intercept, x_fit, y_fit

def get_EMcounts_to_Scicounts_slope_intercept(ata):
    xx = ata['TOTCORR_SUM'].values
    xx_fit = np.linspace(xx.min(), xx.max(), 200)
    yy2 = (ata['SNRSC652'].values)**2
    coeff_median_xx2 = np.polyfit(xx, yy2, 1)
    relation2_slope2, relation2_intercept2 = coeff_median_xx2
    return relation2_slope2, 0.0 # intercept must be zero for log fits. Also it *should* be zero....no counts on EM should be no counts on Sci.

def compute_exposure_info(fits, gmag, thresh):
    # mag_to_EMcounts_slope = -0.326
    # mag_to_EMcounts_intercept = 8.354
    # EMcounts_to_SCIcounts_slope = 0.0007321
    # EMcounts_to_SCIcounts_intercept = 605

    mag_to_EMcounts_slope = fits['mag_to_EMcounts_slope']
    mag_to_EMcounts_intercept = fits['mag_to_EMcounts_intercept']
    EMcounts_to_SCIcounts_slope = fits['EMcounts_to_SCIcounts_slope']
    EMcounts_to_SCIcounts_intercept = fits['EMcounts_to_SCIcounts_intercept']
    
    thresh_raw = (thresh*10**9) #EM cuts off at MegaPhotons per Angstrom. Each EM band is 100nm wide = 1000 Angstrom. Including the "Mega" together this makes 10**9
    
    lograte = mag_to_EMcounts_slope*gmag + mag_to_EMcounts_intercept # the fitted linear relationship of gmag to log count rate
    rate = (10**lograte) # counts/sec at gmag
    time = thresh_raw / rate # time in sec
    spectrum_counts_g = EMcounts_to_SCIcounts_slope*thresh_raw + EMcounts_to_SCIcounts_intercept # the fitted linear relationship of EM counts to L1 counts at 652nm

    return time, spectrum_counts_g

def build_grids(fits, test_thresh, test_gmag):

    per_g_times = []
    per_g_snrs = []
    per_g_ob = []
    test_threshholds = []
    for g in range(len(test_gmag)):
        green_snr2 = []
        pred_times = []
        for t in range(len(test_thresh)):
            time, spectrum_counts_g = compute_exposure_info(fits, test_gmag[g], test_thresh[t])
            green_snr2.append(spectrum_counts_g) #int(np.sqrt(spectrum_counts_g)))
            pred_times.append(time)
        test_threshholds.append(test_thresh)
        per_g_times.append(pred_times)
        per_g_snrs.append(green_snr2)
        per_g_ob.append(test_thresh)
    return per_g_times, per_g_snrs, per_g_ob, test_threshholds

def stretched_exp(x, A, tau, beta, C):
    return A * np.exp(-(x / tau)**beta) + C

def fit_teff_bins(ata, teff_bins, min_points=30):

    x_all = (ata['SNRSC652'].values)**2
    y_all = ata['CCFERV_MPS'].values
    teff_all = ata['TARGTEFF'].values

    base_mask = (
        (x_all < 1e6) &
        (x_all > 1e1) &
        (y_all < 10) &
        (teff_all <= max(teff_bins))
    )

    nbins = len(teff_bins) - 1
    cmap = pt.cm.jet_r
    colors = cmap(np.linspace(0.15, 0.85, nbins))

    fit_results = []

    for i in range(nbins):

        tmin = teff_bins[i]
        tmax = teff_bins[i+1]

        if i == nbins - 1:
            teff_mask = (teff_all >= tmin) & (teff_all <= tmax)
        else:
            teff_mask = (teff_all >= tmin) & (teff_all < tmax)

        mask = base_mask & teff_mask
        x = x_all[mask]
        y = y_all[mask]

        popt = None

        if len(x) >= min_points:
            
            ymask = y > 0
            x = x[ymask]
            y = y[ymask]

            # Initial guesses
            A0 = np.max(y) - np.min(y)
            tau0 = np.median(x)
            beta0 = 0.8  # <1 gives longer tail
            C0 = np.min(y)

#             try:
            popt, _ = curve_fit(
                stretched_exp,
                x,
                y,
                p0=[A0, tau0, beta0, C0],
                bounds=([0, 0, 0.3, 0], [np.inf, np.inf, 3, np.inf]),
                maxfev=50000
            )
#             except RuntimeError:
#                 popt = None

        fit_results.append({
            "tmin": tmin,
            "tmax": tmax,
            "x": x,
            "y": y,
            "color": colors[i],
            "params": popt
        })

    return fit_results

