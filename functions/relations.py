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

def compute_exposure_info(equations, gmag, thresh):

    mag_to_EMcounts_slope = equations['mag_to_EMcounts_slope']
    mag_to_EMcounts_intercept = equations['mag_to_EMcounts_intercept']
    EMcounts_to_SCIcounts_slope = equations['EMcounts_to_SCIcounts_slope']
    EMcounts_to_SCIcounts_intercept = equations['EMcounts_to_SCIcounts_intercept']
    
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

def model(x, A, alpha, C):
    return A * x**(-alpha) + C

def stretched_exp(x, A, tau, beta, C):
    return A * np.exp(-(x / tau)**beta) + C

def fit_teff_bins(ata, teff_bins, min_points=30, max_erv=30):

    x_all = (ata['SNRSC652'].values)**2
    y_all = ata['CCFERV_MPS'].values
    teff_all = ata['TARGTEFF'].values

    base_mask = (
        (x_all < 1e6) &
        (x_all > 1e1) &
        (y_all < max_erv) &
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
            A0 = (np.max(y) - np.min(y)) * np.median(x)**0.3
            alpha0 = 0.5
            C0 = np.min(y)

            popt, _ = curve_fit(
                model,
                x,
                y,
                p0=[A0, alpha0, C0],
                bounds=([0, 0.01, 0], [np.inf, 10, np.inf]),
                maxfev=50000
            )

        fit_results.append({
            "tmin": tmin,
            "tmax": tmax,
            "x": x,
            "y": y,
            "color": colors[i],
            "params": popt
        })

    return fit_results

def predict_rv_uncertainty(fit_results, temperature, mycounts):
    """
    Predict y-value (CCFERV) from x (mycounts) and a given temperature,
    using the appropriate temperature bin fit.

    Parameters
    ----------
    fit_results : list of dicts
        Output of fit_teff_bins()
    temperature : float
        Temperature (TARGTEFF) to choose bin
    mycounts : float
        Value to plug into x = (SNRSC652)^2

    Returns
    -------
    float
        Predicted y-value, rounded to 2 decimals
    """

    # Find the first bin that contains the temperature
    for res in fit_results:
        if res["tmin"] <= temperature < res["tmax"] or (
            temperature == res["tmax"] and res == fit_results[-1]
        ):
            if res["params"] is None:
                raise ValueError("No fit available for this bin.")
            A, alpha, C = res["params"]
            x_val = mycounts**2
            y_val = model(x_val, A, alpha, C)
            return np.round(y_val, 2)

    # If no bin matches
    raise ValueError("Temperature not within any bin range.")
    
def compute_exptime_from_rv_linear(RV_target, Teff, Gmag, fit_results, equations):
    """
    Compute exposure time and threshold needed to reach a desired RV uncertainty
    for a star of given Teff and G magnitude, using the RV-L1 counts fit
    and recompute_exptime2 linear relationships.
    """

    # --- Step 1: select Teff bin ---
    for res in fit_results:
        if res["tmin"] <= Teff < res["tmax"] or (
            Teff == res["tmax"] and res == fit_results[-1]
        ):
            if res["params"] is None:
                raise ValueError("No RV fit available for this Teff bin.")
            A, alpha, C = res["params"]
            break
    else:
        raise ValueError("Temperature not within any bin range.")

    if RV_target <= C:
        raise ValueError("RV_target must be greater than C for this fit.")

    # Inverse of y = A*x**(-alpha) + C  =>  x = ((RV_target - C)/A)**(-1/alpha)
    L1_squared = ((RV_target - C) / A)**(-1 / alpha)
    thresh_raw = (L1_squared - equations['EMcounts_to_SCIcounts_intercept']) / equations['EMcounts_to_SCIcounts_slope']
    lograte = equations['mag_to_EMcounts_slope']*Gmag + equations['mag_to_EMcounts_intercept'] 
    rate = (10**lograte)
    exptime = thresh_raw / rate
    threshOB = np.round(thresh_raw / 1e9,3)
    
    time, spectrum_counts_g = compute_exposure_info(equations, Gmag, threshOB)

    return int(exptime), threshOB, int(spectrum_counts_g)