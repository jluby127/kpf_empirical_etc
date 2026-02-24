import numpy as np
import pandas as pd
import matplotlib.pyplot as pt
import math
from scipy.stats import binned_statistic
from functions.relations import model
from matplotlib.ticker import LogFormatterSciNotation


def gmag_to_EMcountrate(ata, relation1_slope, relation1_intercept, x_fit, y_fit):
    x = np.array(ata['GAIAMAG'])
    y = np.array(np.log10(ata['TOTCORR_SUM'] / ata['EXPTIME']))
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    pt.plot(x, y, ',k')
    pt.plot(x_fit, y_fit, '.r', label='Binned median')

    x_fit2 = np.linspace(x_fit.min(), x_fit.max(), 200)
    pt.plot(x_fit2, relation1_slope * x_fit2 + relation1_intercept, 'r-')
    pt.text(0.65, 0.95, f'75th%tile binned fit\ny = {relation1_slope:.3f} x + {relation1_intercept:.3f}',
            color='red', transform=pt.gca().transAxes, va='top')

    size=15
    pt.xlabel("G Mag", fontsize=size)
    pt.ylabel("log(EM Counts/sec)", fontsize=size)
    pt.tick_params(axis='both', labelsize=size)
    pt.show()

def EMcounts_to_Scicounts(ata, relation2_slope2, relation2_intercept2):

    xx = ata['TOTCORR_SUM'].values
    xx_fit = np.linspace(xx.min(), xx.max(), 200)
    yy2 = (ata['SNRSC652'].values)**2

    pt.plot(xx, yy2, '.g', alpha=1, zorder=3)
    pt.plot(xx_fit, relation2_slope2 * xx_fit + relation2_intercept2, 'k-', zorder=3)
    pt.text(0.05, 0.95, f'@652nm: Fit\ny = {relation2_slope2:.6f} x + {relation2_intercept2:.0f}', color='g', transform=pt.gca().transAxes, va='top')

    size=15
    pt.xlabel("ExpMeter Counts Sum Channels", fontsize=size)
    pt.ylabel("Spectrum Counts", fontsize=size)
    pt.tick_params(axis='both', labelsize=size)
    pt.ylim(-10**5, 0.18*10**7)
    pt.show()

def exptime_per_counts_or_OBthresh(per_g_times, per_g_snrs, per_g_ob, test_gmag, test_threshholds):

    size = 12
    fig, ax1 = pt.subplots(figsize=(10, 4))

    slopes = []
    intercepts = []
    gmag_labels = []
    nb_lines = len(per_g_times)
    colors = ['darkblue', 'purple', 'blueviolet', 'blue', 'darkgreen', 'yellowgreen', 'gold', 'darkorange', 'magenta', 'red', 'maroon']


    # ---------- MAIN PANEL ----------
    for p in range(len(per_g_times)):
        times = np.array(per_g_times[p])
        counts = np.array(per_g_snrs[p])

        mask = [True] * len(times)
        times_masked = times[mask]
        counts_masked = counts[mask]
        domain_mask = (counts_masked > 0) #& (counts_masked.isfinite())

        # --- Log-log linear fit ---
        log_x = np.log10(counts_masked[domain_mask])
        log_y = np.log10(times_masked[domain_mask])
        slope, intercept = np.polyfit(log_x, log_y, 1)

        slopes.append(slope)
        intercepts.append(intercept)
        gmag_labels.append(test_gmag[p])

        ax1.plot(counts_masked, times_masked, '-', color=colors[p], label=str(test_gmag[p]))


    ax1.set_xlabel("L1 Avg Per Pixel Counts near 652nm", fontsize=size)
    ax1.set_ylabel("Exposure Time (s)", fontsize=size)
    ax1.tick_params(axis='both', labelsize=size)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # ---------- TOP AXIS CONVERSION ----------
    # Flatten arrays to determine relation between counts and OB threshold
    all_counts = []
    all_thresh = []

    for p in range(len(per_g_times)):
        all_counts.extend(per_g_snrs[p])
        all_thresh.extend(test_threshholds[p])

    all_counts = np.array(all_counts)
    all_thresh = np.array(all_thresh)

    # Fit relation in log space
    m, b = np.polyfit(np.log10(all_counts), np.log10(all_thresh), 1)

    def counts_to_thresh(x):
        return 10**b * x**m

    def thresh_to_counts(x):
        return (x / 10**b)**(1/m)

    secax = ax1.secondary_xaxis('top', functions=(counts_to_thresh, thresh_to_counts))
    secax.set_xlabel(r"$\mathit{exp\_meter\_threshold}$ value in OB", fontsize=size)
    secax.set_xscale('log')
    secax.tick_params(axis='x', labelsize=size)

    # ---------- Horizontal guide lines ----------
    for yline in [60, 300, 1200, 3600]:
        ax1.axhline(yline, color='gray', linestyle='--')

    ax1.set_ylim(3, 4500)

    # ---------- Table with intercepts ----------
    avg_slope = np.mean(slopes)

    table_data = [[f"{g}", f"{intercept:.2f}"] for g, intercept in zip(gmag_labels, intercepts)]
    column_labels = ["Gmag", "Intercept"]

    table_ax = fig.add_axes([0.78, 0.1, 0.18, 0.8])
    table_ax.axis('off')

    table_data_rev = table_data[::-1]
    colors_rev = colors[::-1]

    tbl = table_ax.table(
        cellText=table_data_rev,
        colLabels=column_labels,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    # Color rows to match curves
    for i, color in enumerate(colors_rev):
        for j in range(len(column_labels)):
            tbl[(i+1, j)].set_facecolor(color)
            tbl[(i+1, j)].get_text().set_color('white')

    # Slope text
    table_ax.text(
        0.5, 0.82,
        f"Log-Log \n Slopes = {avg_slope:.2f}",
        ha='center', va='bottom',
        fontsize=size,
        fontweight='bold'
    )

    fig.tight_layout(rect=[0, 0, 0.77, 0.93])

    pt.show()

def plot_indv_counts_to_err_fits(fit_results, nx_fit=1000, index=None):
    """
    Plot individual panels with scatter, fitted model lines, and equations.
    
    Parameters
    ----------
    fit_results : list of dicts
        Output from fit_teff_bins(), must contain keys: x, y, params, color, tmin, tmax
    nx_fit : int
        Number of points used to plot the fit line
    index : int, optional
        If given, plot only the single panel for fit_results[index]. Otherwise plot all panels.
    """
    
    if index is not None:
        fit_results = [fit_results[index]]
        nbins = 1
        ncols = 1
        nrows = 1
        figsize = (8, 6)
        font_scale = 1.5
    else:
        nbins = len(fit_results)
        ncols = math.ceil(np.sqrt(nbins))
        nrows = math.ceil(nbins / ncols)
        figsize = (4*ncols, 4*nrows)
        font_scale = 1

    fig, axes = pt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=True, sharey=True
    )

    axes = np.array(axes).reshape(-1)

    for i, res in enumerate(fit_results):

        axes[i].scatter(res["x"], res["y"],
                        s=10, alpha=0.7,
                        color=res["color"])

        axes[i].axhline(0.5, color='gray', linestyle='--')
        axes[i].axhline(1.0, color='gray', linestyle='--')

        axes[i].set_title(
            f'{int(res["tmin"])}–{int(res["tmax"])} K '
            f'(N={len(res["x"])})',
            fontsize=10*font_scale
        )

        # --- Add fit line if params exist ---
        if res["params"] is not None and len(res["x"]) > 0:
            A, alpha, C = res["params"]
            x_fit = np.linspace(res["x"].min(), res["x"].max(), nx_fit)
            y_fit = model(x_fit, A, alpha, C)

            axes[i].plot(x_fit, y_fit,
                         color='k',
                         linewidth=2)

            # Add equation to title (after temp range and N count)
            eq_text = rf"$y = {A:.2e} \, x^{{-{alpha:.2f}}} + {C:.3f}$"
            axes[i].set_title(
                f'{int(res["tmin"])}–{int(res["tmax"])} K '
                f'(N={len(res["x"])})\n{eq_text}',
                color='k',
                fontsize=8*font_scale
            )
        axes[i].set_xscale('log')
        axes[i].xaxis.set_major_formatter(LogFormatterSciNotation())
        axes[i].tick_params(axis='both', labelsize=10*font_scale)

    for j in range(nbins, len(axes)):
        axes[j].axis("off")

    fig.supxlabel('Counts on L1 File near 652nm', fontsize=12*font_scale)
    fig.supylabel('CCFERV (m/s)', fontsize=12*font_scale)
    fig.tight_layout()
    pt.show()
    
def plot_global_counts_to_err_fits(fit_results):

    fig, ax = pt.subplots(figsize=(8, 6))

    # # --- Background faint cloud ---
    # for res in fit_results:
    #     ax.scatter(res["x"], res["y"],s=5, color='k', alpha=0.5)

    # Determine global x-range
    x_min = min([res["x"].min() for res in fit_results if len(res["x"]) > 0])
    x_max = max([res["x"].max() for res in fit_results if len(res["x"]) > 0])
    x_fit = np.linspace(x_min, x_max, 600)

    # --- Plot fits and prepare equation labels for title ---
    title_lines = []  # list of (eq_text, color)

    for res in fit_results:

        if res["params"] is not None:
            A, alpha, C = res["params"]
            y_fit = model(x_fit, A, alpha, C)

            # Plot curve
            ax.plot(
                x_fit,
                y_fit,
                color=res["color"],
                linewidth=3,
                label=f'{int(res["tmin"])}–{int(res["tmax"])} K'
            )

            # Format equation for title line
            eq_text = (
                rf"{int(res['tmin'])}–{int(res['tmax'])} K:  "
                rf"$y = {A:.2e} \, x^{{-{alpha:.2f}}} + {C:.3f}$"
            )
            title_lines.append((eq_text, res["color"]))

    # Draw title: one line per fit, each with its own color
    title_y = 1.4
    title_step = 0.06
    for eq_text, color in title_lines:
        ax.text(0.5, title_y, eq_text,
                transform=ax.transAxes,
                color=color,
                fontsize=11,
                ha='center', va='top')
        title_y -= title_step

    ax.axhline(0.5, color='gray', linestyle='--')
    ax.axhline(1.0, color='gray', linestyle='--')

    ax.set_xlabel('L1 Counts near 652nm', fontsize=14)
    ax.set_ylabel('CCFERV (m/s)', fontsize=14)
    ax.set_ylim(0.1,100)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(alpha=0.2)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(LogFormatterSciNotation())

    pt.tight_layout()
    pt.show()