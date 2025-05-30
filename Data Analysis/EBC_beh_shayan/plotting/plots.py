import numpy as np
import matplotlib.pyplot as plt
import h5py as hpy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from skimage.segmentation import find_boundaries
from plotting.fig1_mask import adjust_contrast

def plotting_roi_masks(masks_func):
    masks = {}

    for x in range(len(masks_func[0])):
        for y in range(len(masks_func)):
            if masks_func[y][x] > 0.2:
                pixel_value = masks_func[y][x]
                int_pixel_value = int(pixel_value)

                # Initialize the mask for this pixel value if it doesn't exist
                if int_pixel_value not in masks:
                    masks[int_pixel_value] = np.zeros((len(masks_func), len(masks_func[0])))

                # Set the value at the corresponding position to 1.0
                masks[int_pixel_value][y][x] = 1.0
    return masks

def plot_masks_functions(mask_file, ax_max, ax_mean, ax_mask):

    with hpy.File(mask_file) as f:
        max_func = f["max_func"][:]
        mean_func = f["mean_func"][:]
        masks_func = f["masks_func"][:]  # Assuming it contains multiple masks.
        f.close()


    max_func = adjust_contrast(max_func)
    mean_func = adjust_contrast(mean_func)
    masks_func = adjust_contrast(masks_func)

    f = max_func

    cmap = mcolors.LinearSegmentedColormap.from_list("green_black", ["black", "green"])
    func_img = np.zeros(
            (f.shape[0], f.shape[1], 3), dtype='int32')
    func_img[:, :, 1] = adjust_contrast(f)

    mean_img = np.zeros(
            (mean_func.shape[0], mean_func.shape[1], 3), dtype='int32')
    mean_img[:, :, 1] = adjust_contrast(mean_func)

    # x_all, y_all = np.where(find_boundaries(masks_func))
    # for x,y in zip(x_all, y_all):
    #     func_img[x,y,:] = np.array([255,255,255])

    # Plot max function
    ax_max.matshow(func_img)
    ax_max.set_title('Max')
    ax_max.axis('off')  # Turn off axes for cleaner visuals

    # Plot mean function
    ax_mean.matshow(mean_img)
    ax_mean.set_title('Mean')
    ax_mean.axis('off')

    # Plot first mask
    ax_mask.matshow(masks_func, cmap=cmap, alpha = 0.7)  # Display the first mask
    ax_mask.set_title('Masks')
    ax_mask.axis('off')

def plot_trial_averages_sig(trials, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, title_suffix, event, pooled, ax):
    x_min, x_max = -150, 550

    if title_suffix == "Short":
        suffix_color = "blue"
    if title_suffix == "Long":
        suffix_color = "lime"

    # Extract time and trial id
    time = aligned_time[list(aligned_time.keys())[0]]
    id = list(aligned_time.keys())[0]

    # Mask and filter data
    mask = (time >= x_min) & (time <= x_max)
    crp_filtered = crp_avg[mask]
    crp_sem_filtered = crp_sem[mask]
    crn_filtered = crn_avg[mask]
    crn_sem_filtered = crn_sem[mask]

    # Calculate y-axis limits
    y_min = min(
        np.min(crp_filtered - crp_sem_filtered),
        np.min(crn_filtered - crn_sem_filtered)
    )
    y_max = max(
        np.max(crp_filtered + crp_sem_filtered),
        np.max(crn_filtered + crn_sem_filtered)
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    trial_type = "Pooled" if pooled else "Grand"
    ax.set_title(f"{trial_type} Average of {title_suffix} Trials for {event} significant ROIs")
    ax.set_xlabel("Time for LED Onset (ms)")
    ax.set_ylabel("Mean df/f (+/- SEM)")

    ax.plot(time, crp_avg, label="CR+", color="red")
    ax.fill_between(time, crp_avg - crp_sem, crp_avg + crp_sem, color="red", alpha=0.1)

    ax.plot(time, crn_avg, label="CR-", color="blue")
    ax.fill_between(time, crn_avg - crn_sem, crn_avg + crn_sem, color="blue", alpha=0.1)

    # Add shaded regions for LED and AirPuff
    ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
    ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
               color=suffix_color, alpha=0.5, label="AirPuff")

    ax.legend()


def plot_trial_averages_side_by_side(
    ax1, ax2, n_value_p1, n_value_n1, aligned_time1, crp_avg1, crp_sem1, crn_avg1, crn_sem1, 
    n_value_p2, n_value_n2, aligned_time2, crp_avg2, crp_sem2, crn_avg2, crn_sem2, 
    trials, title_suffix1, title_suffix2, pooled=False
):
    x_min, x_max = -100, 600
    colors = {"Short": "blue", "Long": "lime"}
    
    # Calculate global y-axis limits
    global_y_min = float("inf")
    global_y_max = float("-inf")

    # First loop to determine global y-axis limits
    for n_value_p, n_value_n, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem in [
        (n_value_p1, n_value_n1, aligned_time1, crp_avg1, crp_sem1, crn_avg1, crn_sem1),
        (n_value_p2, n_value_n2, aligned_time2, crp_avg2, crp_sem2, crn_avg2, crn_sem2),
    ]:
        time = aligned_time[list(aligned_time.keys())[0]]
        mask = (time >= x_min) & (time <= x_max)
        crp_filtered = crp_avg[mask]
        crp_sem_filtered = crp_sem[mask]
        crn_filtered = crn_avg[mask]
        crn_sem_filtered = crn_sem[mask]

        global_y_min = min(global_y_min, np.min(crp_filtered - crp_sem_filtered), np.min(crn_filtered - crn_sem_filtered))
        global_y_max = max(global_y_max, np.max(crp_filtered + crp_sem_filtered), np.max(crn_filtered + crn_sem_filtered))

    # Second loop to plot data
    # Second loop to plot data
    for (ax, n_value_p, n_value_n, aligned_time, crp_avg, crp_sem, crn_avg, crn_sem, title_suffix) in [
        (ax1, n_value_p1, n_value_n1, aligned_time1, crp_avg1, crp_sem1, crn_avg1, crn_sem1, title_suffix1),
        (ax2, n_value_p2, n_value_n2, aligned_time2, crp_avg2, crp_sem2, crn_avg2, crn_sem2, title_suffix2),
    ]:

        suffix_color = colors.get(title_suffix, "gray")

        # Extract time and trial id
        time = aligned_time[list(aligned_time.keys())[0]]
        id = list(aligned_time.keys())[0]

        # Mask and filter data
        mask = (time >= x_min) & (time <= x_max)
        crp_filtered = crp_avg[mask]
        crn_filtered = crn_avg[mask]

        # Plot data
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(global_y_min, global_y_max)  # Use global y-axis limits
        trial_type = "Pooled" if pooled else "Grand"
        n_name = "df/f signals" if pooled else "Trial averages"
        ax.set_title(f"{trial_type} Average of df/f signals for {title_suffix} Trials")
        ax.set_xlabel("Time for LED Onset (ms)")
        ax.set_ylabel("Mean df/f (+/- SEM)")

        ax.plot(time, crp_avg, label=f"CR+- : {n_value_p} {n_name}", color="red")
        ax.fill_between(time, crp_avg - crp_sem, crp_avg + crp_sem, color="red", alpha=0.1)

        ax.plot(time, crn_avg, label=f"CR- : {n_value_n} {n_name}", color="blue")
        ax.fill_between(time, crn_avg - crn_sem, crn_avg + crn_sem, color="blue", alpha=0.1)

        # Add shaded regions for LED and AirPuff
        ax.axvspan(0, trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", alpha=0.5, label="LED")
        ax.axvspan(trials[id]["AirPuff"][0] - trials[id]["LED"][0], trials[id]["AirPuff"][1] - trials[id]["LED"][0], 
                   color=suffix_color, alpha=0.5, label="AirPuff")

        ax.legend()


def plot_heatmaps_side_by_side(heat_arrays, aligned_times, titles, trials, color_maps, axes):
    for i, (arr, aligned_time, title, color_map, ax) in enumerate(zip(heat_arrays, aligned_times, titles, color_maps, axes)):
        time_array = aligned_time[list(aligned_time.keys())[0]]
        print(time_array)
        id = list(aligned_time.keys())[0]
        time_limits = [-100, 600]

        # Set the min and max for the color scale based on the actual data
        vmin = arr.min()
        vmax = arr.max()

        # Plot heatmap with dynamic color scaling
        im = ax.imshow(arr, aspect="auto", cmap=color_map,
                       extent=[time_limits[0], time_limits[-1], 0, arr.shape[0]]
                       , vmin=vmin, vmax=vmax)

        # Add colorbar
        # plt.colorbar(im, ax=ax)

        airpuff_color = "lime" if "Long" in title else "blue"

        ax.axvline(trials[id]["AirPuff"][0] - trials[id]["LED"][0], color=airpuff_color, linestyle=':', linewidth=2, label="AirPuff")  # Dotted line for AirPuff
        ax.axvline(trials[id]["AirPuff"][1] - trials[id]["LED"][0], color=airpuff_color, linestyle=':', linewidth=2)  # Dotted line for AirPuff end
        ax.axvline(0, color="gray", linestyle=':', linewidth=2, label="LED")  # Dotted line for LED start
        ax.axvline(trials[id]["LED"][1] - trials[id]["LED"][0], color="gray", linestyle=':', linewidth=2)  # Dotted line for LED end

        ax.set_xlim(-100, 600)
        ax.set_title(title)
        ax.set_xlabel("Time for LED Onset(ms)")
        ax.set_ylabel("Number of ROIs")

def plot_single_condition_scatter(baseline_avg, event_avg, significant_rois, color, label, title):

    plt.figure(figsize=(8, 6))

    base_plot  = []
    event_plot = []

    for roi in baseline_avg:
        roi_color = 'lime' if roi in significant_rois else color
        base = baseline_avg[roi]
        event = event_avg[roi]
        # base_plot .append(np.nanmean(base, axis=0))
        # event_plot.append( np.nanmean(event, axis=0))
        base_plot = np.nanmean(base, axis=0)
        event_plot = np.nanmean(event, axis=0)

        plt.scatter(base_plot, event_plot, color=roi_color)
    # Labels, title, and legend
    plt.xlabel("Baseline Interval Average dF/F")
    plt.ylabel("Event Interval Average dF/F")
    plt.title(title)
    plt.grid(alpha=0.3)
    # plt.show()

def plot_fec_trial(ax, time, mean1, std1, mean0, std0, led, airpuff, y_lim, title):
    ax.plot(time, mean1, label="CR+", color="red")
    ax.fill_between(time, mean1 - std1, mean1 + std1, color="red", alpha=0.1)
    ax.plot(time, mean0, label="CR-", color="blue")
    ax.fill_between(time, mean0 - std0, mean0 + std0, color="blue", alpha=0.1)
    ax.axvspan(0, led[1] - led[0], color="gray", alpha=0.5, label="LED")
    ax.axvspan(airpuff[0] - led[0], airpuff[1] - led[0], color="blue" if "Short" in title else "lime", alpha=0.5, label="AirPuff")
    ax.set_title(title)
    ax.set_xlabel("Time for LED Onset (ms)")
    ax.set_ylabel("FEC")
    ax.set_xlim(-100, 600)
    # ax.set_ylim(y_lim, 1.1)
    ax.legend()


def plot_histogram(ax, data, bins, color, edgecolor, alpha, title, xlabel, ylabel):
    ax.hist(data, bins=bins, color=color, edgecolor=edgecolor, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_scatter(ax, x_data, y_data, color, alpha, label, title, xlabel, ylabel):
    ax.scatter(x_data, y_data, color=color, alpha=alpha, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_hexbin(ax, x_data, y_data, gridsize, cmap, mincnt, alpha, colorbar_label, title, xlabel, ylabel):
    hb = ax.hexbin(x_data, y_data, gridsize=gridsize, cmap=cmap, mincnt=mincnt, alpha=alpha)
    # cbar = plt.colorbar(hb, ax=ax)
    # cbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_mouse_summary(ax0, ax1, stacked_short_crp, stacked_short_crn, stacked_long_crp, stacked_long_crn, time_short, time_long, short_title, long_title, plot_type):
    avg_short_crp = np.mean(stacked_short_crp, axis=0)
    avg_short_crn = np.mean(stacked_short_crn, axis=0)
    avg_long_crp = np.mean(stacked_long_crp, axis=0)
    avg_long_crn = np.mean(stacked_long_crn, axis=0)

    number_short_crp = len(stacked_short_crp)
    number_short_crn = len(stacked_short_crn)
    number_long_crp = len(stacked_long_crp)
    number_long_crn = len(stacked_long_crn)

    sem_short_crp = np.std(stacked_short_crp, axis=0) / np.sqrt(number_short_crp)
    sem_short_crn = np.std(stacked_short_crn, axis=0) / np.sqrt(number_short_crn)
    sem_long_crp = np.std(stacked_long_crp, axis=0) / np.sqrt(number_long_crp)
    sem_long_crn = np.std(stacked_long_crn, axis=0) / np.sqrt(number_long_crn)

    # Short plot
    ax0.plot(time_short, avg_short_crn, color='blue', label=f'CR- (n={number_short_crn})')
    ax0.fill_between(time_short, avg_short_crn - sem_short_crn, avg_short_crn + sem_short_crn, color='blue', alpha=0.1)
    ax0.plot(time_short, avg_short_crp, color='red', label=f'CR+ (n={number_short_crp})')
    ax0.fill_between(time_short, avg_short_crp - sem_short_crp, avg_short_crp + sem_short_crp, color='red', alpha=0.1)
    ax0.set_title(short_title, fontsize=10)
    ax0.axvspan(0, 50, color="gray", alpha=0.3, label="LED")
    ax0.axvspan(200, 220, color="blue", alpha=0.3, label="Air Puff")
    ax0.legend()
    ax0.set_ylabel(f"{plot_type} (+/- SEM)")
    ax0.set_xlabel("Time from LED Onset")

    # Long plot
    ax1.plot(time_long, avg_long_crn, color='blue', label=f'CR- (n={number_long_crn})')
    ax1.fill_between(time_long, avg_long_crn - sem_long_crn, avg_long_crn + sem_long_crn, color='blue', alpha=0.1)
    ax1.plot(time_long, avg_long_crp, color='red', label=f'CR+ (n={number_long_crp})')
    ax1.fill_between(time_long, avg_long_crp - sem_long_crp, avg_long_crp + sem_long_crp, color='red', alpha=0.1)
    ax1.set_title(long_title, fontsize=10)
    ax1.axvspan(0, 50, color="gray", alpha=0.3, label="LED")
    ax1.axvspan(400, 420, color="lime", alpha=0.3, label="Air Puff")
    ax0.set_ylabel(f"{plot_type} (+/- SEM)")
    ax1.set_xlabel("Time from LED Onset")
    ax1.legend()
    # Set x limits
    ax0.set_xlim(-100, 600)
    ax1.set_xlim(-100, 600)

    # Remove top and right spines
    for ax in [ax0, ax1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Adjust y-axis limits to be the same for both plots in the row
    ymin = min(ax0.get_ylim()[0], ax1.get_ylim()[0])
    ymax = max(ax0.get_ylim()[1], ax1.get_ylim()[1])
    ax0.set_ylim(ymin, ymax)
    ax1.set_ylim(ymin, ymax)

def get_color_map_shades(index, total_number, num_shades, colormap="rainbow"):
    # Normalize the index to range [0,1] based on the max value (234)
    normalized_index = index / total_number
    
    # Get the color from the colormap
    cmap = cm.get_cmap(colormap)
    base_color = cmap(normalized_index)
    
    # Convert to RGB
    r, g, b, _ = base_color

    # Generate shades from dim to dark
    shades = [(r * i, g * i, b * i) for i in np.linspace(1, 0.5, num_shades)]
    
    # Create and return the custom colormap
    # return mcolors.LinearSegmentedColormap.from_list("custom_shades", shades)
    return shades
