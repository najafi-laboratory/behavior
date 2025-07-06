import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from itertools import groupby

def plot_trial_outcome_vs_type_with_isi(session_props):
    trial_types = session_props['trial_type']
    isi_values = session_props['trial_isi']
    isi_devider = session_props['isi_devider']
    outcomes = session_props['outcome']
    warm_up = session_props['warm_up']  # Get warmup trial indicators
    opto_tag = session_props['opto_tag']  # Get opto tag indicators
    block_type = session_props['block_type']  # Get block type indicators
    block_length = session_props['block_length']
    
    # Extract ISI settings
    isi_settings = session_props['isi_settings']
    
    n_trials = len(trial_types)
    
    # Create trial indices with gaps
    gap_size = 0.7  # Size of gap between trials
    trial_indices = np.arange(0, n_trials * (1 + gap_size), 1 + gap_size)

    # Set up figure with gridspec
    fig = plt.figure(figsize=(50, 15))
    gs = gridspec.GridSpec(5, 6, height_ratios=[1, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
    
    # Define outcome color mapping
    outcome_colors = {
        'Reward': '#63f250',         # Green
        'RewardNaive': '#115e07',    # Darker green
        'Punish': '#e74c3c',         # Red
        'PunishNaive': '#6b201f',    # Darker red
        'WrongInitiation': '#f39c12', # Orange
        'ChangingMindReward': '#9b59b6', # Purple
        'EarlyChoice': '#3498db',    # Blue
        'DidNotChoose': '#95a5a6',   # Gray
        'Other': '#7f8c8d'           # Darker gray
    }
    
    # Define outcome marker mapping
    outcome_markers = {
        'Reward': 'o',               # Circle
        'RewardNaive': 'o',          # Circle
        'Punish': 'X',               # X
        'PunishNaive': 'X',          # X
        'WrongInitiation': 's',      # Square
        'ChangingMindReward': 'p',   # Pentagon
        'EarlyChoice': 'D',          # Diamond
        'DidNotChoose': '_',         # Horizontal line
        'Other': '+'                 # Plus
    }
    
     # Define colors and labels for block types
    shade_config = {
        0: {'color': 'red', 'label': '50/50 Blocks'},
        1: {'color': 'orange', 'label': 'Short'},
        2: {'color': 'pink', 'label': 'Long'},
    }
    
    #----------------------------------------------------------------------------------
    # First plot: Trial type with outcome coloring (left side of the figure)
    #----------------------------------------------------------------------------------
    ax0 = fig.add_subplot(gs[0:2, 0:5])  # span left width on top
    
    # Add shaded background for warmup trials
    warmup_end_idx = -1  # Initialize to -1
    for i, is_warmup in enumerate(warm_up):
        if is_warmup:
            warmup_end_idx = i
    
    if warmup_end_idx >= 0:  # If there are warmup trials
        warmup_end_x = trial_indices[warmup_end_idx] + 0.5  # Add half spacing for better visual
        ax0.axvspan(-1, warmup_end_x, alpha=0.15, color='lightblue', label='Warmup Trials')
        
        # Add text label for warmup area
        mid_x = warmup_end_x / 2
        ax0.text(mid_x, 0.4, 'Warmup Trials', 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 transform=ax0.get_xaxis_transform(),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
    # Iterate through grouped block types to find consecutive segments
    i = 0
    while i < len(block_type):
        current_type = block_type[i]
        if current_type in shade_config:
            # Find end of this consecutive block
            start_idx = i
            while i + 1 < len(block_type) and block_type[i + 1] == current_type:
                i += 1
            end_idx = i

            # Compute x positions for shading
            start_x = trial_indices[start_idx] - 0.5
            end_x = trial_indices[end_idx] + 0.5

            # Add shaded region
            label = shade_config[current_type]['label']
            color = shade_config[current_type]['color']
            ax0.axvspan(start_x, end_x, alpha=0.15, color=color,
                        label=label if start_idx == block_type.index(current_type) else "")

            # Add text label in the middle
            mid_x = (start_x + end_x) / 2
            ax0.text(mid_x, 0.5, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax0.get_xaxis_transform(),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        i += 1
    
    # Add shaded background for opto trials (opto_tag = 1)
    has_opto_trials = False
    for i, (tag, x_pos) in enumerate(zip(opto_tag, trial_indices)):
        if tag == 1:
            has_opto_trials = True
            # Add shading for this specific trial
            ax0.axvspan(x_pos - 0.4, x_pos + 0.4, alpha=0.2, color='royalblue')
    
    # Plot trial types with outcome-based coloring
    for i, (t_type, outcome, x_pos) in enumerate(zip(trial_types, outcomes, trial_indices)):
        # Set default values for unknown types
        marker = 'o'
        color = 'gray'
        y_pos = 0
        
        # Get color based on outcome
        if outcome in outcome_colors:
            color = outcome_colors[outcome]
        
        # Get marker based on outcome
        if outcome in outcome_markers:
            marker = outcome_markers[outcome]
            
        # Set y-position based on trial type
        if t_type == 1:  # Left
            y_pos = 1
        elif t_type == 2:  # Right
            y_pos = 2

        # Add vertical dashed line for block transitions
        if i + 1 < len(block_type) and block_type[i] != block_type[i + 1]:
            ax0.axvline(
            x=(x_pos + trial_indices[i + 1]) / 2,
            color='black',
            linestyle='--',
            alpha=0.5,
            linewidth=1,
            label=None
            )

            
        # Plot the point
        ax0.plot(x_pos, y_pos, marker=marker, color=color, markersize=10)

    
    # Add horizontal lines to separate trial types
    ax0.axhline(y=1.5, color='black', linestyle='--', alpha=0.3)
    
    # Set y-axis labels and ticks
    ax0.set_yticks([1, 2])
    ax0.set_yticklabels(['Left', 'Right'])
    ax0.set_xlabel('Trial Index')
    ax0.set_ylabel('Trial Type')
    ax0.set_title('Trial Type and Outcome Across Trials', fontsize=14)
    ax0.set_xlim(-1, max(trial_indices) + 1)
    
    # Add custom x-ticks at every 5th trial
    x_tick_indices = [trial_indices[i] for i in range(0, n_trials, 5)]
    x_tick_labels = [str(i) for i in range(0, n_trials, 5)]
    ax0.set_xticks(x_tick_indices)
    ax0.set_xticklabels(x_tick_labels)
    
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    
    # Create custom legend for outcomes
    legend_elements = []
    
    # Create a set of unique outcomes present in the data
    unique_outcomes = set(outcomes)
    
    # Add legend elements for each outcome that appears in the data
    for outcome in unique_outcomes:
        if outcome in outcome_colors and outcome in outcome_markers:
            legend_elements.append(
                Line2D([0], [0], 
                       marker=outcome_markers[outcome], 
                       color='w',
                       markerfacecolor=outcome_colors[outcome],
                       markersize=10,
                       label=outcome)
            )
    
    # Add warmup indicator to legend if warmup trials exist
    if warmup_end_idx >= 0:
        legend_elements.append(
            Patch(facecolor='lightblue', alpha=0.15, label='Warmup Trials')
        )
        
    # Add opto trial indicator to legend if opto trials exist
    if has_opto_trials:
        legend_elements.append(
            Patch(facecolor='royalblue', alpha=0.2, label='Opto Trials')
        )
    
    # Add the legend to the plot
    ax0.legend(handles=legend_elements, loc='best', 
               bbox_to_anchor=(1, 1), ncol=2)
    
    #----------------------------------------------------------------------------------
    # ISI Distribution Plots (right side of the figure)
    #----------------------------------------------------------------------------------
    # Function to plot normal distribution based on min, max, mean
    def plot_normal_distribution(ax, mean, min_val, max_val, color, title):
        # Calculate standard deviation based on the range (assuming the range is ±3 sigma)
        std_dev = (max_val - min_val) / 6
        
        # Create x values for the normal distribution
        x = np.linspace(min_val * 0.8, max_val * 1.2, 1000)
        
        # Calculate the normal distribution values
        y = stats.norm.pdf(x, mean, std_dev)
        
        # Plot the distribution
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, y, color=color, alpha=0.3)
        
        # Add vertical line for mean
        ax.axvline(mean, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean:.2f}s')
        
        # Add vertical lines for min and max
        ax.axvline(min_val, color='gray', linestyle=':', alpha=0.7, label=f'Min: {min_val:.2f}s')
        ax.axvline(max_val, color='gray', linestyle=':', alpha=0.7, label=f'Max: {max_val:.2f}s')
        
        # Set labels and title
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend(loc='upper right')
    
    # Plot short ISI distribution
    ax1 = fig.add_subplot(gs[0, 5])
    short_isi = isi_settings['short']
    plot_normal_distribution(
        ax1, 
        short_isi['mean'], 
        short_isi['min'], 
        short_isi['max'], 
        '#3498db',  # Blue
        'Short ISI Distribution'
    )
    
    # Plot long ISI distribution
    ax2 = fig.add_subplot(gs[1, 5])
    long_isi = isi_settings['long']
    plot_normal_distribution(
        ax2, 
        long_isi['mean'], 
        long_isi['min'], 
        long_isi['max'], 
        '#e74c3c',  # Red
        'Long ISI Distribution'
    )


    #----------------------------------------------------------------------------------
    # ISI Distribution for Trials Plots (bottom left of the figure)
    #----------------------------------------------------------------------------------
    # Plot distribiutin of ISI for each trial
    ax3 = fig.add_subplot(gs[2:4, 0:5])
    isi_values = np.array(isi_values)

    # Add shaded background for warmup trials
    warmup_end_idx = -1  # Initialize to -1
    for i, is_warmup in enumerate(warm_up):
        if is_warmup:
            warmup_end_idx = i
    
    if warmup_end_idx >= 0:  # If there are warmup trials
        warmup_end_x = trial_indices[warmup_end_idx] + 0.5  # Add half spacing for better visual
        ax3.axvspan(-1, warmup_end_x, alpha=0.15, color='lightblue', label='Warmup Trials')
        
        # Add text label for warmup area
        mid_x = warmup_end_x / 2
        ax3.text(mid_x, 0.4, 'Warmup Trials', 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 transform=ax3.get_xaxis_transform(),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Iterate through grouped block types to find consecutive segments
    i = 0
    while i < len(block_type):
        current_type = block_type[i]
        if current_type in shade_config:
            # Find end of this consecutive block
            start_idx = i
            while i + 1 < len(block_type) and block_type[i + 1] == current_type:
                i += 1
            end_idx = i

            # Compute x positions for shading
            start_x = trial_indices[start_idx] - 0.5
            end_x = trial_indices[end_idx] + 0.5

            # Add shaded region
            label = shade_config[current_type]['label']
            color = shade_config[current_type]['color']
            ax3.axvspan(start_x, end_x, alpha=0.15, color=color,
                        label=label if start_idx == block_type.index(current_type) else "")

            # Add text label in the middle
            mid_x = (start_x + end_x) / 2
            ax3.text(mid_x, 0.5, label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax3.get_xaxis_transform(),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        i += 1
    
    # Add shaded background for opto trials (opto_tag = 1)
    has_opto_trials = False
    for i, (tag, x_pos) in enumerate(zip(opto_tag, trial_indices)):
        if tag == 1:
            has_opto_trials = True
            # Add shading for this specific trial
            ax3.axvspan(x_pos - 0.4, x_pos + 0.4, alpha=0.2, color='royalblue')
    
    # Plot trial types with outcome-based coloring and timing
    for i, (t_type, outcome, x_pos) in enumerate(zip(trial_types, outcomes, trial_indices)):
        # Set default values for unknown types
        marker = 'o'
        color = 'gray'
        y_pos = isi_values[i]  # Use ISI value for y-position
        
        # Get color based on outcome
        if outcome in outcome_colors:
            color = outcome_colors[outcome]
        
        # Get marker based on outcome
        if outcome in outcome_markers:
            marker = outcome_markers[outcome]

        # Add vertical dashed line for block transitions
        if i + 1 < len(block_type) and block_type[i] != block_type[i + 1]:
            ax3.axvline(
            x=(x_pos + trial_indices[i + 1]) / 2,
            color='black',
            linestyle='--',
            alpha=0.5,
            linewidth=1,
            label=None
            )
            
        # Plot the point
        ax3.plot(x_pos, y_pos, marker=marker, color=color, markersize=10)
    
    # Add horizontal lines to separate trial types
    ax3.axhline(y=isi_devider, color='black', linestyle='--', alpha=0.3)
    
    # Set y-axis labels and ticks
    ax3.set_xlabel('Trial Index')
    ax3.set_ylabel('ISI (s)')
    ax3.set_title('Trial Type and Outcome Across Trials and ISI', fontsize=14)
    ax3.set_xlim(-1, max(trial_indices) + 1)
    
    # Add custom x-ticks at every 5th trial
    x_tick_indices = [trial_indices[i] for i in range(0, n_trials, 5)]
    x_tick_labels = [str(i) for i in range(0, n_trials, 5)]
    ax3.set_xticks(x_tick_indices)
    ax3.set_xticklabels(x_tick_labels)
    
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    #----------------------------------------------------------------------------------
    # ISI Distribution Plots (bottom right of the figure)
    #----------------------------------------------------------------------------------
    def plot_isi_distribution_from_values(ax, isi_values, color, title):
        """
        Plots a normal distribution based on actual ISI values using provided axis.

        Args:
            ax (matplotlib.axes.Axes): Axis to plot on.
            isi_values (array-like): Array of ISI values for one type (e.g., short or long).
            color (str): Hex or named color for the curve and fill.
            title (str): Title of the subplot.
        """
        isi_values = np.array(isi_values)
        if len(isi_values) == 0:
            ax.set_title(f"{title} (No Data)")
            return

        mean_val = np.mean(isi_values)
        min_val = np.min(isi_values)
        max_val = np.max(isi_values)
        std_dev = (max_val - min_val) / 6 if max_val > min_val else 1e-6

        x = np.linspace(min_val * 0.8, max_val * 1.2, 1000)
        y = stats.norm.pdf(x, mean_val, std_dev)

        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, y, color=color, alpha=0.3)
        ax.axvline(mean_val, color='black', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}s')
        ax.axvline(min_val, color='gray', linestyle=':', alpha=0.7, label=f'Min: {min_val:.2f}s')
        ax.axvline(max_val, color='gray', linestyle=':', alpha=0.7, label=f'Max: {max_val:.2f}s')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper right')

    # Separate ISI values by type
    trial_types = np.array(trial_types )
    short_isi = isi_values[trial_types == 1]
    long_isi = isi_values[trial_types == 2]

    # Plot short ISI distribution
    ax4 = fig.add_subplot(gs[2, 5])
    plot_isi_distribution_from_values(
        ax4,
        short_isi,  # ← actual ISI values, not stats
        '#3498db',         # blue
        'Experimental Short ISI Distribution'
    )

    # Plot long ISI distribution
    ax5 = fig.add_subplot(gs[3, 5])
    plot_isi_distribution_from_values(
        ax5,
        long_isi,   # ← actual ISI values, not stats
        '#e74c3c',         # red
        'Experimental Long ISI Distribution'
    )

    # Summary of the trials and outcomes in pie chart
    #----------------------------------------------------------------------------------
    # Pie chart for trial outcomes (last row of the figure)
    #----------------------------------------------------------------------------------
    # main function to plot pie chart
    def create_donut_pie_chart(data, labels, ax, colors=None, inner_radius=0.8, title = None):
        """
        Draws a donut (ring-shaped) pie chart on a given matplotlib axis.

        Parameters:
        - data: List of values for the pie chart.
        - labels: List of labels corresponding to each value.
        - ax: Matplotlib axis to draw the pie chart on.
        - colors: List of colors for each wedge (optional).
        - inner_radius: Float between 0 and 1, size of the hole in the center.

        Returns:
        - ax: The axis containing the donut pie chart.
        """

        # Combine label and value for display
        display_labels = [f"{label} ({value})" for label, value in zip(labels, data)]

        wedges, texts = ax.pie(data, labels=display_labels, colors=colors, startangle=90,
                            wedgeprops=dict(width=1-inner_radius))
        ax.set_aspect('equal')  # Keep it circular
        ax.set_title(title, fontsize=14)

        return ax
    
    # function for creating 2 half pie chart
    def create_split_pie_chart(left_data, left_labels, right_data, right_labels, ax, 
                          left_colors=None, right_colors=None, inner_radius=0.8, title=None,
                          left_title="Left Data", right_title="Right Data"):
        """
        Draws a split donut pie chart with left and right data sets on the same axis,
        separated by a vertical line.
        
        Parameters:
        - left_data: List of values for the left half of the pie chart.
        - left_labels: List of labels corresponding to each left value.
        - right_data: List of values for the right half of the pie chart.
        - right_labels: List of labels corresponding to each right value.
        - ax: Matplotlib axis to draw the pie chart on.
        - left_colors: List of colors for each left wedge (optional).
        - right_colors: List of colors for each right wedge (optional).
        - inner_radius: Float between 0 and 1, size of the hole in the center.
        - title: Title for the chart (optional).
        - left_title: Title for the left legend (optional).
        - right_title: Title for the right legend (optional).
        
        Returns:
        - ax: The axis containing the split donut pie chart.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        
        # Calculate dummy values to make each half span 180 degrees
        dummy_left = sum(left_data)
        dummy_right = sum(right_data)
        
        # Prepare pie data with dummy wedges
        left_pie_data = left_data + [dummy_left]
        right_pie_data = right_data + [dummy_right]
        
        # Prepare display labels with values for the legend
        left_display_labels = [f"{label} ({value})" for label, value in zip(left_labels, left_data)]
        right_display_labels = [f"{label} ({value})" for label, value in zip(right_labels, right_data)]
        
        # Set up colors if not provided, adding 'none' for dummy wedges
        if left_colors is None:
            left_colors = plt.cm.Pastel1(np.linspace(0, 0.5, len(left_data)))
        left_pie_colors = list(left_colors) + ['none']
        
        if right_colors is None:
            right_colors = plt.cm.Pastel2(np.linspace(0, 0.5, len(right_data)))
        right_pie_colors = list(right_colors) + ['none']
        
        # Remove all spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Remove tick marks and labels
        ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                    left=False, right=False, labelbottom=False, labelleft=False)
        
        # Draw left side pie chart (starts at 90°, counter-clockwise)
        wedges_left, _ = ax.pie(
            left_pie_data, 
            labels=None,
            colors=left_pie_colors,
            startangle=90,
            counterclock=True,
            wedgeprops=dict(width=1-inner_radius),
            radius=1.0,
            center=(0, 0)
        )
        
        # Draw right side pie chart (starts at 270°, counter-clockwise)
        wedges_right, _ = ax.pie(
            right_pie_data, 
            labels=None,
            colors=right_pie_colors,
            startangle=270,
            counterclock=True,
            wedgeprops=dict(width=1-inner_radius),
            radius=1.0,
            center=(0, 0)
        )
        
        # Add a vertical line to separate the halves
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.7, zorder=10)
        
        # Create legend elements, excluding dummy wedges
        left_legend_elements = [Patch(facecolor=wedge.get_facecolor(), edgecolor='w', label=label) 
                                for wedge, label in zip(wedges_left[:-1], left_display_labels)]
        right_legend_elements = [Patch(facecolor=wedge.get_facecolor(), edgecolor='w', label=label) 
                                for wedge, label in zip(wedges_right[:-1], right_display_labels)]
        
        # Add two separate legends
        left_legend = ax.legend(handles=left_legend_elements, loc='center left', 
                                bbox_to_anchor=(-1, 0.5), title=left_title, frameon=False)
        ax.add_artist(left_legend)
        
        ax.legend(handles=right_legend_elements, loc='center right', 
                bbox_to_anchor=(2, 0.5), title=right_title, frameon=False)
        
        # Set aspect ratio to keep it circular
        ax.set_aspect('equal')
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=14)
        
        return ax
    
    # count Opto vs Non-Opto trials
    opto_count = sum(1 for tag in opto_tag if tag == 1)
    non_opto_count = len(opto_tag) - opto_count
    opto_labels = ['Opto Trials', 'Non-Opto Trials']
    opto_data = [opto_count, non_opto_count]
    opto_colors = ['#3498db', '#e74c3c']  # Blue and Red

    # plot
    ax6 = fig.add_subplot(gs[4, 0])
    create_donut_pie_chart(opto_data, opto_labels, ax6, 
                           colors=opto_colors, inner_radius=0.8, title= 'opto vs Non-Opto Trials')
    
    # plot Majority and Minority blocks
    ax7 = fig.add_subplot(gs[4, 1])
    # Count blocks of each type non warm-up trials
    filterd_trials = [trial for i, trial in enumerate(trial_types) if warm_up[i] == 0]
    short_maj_outcomes = [outcome for i, outcome in enumerate(filterd_trials) if block_type[i] == 1]
    long_maj_outcomes = [outcome for i, outcome in enumerate(filterd_trials) if block_type[i] == 2]

    # count outcomes for short and long majority blocks
    short_maj_counts = {outcome: short_maj_outcomes.count(outcome) for outcome in set(short_maj_outcomes)}
    long_maj_counts = {outcome: long_maj_outcomes.count(outcome) for outcome in set(long_maj_outcomes)}

    short_maj_labels = ['short', 'long']
    short_maj_data = [count for outcome, count in zip(short_maj_counts.keys(), short_maj_counts.values())]

    long_maj_labels = ['short', 'long']
    long_maj_data = [count for outcome, count in zip(long_maj_counts.keys(), long_maj_counts.values())]

    # get colors for short and long majority blocks
    maj_color = ['#3498db', '#e74c3c']  # Blue and Red

    create_split_pie_chart(short_maj_data, short_maj_labels, long_maj_data, long_maj_labels, ax7,
                                left_colors=maj_color, right_colors=maj_color,
                                inner_radius=0.8, title='Majority Distribution \n'+'Short|Long')
    
    # outcome distribution all trials
    outcome_counts = {outcome: outcomes.count(outcome) for outcome in set(outcomes)}
    outcome_labels = list(outcome_counts.keys())
    outcome_data = list(outcome_counts.values())
    outcome_colors_list = [outcome_colors[outcome] for outcome in outcome_labels if outcome in outcome_colors]

    # plot
    ax8 = fig.add_subplot(gs[4, 2])
    create_donut_pie_chart(outcome_data, outcome_labels, ax8, 
                           colors=outcome_colors_list, inner_radius=0.8, title= 'Outcome Distribution for All Trials')
    
    # Outcome distribution for non-opto and non-warmup trials based on the side choosen
    filterd_outcomes = [outcome for i, outcome in enumerate(outcomes) if warm_up[i] == 0 and opto_tag[i] == 0]
    short_isi_outcomes = [outcome for i, outcome in enumerate(filterd_outcomes) if trial_types[i] == 1]
    long_isi_outcomes = [outcome for i, outcome in enumerate(filterd_outcomes) if trial_types[i] == 2]
    # count outcomes for short and long ISI trials
    short_isi_counts = {outcome: short_isi_outcomes.count(outcome) for outcome in set(short_isi_outcomes)}
    long_isi_counts = {outcome: long_isi_outcomes.count(outcome) for outcome in set(long_isi_outcomes)}
    # remove punish naive and reward naive from the list of outcomes
    short_isi_labels = [outcome for outcome in short_isi_counts.keys() if outcome not in ['PunishNaive', 'RewardNaive']]
    short_isi_data = [count for outcome, count in zip(short_isi_counts.keys(), short_isi_counts.values()) if outcome not in ['PunishNaive', 'RewardNaive']]

    long_isi_labels = [outcome for outcome in long_isi_counts.keys() if outcome not in ['PunishNaive', 'RewardNaive']]
    long_isi_data = [count for outcome, count in zip(long_isi_counts.keys(), long_isi_counts.values()) if outcome not in ['PunishNaive', 'RewardNaive']]

    # get colors for short and long ISI trials (keep colors of punish naive and reward naive out)
    filtered_outcome_colors_short_isi = {k: v for k, v in outcome_colors.items() if k not in ['PunishNaive', 'RewardNaive']}
    filtered_outcome_colors_short_isi = [filtered_outcome_colors_short_isi[outcome] for outcome in short_isi_labels if outcome in filtered_outcome_colors_short_isi]
    filtered_outcome_colors_long_isi = {k: v for k, v in outcome_colors.items() if k not in ['PunishNaive', 'RewardNaive']}
    filtered_outcome_colors_long_isi = [filtered_outcome_colors_long_isi[outcome] for outcome in long_isi_labels if outcome in filtered_outcome_colors_long_isi]


    # plot short ISI trials
    ax9 = fig.add_subplot(gs[4, 3])
    create_split_pie_chart(short_isi_data, short_isi_labels, long_isi_data, long_isi_labels, ax9,
                             left_colors=filtered_outcome_colors_short_isi, right_colors=filtered_outcome_colors_long_isi,
                             inner_radius=0.8, title='Decision distribution for non-warmup and non-opto \n'+'Left|Right')

    
    if sum(opto_tag) > 0:
        # Outcome distribution for just opto trials (short ISI vs long ISI)
        opto_outcomes = [outcome for i, outcome in enumerate(outcomes) if warm_up[i] == 0 and opto_tag[i] == 1]
        opto_short_isi_outcomes = [outcome for i, outcome in enumerate(opto_outcomes) if trial_types[i] == 1]
        opto_long_isi_outcomes = [outcome for i, outcome in enumerate(opto_outcomes) if trial_types[i] == 2]
        # count outcomes for short and long ISI trials
        opto_short_isi_counts = {outcome: opto_short_isi_outcomes.count(outcome) for outcome in set(opto_short_isi_outcomes)}
        opto_long_isi_counts = {outcome: opto_long_isi_outcomes.count(outcome) for outcome in set(opto_long_isi_outcomes)}
        # remove punish naive and reward naive from the list of outcomes
        opto_short_isi_labels = [outcome for outcome in opto_short_isi_counts.keys() if outcome not in ['PunishNaive', 'RewardNaive']]
        opto_short_isi_data = [count for outcome, count in zip(opto_short_isi_counts.keys(), opto_short_isi_counts.values()) if outcome not in ['PunishNaive', 'RewardNaive']]

        opto_long_isi_labels = [outcome for outcome in opto_long_isi_counts.keys() if outcome not in ['PunishNaive', 'RewardNaive']]
        opto_long_isi_data = [count for outcome, count in zip(opto_long_isi_counts.keys(), opto_long_isi_counts.values()) if outcome not in ['PunishNaive', 'RewardNaive']]

        # get colors for short and long ISI trials (keep colors of punish naive and reward naive out)
        filtered_opto_outcome_colors_short_isi = {k: v for k, v in outcome_colors.items() if k not in ['PunishNaive', 'RewardNaive']}
        filtered_opto_outcome_colors_short_isi = [filtered_opto_outcome_colors_short_isi[outcome] for outcome in opto_short_isi_labels if outcome in filtered_opto_outcome_colors_short_isi]
        filtered_opto_outcome_colors_long_isi = {k: v for k, v in outcome_colors.items() if k not in ['PunishNaive', 'RewardNaive']}
        filtered_opto_outcome_colors_long_isi = [filtered_opto_outcome_colors_long_isi[outcome] for outcome in opto_long_isi_labels if outcome in filtered_opto_outcome_colors_long_isi]

        # plot opto trials
        ax10 = fig.add_subplot(gs[4, 4])    
        create_split_pie_chart(opto_short_isi_data, opto_short_isi_labels, opto_long_isi_data, opto_long_isi_labels, ax10,
                                left_colors=filtered_opto_outcome_colors_short_isi, right_colors=filtered_opto_outcome_colors_long_isi,
                                inner_radius=0.8, title='Decision distribution for opto \n'+'Left|Right')

    #----------------------------------------------------------------------------------
    
    # Summary of the session
    #----------------------------------------------------------------------------------
    # text summary of the session
    #----------------------------------------------------------------------------------
    ax11 = fig.add_subplot(gs[4, 5])
    ax11.axis('off')  # Turn off the axis frame, ticks, etc.

    # Add a text box
    info_text = "Summary:\n" + \
               "- Subject: {}\n".format(session_props['subject']) + \
               "- Session: {}\n".format(session_props['session_date']) + \
               "- Experimenter: {}\n".format(session_props['experimentor']) + \
               "- Version: {}\n".format(session_props['version']) + \
               "- Training Level: {}\n".format(session_props['difficulty']) + \
               "- Anti Bias: {}\n".format(session_props['Anti_bias']) + \
               "- Number of Trials: {}\n".format(len(session_props['trial_type'])) + \
               "- Number of Warmup Trials: {}\n".format(session_props['n_warm_up_trials']) + \
               "- Max Same Side: {}\n".format(session_props['n_max_same_side'])
    ax11.text(0.5, 0.5, info_text,
            fontsize=16,
            ha='center', va='center',
            bbox=dict(boxstyle="round", facecolor='whitesmoke', edgecolor='gray'))

    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.8)  # Add space between subplots
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.close('all')
    
    return fig