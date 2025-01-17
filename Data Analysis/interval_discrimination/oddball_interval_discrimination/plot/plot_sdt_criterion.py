import numpy as np
from scipy.stats import sem

def run(ax, subject_session_data, start_from='std'):
    
     # Plotting d' and c over time (sessions)
    # plt.figure(figsize=(10, 5))
    
    dates = subject_session_data['dates']
    
    criterions = subject_session_data['criterion']
    
    
    # Plot d' (Sensitivity) over sessions
    # plt.subplot(1, 2, 1)
    ax.plot(criterions, marker='o', label="c (Criterion)", color='r')
    ax.set_xticks(np.arange(0, len(dates)))
    ax.set_xticklabels(dates, rotation=45)    
    ax.set_xlabel('Session')
    ax.set_ylabel('Criterion (c)')
    ax.set_title("Decision Criterion (c) Over Sessions")
    # Add a caption (figure description) at a specific position
    # ax.text(0, 0, 'Figure 1: A simple plot of x vs x^2', ha='center', va='center', fontsize=12)
    ax.grid(True)
    
    # # Plot c (Criterion) over sessions
    # plt.subplot(1, 2, 2)
    # plt.plot(criterions, marker='o', label="c (Criterion)", color='r')
    # plt.xlabel('Session')
    # plt.ylabel('Criterion (c)')
    # plt.title("Decision Criterion (c) Over Sessions")
    # plt.grid(True)
    
    # plt.tight_layout()
    # plt.show()