import numpy as np
from scipy.stats import sem

def run(ax, subject_session_data, start_from='std'):
    
     # Plotting d' and c over time (sessions)
    # plt.figure(figsize=(10, 5))
    
    dates = subject_session_data['dates']
    
    d_primes = subject_session_data['d_prime']
    
    
    # Plot d' (Sensitivity) over sessions
    # plt.subplot(1, 2, 1)
    ax.plot(d_primes, marker='o', label="d' (Sensitivity)")
    ax.set_xticks(np.arange(0, len(dates)))
    ax.set_xticklabels(dates, rotation=45)
    ax.set_xlabel('Session')
    ax.set_ylabel('d\'')
    ax.set_title("Sensitivity (d') Over Sessions")
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