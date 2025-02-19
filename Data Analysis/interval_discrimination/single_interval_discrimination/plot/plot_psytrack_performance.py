import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

COLORS = {'bias' : '#FAA61A', 
          's1' : "#A9373B", 's2' : "#2369BD", 
          's_a' : "#A9373B", 's_b' : "#2369BD", 
          'sR' : "#A9373B", 'sL' : "#2369BD",
          'cR' : "#A9373B", 'cL' : "#2369BD",
          'c' : '#59C3C3', 'h' : '#9593D9', 's_avg' : '#99CC66',
          'emp_perf': '#E32D91', 'emp_bias': '#9252AB'}
ZORDER = {'bias' : 2, 
          's1' : 3, 's2' : 3, 
          's_a' : 3, 's_b' : 3, 
          'sR' : 3, 'sL' : 3,
          'cR' : 3, 'cL' : 3,
          'c' : 1, 'h' : 1, 's_avg' : 1}

def run(ax, subject_session_data, xval_pL=None, sigma=50, figsize=(5, 1.5)):
    '''Plots empirical and (optional) cross-validated prediction of performance.
    
    Args:
        dat: a standard Psytrack input dataset.
        xval_pL: array of cross-validated P(y=0) for each trial in dat, the
            output of crossValidation().
        sigma: option passed to gaussian_filter controling smoothing of
            performance curve.
        figsize: size of figure.
    
    Returns:
        fig: The figure, to be modified further if necessary.
    '''
    dat = subject_session_data

    if "correct" not in dat or "answer" not in dat:
        raise Exception("Please define a `correct` {0,1} and an `answer` {1,2} "
                        "field in `dat`.")
  
    N = len(dat['y'])
    if 2 in np.unique(dat['answer']):
        answerR = (dat['answer'] == 2).astype(float)
    else:
        answerR = (dat['answer'] == 1).astype(float)

    ### Plotting
    # fig = ax.figure(figsize=figsize)        
    
    # Smoothing vector for errorbars
    QQQ = np.zeros(10001)
    QQQ[5000] = 1
    QQQ = gaussian_filter(QQQ, sigma)

    # Calculate smooth representation of binary accuracy
    raw_correct = dat['correct'].astype(float)
    smooth_correct = gaussian_filter(raw_correct, sigma)
    ax.plot(smooth_correct, c=COLORS['emp_perf'], lw=3, zorder=4)

    # Calculate errorbars on empirical performance
    perf_errorbars = np.sqrt(
        np.sum(QQQ**2) * gaussian_filter(
            (raw_correct - smooth_correct)**2, sigma))
    ax.fill_between(range(N),
                     smooth_correct - 1.96 * perf_errorbars,
                     smooth_correct + 1.96 * perf_errorbars,
                     facecolor=COLORS['emp_perf'], alpha=0.3, zorder=3)

    # Calculate the predicted accuracy
    if xval_pL is not None:
        pred_correct = np.abs(answerR - xval_pL)
        smooth_pred_correct = gaussian_filter(pred_correct, sigma)
        ax.plot(smooth_pred_correct, c='k', alpha=0.75, lw=2, zorder=6)

    # Plot vertical session lines
    if 'dayLength' in dat and dat['dayLength'] is not None:
        days = np.cumsum(dat['dayLength'])
        i = 0
        for d in days:
            # ax.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
            if i < len(dat['dates'])-1:
                if dat['dates'][i+1] == dat['move_correct_spout']:
                    ax.axvline(d, color='violet', lw=3, zorder=0)
                else:
                    ax.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
            else:                    
                ax.axvline(d, c='k', lw=0.5, alpha=0.5, zorder=0)
            i += 1
    
    # Add plotting details
    ax.axhline(0.5, c='k', ls='--', lw=1, alpha=0.5, zorder=1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(np.concatenate((np.array([0]), days[0:-1])), dat['dates'], rotation=45)
    # ax.xlim(0, N); ax.ylim(0.3, 1.0)
    # ax.set_xlim(0, N); ax.set_ylim(0.3, 1.0)
    ax.set_xlim(0, N); ax.set_ylim(0.0, 1.0)
    # ax.xlabel('Trial #'); ax.ylabel('Performance')
    ax.set_xlabel('trials (concatenated sessions)'); ax.set_ylabel('Performance')
