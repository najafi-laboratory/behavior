# ZS02 EBC Analysis Summary

Analysis scripts and final report outputs for the ZS02 EBC control/chemo summary.

## Main Report

Open `ZS02_final_report.html`.

Use `generate_zs02_final_report.py` to rebuild the analysis outputs, figure previews, composite PDFs, and the single final HTML report.

Fast rebuild from existing PDFs:

```bash
python3 generate_zs02_final_report.py
```

Full rebuild after adding new `.mat` files:

```bash
python3 generate_zs02_final_report.py --run-analysis
```

The HTML report links to the final figure PDFs and uses selected PNG previews for the report gallery.

## Install Dependencies

```bash
python3 -m pip install -r requirements.txt
```

## Main Scripts

- `001_BarPlotCR_Classification.py`
- `002_AvgFEC_AllSessions_bars.py`
- `003_AvgAllSessions_CRClassificationErrorBars.py`
- `004_AvgAllFEC.py`
- `005_FEC_AvgClassified_CR_V_2.py`
- `006_GranAvgAll_FEC.py`
- `007_GrandAvgClassified_BySession.py`
- `008_Pooled_Line_CRamplitude_Dist_V_3.py`
- `009_trans6_8RowSummary_V_3.py`
- `010_CR_onsetTiming_ChemoControl.py`
- `011_CR_onset_trial_by_trial_QC.py`
- `011_CR_onset_ChemoControl_trial_by_trial_QC.py`
- `012_ProbeTrials_Analysis.py`
- `013_UR_onset_detection.py`
- `generate_zs02_final_report.py`

Raw `.mat` files are intentionally not included in the recommended GitHub upload. Keep raw data on the lab drive or another approved data archive.

See `GITHUB_UPLOAD_GUIDE.md` for the exact recommended upload list.
