#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parent
MOUSE_ID = ROOT.name.replace("_Summary", "")
ASSET_DIR = ROOT / "modular_report_assets"
REPORT_HTML = ROOT / f"{MOUSE_ID}_modular_report.html"
DATE_RANGE_RE = re.compile(r"_(\d{8})_to_(\d{8})\.pdf$")


CORE_SCRIPTS = [
    "001_BarPlotCR_Classification.py",
    "002_AvgFEC_AllSessions_bars.py",
    "003_AvgAllSessions_CRClassificationErrorBars.py",
    "004_AvgAllFEC.py",
    "005_FEC_AvgClassified_CR_V_2.py",
    "006_GranAvgAll_FEC.py",
    "007_GrandAvgClassified_BySession.py",
    "008_Pooled_Line_CRamplitude_Dist_V_3.py",
    "009_trans6_8RowSummary_V_3.py",
    "010_CR_onsetTiming_Distribution_SanityCheck.py",
    "011_CR_onset_trial_by_trial_QC.py",
    "012_ProbeTrials_Analysis.py",
    "013_UR_onset_detection.py",
]

CHEMO_SCRIPTS = [
    "010_CR_onsetTiming_ChemoControl.py",
    "011_CR_onset_ChemoControl_trial_by_trial_QC.py",
]

OPTO_SCRIPTS = [
    "013_OptoTrialByTrial_FEC_Summary.py",
    "016_PooledOptoVsNonOpto_AvgFEC.py",
    "017_Make_PI_Summary_PDF.py",
    "018_PooledAvgFEC_OptoReportFigures.py",
    "019_GrandAvgFEC_OptoReportFigures.py",
    "020_FECMagnitudeDistribution_OptoReportFigures.py",
    "021_OptoFraction_PerSession.py",
    "022_EpochAnalysis_ControlOnly.py",
]


@dataclass(frozen=True)
class FigureSpec:
    title: str
    patterns: tuple[str, ...]
    classes: tuple[str, ...] = ()
    optional: bool = True


@dataclass(frozen=True)
class SectionSpec:
    title: str
    description: str
    figures: tuple[FigureSpec, ...]
    requires: str | None = None


def log(message: str) -> None:
    print(f"[modular-report] {message}", flush=True)


def get_field(obj, name, default=None):
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def safe_array(x):
    if x is None:
        return np.array([], dtype=float)
    arr = np.asarray(x).squeeze()
    try:
        return arr.astype(float)
    except Exception:
        return arr


def scalar_bool(x) -> bool:
    arr = safe_array(x)
    if arr.size == 0:
        return False
    return bool(np.any(arr.astype(float) != 0))


def ensure_trial_list(raw_trials):
    if raw_trials is None:
        return []
    if isinstance(raw_trials, np.ndarray):
        return list(raw_trials.flat)
    if isinstance(raw_trials, (list, tuple)):
        return list(raw_trials)
    return [raw_trials]


def parse_date_token(path: Path) -> str | None:
    m = re.search(r"20\d{6}", path.name)
    return m.group(0) if m else None


def is_chemo_session(session_data) -> bool:
    if scalar_bool(get_field(session_data, "Chemogenetics", 0)):
        return True
    trial_settings = get_field(session_data, "TrialSettings", None)
    settings = ensure_trial_list(trial_settings)
    for setting in settings:
        gui = get_field(setting, "GUI", None)
        if scalar_bool(get_field(gui, "ChemogeneticsEnabled", 0)):
            return True
    return False


def session_has_opto(session_data) -> bool:
    raw_events = get_field(session_data, "RawEvents", None)
    trials = ensure_trial_list(get_field(raw_events, "Trial", None))
    for trial in trials:
        data = get_field(trial, "Data", None)
        if scalar_bool(get_field(data, "IsOptoTrial", 0)):
            return True
    trial_settings = get_field(session_data, "TrialSettings", None)
    settings = ensure_trial_list(trial_settings)
    for setting in settings:
        gui = get_field(setting, "GUI", None)
        if scalar_bool(get_field(gui, "OptoEnabled", 0)):
            return True
    return False


def summarize_sessions() -> dict[str, object]:
    mat_files = sorted(ROOT.glob("*_EBC_*.mat"))
    dates = [parse_date_token(path) for path in mat_files]
    dates = [d for d in dates if d is not None]
    chemo_sessions = 0
    opto_sessions = 0
    readable_sessions = 0

    for path in mat_files:
        try:
            data = loadmat(path, squeeze_me=True, struct_as_record=False)
            session_data = data.get("SessionData")
            if session_data is None:
                continue
            readable_sessions += 1
            chemo_sessions += int(is_chemo_session(session_data))
            opto_sessions += int(session_has_opto(session_data))
        except Exception as exc:
            log(f"metadata skipped for {path.name}: {exc}")

    first = min(dates) if dates else None
    last = max(dates) if dates else None
    return {
        "mat_count": len(mat_files),
        "readable_sessions": readable_sessions,
        "control_sessions": max(0, readable_sessions - chemo_sessions),
        "chemo_sessions": chemo_sessions,
        "opto_sessions": opto_sessions,
        "first": first,
        "last": last,
        "has_chemo": chemo_sessions > 0,
        "has_opto": opto_sessions > 0 or any(ROOT.glob("*Opto*.pdf")),
    }


def date_range_key(path: Path) -> tuple[int, int, float]:
    m = DATE_RANGE_RE.search(path.name)
    if m:
        return int(m.group(2)), int(m.group(1)), path.stat().st_mtime
    token = parse_date_token(path)
    return int(token or 0), 0, path.stat().st_mtime


def find_latest_pdf(patterns: Iterable[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(ROOT / pattern)))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=date_range_key)


def safe_asset_name(pdf: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", pdf.stem)
    return f"{stem}.png"


def render_preview(pdf: Path) -> Path:
    ASSET_DIR.mkdir(exist_ok=True)
    png = ASSET_DIR / safe_asset_name(pdf)
    if png.exists() and png.stat().st_mtime >= pdf.stat().st_mtime:
        return png
    cmd = [
        "sips",
        "-s",
        "format",
        "png",
        "-Z",
        "2600",
        str(pdf),
        "--out",
        str(png),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return png


def run_scripts(summary: dict[str, object]) -> None:
    scripts = list(CORE_SCRIPTS)
    if summary["has_chemo"]:
        scripts.extend(CHEMO_SCRIPTS)
    if summary["has_opto"]:
        scripts.extend(OPTO_SCRIPTS)

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

    seen: set[str] = set()
    for script in scripts:
        if script in seen:
            continue
        seen.add(script)
        if not (ROOT / script).exists():
            continue
        log(f"running {script}")
        subprocess.run([sys.executable, script], cwd=ROOT, check=True, env=env)


def section_specs() -> tuple[SectionSpec, ...]:
    return (
        SectionSpec(
            "Core Session Summaries",
            "Control sessions are always counted. Chemo/opto variants are included only when detected.",
            (
                FigureSpec("Per-session CR fractions", ("BarPlt_FECFraction_ClassifiedCRs_AllSessions.pdf", "BarPlt_FECFraction_ClassifiedCRs_OptoVsControl_AllSessions.pdf"), ("wide",)),
                FigureSpec("Average classified CR fractions", ("AvgBarPlot_ClassifiedCRFractions_*.pdf",)),
                FigureSpec("Classified CR fractions with error bars", ("AvgCircles_ClassifiedCRFractions_*.pdf",)),
                FigureSpec("Pooled FEC, raw", ("PooledAvgFEC_Raw_AllAndCRclassified_*ShortLong_*.pdf", "PooledAvgFEC_Raw_ShortLong_*.pdf",), ("tall",)),
                FigureSpec("Pooled FEC, baseline-matched", ("PooledAvgFEC_BaselineMatched_AllAndCRclassified_*ShortLong_*.pdf",), ("tall",)),
                FigureSpec("Grand average FEC, raw", ("GrandAvgFEC_BySession_Raw_AllAndCRclassified_*ShortLong_*.pdf", "GrandAvgFEC_BySession_ShortLong_*.pdf")),
                FigureSpec("Grand average FEC, baseline-matched", ("GrandAvgFEC_BySession_BaselineMatched_AllAndCRclassified_*ShortLong_*.pdf",)),
            ),
        ),
        SectionSpec(
            "CR And FEC Distributions",
            "CR onset and FEC magnitude distribution summaries.",
            (
                FigureSpec("FEC magnitude distribution, raw", ("Baseline_CRonset_Distributions_Raw_*.pdf", "FECMagnitudeDistribution_Raw_OptoShortLong_*.pdf"), ("tall",)),
                FigureSpec("FEC magnitude distribution, baseline-matched", ("Baseline_CRonset_Distributions_BaselineMatched_*.pdf", "FECMagnitudeDistribution_BaselineMatched_OptoShortLong_*.pdf"), ("tall",)),
                FigureSpec("CR onset sanity distribution", ("CR_onset_distribution_sanity_*.pdf",)),
                FigureSpec("CR onset superimposed short/long", ("CR_onset_superimposed_short_long_*.pdf",)),
                FigureSpec("CR onset latency table", ("CR_onset_latency_summary_table_*.pdf",)),
                FigureSpec("Pooled CR amplitude lines", ("Pooled_Line_CR_Distributions_ModifiedWindow_*.pdf",)),
                FigureSpec("Pooled CR amplitude cumulative", ("Pooled_Cumulative_CR_Distributions_ModifiedWindow_*.pdf",)),
            ),
        ),
        SectionSpec(
            "Chemo-Control Analysis",
            "Included only when chemogenetics sessions or chemo-control outputs exist.",
            (
                FigureSpec("Control vs chemo CR amplitude lines", ("Pooled_ControlChemo_CRAmplitude_Line_*.pdf", "Pooled_ControlChemo_CRAmplitude_Distributions_*.pdf")),
                FigureSpec("Control vs chemo CR amplitude cumulative", ("Pooled_ControlChemo_CRAmplitude_Cumulative*.pdf",)),
                FigureSpec("Good CR baseline-subtracted traces", ("CR_onset_goodCR_average_traces_baselined_*.pdf",)),
                FigureSpec("Poor CR baseline-subtracted traces", ("CR_onset_poorCR_average_traces_baselined_*.pdf",)),
            ),
            requires="chemo",
        ),
        SectionSpec(
            "Opto Analysis",
            "Included only when opto trials/sessions or opto outputs exist.",
            (
                FigureSpec("Opto trial fraction per session", ("OptoFraction_PerSession_*.pdf",)),
                FigureSpec("Opto trial-by-trial FEC", ("OptoTrialByTrial_FEC_*.pdf",), ("wide", "tall")),
                FigureSpec("Pooled opto vs non-opto FEC", ("Pooled_OptoVsNonOpto_AvgFEC.pdf",)),
                FigureSpec("Opto PI summary", ("LS02_PI_Summary_Opto_CR_FEC.pdf",)),
                FigureSpec("Opto epoch raw", ("EpochAnalysis_ControlOnly_Raw.pdf",)),
                FigureSpec("Opto epoch baseline-matched", ("EpochAnalysis_ControlOnly_BaselineMatched.pdf",)),
            ),
            requires="opto",
        ),
        SectionSpec(
            "Probe And Transition Analysis",
            "Probe, transition, and within-block adaptation summaries.",
            (
                FigureSpec("Probe fraction per session", ("Probe_01_FractionPerSession.pdf",)),
                FigureSpec("Probe fraction at transitions", ("Probe_02_FractionAtTransitions.pdf",)),
                FigureSpec("Probe FEC triplets", ("Probe_03_FEC_Triplets.pdf",)),
                FigureSpec("Probe FEC triplets, baseline-matched", ("Probe_03_FEC_Triplets_BaselineMatched.pdf",)),
                FigureSpec("Transition summary", ("Professor_Report/within_block_adaptation/POOLED_Transitions_Summary_FIXED.pdf", "*Transition*/POOLED_Transitions_Summary_FIXED.pdf", "*Transition*/POOLED_Transition_Summary_NoDuplicateRows.pdf"), ("wide", "tall")),
                FigureSpec("Transition block-vs-trial mean", ("Professor_Report/within_block_adaptation/POOLED_BlockVsTrial_MEAN.pdf", "*Transition*/POOLED_BlockVsTrial_MEAN.pdf")),
            ),
        ),
        SectionSpec(
            "UR And QC",
            "UR onset summaries and trial-level QC figures.",
            (
                FigureSpec("UR onset distribution", ("UR_onset_distribution_*.pdf",)),
                FigureSpec("UR onset average traces", ("UR_onset_average_traces_*.pdf",)),
                FigureSpec("UR onset summary table", ("UR_onset_summary_table_*.pdf",)),
                FigureSpec("Latest CR trial-by-trial QC", ("CR_onset_trial_by_trial_QC_*.pdf",)),
                FigureSpec("Latest UR trial-by-trial QC", ("UR_onset_QC_*.pdf",)),
            ),
        ),
    )


def collect_report_items(summary: dict[str, object]) -> list[tuple[SectionSpec, list[tuple[FigureSpec, Path, Path]]]]:
    sections = []
    for section in section_specs():
        if section.requires == "chemo" and not summary["has_chemo"] and not any(ROOT.glob("*Chemo*.pdf")):
            continue
        if section.requires == "opto" and not summary["has_opto"]:
            continue
        items = []
        for fig in section.figures:
            pdf = find_latest_pdf(fig.patterns)
            if pdf is None:
                if not fig.optional:
                    raise FileNotFoundError(f"No PDF for {fig.title}: {fig.patterns}")
                continue
            png = render_preview(pdf)
            items.append((fig, pdf, png))
        if items:
            sections.append((section, items))
    return sections


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def figure_html(idx: int, fig: FigureSpec, pdf: Path, png: Path) -> str:
    classes = " ".join(("figure-card",) + fig.classes)
    return f"""          <article class="{html.escape(classes)}" id="fig{idx}">
            <div class="figure-title"><h3>{html.escape(fig.title)}</h3><a href="{html.escape(rel(pdf))}">Open PDF</a></div>
            <a class="preview" href="{html.escape(rel(pdf))}"><img src="{html.escape(rel(png))}" alt="{html.escape(fig.title)} preview"></a>
          </article>"""


def write_html(summary: dict[str, object], sections: list[tuple[SectionSpec, list[tuple[FigureSpec, Path, Path]]]]) -> None:
    today = dt.date.today().strftime("%B %-d, %Y") if os.name != "nt" else dt.date.today().strftime("%B %#d, %Y")
    first = summary["first"] or "unknown"
    last = summary["last"] or "unknown"
    phenotype_bits = [f"control={summary['control_sessions']}"]
    if summary["chemo_sessions"]:
        phenotype_bits.append(f"chemo={summary['chemo_sessions']}")
    if summary["opto_sessions"]:
        phenotype_bits.append(f"opto={summary['opto_sessions']}")
    phenotype = ", ".join(phenotype_bits)

    nav_parts = []
    body_parts = []
    fig_idx = 1
    for section, items in sections:
        section_id = re.sub(r"[^a-z0-9]+", "-", section.title.lower()).strip("-")
        nav_parts.append(f'      <a href="#{html.escape(section_id)}">{html.escape(section.title)}</a>')
        cards = []
        for fig, pdf, png in items:
            cards.append(figure_html(fig_idx, fig, pdf, png))
            fig_idx += 1
        body_parts.append(
            f"""      <section class="section" id="{html.escape(section_id)}">
        <div class="section-head">
          <h2>{html.escape(section.title)}</h2>
          <p>{html.escape(section.description)}</p>
        </div>
        <div class="figure-grid">
{os.linesep.join(cards)}
        </div>
      </section>"""
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(MOUSE_ID)} Modular Report</title>
  <style>
    :root {{ --ink:#1f2933; --muted:#65717f; --line:#d7dde3; --paper:#fff; --soft:#f5f7fa; --accent:#0057b8; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; color:var(--ink); background:#eef2f6; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; line-height:1.45; }}
    header {{ background:var(--paper); border-bottom:1px solid var(--line); padding:28px 36px 22px; position:sticky; top:0; z-index:5; }}
    h1 {{ margin:0; font-size:clamp(26px,4vw,42px); line-height:1.08; letter-spacing:0; }}
    .subtitle {{ margin:10px 0 0; color:var(--muted); max-width:980px; font-size:15px; }}
    .layout {{ display:grid; grid-template-columns:210px minmax(0,1fr); gap:22px; padding:22px; max-width:1680px; margin:0 auto; }}
    nav {{ position:sticky; top:136px; align-self:start; background:var(--paper); border:1px solid var(--line); border-radius:8px; padding:12px; max-height:calc(100vh - 160px); overflow:auto; }}
    nav a {{ display:block; color:var(--muted); text-decoration:none; padding:7px 8px; border-radius:6px; font-size:13px; }}
    nav a:hover {{ color:var(--accent); background:#e7f0ff; }}
    main {{ display:grid; gap:18px; }}
    .section {{ background:var(--paper); border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .section-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:18px; padding:18px 20px; border-bottom:1px solid var(--line); background:#fbfcfe; }}
    .section-head h2 {{ margin:0; font-size:18px; letter-spacing:0; }}
    .section-head p {{ margin:0; color:var(--muted); font-size:13px; }}
    .figure-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:18px; padding:18px; }}
    .figure-card {{ border:1px solid var(--line); border-radius:8px; background:var(--paper); overflow:hidden; }}
    .figure-card.wide {{ grid-column:1 / -1; }}
    .figure-title {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 14px; border-bottom:1px solid var(--line); background:var(--soft); }}
    .figure-title h3 {{ margin:0; font-size:15px; font-weight:650; }}
    .figure-title a {{ color:var(--accent); text-decoration:none; font-size:13px; white-space:nowrap; }}
    .preview {{ display:block; padding:12px; background:#fff; }}
    .preview img {{ display:block; width:100%; max-height:760px; object-fit:contain; border:1px solid #edf0f3; background:#fff; }}
    .figure-card.tall .preview img {{ max-height:1080px; }}
    footer {{ color:var(--muted); font-size:12px; padding:10px 4px 24px; }}
    @media (max-width:980px) {{ header {{ position:static; padding:22px 18px; }} .layout {{ display:block; padding:14px; }} nav {{ position:static; margin-bottom:14px; max-height:none; }} .figure-grid {{ grid-template-columns:1fr; padding:12px; }} .section-head {{ display:block; }} .section-head p {{ margin-top:6px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(MOUSE_ID)} Modular EBC Report</h1>
    <p class="subtitle">Detected {summary['mat_count']} session files ({phenotype}); date range {first} to {last}. Sections are included only when matching data or outputs are present.</p>
  </header>
  <div class="layout">
    <nav aria-label="Report navigation">
{os.linesep.join(nav_parts)}
    </nav>
    <main>
{os.linesep.join(body_parts)}
      <footer>Generated locally on {today}.</footer>
    </main>
  </div>
</body>
</html>
"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")
    log(f"wrote {REPORT_HTML.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a modular EBC report for this mouse folder.")
    parser.add_argument("--run-analysis", action="store_true", help="Run detected analysis scripts before building the report.")
    args = parser.parse_args()

    summary = summarize_sessions()
    log(
        f"detected sessions={summary['mat_count']} control={summary['control_sessions']} "
        f"chemo={summary['chemo_sessions']} opto={summary['opto_sessions']}"
    )
    if args.run_analysis:
        run_scripts(summary)
        summary = summarize_sessions()
    sections = collect_report_items(summary)
    if not sections:
        raise FileNotFoundError("No reportable PDFs found. Run with --run-analysis or generate figures first.")
    write_html(summary, sections)


if __name__ == "__main__":
    main()
