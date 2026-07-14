#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import glob
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "fig_report_assets"
TRANSITION_DIR = ROOT / "ZS02_Transition_Trial_By_Trial_Adaptation_FIXED"
REPORT_HTML = ROOT / "ZS02_final_report.html"


ANALYSIS_SCRIPTS = [
    "001_BarPlotCR_Classification.py",
    "002_AvgFEC_AllSessions_bars.py",
    "003_AvgAllSessions_CRClassificationErrorBars.py",
    "004_AvgAllFEC.py",
    "005_FEC_AvgClassified_CR_V_2.py",
    "006_GranAvgAll_FEC.py",
    "007_GrandAvgClassified_BySession.py",
    "008_Pooled_Line_CRamplitude_Dist_V_3.py",
    "010_CR_onsetTiming_ChemoControl.py",
    "012_ProbeTrials_Analysis.py",
    # V3 currently contains the complete FEC epoch + transition metric outputs
    # that Figures 16/17 need. Keep this explicit until V4 gains parity.
    "009_trans6_8RowSummary_V_3.py",
]


DATE_RANGE_RE = re.compile(r"_(\d{8})_to_(\d{8})\.pdf$")


@dataclass(frozen=True)
class PdfRule:
    target: str
    patterns: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class FigureCard:
    id: str
    title: str
    pdf: str
    png: str
    alt: str
    classes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Subhead:
    text: str


@dataclass(frozen=True)
class Section:
    title: str
    description: str
    items: tuple[FigureCard | Subhead, ...]


PDF_RULES = [
    PdfRule("fig1.pdf", ("BarPlt_FECFraction_ClassifiedCRs_AllSessions.pdf",), required=False),
    PdfRule("fig2.pdf", ("AvgBarPlot_ClassifiedCRFractions_*.pdf",)),
    PdfRule("fig3.pdf", ("AvgCircles_ClassifiedCRFractions_*.pdf",)),
    PdfRule("fig4.pdf", ("PooledAvgFEC_Raw_AllAndCRclassified_ShortLong_*.pdf",)),
    PdfRule("fig5.pdf", ("PooledAvgFEC_BaselineMatched_AllAndCRclassified_ShortLong_*.pdf",)),
    PdfRule("fig6.pdf", ("GrandAvgFEC_BySession_Raw_AllAndCRclassified_ShortLong_*.pdf",)),
    PdfRule("fig7.pdf", ("GrandAvgFEC_BySession_BaselineMatched_AllAndCRclassified_ShortLong_*.pdf",)),
    PdfRule("fig8.pdf", ("Baseline_CRonset_Distributions_Raw_*.pdf",)),
    PdfRule("fig9.pdf", ("Baseline_CRonset_Distributions_BaselineMatched_*.pdf",)),
    PdfRule("fig10.pdf", ("Baseline_CRonset_CumulativeDistributions_Raw_*.pdf",)),
    PdfRule("fig11.pdf", ("Baseline_CRonset_CumulativeDistributions_BaselineMatched_*.pdf",)),
    PdfRule("fig14.pdf", ("Probe_01_FractionPerSession.pdf",)),
]


REPORT_FIGURES = [
    "fig1.pdf",
    "fig2.pdf",
    "fig3.pdf",
    "fig4.pdf",
    "fig5.pdf",
    "fig6.pdf",
    "fig7.pdf",
    "fig8.pdf",
    "fig9.pdf",
    "fig10.pdf",
    "fig11.pdf",
    "fig14.pdf",
    "Probe_03_FEC_Triplets_Raw.pdf",
    "Probe_03_FEC_Triplets_BaselineMatched.pdf",
    "fig16_trial_adaptation_raw_3row.pdf",
    "fig17_trial_adaptation_baseline_matched_3row.pdf",
    "fig17_transition_clean_summary_with_probe.pdf",
]


def log(message: str) -> None:
    print(f"[report] {message}", flush=True)


def run_command(args: list[str]) -> None:
    log("running " + " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def run_analysis_scripts() -> None:
    for script in ANALYSIS_SCRIPTS:
        path = ROOT / script
        if not path.exists():
            raise FileNotFoundError(f"Missing analysis script: {script}")
        run_command([sys.executable, script])


def date_range_key(path: Path) -> tuple[int, int, float]:
    m = DATE_RANGE_RE.search(path.name)
    if m:
        return int(m.group(2)), int(m.group(1)), path.stat().st_mtime
    return 0, 0, path.stat().st_mtime


def find_latest_pdf(patterns: Iterable[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path(p) for p in glob.glob(str(ROOT / pattern)))
    matches = [p for p in matches if p.is_file()]
    if not matches:
        return None
    return max(matches, key=date_range_key)


def copy_pdf_rules() -> None:
    for rule in PDF_RULES:
        source = find_latest_pdf(rule.patterns)
        target = ROOT / rule.target
        if source is None:
            if rule.required or not target.exists():
                raise FileNotFoundError(
                    f"No PDF found for {rule.target}; searched {', '.join(rule.patterns)}"
                )
            log(f"kept existing {rule.target}; no source matched {', '.join(rule.patterns)}")
            continue
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        log(f"{rule.target} <- {source.name}")


def sips_render(pdf_path: Path, png_path: Path, max_px: int = 2600) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sips",
        "-s",
        "format",
        "png",
        "-Z",
        str(max_px),
        str(pdf_path),
        "--out",
        str(png_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def render_pdf_temp(pdf_path: Path, max_px: int = 2600) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        png_path = Path(tmp) / f"{pdf_path.stem}.png"
        sips_render(pdf_path, png_path, max_px=max_px)
        with Image.open(png_path) as img:
            return img.convert("RGB")


def crop_white_border(image: Image.Image, padding: int = 18) -> Image.Image:
    gray = ImageOps.grayscale(image)
    mask = gray.point(lambda p: 0 if p > 248 else 255)
    bbox = mask.getbbox()
    if bbox is None:
        return image
    left = max(0, bbox[0] - padding)
    top = max(0, bbox[1] - padding)
    right = min(image.width, bbox[2] + padding)
    bottom = min(image.height, bbox[3] + padding)
    return image.crop((left, top, right, bottom))


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def stack_images_vertically(
    images: list[Image.Image],
    out_pdf: Path,
    out_png: Path,
    gap: int = 26,
    margin: int = 28,
) -> None:
    width = max(img.width for img in images)
    normalized = [resize_to_width(img, width) for img in images]
    total_height = margin * 2 + gap * (len(normalized) - 1) + sum(img.height for img in normalized)
    canvas = Image.new("RGB", (width + margin * 2, total_height), "white")
    y = margin
    for img in normalized:
        canvas.paste(img, (margin, y))
        y += img.height + gap
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, "PNG", optimize=True)
    canvas.save(out_pdf, "PDF", resolution=220.0)
    log(f"built {out_pdf.name}")


def build_composite_pdf(target_pdf: str, target_png: str, sources: list[Path], crop: bool = True) -> None:
    target_pdf_path = ROOT / target_pdf
    target_png_path = ASSET_DIR / target_png
    missing = [src for src in sources if not src.exists()]
    if missing and target_pdf_path.exists():
        log(
            f"kept existing {target_pdf}; missing source(s): "
            + ", ".join(str(src.relative_to(ROOT)) for src in missing)
        )
        if not target_png_path.exists():
            sips_render(target_pdf_path, target_png_path, max_px=2600)
        return
    if missing:
        raise FileNotFoundError(
            "Missing composite source(s):\n  "
            + "\n  ".join(str(src) for src in missing)
        )

    images = []
    for src in sources:
        img = render_pdf_temp(src, max_px=3000)
        images.append(crop_white_border(img) if crop else img)
    stack_images_vertically(images, target_pdf_path, target_png_path)


def build_composites() -> None:
    transition_raw = TRANSITION_DIR / "POOLED_Transition_FEC_EpochAverages_Raw.pdf"
    transition_bl = TRANSITION_DIR / "POOLED_Transition_FEC_EpochAverages_BaselineMatched.pdf"
    raw_metrics = TRANSITION_DIR / "POOLED_Transition_Raw_BLSub_Metrics.pdf"
    bl_metrics = TRANSITION_DIR / "POOLED_Transition_BLSub_BaselineMatched_Metrics.pdf"
    clean_summary = TRANSITION_DIR / "POOLED_Transition_Summary_NoDuplicateRows.pdf"

    build_composite_pdf(
        "fig16_trial_adaptation_raw_3row.pdf",
        "fig16_trial_adaptation_raw_3row.png",
        [transition_raw, raw_metrics],
    )
    build_composite_pdf(
        "fig17_trial_adaptation_baseline_matched_3row.pdf",
        "fig17_trial_adaptation_baseline_matched_3row.png",
        [transition_bl, bl_metrics],
    )
    build_composite_pdf(
        "fig17_transition_clean_summary_with_probe.pdf",
        "fig17_transition_clean_summary_with_probe.png",
        [clean_summary, ROOT / "Probe_02_FractionAtTransitions.pdf"],
    )


def render_report_previews() -> None:
    ASSET_DIR.mkdir(exist_ok=True)
    for pdf_name in REPORT_FIGURES:
        pdf = ROOT / pdf_name
        if not pdf.exists():
            raise FileNotFoundError(f"Missing report PDF: {pdf_name}")
        png = ASSET_DIR / f"{pdf.stem}.png"
        if png.exists() and png.stat().st_mtime >= pdf.stat().st_mtime:
            continue
        sips_render(pdf, png, max_px=2600)
        log(f"preview {png.relative_to(ROOT)}")


def nav_links(sections: list[Section]) -> str:
    links: list[str] = []
    for section in sections:
        for item in section.items:
            if isinstance(item, FigureCard):
                m = re.match(r"fig(\d+)$", item.id)
                label = f"Fig {m.group(1)}" if m else item.title
                if item.id == "transition-summary":
                    label = "Transition summary"
                links.append(f'      <a href="#{html.escape(item.id)}">{html.escape(label)}</a>')
    return "\n".join(links)


def figure_card(card: FigureCard) -> str:
    classes = " ".join(("figure-card",) + card.classes)
    return f"""          <article class="{html.escape(classes)}" id="{html.escape(card.id)}">
            <div class="figure-title"><h3>{html.escape(card.title)}</h3><a href="{html.escape(card.pdf)}">Open PDF</a></div>
            <a class="preview" href="{html.escape(card.pdf)}"><img src="{html.escape(card.png)}" alt="{html.escape(card.alt)}"></a>
          </article>"""


def render_sections(sections: list[Section]) -> str:
    rendered: list[str] = []
    for section in sections:
        desc = f"\n          <p>{html.escape(section.description)}</p>" if section.description else ""
        parts = [
            '      <section class="section">',
            '        <div class="section-head">',
            f"          <h2>{html.escape(section.title)}</h2>{desc}",
            '        </div>',
            '        <div class="figure-grid">',
        ]
        for item in section.items:
            if isinstance(item, Subhead):
                parts.append(f'          <div class="grid-subhead">{html.escape(item.text)}</div>')
            else:
                parts.append(figure_card(item))
        parts.extend(["        </div>", "      </section>"])
        rendered.append("\n".join(parts))
    return "\n\n".join(rendered)


def report_sections() -> list[Section]:
    return [
        Section(
            "Core Session Summaries",
            "Classification, pooled FEC, and session-level overview figures.",
            (
                FigureCard("fig1", "Figure 1", "fig1.pdf", "fig_report_assets/fig1.png", "Figure 1 preview", ("wide",)),
                FigureCard("fig2", "Figure 2", "fig2.pdf", "fig_report_assets/fig2.png", "Figure 2 preview"),
                FigureCard("fig3", "Figure 3", "fig3.pdf", "fig_report_assets/fig3.png", "Figure 3 preview"),
                Subhead("FEC Traces"),
                Subhead("Pooled Avg"),
                FigureCard("fig4", "Figure 4 - Pooled Avg raw", "fig4.pdf", "fig_report_assets/fig4.png", "Figure 4 pooled average raw preview", ("tall",)),
                FigureCard("fig5", "Figure 5 - Pooled Avg baseline-matched", "fig5.pdf", "fig_report_assets/fig5.png", "Figure 5 pooled average baseline-matched preview", ("tall",)),
            ),
        ),
        Section(
            "Grand Avg",
            "",
            (
                FigureCard("fig6", "Figure 6 - Grand Avg", "fig6.pdf", "fig_report_assets/fig6.png", "Figure 6 preview", ("tall",)),
                FigureCard("fig7", "Figure 7 - Grand Avg", "fig7.pdf", "fig_report_assets/fig7.png", "Figure 7 preview", ("tall",)),
                Subhead("Distribution of FEC magnitude"),
                FigureCard("fig8", "Figure 8", "fig8.pdf", "fig_report_assets/fig8.png", "Figure 8 preview", ("tall",)),
                FigureCard("fig9", "Figure 9", "fig9.pdf", "fig_report_assets/fig9.png", "Figure 9 preview", ("tall",)),
                Subhead("Cumulative distribution of FEC magnitude"),
                FigureCard("fig10", "Raw cumulative distribution", "fig10.pdf", "fig_report_assets/fig10.png", "Raw cumulative FEC magnitude distribution preview", ("tall",)),
                FigureCard("fig11", "Baseline-matched cumulative distribution", "fig11.pdf", "fig_report_assets/fig11.png", "Baseline-matched cumulative FEC magnitude distribution preview", ("tall",)),
            ),
        ),
        Section(
            "Probe Analysis",
            "Probe behavior and transition adaptation summaries.",
            (
                FigureCard("fig14", "Probe Fraction", "fig14.pdf", "fig_report_assets/fig14.png", "Probe fraction preview", ("compact-centered",)),
                Subhead("Probe average FEC"),
                FigureCard("probe-avg-raw", "Probe average FEC - Raw", "Probe_03_FEC_Triplets_Raw.pdf", "fig_report_assets/Probe_03_FEC_Triplets_Raw.png", "Raw probe average FEC preview"),
                FigureCard("probe-avg-baseline-matched", "Probe average FEC - Baseline-matched", "Probe_03_FEC_Triplets_BaselineMatched.pdf", "fig_report_assets/Probe_03_FEC_Triplets_BaselineMatched.png", "Baseline-matched probe average FEC preview"),
                Subhead("FEC Epoch analysis"),
                FigureCard("fig16", "Figure 16 - Trial adaptation raw", "fig16_trial_adaptation_raw_3row.pdf", "fig_report_assets/fig16_trial_adaptation_raw_3row.png", "Raw trial adaptation 3-row preview"),
                FigureCard("fig17", "Figure 17 - Baseline-matched trial adaptation", "fig17_trial_adaptation_baseline_matched_3row.pdf", "fig_report_assets/fig17_trial_adaptation_baseline_matched_3row.png", "Baseline-matched trial adaptation 3-row preview"),
                FigureCard("transition-summary", "Trial-by-trial transition summary", "fig17_transition_clean_summary_with_probe.pdf", "fig_report_assets/fig17_transition_clean_summary_with_probe.png", "Trial-by-trial transition summary preview", ("wide", "tall")),
            ),
        ),
    ]


def write_html() -> None:
    sections = report_sections()
    today = dt.date.today().strftime("%B %-d, %Y") if os.name != "nt" else dt.date.today().strftime("%B %#d, %Y")
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ZS02 Final Report</title>
  <style>
    :root {{
      --ink: #1f2933;
      --muted: #65717f;
      --line: #d7dde3;
      --paper: #ffffff;
      --soft: #f5f7fa;
      --accent: #0057b8;
      --accent-soft: #e7f0ff;
      --control: #111111;
      --chemo: #B00000;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: #eef2f6;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: var(--paper);
      border-bottom: 1px solid var(--line);
      padding: 28px 36px 22px;
      position: sticky;
      top: 0;
      z-index: 5;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(26px, 4vw, 42px);
      letter-spacing: 0;
      line-height: 1.08;
    }}
    .subtitle {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 980px;
      font-size: 15px;
    }}
    .injection-info {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 22px;
      margin-top: 14px;
      padding: 10px 14px;
      max-width: 980px;
      background: var(--accent-soft);
      border-left: 4px solid var(--accent);
      border-radius: 6px;
      font-size: 14px;
    }}
    .injection-info strong {{ color: var(--ink); }}
    .injection-title {{
      flex-basis: 100%;
      font-weight: 700;
      margin-bottom: 2px;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 16px;
      font-size: 13px;
      color: var(--muted);
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 7px;
      white-space: nowrap;
    }}
    .swatch {{
      width: 22px;
      height: 3px;
      display: inline-block;
      border-radius: 2px;
      background: var(--ink);
    }}
    .swatch.control {{ background: var(--control); }}
    .swatch.chemo {{ background: var(--chemo); }}
    .layout {{
      display: grid;
      grid-template-columns: 190px minmax(0, 1fr);
      gap: 22px;
      padding: 22px;
      max-width: 1680px;
      margin: 0 auto;
    }}
    nav {{
      position: sticky;
      top: 136px;
      align-self: start;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: calc(100vh - 160px);
      overflow: auto;
    }}
    nav a {{
      display: block;
      color: var(--muted);
      text-decoration: none;
      padding: 7px 8px;
      border-radius: 6px;
      font-size: 13px;
    }}
    nav a:hover {{
      color: var(--accent);
      background: var(--accent-soft);
    }}
    main {{
      display: grid;
      gap: 18px;
    }}
    .section {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .section-head {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .section-head h2 {{
      margin: 0;
      font-size: 18px;
      letter-spacing: 0;
    }}
    .section-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      padding: 18px;
    }}
    .figure-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--paper);
      overflow: hidden;
    }}
    .figure-card.wide {{
      grid-column: 1 / -1;
    }}
    .figure-card.compact-centered {{
      grid-column: 1 / -1;
      justify-self: center;
      width: min(760px, 100%);
    }}
    .grid-subhead {{
      grid-column: 1 / -1;
      margin: 4px 2px -4px;
      color: var(--ink);
      font-size: 16px;
      font-weight: 700;
    }}
    .figure-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line);
      background: var(--soft);
    }}
    .figure-title h3 {{
      margin: 0;
      font-size: 15px;
      font-weight: 650;
    }}
    .figure-title a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
      white-space: nowrap;
    }}
    .figure-title a:hover {{ text-decoration: underline; }}
    .preview {{
      display: block;
      padding: 12px;
      background: #ffffff;
    }}
    .preview img {{
      display: block;
      width: 100%;
      max-height: 760px;
      object-fit: contain;
      border: 1px solid #edf0f3;
      background: white;
    }}
    .figure-card.tall .preview img {{
      max-height: 1080px;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      padding: 10px 4px 24px;
    }}
    @media (max-width: 980px) {{
      header {{
        position: static;
        padding: 22px 18px;
      }}
      .layout {{
        display: block;
        padding: 14px;
      }}
      nav {{
        position: static;
        margin-bottom: 14px;
        max-height: none;
      }}
      .figure-grid {{
        grid-template-columns: 1fr;
        padding: 12px;
      }}
      .section-head {{
        display: block;
      }}
      .section-head p {{
        margin-top: 6px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>ZS02_WT_ThalDCN_PFC:Di;<br>DCN: cre ; PFC: RetroFlexDi; (YH43_WT)</h1>
    <p class="subtitle">
      Final HTML gallery for selected report figures. Key comparison panels link to the full-resolution PDF for presentation or detailed review.
    </p>
    <div class="injection-info" aria-label="DCZ administration information">
      <div class="injection-title">Chemo injection info:</div>
      <span><strong>Dose:</strong> 0.8 mg/kg</span>
      <span><strong>Working concentration:</strong> 0.2 mg/mL</span>
      <span><strong>Injection volume:</strong> 80 &micro;L (8 units, U-100 insulin syringe)</span>
      <span><strong>DCZ delivered:</strong> 16 &micro;g (20 g mouse)</span>
      <span><strong>Behavior session start:</strong> ______ (15 min post-injection)</span>
    </div>
    <div class="legend">
      <span><i class="swatch control"></i>Control traces / summaries</span>
      <span><i class="swatch chemo"></i>Chemo traces / summaries</span>
      <span>Missing-video session note: 06/15/2026 is retained where applicable but does not contribute usable FEC data.</span>
    </div>
  </header>

  <div class="layout">
    <nav aria-label="Figure navigation">
{nav_links(sections)}
    </nav>

    <main>
{render_sections(sections)}

      <footer>
        Generated locally from selected report PDFs on {today}.
      </footer>
    </main>
  </div>
</body>
</html>
"""
    REPORT_HTML.write_text(html_text, encoding="utf-8")
    log(f"wrote {REPORT_HTML.name}")


def validate_report_assets() -> None:
    missing: list[str] = []
    for fig in REPORT_FIGURES:
        if not (ROOT / fig).exists():
            missing.append(fig)
        png = ASSET_DIR / f"{Path(fig).stem}.png"
        if not png.exists():
            missing.append(str(png.relative_to(ROOT)))
    if missing:
        raise FileNotFoundError("Missing report outputs:\n  " + "\n  ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the single ZS02 final HTML report.")
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run all analysis scripts before refreshing report PDFs and HTML.",
    )
    parser.add_argument(
        "--skip-composites",
        action="store_true",
        help="Skip rebuilding composite PDFs; only refresh mapped PDFs, previews, and HTML.",
    )
    args = parser.parse_args()

    ASSET_DIR.mkdir(exist_ok=True)
    if args.run_analysis:
        run_analysis_scripts()
    copy_pdf_rules()
    if not args.skip_composites:
        build_composites()
    render_report_previews()
    write_html()
    validate_report_assets()
    log("done")


if __name__ == "__main__":
    main()
