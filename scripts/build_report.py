#!/usr/bin/env python3
"""Build the leaderboard tables and the noise-robustness plot from stored results.

    python scripts/build_report.py            # writes docs/RESULTS.md + docs/speed_vs_mae_buckeye.png
    python scripts/build_report.py --headline # also prints the README headline tables

Inputs: ``results_full/`` (local systems), ``results_api/`` (hosted APIs),
``results_robustness/`` (TIMIT + Gaussian noise) and ``results_speed/``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS = [("buckeye", "Buckeye (conversational)"),
            ("fluency_bank_stutter", "FluencyBank (stuttered speech)"),
            ("timit", "TIMIT (read speech)")]
COLLARS = ["25ms", "50ms", "100ms", "200ms"]
SNRS = [None, 30.0, 20.0, 10.0, 5.0, 0.0]

# Result-file model name prefix -> public display name.
NAMES = {
    "mfa-english_mfa-beam400-4000": "mfa_widebeam",
    "mwa-buckeye": "mwa_buckeye", "mwa-timit": "mwa_timit",
    "nyra_fa-": "nyra_fa", "nyra_fa_pro-": "nyra_fa_pro", "qwen_fa-": "qwen_fa",
    "mfa-": "mfa", "ctc_segmentation-": "ctc_segmentation", "nemo_fa-": "nemo_fa",
    "crisperwhisper-transformers-large-v2_verbatimize_longform_with_timing_release":
        "crisperwhisper_2.0_large",
    "whisperx-": "whisperx", "canary-": "canary", "fusealign-": "fusealign",
}
# Speed comes from the deployment backend where it differs from the scored one.
SPEED_FILE_OVERRIDE = {
    "crisperwhisper_2.0_large":
        "speed_crisperwhisper-ct2-large-v2_verbatimize_longform_with_timing_release_buckeye.json",
}
# Public label: system name plus the company behind it.
LABELS = {
    "elevenlabs_fa": "Cue (ElevenLabs)",
    "mfa": "`mfa` (MFA 3.3.9, `english_mfa` 3.1.0, default beam)",
    "mfa_widebeam": "`mfa` (MFA 3.3.9, `english_mfa` 3.1.0, wide beam)",
    "nyra_fa": "`nyra_fa` (nyra labs)",
    "crisperwhisper_2.0_large": "`crisperwhisper_2.0_large` (nyra labs)",
    "qwen_fa": "`qwen_fa` (Qwen)",
    "mms_fa": "`mms_fa` (Meta)",
    "fusealign": "`fusealign` (Zyphra)",
    "mwa_buckeye": "`mwa_buckeye` (MLSpeech)",
    "mwa_timit": "`mwa_timit` (MLSpeech)",
    "olign_v1": "`olign_v1` (Olewave)",
    "nemo_fa": "`nemo_fa` (NVIDIA)",
    "canary": "`canary` (NVIDIA)",
    "nemotron_transcribe": "`nemotron_transcribe` (NVIDIA)",
    "fish_transcribe_1_pro": "`fish_transcribe_1_pro` (Fish Audio)",
    "xai_grok_stt": "`xai_grok_stt` (xAI)",
    "mai_transcribe_2": "`mai_transcribe_2` (Microsoft)",
    "assemblyai_u35_pro": "`assemblyai_u35_pro` (AssemblyAI)",
    "assemblyai_u2_disfl": "`assemblyai_u2_disfl` (AssemblyAI)",
    "cartesia_ink_whisper": "`cartesia_ink_whisper` (Cartesia)",
    "deepgram_nova3": "`deepgram_nova3` (Deepgram)",
}
# Rows left out of the README headline (kept in docs/RESULTS.md).
NOT_IN_HEADLINE = {"assemblyai_u2_disfl"}
# Rows left out of every report.
EXCLUDED = {"nyra_fa_pro"}


def display(model: str) -> str:
    for prefix, name in NAMES.items():
        if model == prefix or model.startswith(prefix):
            return name
    return model


PLOT_LABELS = {"mfa": "mfa (default beam)", "mfa_widebeam": "mfa (wide beam)"}


def label(name: str, markdown: bool = True) -> str:
    if not markdown and name in PLOT_LABELS:
        return PLOT_LABELS[name]
    text = LABELS.get(name, f"`{name}`")
    return text if markdown else text.replace("`", "")


def load(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def clean_rows(dataset: str) -> List[Dict]:
    rows = []
    for d in ("results_full", "results_api"):
        for p in glob.glob(os.path.join(ROOT, d, f"*_{dataset}.json")):
            r = load(p)
            if "overall" not in r or "crisperwhisper-ct2" in r["model"]:
                continue
            o, md = r["overall"], r.get("metadata") or {}
            name = display(r["model"])
            if name in EXCLUDED:
                continue
            rows.append({
                "name": name,
                "aligner": md.get("system_type") == "forced_aligner",
                "hosted": md.get("deployment") == "hosted_api",
                "recall": o["word_recall"],
                "mae": o["mae_ms"]["overall"],
                "medae": o["median_ae_ms"]["overall"],
                "miou": o["miou"],
                "acc": {c: o["f1_matched"][c]["f1"] for c in COLLARS},
                "f1": {c: o["f1_standard"][c]["f1"] for c in COLLARS},
                "rtf": speed(name, r["model"]),
            })
    return rows


def speed(name: str, model: str) -> Optional[float]:
    fname = SPEED_FILE_OVERRIDE.get(name, f"speed_{model}_buckeye.json")
    p = os.path.join(ROOT, "results_speed", fname)
    return load(p)["rtf"] if os.path.exists(p) else None


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def fmt_speed(rtf: Optional[float]) -> str:
    if rtf is None:
        return "–"
    x = 1 / rtf
    return f"{x:.0f}×" if x >= 10 else f"{x:.1f}×"


def kind(r: Dict) -> str:
    return "hosted API" if r["hosted"] else "local"


def system_type(r: Dict) -> str:
    return "forced aligner" if r["aligner"] else "ASR system"


def acc_table(rows: List[Dict], headline: bool) -> str:
    rows = [r for r in rows if not (headline and r["name"] in NOT_IN_HEADLINE)]
    out = ["| System | Type | Deployment | Word recall | MAE (ms) | mIoU | Acc@25ms | Acc@50ms "
           "| Acc@100ms | Acc@200ms | Speed (× real time) |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: -r["acc"]["50ms"]):
        accs = " | ".join(pct(r["acc"][c]) for c in COLLARS)
        out.append(f"| {label(r['name'])} | {system_type(r)} | {kind(r)} | {pct(r['recall'])} "
                   f"| {r['mae']:.1f} | {pct(r['miou'])} | {accs} | {fmt_speed(r['rtf'])} |")
    out.append("")
    return "\n".join(out)


def f1_table(rows: List[Dict]) -> str:
    out = ["| System | Type | Deployment | Word recall | MedAE (ms) | F1@25ms | F1@50ms | F1@100ms "
           "| F1@200ms |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: -r["f1"]["50ms"]):
        f1s = " | ".join(pct(r["f1"][c]) for c in COLLARS)
        out.append(f"| {label(r['name'])} | {system_type(r)} | {kind(r)} | {pct(r['recall'])} "
                   f"| {r['medae']:.1f} | {f1s} |")
    return "\n".join(out)


def noise_rows() -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for p in glob.glob(os.path.join(ROOT, "results_robustness", "*.json")):
        r = load(p)
        if "overall" not in r:
            continue
        snr = (r.get("condition") or {}).get("snr_db")
        o = r["overall"]
        if display(r["model"]) in EXCLUDED:
            continue
        data.setdefault(display(r["model"]), {})[snr] = {
            "f1": o["f1_standard"]["50ms"]["f1"],
            "acc": o["f1_matched"]["50ms"]["f1"],
            "recall": o["word_recall"],
        }
    return data


def noise_table(data: Dict[str, Dict], key: str) -> str:
    head = " | ".join("clean" if s is None else f"{s:.0f} dB" for s in SNRS)
    out = [f"| System | {head} |", "| --- |" + " ---: |" * len(SNRS)]
    for name in sorted(data, key=lambda n: -data[n].get(None, {}).get("f1", 0)):
        vals = [pct(data[name][s][key]) if s in data[name] else "–" for s in SNRS]
        out.append(f"| {label(name)} | " + " | ".join(vals) + " |")
    return "\n".join(out)


# Label offsets (points) that keep neighbouring annotations apart.
SPEED_LABEL_OFFSET = {
    "nyra_fa": (-8, 0, "right"), "qwen_fa": (7, 0, "left"),
    "ctc_segmentation": (0, -11, "center"), "nemo_fa": (7, 0, "left"),
    "elevenlabs_fa": (7, -6, "left"), "mfa": (7, 0, "left"), "mfa_widebeam": (7, 0, "left"),
    "fish_transcribe_1_pro": (-6, 6, "right"), "crisperwhisper_2.0_large": (7, 0, "left"),
    "xai_grok_stt": (-6, 4, "right"), "mai_transcribe_2": (-7, 0, "right"),
    "assemblyai_u35_pro": (7, 0, "left"), "cartesia_ink_whisper": (-7, 0, "right"),
    "deepgram_nova3": (7, 0, "left"), "whisperx": (-7, 0, "right"),
    "canary": (7, 0, "left"), "olign_v1": (7, -5, "left"), "mwa_buckeye": (7, 0, "left"),
    "mwa_timit": (7, 0, "left"), "mms_fa": (7, -6, "left"), "fusealign": (7, 0, "left"),
}


def speed_plot(rows: List[Dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    style = {
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 10,
        "axes.titlesize": 10.5, "axes.linewidth": 0.8, "xtick.direction": "out",
        "ytick.direction": "out", "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.frameon": False,
    }
    colors = {True: "#0072B2", False: "#D55E00"}
    markers = {True: "o", False: "s"}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=200)
        ax.axvline(1, color="0.55", lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(1.06, 0.985, "real time", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="left", fontsize=7.5, color="0.45")
        for r in rows:
            if not r["rtf"]:
                continue
            x, y, c = 1 / r["rtf"], r["mae"], colors[r["aligner"]]
            ax.scatter(x, y, s=42, marker=markers[r["aligner"]], zorder=3, linewidths=1.1,
                       edgecolors=c, facecolors="white" if r["hosted"] else c)
            dx, dy, ha = SPEED_LABEL_OFFSET.get(r["name"], (8, -3, "left"))
            weight = "bold" if r["name"] == "nyra_fa" else "normal"
            ax.annotate(label(r["name"], markdown=False), (x, y),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                        fontsize=7, color="0.15", weight=weight)

        ax.set_xscale("log")
        ax.set_xlim(0.5, 1500)
        ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" + r"$\times$"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(155, 10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(which="major", color="0.88", lw=0.6)
        ax.grid(which="minor", axis="x", color="0.94", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"Processing speed ($\times$ real time, single clip, log scale)")
        ax.set_ylabel("Mean absolute boundary error (ms)")
        ax.set_title("Word-boundary error vs. speed on Buckeye (upper right is better)",
                     loc="left")
        ax.annotate("", xy=(0.975, 0.07), xytext=(0.86, 0.07), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=0.8))
        ax.annotate("", xy=(0.975, 0.2), xytext=(0.975, 0.07), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="0.5", lw=0.8))
        ax.text(0.955, 0.1, "better", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, style="italic", color="0.4")

        handles = [
            Line2D([], [], ls="", marker="o", ms=6, mfc=colors[True], mec=colors[True],
                   label="Forced aligner"),
            Line2D([], [], ls="", marker="s", ms=6, mfc=colors[False], mec=colors[False],
                   label="ASR system"),
            Line2D([], [], ls="", marker="o", ms=6, mfc="0.35", mec="0.35", label="Local"),
            Line2D([], [], ls="", marker="o", ms=6, mfc="white", mec="0.35", label="Hosted API"),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.11, 1.0), fontsize=7.5, ncol=2,
                  handletextpad=0.3, columnspacing=1.0)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headline", action="store_true", help="Print the README headline tables.")
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)

    noise = noise_rows()
    speed_plot([r for r in clean_rows("buckeye") if r["name"] not in NOT_IN_HEADLINE],
               os.path.join(ROOT, "docs", "speed_vs_mae_buckeye.png"))

    parts = ["# Full results\n",
             "Generated by `scripts/build_report.py`. See the README for metric definitions.\n"]
    for ds, title in DATASETS:
        rows = clean_rows(ds)
        parts += [f"## {title}\n", "### Timing accuracy (matched words)\n", acc_table(rows, False),
                  "### F1 (recognition and timing combined)\n", f1_table(rows), ""]
    parts += ["## Noise robustness (TIMIT + Gaussian noise)\n",
              "### F1@50ms\n", noise_table(noise, "f1"), "",
              "### Acc@50ms (matched words only)\n", noise_table(noise, "acc"), "",
              "### Word recall\n", noise_table(noise, "recall"), ""]
    with open(os.path.join(ROOT, "docs", "RESULTS.md"), "w") as f:
        f.write("\n".join(parts))

    if args.headline:
        for ds, title in DATASETS[:2]:
            print(f"### {title}\n")
            print(acc_table(clean_rows(ds), True))
        print(noise_table(noise, "f1"))


if __name__ == "__main__":
    main()
