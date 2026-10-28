"""
Operational Gradio dashboard for CCTV incident triage
(1 still frame every 30 seconds).

Loads weights/best.pt and training artifacts. Never trains.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "weights" / "best.pt"
ARTIFACTS = ROOT / "artifacts"
PRED_PATH = ARTIFACTS / "test_predictions.json"
SWEEP_PATH = ARTIFACTS / "cost_sweep.json"
EVAL_PATH = ARTIFACTS / "eval_test.json"

CLASSES = ("normal", "violence", "weaponized")
CLASS_LABELS = {
    "normal": "Calm",
    "violence": "Violence",
    "weaponized": "Weapon",
}
POSITIVE = ("violence", "weaponized")
DECISION_LABELS = {
    "auto-clear": "Ignore",
    "human-review": "Send to operator",
    "auto-alert": "Raise alert",
}


def _require(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Missing required artifact: {path}")
    return path


_require(WEIGHTS_PATH)
_require(PRED_PATH)
_require(SWEEP_PATH)
_require(EVAL_PATH)

sweep = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
eval_payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
records = json.loads(PRED_PATH.read_text(encoding="utf-8"))["records"]

y_true = np.array([r["true"] for r in records])
p_pos = np.array([r["p_positive"] for r in records], dtype=float)

is_normal = y_true == "normal"
is_violence = y_true == "violence"
is_weaponized = y_true == "weaponized"
n_normal = int(is_normal.sum())
n_violence = int(is_violence.sum())
n_weaponized = int(is_weaponized.sum())

COSTS = sweep.get("costs_usd") or sweep["costs_eur"]
PROD = sweep["production"]
DEFAULT_CLEAR = float(sweep["operating_point"]["t_clear"])
DEFAULT_ALERT = float(sweep["operating_point"]["t_alert"])
COST_REVIEW = float(COSTS["review"])
FRAME_INTERVAL_S = float(PROD.get("frame_interval_seconds", 30))
PILOT_CAMERAS = int(PROD.get("pilot_cameras", 5))
COVERAGE_HOURS = float(PROD.get("coverage_hours_per_day", 8))
FRAMES_PER_CAM_DAY = COVERAGE_HOURS * 3600.0 / FRAME_INTERVAL_S
FRAMES_PILOT_DAY = FRAMES_PER_CAM_DAY * PILOT_CAMERAS

_sweep_rows = [row for row in sweep["single_threshold_sweep"] if row["threshold"] <= 1.0]
if len(_sweep_rows) > 400:
    idx = np.linspace(0, len(_sweep_rows) - 1, 400).astype(int)
    _sweep_rows = [_sweep_rows[i] for i in idx]
SWEEP_T = np.array([r["threshold"] for r in _sweep_rows], dtype=float)
SWEEP_COST_1K = np.array(
    [r["expected_cost_per_1000_production"] for r in _sweep_rows], dtype=float
)
ARGMIN_T = float(sweep["argmin_production"]["threshold"])

model = YOLO(str(WEIGHTS_PATH))
name_to_idx = {v: int(k) for k, v in model.names.items()}


def _clamp_band(t_clear: float, t_alert: float) -> tuple[float, float]:
    t_clear = float(np.clip(t_clear, 0.0, 1.0))
    t_alert = float(np.clip(t_alert, 0.0, 1.0))
    if t_clear > t_alert:
        t_clear, t_alert = t_alert, t_clear
    return t_clear, t_alert


def decide(score: float, t_clear: float, t_alert: float) -> str:
    if score < t_clear:
        return "auto-clear"
    if score >= t_alert:
        return "auto-alert"
    return "human-review"


def band_metrics(t_clear: float, t_alert: float) -> dict:
    t_clear, t_alert = _clamp_band(t_clear, t_alert)
    clear = p_pos < t_clear
    alert = p_pos >= t_alert
    review = ~(clear | alert)

    review_given = {
        "normal": float(review[is_normal].mean()) if n_normal else 0.0,
        "violence": float(review[is_violence].mean()) if n_violence else 0.0,
        "weaponized": float(review[is_weaponized].mean()) if n_weaponized else 0.0,
    }
    review_share_prod = (
        PROD["p_normal"] * review_given["normal"]
        + PROD["p_violence"] * review_given["violence"]
        + PROD["p_weaponized"] * review_given["weaponized"]
    )
    auto_fpr = float(alert[is_normal].mean()) if n_normal else 0.0
    auto_miss_v = float(clear[is_violence].mean()) if n_violence else 0.0
    auto_miss_w = float(clear[is_weaponized].mean()) if n_weaponized else 0.0

    error = (
        PROD["p_normal"] * auto_fpr * COSTS["fp"]
        + PROD["p_violence"] * auto_miss_v * COSTS["fn_violence"]
        + PROD["p_weaponized"] * auto_miss_w * COSTS["fn_weaponized"]
    )
    e_cost = error + review_share_prod * COST_REVIEW

    decisions = np.where(clear, "clear", np.where(alert, "alert", "review"))
    matrix = []
    for label in CLASSES:
        mask = y_true == label
        matrix.append(
            [
                int((mask & (decisions == "clear")).sum()),
                int((mask & (decisions == "review")).sum()),
                int((mask & (decisions == "alert")).sum()),
            ]
        )

    auto = clear | alert
    tp = fp = tn = fn = 0
    for t, c, a in zip(y_true[auto], clear[auto], alert[auto]):
        t_pos = t in POSITIVE
        if t_pos and a:
            tp += 1
        elif (not t_pos) and a:
            fp += 1
        elif (not t_pos) and c:
            tn += 1
        elif t_pos and c:
            fn += 1

    return {
        "t_clear": t_clear,
        "t_alert": t_alert,
        "review_share_test": float(review.mean()),
        "review_share_production": review_share_prod,
        "expected_cost_per_1000": 1000.0 * e_cost,
        "error_cost_per_1000": 1000.0 * error,
        "review_cost_per_1000": 1000.0 * review_share_prod * COST_REVIEW,
        "auto_fpr": auto_fpr,
        "auto_miss_violence": auto_miss_v,
        "auto_miss_weaponized": auto_miss_w,
        "decision_matrix": matrix,
        "auto_binary": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_auto": int(auto.sum()),
        "n_review": int(review.sum()),
    }


def plot_cost_curve(t_clear: float, t_alert: float, band_cost_1k: float):
    t_clear, t_alert = _clamp_band(t_clear, t_alert)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(SWEEP_T, SWEEP_COST_1K, color="#1f4e79", lw=1.6, label="Machine alone")
    ax.axvline(ARGMIN_T, color="#888888", ls="--", lw=1.0, label="Cheapest machine-only cutoff")
    ax.axvline(t_clear, color="#2a9d8f", ls="-", lw=1.3, label="Ignore below")
    ax.axvline(t_alert, color="#c45c26", ls="-", lw=1.3, label="Alert from")
    ax.axhline(
        band_cost_1k,
        color="#2a9d8f",
        ls=":",
        lw=1.3,
        label=f"Your rules now: {band_cost_1k:.0f} / 1,000 frames",
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Incident score cutoff")
    ax.set_ylabel("Expected cost per 1,000 frames")
    ax.set_title("What different cutoffs would cost")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(m: dict):
    """Classic 2×2 on automatic decisions only (operator queue excluded)."""
    b = m["auto_binary"]
    # rows = actual (incident, calm), cols = predicted (alert, ignore)
    data = np.array([[b["tp"], b["fn"]], [b["fp"], b["tn"]]], dtype=float)
    cell_names = [["True positive", "False negative"], ["False positive", "True negative"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    im = ax.imshow(data, cmap="Blues")
    ax.set_xticks([0, 1], ["Alert", "Ignore"])
    ax.set_yticks([0, 1], ["Actually incident", "Actually calm"])
    ax.set_xlabel("Automatic decision")
    ax.set_ylabel("Ground truth")
    ax.set_title(
        f"Confusion matrix (auto only)\n"
        f"{m['n_review']} frames in operator queue not counted here"
    )
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cell_names[i][j]}\n{int(data[i, j])}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def ops_summary(m: dict) -> str:
    b = m["auto_binary"]
    reviews_pilot_day = m["review_share_production"] * FRAMES_PILOT_DAY
    reviewer_hours = reviews_pilot_day / 100.0
    reviewers_8h = reviewer_hours / COVERAGE_HOURS
    return "\n".join(
        [
            "### Live outlook with these rules",
            "",
            f"| | |",
            f"|---|---|",
            f"| **Expected cost** | **${m['expected_cost_per_1000']:.0f}** per 1,000 frames "
            f"(mistakes ${m['error_cost_per_1000']:.0f} + operator time ${m['review_cost_per_1000']:.0f}) |",
            f"| **Operator load** | **{100 * m['review_share_production']:.1f}%** of frames "
            f"(~**{reviews_pilot_day:,.0f}** looks / day-shift on **{PILOT_CAMERAS}** cameras "
            f"at 1 frame / {FRAME_INTERVAL_S:.0f}s → ~**{reviewers_8h:.1f}** people on "
            f"{COVERAGE_HOURS:.0f}h shifts; 24/7 is a future step) |",
            f"| **False positives** | **{100 * m['auto_fpr']:.1f}%** of calm frames auto-alerted |",
            f"| **False negatives (violence)** | **{100 * m['auto_miss_violence']:.1f}%** auto-ignored |",
            f"| **False negatives (weapon)** | **{100 * m['auto_miss_weaponized']:.1f}%** auto-ignored |",
            "",
            f"Automatic decisions on the evaluation set: "
            f"**TP={b['tp']}**, **FP={b['fp']}**, **TN={b['tn']}**, **FN={b['fn']}** "
            f"({m['n_review']} frames sent to an operator instead).",
        ]
    )


def model_quality_markdown() -> str:
    prf = eval_payload["finetuned"]["per_class"]
    binary = eval_payload["finetuned"]["binary"]
    lines = [
        "How often the model names the right scene type when forced to pick one label "
        "(before the ignore / operator / alert rules).",
        "",
        "| Scene type | When it says this, how often it's right | Of real cases, how many it finds | Frames |",
        "|---|---:|---:|---:|",
    ]
    for label in CLASSES:
        s = prf[label]
        lines.append(
            f"| {CLASS_LABELS[label]} | {100 * s['precision']:.0f}% | "
            f"{100 * s['recall']:.0f}% | {s['support']} |"
        )
    lines += [
        "",
        f"For **incident vs calm** overall: "
        f"when it flags an incident it is right **{100 * binary['precision']:.0f}%** of the time; "
        f"it finds **{100 * binary['recall']:.0f}%** of real incidents.",
        "",
        f"*Cost assumptions in use (USD):* false alarm = ${COSTS['fp']}, "
        f"missed violence = ${COSTS['fn_violence']}, "
        f"missed weapon = ${COSTS['fn_weaponized']}, "
        f"one operator look = ${COST_REVIEW}. "
        f"Incidents assumed in **{100 * PROD['p_positive']:.3f}%** of live frames. "
        f"Pilot sampling: 1 frame / {FRAME_INTERVAL_S:.0f}s on {PILOT_CAMERAS} cameras, "
        f"{COVERAGE_HOURS:.0f}h day shift.",
    ]
    return "\n".join(lines)


def update_dashboard(t_clear: float, t_alert: float):
    m = band_metrics(t_clear, t_alert)
    return (
        ops_summary(m),
        plot_confusion_matrix(m),
        plot_cost_curve(m["t_clear"], m["t_alert"], m["expected_cost_per_1000"]),
        m["t_clear"],
        m["t_alert"],
    )


def predict_image(image, t_clear: float, t_alert: float):
    t_clear, t_alert = _clamp_band(t_clear, t_alert)
    if image is None:
        return "Drop a camera frame on the left, then press **Check frame**.", None

    result = model.predict(image, imgsz=224, verbose=False)[0]
    probs = result.probs.data.detach().cpu().numpy().astype(float)
    pn = float(probs[name_to_idx["normal"]])
    pv = float(probs[name_to_idx["violence"]])
    pw = float(probs[name_to_idx["weaponized"]])
    score = pv + pw
    key = decide(score, t_clear, t_alert)
    action = DECISION_LABELS[key]

    text = "\n".join(
        [
            f"## Action: **{action}**",
            "",
            f"**Incident score:** {100 * score:.1f}% "
            f"(ignore below {100 * t_clear:.0f}% · alert from {100 * t_alert:.0f}%)",
            "",
            "| What it looks like | Confidence |",
            "|---|---:|",
            f"| Calm | {100 * pn:.1f}% |",
            f"| Violence | {100 * pv:.1f}% |",
            f"| Weapon | {100 * pw:.1f}% |",
        ]
    )

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    labels = [CLASS_LABELS[c] for c in CLASSES]
    ax.bar(labels, [pn, pv, pw], color=["#5c5c5c", "#c45c26", "#1f4e79"])
    ax.axhline(t_clear, color="#2a9d8f", ls="--", lw=1.1, label="Ignore below")
    ax.axhline(t_alert, color="#c45c26", ls="--", lw=1.1, label="Alert from")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title(action)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return text, fig


QUALITY_MD = model_quality_markdown()
_init = band_metrics(DEFAULT_CLEAR, DEFAULT_ALERT)

with gr.Blocks(title="Camera incident desk") as demo:
    gr.Markdown(
        """
# Camera incident desk
Watching a **5-camera day-shift pilot** at **one frame every 30 seconds**.
The system either **ignores** the frame, **sends it to an operator**, or
**raises an alert**.
"""
    )

    with gr.Tabs():
        with gr.Tab("Check a frame"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_in = gr.Image(type="pil", label="Camera frame")
                    predict_btn = gr.Button("Check frame", variant="primary")
                with gr.Column(scale=1):
                    pred_md = gr.Markdown(
                        "Upload a frame to see whether it would be ignored, "
                        "sent to an operator, or raised as an alert."
                    )
                    pred_plot = gr.Plot(label="Confidence breakdown")

        with gr.Tab("Operating rules"):
            gr.Markdown(
                "Move the two dials. Everything below updates for the evaluation set "
                "and the live cost assumptions."
            )
            with gr.Row():
                t_clear = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_CLEAR,
                    step=0.001,
                    label="Ignore below this incident score",
                    info="Frames weaker than this are dropped automatically. Default = cost-curve minimum.",
                )
                t_alert = gr.Slider(
                    0.0,
                    1.0,
                    value=DEFAULT_ALERT,
                    step=0.001,
                    label="Raise alert from this incident score",
                    info="Frames at or above this go straight to alert. Default = same cost-curve minimum.",
                )
            metrics_out = gr.Markdown(ops_summary(_init))
            with gr.Row():
                matrix_plot = gr.Plot(
                    value=plot_confusion_matrix(_init),
                    label="Confusion matrix",
                )
                curve_plot = gr.Plot(
                    value=plot_cost_curve(
                        _init["t_clear"],
                        _init["t_alert"],
                        _init["expected_cost_per_1000"],
                    ),
                    label="Cost of different cutoffs",
                )

        with gr.Tab("Model quality"):
            gr.Markdown(QUALITY_MD)

    def _on_thresholds(tc, ta):
        return update_dashboard(tc, ta)

    for slider in (t_clear, t_alert):
        slider.release(
            _on_thresholds,
            inputs=[t_clear, t_alert],
            outputs=[metrics_out, matrix_plot, curve_plot, t_clear, t_alert],
        )

    predict_btn.click(
        predict_image,
        inputs=[image_in, t_clear, t_alert],
        outputs=[pred_md, pred_plot],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
