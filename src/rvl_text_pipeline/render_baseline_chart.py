from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    {
        "run_id": "smoke_20260406_160133",
        "label": "Smoke v1",
        "model": "qwen2.5:14b",
        "sample_size": 16,
        "config": "prompt v1 / no few-shot",
        "kind": "smoke",
    },
    {
        "run_id": "smoke_v2_20260406_171902",
        "label": "Smoke v2",
        "model": "qwen2.5:14b",
        "sample_size": 16,
        "config": "prompt v2 / few-shot / min schema",
        "kind": "smoke",
    },
    {
        "run_id": "run_001",
        "label": "Run 001",
        "model": "qwen2.5:14b",
        "sample_size": 100,
        "config": "prompt v1 / no few-shot",
        "kind": "official",
    },
    {
        "run_id": "run_002",
        "label": "Run 002",
        "model": "qwen2.5:14b",
        "sample_size": 100,
        "config": "prompt v2 / few-shot / num_predict=512",
        "kind": "official",
    },
    {
        "run_id": "run_003_qwen32b",
        "label": "Run 003",
        "model": "qwen2.5:32b",
        "sample_size": 100,
        "config": "same sample as run_001 / prompt v2",
        "kind": "official",
    },
]

PERCENT_SERIES = [
    ("coverage", "Cobertura", "#2a9d8f"),
    ("accuracy", "Accuracy liquida", "#f4a261"),
    ("strict_accuracy", "Accuracy total", "#457b9d"),
    ("macro_f1", "Macro-F1", "#e76f51"),
]
ERROR_SERIES = [
    ("ocr_error", "OCR error", "#6c757d"),
    ("validation_error", "Erro de validacao", "#c1121f"),
]
RUNTIME_COLOR = "#7b2cbf"


def load_run_metrics(run_id: str) -> dict:
    metrics_path = ROOT / "data" / "processed" / "saida-ocr-class" / run_id / "evaluation" / "metrics.json"
    log_path = ROOT / "data" / "processed" / "saida-ocr-class" / run_id / "logs" / "pipeline.log"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    duration_min = load_duration_minutes(log_path)

    return {
        "coverage": metrics["coverage"] * 100.0,
        "accuracy": metrics["accuracy"] * 100.0,
        "strict_accuracy": metrics["strict_accuracy"] * 100.0,
        "macro_f1": metrics["macro_f1"] * 100.0,
        "ocr_error": int(metrics["status_counts"].get("ocr_error", 0)),
        "validation_error": int(metrics["status_counts"].get("validation_error", 0)),
        "duration_min": duration_min,
        "num_total": int(metrics["num_total"]),
        "num_ok": int(metrics["num_ok"]),
    }


def load_duration_minutes(log_path: Path) -> float:
    stamps: list[dt.datetime] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\[([^\]]+)\]", line)
        if not match:
            continue
        stamps.append(dt.datetime.fromisoformat(match.group(1).replace("Z", "+00:00")))
    if len(stamps) < 2:
        return 0.0
    return round((stamps[-1] - stamps[0]).total_seconds() / 60.0, 2)


def svg_text(x: float, y: float, text: str, size: int = 14, weight: str = "400", fill: str = "#111", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Liberation Sans, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def svg_multiline_text(x: float, y: float, lines: list[str], size: int = 13, fill: str = "#111", anchor: str = "middle") -> str:
    tspans = []
    for i, line in enumerate(lines):
        dy = "0" if i == 0 else "1.25em"
        tspans.append(
            f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(line)}</tspan>'
        )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Liberation Sans, Arial, sans-serif" '
        f'font-size="{size}" fill="{fill}" text-anchor="{anchor}">{"".join(tspans)}</text>'
    )


def draw_panel_title(x: float, y: float, title: str) -> str:
    return svg_text(x, y, title, size=20, weight="700")


def bar_y(value: float, max_value: float, y: float, height: float) -> float:
    return y + height - (value / max_value) * height


def draw_axes(x: float, y: float, width: float, height: float, max_value: float, ticks: list[int], suffix: str = "") -> str:
    parts = [
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + height:.1f}" stroke="#444" stroke-width="1.5"/>',
        f'<line x1="{x:.1f}" y1="{y + height:.1f}" x2="{x + width:.1f}" y2="{y + height:.1f}" stroke="#444" stroke-width="1.5"/>',
    ]
    for tick in ticks:
        yy = bar_y(tick, max_value, y, height)
        parts.append(f'<line x1="{x:.1f}" y1="{yy:.1f}" x2="{x + width:.1f}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(svg_text(x - 12, yy + 5, f"{tick}{suffix}", size=12, anchor="end", fill="#555"))
    return "".join(parts)


def draw_legend(x: float, y: float, series: list[tuple[str, str, str]], columns: int = 2) -> str:
    parts: list[str] = []
    col_width = 200
    row_height = 26
    for idx, (_, label, color) in enumerate(series):
        row = idx // columns
        col = idx % columns
        xx = x + col * col_width
        yy = y + row * row_height
        parts.append(f'<rect x="{xx:.1f}" y="{yy - 11:.1f}" width="18" height="18" rx="3" fill="{color}"/>')
        parts.append(svg_text(xx + 28, yy + 3, label, size=13))
    return "".join(parts)


def build_svg() -> str:
    runs = []
    for run in RUNS:
        metrics = load_run_metrics(run["run_id"])
        runs.append({**run, **metrics})

    width = 1520
    height = 1180
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<rect x="28" y="28" width="1464" height="1124" rx="18" fill="#ffffff" stroke="#d9d9d9"/>',
        svg_text(72, 82, "Comparacao analitica dos cenarios OCR -> LLM em RVL-CDIP", size=28, weight="700"),
        svg_text(72, 112, "Tabela consolidada dos testes locais sem treinamento adicional", size=16, fill="#555"),
    ]

    parts.append(draw_legend(72, 136, PERCENT_SERIES + ERROR_SERIES + [("runtime", "Runtime (min)", RUNTIME_COLOR)], columns=3))

    parts.append('<rect x="62" y="186" width="1396" height="308" rx="14" fill="#fcfcfc" stroke="#e6e6e6"/>')
    parts.append('<rect x="62" y="524" width="1396" height="230" rx="14" fill="#fcfcfc" stroke="#e6e6e6"/>')
    parts.append('<rect x="62" y="784" width="1396" height="318" rx="14" fill="#fcfcfc" stroke="#e6e6e6"/>')

    parts.append(draw_panel_title(86, 222, "A. Metricas de qualidade (%)"))
    parts.append(draw_panel_title(86, 560, "B. Contagem de erros"))
    parts.append(draw_panel_title(86, 820, "C. Tempo e notas analiticas"))

    # Panel A
    chart_x = 110
    chart_y = 250
    chart_w = 1280
    chart_h = 195
    parts.append(draw_axes(chart_x, chart_y, chart_w, chart_h, 100.0, [0, 20, 40, 60, 80, 100], suffix="%"))

    group_w = 220
    bar_w = 30
    intra_gap = 10
    for i, run in enumerate(runs):
        group_x = chart_x + 34 + i * group_w
        if run["kind"] == "official":
            parts.append(f'<rect x="{group_x - 18:.1f}" y="{chart_y - 18:.1f}" width="168" height="{chart_h + 74:.1f}" rx="10" fill="#f7fbff" stroke="#d9ecff"/>')
        else:
            parts.append(f'<rect x="{group_x - 18:.1f}" y="{chart_y - 18:.1f}" width="168" height="{chart_h + 74:.1f}" rx="10" fill="#fffaf3" stroke="#f3e5bf"/>')
        for j, (key, _, color) in enumerate(PERCENT_SERIES):
            xx = group_x + j * (bar_w + intra_gap)
            yy = bar_y(run[key], 100.0, chart_y, chart_h)
            hh = chart_y + chart_h - yy
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{bar_w}" height="{hh:.1f}" rx="4" fill="{color}"/>')
            parts.append(svg_text(xx + bar_w / 2, yy - 6, f"{run[key]:.1f}", size=11, anchor="middle", fill="#333"))
        parts.append(svg_multiline_text(group_x + 62, chart_y + chart_h + 28, [run["label"], run["model"], f"n={run['sample_size']}"], size=12))

    parts.append(svg_text(1292, 224, "Smokes em amarelo, runs oficiais em azul", size=12, fill="#666", anchor="end"))

    # Panel B
    err_x = 110
    err_y = 590
    err_w = 900
    err_h = 115
    parts.append(draw_axes(err_x, err_y, err_w, err_h, 14.0, [0, 2, 4, 6, 8, 10, 12, 14]))
    err_group_w = 170
    err_bar_w = 40
    for i, run in enumerate(runs):
        gx = err_x + 44 + i * err_group_w
        for j, (key, _, color) in enumerate(ERROR_SERIES):
            xx = gx + j * 56
            yy = bar_y(float(run[key]), 14.0, err_y, err_h)
            hh = err_y + err_h - yy
            parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{err_bar_w}" height="{hh:.1f}" rx="4" fill="{color}"/>')
            parts.append(svg_text(xx + err_bar_w / 2, yy - 6, str(run[key]), size=12, anchor="middle"))
        parts.append(svg_multiline_text(gx + 28, err_y + err_h + 24, [run["label"]], size=12))

    parts.append(svg_text(1340, 620, "Leitura rapida", size=15, weight="700"))
    parts.append(svg_text(1340, 650, "O prompt v2 eliminou erros de validacao", size=13))
    parts.append(svg_text(1340, 676, "nos runs oficiais (run_002 e run_003).", size=13))
    parts.append(svg_text(1340, 715, "Os erros de OCR ficaram estaveis em 3/100", size=13))
    parts.append(svg_text(1340, 741, "ao longo dos tres runs completos.", size=13))

    # Panel C
    rt_x = 110
    rt_y = 854
    rt_w = 900
    rt_h = 170
    max_runtime = max(run["duration_min"] for run in runs)
    parts.append(draw_axes(rt_x, rt_y, rt_w, rt_h, max_runtime + 4.0, [0, 5, 10, 15, 20, 25, 30]))
    rt_group_w = 170
    rt_bar_w = 72
    for i, run in enumerate(runs):
        xx = rt_x + 46 + i * rt_group_w
        yy = bar_y(run["duration_min"], max_runtime + 4.0, rt_y, rt_h)
        hh = rt_y + rt_h - yy
        parts.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{rt_bar_w}" height="{hh:.1f}" rx="5" fill="{RUNTIME_COLOR}"/>')
        parts.append(svg_text(xx + rt_bar_w / 2, yy - 7, f"{run['duration_min']:.2f}", size=12, anchor="middle"))
        parts.append(svg_multiline_text(xx + rt_bar_w / 2, rt_y + rt_h + 24, [run["label"]], size=12))

    parts.append(svg_text(1056, 860, "Notas analiticas", size=18, weight="700"))
    notes = [
        "Melhor resultado oficial: run_002",
        "  Accuracy liquida: 56.70%",
        "  Accuracy total: 55.00%",
        "  Macro-F1: 55.89%",
        "",
        "O efeito do prompt foi maior que o ganho por escalar o modelo:",
        "  run_001 -> run_002: +17.00 p.p. em accuracy total",
        "  erros de validacao: 13 -> 0",
        "",
        "qwen2.5:32b nao melhorou o baseline:",
        "  run_003 vs run_002: -1.00 p.p. em accuracy total",
        "  tempo relativo: ~5.0x mais lento",
        "",
        "Leitura filtrada (sem classes 3, 8 e 12):",
        "  run_002 = run_003 = 63.75% de accuracy total",
    ]
    note_y = 890
    for line in notes:
        if not line:
            note_y += 12
            continue
        weight = "700" if line.endswith(":") or line.startswith("Best") else "400"
        parts.append(svg_text(1056, note_y, line, size=14, weight=weight))
        note_y += 24

    parts.append(svg_text(72, 1132, "Fonte: artefatos locais de metrics.json, pipeline.log e state.json para smoke_20260406_160133, smoke_v2_20260406_171902, run_001, run_002 e run_003_qwen32b.", size=12, fill="#666"))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    out_path = ROOT / "docs" / "grafico_cenarios_testados.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_svg(), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
