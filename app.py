"""
Gradio app to visualize PatchCore inference results.
- Left: CSV scores table
- Right: 5 rows × 2 cols heatmap grid with prev/next paging
- Click image -> lightbox popup; click outside or Esc to close
"""

import base64
import glob
import io
import os

import gradio as gr
import pandas as pd
from PIL import Image

RESULTS_ROOT = "/nfs/data/1/xning/elec_AI/results/evaluated_results"
PAGE_SIZE = 10

# JS injected via js= on demo.load — runs once at page load, not via innerHTML
LIGHTBOX_JS = """
() => {
  // Create overlay once
  if (document.getElementById('pc-lightbox')) return;

  const overlay = document.createElement('div');
  overlay.id = 'pc-lightbox';
  overlay.style.cssText = [
    'display:none', 'position:fixed', 'inset:0',
    'background:rgba(0,0,0,0.88)', 'z-index:99999',
    'align-items:center', 'justify-content:center',
    'flex-direction:column', 'cursor:zoom-out'
  ].join(';');

  const img = document.createElement('img');
  img.id = 'pc-lightbox-img';
  img.style.cssText = 'max-width:90vw;max-height:85vh;border-radius:6px;box-shadow:0 0 40px #000;cursor:default';

  const caption = document.createElement('p');
  caption.id = 'pc-lightbox-caption';
  caption.style.cssText = 'color:#ddd;margin-top:10px;font-size:14px;text-align:center';

  overlay.appendChild(img);
  overlay.appendChild(caption);
  document.body.appendChild(overlay);

  // Close on background click
  overlay.addEventListener('click', function(e) {
    if (e.target === overlay) overlay.style.display = 'none';
  });

  // Close on Escape
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') overlay.style.display = 'none';
  });

  // Delegate clicks on .pc-thumb images (works even after grid HTML is replaced)
  document.addEventListener('click', function(e) {
    const thumb = e.target.closest('img.pc-thumb');
    if (!thumb) return;
    document.getElementById('pc-lightbox-img').src = thumb.src;
    document.getElementById('pc-lightbox-caption').textContent = thumb.dataset.caption || '';
    const lb = document.getElementById('pc-lightbox');
    lb.style.display = 'flex';
  });
}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def find_result_dirs(root: str) -> list[str]:
    csvs = glob.glob(os.path.join(root, "**", "*_per_image_scores.csv"), recursive=True)
    return sorted({os.path.dirname(c) for c in csvs})


def load_csv(result_dir: str) -> pd.DataFrame:
    csvs = glob.glob(os.path.join(result_dir, "*_per_image_scores.csv"))
    if not csvs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df["filename"] = df["image_path"].apply(os.path.basename)
    df["stem"] = df["filename"].apply(lambda x: os.path.splitext(x)[0])
    df["label"] = df["is_defect"].map({1: "DEFECT", 0: "GOOD"})
    df["anomaly_score"] = df["anomaly_score"].round(4)
    return df.sort_values("anomaly_score", ascending=False).reset_index(drop=True)


def find_heatmap(result_dir: str, stem: str) -> str | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidates = glob.glob(os.path.join(result_dir, f"*{stem}*{ext}"))
        if candidates:
            return candidates[0]
    return None


def img_to_b64(path: str) -> str | None:
    if not path or not os.path.exists(path):
        return None
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ── grid HTML (5 rows × 2 cols) ───────────────────────────────────────────────

def build_grid_html(rows: list[dict]) -> str:
    if not rows:
        return "<p style='color:#888;padding:12px'>No images on this page.</p>"

    cards = []
    for r in rows:
        b64 = r.get("b64")
        caption = f"{r['filename']}  |  score: {r['score']:.4f}  |  {r['label']}"
        if not b64:
            thumb = "<div style='width:100%;aspect-ratio:1;background:#222;display:flex;align-items:center;justify-content:center;color:#555;font-size:12px'>No heatmap</div>"
        else:
            thumb = (
                f'<img class="pc-thumb" '
                f'src="data:image/jpeg;base64,{b64}" '
                f'data-caption="{caption}" '
                f'style="width:100%;height:auto;display:block;cursor:zoom-in">'
            )

        color = "#c0392b" if r["label"] == "DEFECT" else "#27ae60"
        badge = f'<span style="background:{color};color:#fff;padding:1px 5px;border-radius:3px;font-size:10px">{r["label"]}</span>'

        cards.append(f"""
<div style="background:#1e1e1e;border-radius:6px;overflow:hidden">
  {thumb}
  <div style="padding:4px 6px;font-size:11px;color:#ccc;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="{r['filename']}">
    {badge} {r['filename']}
  </div>
  <div style="padding:0 6px 5px;font-size:11px;color:#999">score: {r['score']:.4f}</div>
</div>""")

    items = "\n".join(cards)
    # 5 rows × 2 cols
    return (
        '<div style="display:grid;grid-template-columns:repeat(2,1fr);'
        'gap:8px;background:#111;padding:8px;border-radius:8px">'
        + items + '</div>'
    )


def get_page_data(df: pd.DataFrame, result_dir: str, page: int, filter_label: str):
    if df is None or df.empty:
        return "<p>No data loaded.</p>", "Page 0 / 0", 0

    filtered = df if filter_label == "All" else df[df["label"] == filter_label]
    filtered = filtered.reset_index(drop=True)
    total = len(filtered)
    n_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, n_pages - 1))

    chunk = filtered.iloc[page * PAGE_SIZE: page * PAGE_SIZE + PAGE_SIZE]
    rows = []
    for _, row in chunk.iterrows():
        hmap = find_heatmap(result_dir, row["stem"])
        rows.append({"filename": row["filename"], "score": row["anomaly_score"],
                     "label": row["label"], "b64": img_to_b64(hmap)})

    page_info = f"Page {page + 1} / {n_pages}  ({total} images)"
    return build_grid_html(rows), page_info, page


# ── callbacks ─────────────────────────────────────────────────────────────────

def on_load_dir(result_dir: str):
    df = load_csv(result_dir)
    if df.empty:
        return "No CSV found.", None, pd.DataFrame(), "<p>No data.</p>", "Page 0 / 0", 0

    n_total = len(df)
    n_defect = int((df["is_defect"] == 1).sum())
    n_good = int((df["is_defect"] == 0).sum())
    summary = (
        f"**Total:** {n_total} &nbsp;|&nbsp; "
        f"**Defects:** {n_defect} &nbsp;|&nbsp; "
        f"**Good:** {n_good} &nbsp;|&nbsp; "
        f"**Mean score:** {df['anomaly_score'].mean():.4f}"
    )
    table = df[["filename", "anomaly_score", "label"]].copy()
    html, page_info, page = get_page_data(df, result_dir, 0, "All")
    return summary, df, table, html, page_info, page


def on_page(df, result_dir, page, filter_label, delta):
    if df is None:
        return "<p>No data.</p>", "Page 0 / 0", page
    html, page_info, new_page = get_page_data(df, result_dir, page + delta, filter_label)
    return html, page_info, new_page


def on_filter_change(filter_label, df, result_dir):
    if df is None:
        return pd.DataFrame(), "<p>No data.</p>", "Page 0 / 0", 0
    filtered = df if filter_label == "All" else df[df["label"] == filter_label]
    table = filtered[["filename", "anomaly_score", "label"]].copy()
    html, page_info, page = get_page_data(df, result_dir, 0, filter_label)
    return table, html, page_info, page


# ── build UI ──────────────────────────────────────────────────────────────────

def build_app():
    result_dirs = find_result_dirs(RESULTS_ROOT)
    rel_dirs = [os.path.relpath(d, RESULTS_ROOT) for d in result_dirs]
    dir_map = dict(zip(rel_dirs, result_dirs))

    with gr.Blocks(title="PatchCore Results Viewer") as demo:

        gr.Markdown("# PatchCore Anomaly Detection — Results Viewer")

        df_state = gr.State(None)
        page_state = gr.State(0)

        with gr.Row():
            dir_dropdown = gr.Dropdown(
                choices=rel_dirs, label="Result directory",
                value=rel_dirs[0] if rel_dirs else None, scale=4,
            )
            filter_radio = gr.Radio(
                choices=["All", "DEFECT", "GOOD"], value="All",
                label="Filter", scale=1,
            )

        summary_md = gr.Markdown("")

        with gr.Row():
            # Left: scores table
            with gr.Column(scale=1, min_width=280):
                score_table = gr.Dataframe(
                    headers=["filename", "anomaly_score", "label"],
                    label="All scores",
                    interactive=False,
                    wrap=False,
                    max_height=560,
                )
            # Right: heatmap grid + pagination
            with gr.Column(scale=2):
                grid_html = gr.HTML()
                with gr.Row():
                    prev_btn = gr.Button("◀  Prev", scale=1)
                    with gr.Column(scale=3):
                        page_info_md = gr.Markdown("")
                    next_btn = gr.Button("Next  ▶", scale=1)

        # ── events ────────────────────────────────────────────────────────────

        def load_dir_wrapper(rel_dir):
            return on_load_dir(dir_map.get(rel_dir, rel_dir))

        def filter_wrapper(filter_label, df, rel_dir):
            return on_filter_change(filter_label, df, dir_map.get(rel_dir, rel_dir))

        def prev_wrapper(df, rel_dir, page, filter_label):
            return on_page(df, dir_map.get(rel_dir, rel_dir), page, filter_label, -1)

        def next_wrapper(df, rel_dir, page, filter_label):
            return on_page(df, dir_map.get(rel_dir, rel_dir), page, filter_label, +1)

        dir_dropdown.change(
            fn=load_dir_wrapper, inputs=[dir_dropdown],
            outputs=[summary_md, df_state, score_table, grid_html, page_info_md, page_state],
        )
        filter_radio.change(
            fn=filter_wrapper, inputs=[filter_radio, df_state, dir_dropdown],
            outputs=[score_table, grid_html, page_info_md, page_state],
        )
        prev_btn.click(
            fn=prev_wrapper, inputs=[df_state, dir_dropdown, page_state, filter_radio],
            outputs=[grid_html, page_info_md, page_state],
        )
        next_btn.click(
            fn=next_wrapper, inputs=[df_state, dir_dropdown, page_state, filter_radio],
            outputs=[grid_html, page_info_md, page_state],
        )
        # Load data + inject lightbox JS on startup
        demo.load(
            fn=load_dir_wrapper, inputs=[dir_dropdown],
            outputs=[summary_md, df_state, score_table, grid_html, page_info_md, page_state],
            js=LIGHTBOX_JS,
        )

    return demo


if __name__ == "__main__":
    os.environ.setdefault("GRADIO_TEMP_DIR", "/nfs/data/1/xning/tmp/gradio")
    os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
