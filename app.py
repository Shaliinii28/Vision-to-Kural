# app/app.py
"""
Vision-to-Kural — Gradio Web Application
=========================================

Entry point for the Hugging Face Spaces deployment.
Run locally with:  python app.py
Deploy to HF:      git push space main

The app:
  1. Loads CLIP + projection head + FAISS index on startup.
  2. Accepts an image upload from the user.
  3. Encodes the image, retrieves top-K matching Kurals.
  4. Displays styled result cards with Tamil text, English
     explanation, commentary, chapter, Pal, and match score.

No Sarvam-2B is loaded at runtime — its work is pre-baked
into the FAISS index. Only CLIP runs live.
"""

import logging
import sys
import os
from pathlib import Path

# Ensure we can import inference.py and model.py from the same directory
sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from PIL import Image
from inference import load_models, retrieve_kurals, PAL_META

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("app")

# ─────────────────────────────────────────────
# Load models at startup (once)
# ─────────────────────────────────────────────
log.info("Initialising models…")
load_models(device="cpu")   # HF Spaces free tier = CPU only
log.info("Ready ✓")


# ─────────────────────────────────────────────
# HTML card renderer
# ─────────────────────────────────────────────
def render_kural_card(kural: dict, rank: int) -> str:
    """Render a single Kural result as a styled HTML card."""
    pm       = kural.get("pal_meta", {})
    color    = pm.get("color",  "#1A56A8")
    bg       = pm.get("bg",     "#EFF6FF")
    emoji    = pm.get("emoji",  "📜")
    pal_tamil = pm.get("tamil", "")

    score_pct = int(kural["score"] * 100)

    # Confidence label
    if score_pct >= 70:
        confidence = "Strong match"
        conf_color = "#065F46"
    elif score_pct >= 50:
        confidence = "Good match"
        conf_color = "#78350F"
    else:
        confidence = "Possible match"
        conf_color = "#374151"

    # Best commentary to show (prefer sp — Salamon Pappaiah, most modern)
    commentary = (
        kural.get("commentary_sp") or
        kural.get("commentary_mv") or
        kural.get("commentary_mk") or
        ""
    )
    commentary_html = ""
    if commentary and len(commentary) > 20:
        commentary_html = f"""
        <details style="margin-top:12px;">
          <summary style="cursor:pointer;color:{color};font-size:13px;
                          font-weight:600;user-select:none;">
            📖 Tamil Commentary (expand)
          </summary>
          <p style="margin-top:8px;font-size:13px;color:#374151;
                    line-height:1.7;font-family:'Noto Sans Tamil',serif;">
            {commentary}
          </p>
        </details>"""

    return f"""
    <div style="
      border:2px solid {color};
      border-radius:14px;
      padding:22px 24px;
      margin:14px 0;
      background:{bg};
      box-shadow:0 2px 8px rgba(0,0,0,0.06);
      font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    ">
      <!-- Header row -->
      <div style="display:flex;justify-content:space-between;
                  align-items:flex-start;flex-wrap:wrap;gap:8px;">
        <div>
          <span style="font-weight:800;color:{color};font-size:16px;">
            {emoji} Kural #{kural['number']}
          </span>
          <span style="color:#64748B;font-size:14px;margin-left:8px;">
            — {kural.get('chapter','?')}
          </span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
          <span style="background:{color};color:white;padding:3px 12px;
                       border-radius:999px;font-size:12px;font-weight:700;">
            {score_pct}%
          </span>
          <span style="color:{conf_color};font-size:12px;font-weight:600;">
            {confidence}
          </span>
        </div>
      </div>

      <!-- Tamil kural -->
      <p style="
        font-size:20px;
        margin:14px 0 8px;
        color:#0F172A;
        font-family:'Noto Sans Tamil','Latha',serif;
        line-height:1.8;
        letter-spacing:0.3px;
      ">{kural.get('kural_tamil','')}</p>

      <!-- English translation -->
      <p style="
        color:#334155;
        font-size:14.5px;
        font-style:italic;
        line-height:1.7;
        margin:8px 0 12px;
        padding-left:12px;
        border-left:3px solid {color};
      ">"{kural.get('explanation','')}"</p>

      <!-- Chapter / Pal meta -->
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#64748B;">
        <span>🗂 {kural.get('section','')}</span>
        <span>📚 {kural.get('pal','')} ({pal_tamil})</span>
      </div>

      {commentary_html}
    </div>
    """


def render_no_match() -> str:
    return """
    <div style="text-align:center;padding:40px;color:#64748B;">
      <p style="font-size:32px;">🙏</p>
      <p style="font-size:16px;">No strong match found for this filter.</p>
      <p style="font-size:13px;">Try selecting "All" from the Pal filter,
         or upload a clearer image of human activity.</p>
    </div>
    """


def render_loading() -> str:
    return """
    <div style="text-align:center;padding:40px;color:#64748B;">
      <p style="font-size:32px;animation:pulse 1.5s infinite;">⏳</p>
      <p>Finding your Kural…</p>
    </div>
    """


# ─────────────────────────────────────────────
# Main handler
# ─────────────────────────────────────────────
def find_kurals(image: Image.Image, pal_filter: str, top_k: int) -> str:
    """
    Called by Gradio on every image upload / button click.
    Returns an HTML string of result cards.
    """
    if image is None:
        return """
        <div style="text-align:center;padding:48px;color:#94A3B8;">
          <p style="font-size:48px;margin-bottom:12px;">📷</p>
          <p style="font-size:17px;font-weight:600;color:#475569;">Upload an image to begin</p>
          <p style="font-size:14px;color:#94A3B8;margin-top:8px;">
            Any photo of human activity, nature, or emotion works.
          </p>
        </div>"""

    try:
        results = retrieve_kurals(
            pil_image=image,
            top_k=int(top_k),
            pal_filter=pal_filter if pal_filter != "All" else None,
        )
    except Exception as e:
        log.error(f"Retrieval error: {e}", exc_info=True)
        return f"<p style='color:red;'>Error: {e}</p>"

    if not results:
        return render_no_match()

    cards = "".join(render_kural_card(r, i + 1) for i, r in enumerate(results))

    header = f"""
    <div style="font-size:13px;color:#64748B;margin-bottom:4px;padding:0 4px;">
      Found {len(results)} Kural{'s' if len(results) != 1 else ''} matching your image
    </div>"""

    return header + cards


# ─────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────
TITLE = "Vision-to-Kural | திருக்குறள் Ethical Image Captioning"
DESCRIPTION = """
## 🔍 Vision-to-Kural — திருக்குறள்

**Upload any photo. Discover the 2,000-year-old Tamil wisdom it embodies.**

The [Thirukkural](https://en.wikipedia.org/wiki/Kural), authored by Thiruvalluvar,
contains 1,330 moral couplets organised into 133 chapters across three books:
*Virtue (அறம்)*, *Wealth (பொருள்)*, and *Love (காமம்)*.

This system uses **CLIP + Sarvam-2B + FAISS** to find the Kural that best
resonates with the ethical or emotional essence of your image.
"""

FOOTER = """
<div style="text-align:center;margin-top:24px;padding:16px;
            border-top:1px solid #E2E8F0;color:#94A3B8;font-size:12px;">
  Built with CLIP · Sarvam-2B · FAISS · Gradio · Hosted free on 🤗 HF Spaces<br>
  <em>Thirukkural text is in the public domain.</em>
</div>
"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Tamil&display=swap');

.gradio-container { max-width: 1000px !important; margin: auto; }
#kural-output { min-height: 300px; }

/* Style the upload area */
.upload-container { border-radius: 12px !important; }

/* Animate result appearance */
@keyframes fadeIn {
  from { opacity:0; transform:translateY(8px); }
  to   { opacity:1; transform:translateY(0);   }
}
#kural-output > div { animation: fadeIn 0.3s ease forwards; }
"""

with gr.Blocks(
    title=TITLE,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="teal",
    ),
    css=CSS,
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row(equal_height=False):

        # ── Left column: inputs ──
        with gr.Column(scale=1, min_width=280):
            img_input = gr.Image(
                type="pil",
                label="📷 Upload Image",
                elem_classes=["upload-container"],
            )
            pal_filter = gr.Radio(
                choices=["All", "Virtue 🌿", "Wealth 💰", "Love ❤️"],
                value="All",
                label="Filter by Pal",
                info="அறம் (Virtue) · பொருள் (Wealth) · காமம் (Love)",
            )
            top_k_slider = gr.Slider(
                minimum=1, maximum=5, value=3, step=1,
                label="Number of Kurals to show",
            )
            submit_btn = gr.Button(
                "✨ Find Ethical Kural",
                variant="primary",
                size="lg",
            )

        # ── Right column: outputs ──
        with gr.Column(scale=1, min_width=360):
            output_html = gr.HTML(
                value="""
                <div style='text-align:center;padding:48px;color:#94A3B8;'>
                  <p style='font-size:48px;margin-bottom:12px;'>📷</p>
                  <p style='font-size:17px;font-weight:600;color:#475569;'>
                    Upload an image to begin
                  </p>
                  <p style='font-size:14px;margin-top:8px;'>
                    Any photo of human activity, nature, or emotion works.
                  </p>
                </div>""",
                elem_id="kural-output",
                label="Matched Kurals",
            )

    # ── Parse Pal filter from radio label ──
    def parse_pal(radio_val: str) -> str:
        if "Virtue" in radio_val:  return "Virtue"
        if "Wealth" in radio_val:  return "Wealth"
        if "Love"   in radio_val:  return "Love"
        return "All"

    def on_submit(image, pal_radio, top_k):
        return find_kurals(image, parse_pal(pal_radio), top_k)

    # ── Wire events ──
    submit_btn.click(
        fn=on_submit,
        inputs=[img_input, pal_filter, top_k_slider],
        outputs=output_html,
    )
    img_input.change(
        fn=on_submit,
        inputs=[img_input, pal_filter, top_k_slider],
        outputs=output_html,
    )

    # ── Example images ──
    examples_dir = Path(__file__).parent / "examples"
    if examples_dir.exists():
        example_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
        if example_files:
            gr.Examples(
                examples=[[str(f), "All", 3] for f in example_files[:4]],
                inputs=[img_input, pal_filter, top_k_slider],
                outputs=output_html,
                fn=on_submit,
                label="Try these examples",
            )

    gr.HTML(FOOTER)


# ─────────────────────────────────────────────
# Launch
# ─────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
    )
