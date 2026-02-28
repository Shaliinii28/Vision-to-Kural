
# Vision-to-Kural 🔍 — திருக்குறள்

**Upload any photo. Discover the 2,000-year-old Tamil wisdom it embodies.**

### What it does
Standard image captioning says *"A person helping an old man."*  
This says *"Kural #211 — The world survives because of those who give."*

### How it works
1. Your image is encoded by **CLIP ViT-L/14** → 768-dim vector
2. A trained **MLP** projects it into a 512-dim shared ethical space
3. **FAISS** searches 1,330 Kural vectors and returns the top matches
4. Results show Tamil text, transliteration, English meaning, and commentary

### Models
- `openai/clip-vit-large-patch14` — image encoding (runs live)
- `sarvamai/sarvam-2b-v0.5` — Tamil text encoding (offline, pre-indexed)

### About the Thirukkural
The Thirukkural (திருக்குறள்), by Thiruvalluvar (~3rd century BCE),
contains 1,330 two-line couplets on Virtue, Wealth, and Love.
It is one of the most translated texts in the world and is considered
the moral backbone of Tamil civilisation.

---
*Built with ❤️ using CLIP · Sarvam-2B · FAISS · Gradio*
