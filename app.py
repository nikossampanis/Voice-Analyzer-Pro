# app.py — Chant FFT Analyzer PRO (Psaltic Edition)
# Developed by Nikolaos Sampanis

import io
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import soundfile as sf

from dataclasses import dataclass
from typing import List, Optional, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


st.set_page_config(
    page_title="Chant FFT Analyzer PRO",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
small { opacity: 0.85; }
.card {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 16px 16px 14px 16px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}
.badge {
  display:inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.06);
}
div.stDownloadButton > button {
  border-radius: 12px !important;
  padding: 0.55rem 0.9rem !important;
}
section[data-testid="stSidebar"] {
  border-right: 1px solid rgba(255,255,255,0.10);
}
footer { visibility: hidden; }
</style>"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@dataclass
class EQPoint:
    freq_hz: float
    gain_db: float
    label: str

@dataclass
class AnalysisResult:
    sr: int
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    f0_hz: Optional[float]
    fft_freqs: np.ndarray
    fft_mag_db: np.ndarray
    eq_points: List[EQPoint]
    notes: List[str]


def to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float64)
    return np.mean(y.astype(np.float64), axis=1)

def db20(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), eps))

def peak_dbfs(y: np.ndarray) -> float:
    return float(20.0 * math.log10(np.max(np.abs(y)) + 1e-12))

def rms_dbfs(y: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(y**2) + 1e-12))
    return float(20.0 * math.log10(r + 1e-12))

def trim_silence(y: np.ndarray, sr: int, threshold_db: float = -45.0) -> np.ndarray:
    if len(y) < sr:
        return y
    frame = int(0.02 * sr)
    hop = int(0.01 * sr)
    if frame <= 0 or hop <= 0:
        return y
    energies, idxs = [], []
    for start in range(0, len(y) - frame, hop):
        seg = y[start:start+frame]
        e = 20*np.log10(np.sqrt(np.mean(seg**2)) + 1e-12)
        energies.append(e); idxs.append(start)
    energies = np.array(energies)
    keep = energies > threshold_db
    if not np.any(keep):
        return y
    first = idxs[int(np.argmax(keep))]
    last = idxs[int(len(keep) - 1 - np.argmax(keep[::-1]))] + frame
    first = max(0, first - frame)
    last = min(len(y), last + frame)
    return y[first:last]

def compute_fft_segment(y: np.ndarray, sr: int, n_fft: int = 65536) -> Tuple[np.ndarray, np.ndarray]:
    if len(y) < 4096:
        raise ValueError("Audio too short for FFT analysis.")
    seg_len = min(len(y), n_fft)
    start = max(0, (len(y) - seg_len)//2)
    seg = y[start:start+seg_len]
    w = np.hanning(len(seg))
    X = np.fft.rfft(seg * w, n=n_fft)
    f = np.fft.rfftfreq(n_fft, d=1.0/sr)
    mag_db = db20(X)
    return f, mag_db

def band_median(mag_db: np.ndarray, f: np.ndarray, f1: float, f2: float) -> float:
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return float("nan")
    return float(np.median(mag_db[mask]))

def narrow_peak(mag_db: np.ndarray, f: np.ndarray, target: float, width: float = 1.5) -> float:
    mask = (f >= target - width) & (f <= target + width)
    if not np.any(mask):
        return float("-inf")
    return float(np.max(mag_db[mask]))

def estimate_f0(y: np.ndarray, sr: int) -> Optional[float]:
    if len(y) < sr // 2:
        return None
    win = min(len(y), int(1.2 * sr))
    start = max(0, (len(y) - win)//2)
    seg = y[start:start+win].astype(np.float64)
    seg = seg - np.mean(seg)
    seg = seg / (np.max(np.abs(seg)) + 1e-12)
    ac = np.correlate(seg, seg, mode="full")[len(seg)-1:]
    ac = ac / (np.max(ac) + 1e-12)
    fmin, fmax = 80.0, 300.0
    lag_min = int(sr / fmax)
    lag_max = int(sr / fmin)
    if lag_max <= lag_min + 2 or lag_max >= len(ac):
        return None
    search = ac[lag_min:lag_max]
    peak_idx = int(np.argmax(search)) + lag_min
    if peak_idx <= 0:
        return None
    f0 = sr / peak_idx
    if not (60.0 <= f0 <= 400.0):
        return None
    return float(f0)

def suggest_eq_chant(f: np.ndarray, mag_db: np.ndarray, sr: int, f0: Optional[float],
                     voice_type: str, cathedral_style: str):
    points, notes = [], []
    core = band_median(mag_db, f, 500, 2000)
    if not np.isfinite(core):
        core = float(np.median(mag_db))

    hpf = 70 if voice_type == "Bass" else 85 if voice_type == "Tenor" else 80
    points.append(EQPoint(float(hpf), 0.0, "HPF guide (use High-Pass filter)"))
    notes.append(f"High-Pass: περίπου {hpf} Hz για καθάρισμα rumble/πατημάτων χωρίς να χαθεί η βάση της φωνής.")

    hum50 = narrow_peak(mag_db, f, 50)
    hum100 = narrow_peak(mag_db, f, 100)
    if hum50 > core + 15:
        points.append(EQPoint(50.0, -12.0, "Notch (mains hum)"))
        notes.append("Ισχυρή αιχμή ~50 Hz (mains hum). Πρότεινε notch -10 έως -12 dB.")
    if hum100 > core + 12:
        points.append(EQPoint(100.0, -10.0, "Notch (harmonic hum)"))
        notes.append("Αρμονική βόμβου ~100 Hz. Πρότεινε δεύτερο notch αν χρειάζεται.")

    mud = band_median(mag_db, f, 150, 300)
    if np.isfinite(mud) and mud > core + 3:
        points.append(EQPoint(220.0, -2.0, "Mud cut"))
        notes.append("150–300 Hz: μούχλα/λάσπη. Μικρό cut (~-2 dB @ 220 Hz) καθαρίζει χωρίς να αδυνατίζει.")

    boxy = band_median(mag_db, f, 300, 600)
    if np.isfinite(boxy) and boxy > core + 3:
        points.append(EQPoint(420.0, -1.5, "Boxy cut"))
        notes.append("300–600 Hz: “κουτί/μύτη”. Ελαφρύ cut (~-1.5 dB @ 400–500 Hz) δίνει φυσικότητα.")

    harsh = band_median(mag_db, f, 2000, 4000)
    if np.isfinite(harsh) and harsh > core + 4:
        points.append(EQPoint(3000.0, -2.5, "Harsh cut"))
        notes.append("2–4 kHz: σκληρό/μεταλλικό. Cut ~2.8–3.2 kHz μειώνει κόπωση και κάνει τη χροιά πιο γλυκιά.")
    else:
        points.append(EQPoint(3000.0, -1.5, "Sweetness cut"))
        notes.append("Για “γλυκιά” ψαλτική: πολύ μικρό cut γύρω στα 3 kHz (≈ -1 έως -1.5 dB) συχνά βοηθά.")

    sib = band_median(mag_db, f, 6000, 9000)
    if np.isfinite(sib) and sib > core + 6:
        points.append(EQPoint(7500.0, -2.0, "Sibilance control"))
        notes.append("6–9 kHz: συριστικά. Προτίμησε de-esser ή μικρό cut (~-2 dB @ 7–8 kHz) αν ενοχλεί.")

    air = band_median(mag_db, f, 9000, 12000)
    if np.isfinite(air) and air < core - 10:
        points.append(EQPoint(10000.0, +0.8, "Air (subtle)"))
        notes.append("9–12 kHz: λίγο “αέρας” (+0.5 έως +1 dB) μόνο αν ακούγεται θαμπό.")

    if f0 is not None:
        notes.append(f"Εκτίμηση θεμελιώδους f₀ ≈ {f0:.1f} Hz (για κατανόηση βάσης/αρμονικών).")

    seen, uniq = set(), []
    for p in points:
        key = (p.label, int(round(p.freq_hz)))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq, notes

def plot_fft(freqs: np.ndarray, mag_db: np.ndarray, fmax: int = 12000) -> plt.Figure:
    fig = plt.figure()
    mask = freqs <= fmax
    plt.plot(freqs[mask], mag_db[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB, relative)")
    plt.title("FFT Spectrum (windowed segment)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    return fig

def plot_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 256, fmax: int = 12000):
    if len(y) < n_fft + hop:
        return None
    w = np.hanning(n_fft)
    frames = 1 + (len(y) - n_fft)//hop
    spec = np.empty((n_fft//2 + 1, frames), dtype=np.float64)
    for i in range(frames):
        start = i * hop
        frame = y[start:start+n_fft] * w
        X = np.fft.rfft(frame)
        spec[:, i] = db20(X)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    mask = freqs <= fmax
    fig = plt.figure()
    plt.imshow(
        spec[mask, :],
        aspect="auto",
        origin="lower",
        extent=[0, frames*hop/sr, freqs[mask][0], freqs[mask][-1]]
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Spectrogram (dB)")
    plt.colorbar(label="dB")
    return fig

def fig_to_png_bytes(fig: plt.Figure, dpi: int = 220) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()

def eq_points_to_csv(points: List[EQPoint]) -> bytes:
    lines = ["freq_hz,gain_db,label"]
    for p in points:
        lines.append(f"{p.freq_hz:.2f},{p.gain_db:.2f},{p.label}")
    return ("\n".join(lines)).encode("utf-8")

def build_pdf_report(analysis: AnalysisResult, fft_png: bytes, spec_png: Optional[bytes]) -> bytes:
    styles = getSampleStyleSheet()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    story = []
    story.append(Paragraph("<b>Chant FFT Analyzer PRO – Report</b>", styles["Title"]))
    story.append(Paragraph("Developed by Nikolaos Sampanis", styles["Normal"]))
    story.append(Spacer(1, 10))

    meta = [
        ["Sample rate", f"{analysis.sr} Hz"],
        ["Duration", f"{analysis.duration_s:.2f} s"],
        ["Peak", f"{analysis.peak_dbfs:.1f} dBFS"],
        ["RMS", f"{analysis.rms_dbfs:.1f} dBFS"],
        ["Estimated f₀", f"{analysis.f0_hz:.1f} Hz" if analysis.f0_hz else "—"],
    ]
    t = Table(meta, colWidths=[140, 360])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.8, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>FFT Spectrum</b>", styles["Heading2"]))
    fft_img = RLImage(io.BytesIO(fft_png))
    fft_img.drawWidth = 480
    fft_img.drawHeight = 240
    story.append(fft_img)
    story.append(Spacer(1, 10))

    if spec_png:
        story.append(Paragraph("<b>Spectrogram</b>", styles["Heading2"]))
        sp_img = RLImage(io.BytesIO(spec_png))
        sp_img.drawWidth = 480
        sp_img.drawHeight = 240
        story.append(sp_img)
        story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Suggested EQ Points</b>", styles["Heading2"]))
    eq_rows = [["Freq (Hz)", "Gain (dB)", "Label"]]
    for p in analysis.eq_points:
        eq_rows.append([f"{p.freq_hz:.0f}", f"{p.gain_db:+.1f}", p.label])
    eq_table = Table(eq_rows, colWidths=[100, 100, 300])
    eq_table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.8, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTSIZE", (0,0), (-1,-1), 9),
    ]))
    story.append(eq_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Notes</b>", styles["Heading2"]))
    for n in analysis.notes[:12]:
        story.append(Paragraph("• " + n, styles["Normal"]))

    doc.build(story)
    return buf.getvalue()


st.sidebar.markdown(
    """<div class="card">
      <div style="font-size:1.05rem;"><b>🎙️ Chant FFT Analyzer PRO</b></div>
      <div style="margin-top:6px;" class="badge">Developed by Nikolaos Sampanis</div>
      <div style="margin-top:10px; font-size:0.93rem; line-height:1.35;">
        Upload a WAV/FLAC and get:
        <ul style="margin:8px 0 0 16px;">
          <li>FFT spectrum + Spectrogram</li>
          <li>Estimated f₀ (fundamental)</li>
          <li>Chant-focused EQ suggestions</li>
          <li>PNG + CSV + PDF exports</li>
        </ul>
      </div>
    </div>""",
    unsafe_allow_html=True
)

voice_type = st.sidebar.selectbox("Voice Type", ["Baritone", "Bass", "Tenor"], index=0)
cathedral_style = st.sidebar.selectbox("Target Space (guidance)", ["Hagia Sophia (Cathedral)", "Big Cathedral", "Small Church (dry)"], index=0)
st.sidebar.markdown("---")
fft_size = st.sidebar.selectbox("FFT Size", [16384, 32768, 65536, 131072], index=2)
fmax = st.sidebar.slider("Max Frequency Shown (Hz)", 2000, 16000, 12000, step=500)
trim_db = st.sidebar.slider("Trim silence threshold (dB)", -60, -25, -45, step=1)
show_spectrogram = st.sidebar.toggle("Show Spectrogram", value=True)
spec_fmax = st.sidebar.slider("Spectrogram max freq (Hz)", 2000, 16000, 12000, step=500)
st.sidebar.markdown("---")
st.sidebar.info("Tip: Για ψαλτική, συνήθως θέλουμε μικρά cuts σε 150–300 Hz (mud) και 2–4 kHz (harsh). Απόφυγε μεγάλα boosts.")

st.markdown(
    """<div class="card">
      <div style="display:flex;align-items:baseline;gap:12px;">
        <h1 style="margin:0;">🎙️ Chant FFT Analyzer PRO</h1>
        <div class="badge">Psaltic Edition</div>
      </div>
      <div style="margin-top:6px;font-size:1.02rem;opacity:0.9;">
        Upload your <b>RAW</b> recording, visualize FFT, and get chant-focused EQ guidance.
      </div>
      <div style="margin-top:6px;opacity:0.8;">
        <small><b>Developed by Nikolaos Sampanis</b></small>
      </div>
    </div>""",
    unsafe_allow_html=True
)

st.write("")
uploaded = st.file_uploader("⬆️ Upload WAV / FLAC / AIFF", type=["wav", "flac", "aiff", "aif"])
if uploaded is None:
    st.info("Ανέβασε ένα RAW WAV για να ξεκινήσει η ανάλυση (χωρίς εφέ).")
    st.stop()

data, sr = sf.read(uploaded)
y = to_mono(np.array(data))
duration = len(y) / sr
orig_peak = peak_dbfs(y)
orig_rms = rms_dbfs(y)

y_trim = trim_silence(y, sr, threshold_db=float(trim_db))
y_an = y_trim / (np.max(np.abs(y_trim)) + 1e-12)

f0 = estimate_f0(y_an, sr)
freqs, mag_db = compute_fft_segment(y_an, sr, n_fft=int(fft_size))
eq_points, notes = suggest_eq_chant(freqs, mag_db, sr, f0=f0, voice_type=voice_type, cathedral_style=cathedral_style)

analysis = AnalysisResult(sr=sr, duration_s=duration, peak_dbfs=orig_peak, rms_dbfs=orig_rms, f0_hz=f0,
                         fft_freqs=freqs, fft_mag_db=mag_db, eq_points=eq_points, notes=notes)

tab_overview, tab_fft, tab_eq, tab_exports, tab_howto = st.tabs(["📌 Overview", "📈 FFT & Spectrogram", "🎚️ EQ Guidance", "📦 Exports", "🧠 How to use (Audacity)"])

with tab_overview:
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 Audio Metrics")
        st.write(f"**Sample rate:** {sr} Hz")
        st.write(f"**Duration:** {duration:.2f} s")
        st.write(f"**Peak:** {orig_peak:.1f} dBFS")
        st.write(f"**RMS:** {orig_rms:.1f} dBFS")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎼 Fundamental (f₀)")
        if f0:
            st.write(f"**Estimated f₀:** {f0:.1f} Hz")
            st.caption("Εκτίμηση για καθοδήγηση (όχι tuner).")
        else:
            st.write("—")
            st.caption("Δεν βρέθηκε αξιόπιστο f₀ (ίσως πολύ μικρό δείγμα ή θόρυβος).")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎧 Playback")
        st.audio(uploaded)
        st.caption("Ιδανικά ανέβασε 20–60″ με ήρεμο + forte.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_fft:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 FFT Spectrum")
    fig_fft = plot_fft(freqs, mag_db, fmax=fmax)
    st.pyplot(fig_fft, clear_figure=True)
    st.markdown("</div>", unsafe_allow_html=True)

    fig_sp = None
    if show_spectrogram:
        st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
        st.subheader("🗺️ Spectrogram")
        fig_sp = plot_spectrogram(y_an, sr, n_fft=2048, hop=256, fmax=spec_fmax)
        if fig_sp is not None:
            st.pyplot(fig_sp, clear_figure=True)
        else:
            st.info("Το δείγμα είναι πολύ μικρό για spectrogram με τα τρέχοντα settings.")
        st.markdown("</div>", unsafe_allow_html=True)

with tab_eq:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎚️ Chant-focused EQ Suggestions")
    st.caption("Οι προτάσεις είναι heuristics. Επιβεβαίωσε με A/B ακρόαση.")
    for i, p in enumerate(eq_points, 1):
        if "HPF guide" in p.label:
            st.write(f"**{i}. High-Pass (guide)** → ~**{p.freq_hz:.0f} Hz**")
        else:
            st.write(f"**{i}. {p.label}** → **{p.freq_hz:.0f} Hz**, **{p.gain_db:+.1f} dB**")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:14px;">', unsafe_allow_html=True)
    st.subheader("📌 Explanations")
    for n in notes:
        st.write("• " + n)
    st.markdown("</div>", unsafe_allow_html=True)

with tab_exports:
    fft_png = fig_to_png_bytes(fig_fft, dpi=240)
    spec_png = fig_to_png_bytes(fig_sp, dpi=240) if fig_sp is not None else None
    csv_bytes = eq_points_to_csv(eq_points)
    pdf_bytes = build_pdf_report(analysis, fft_png=fft_png, spec_png=spec_png)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("⬇️ Exports")
    st.download_button("Download FFT Spectrum (PNG)", data=fft_png, file_name="chant_fft_spectrum.png", mime="image/png")
    if spec_png:
        st.download_button("Download Spectrogram (PNG)", data=spec_png, file_name="chant_spectrogram.png", mime="image/png")
    st.download_button("Download EQ Points (CSV)", data=csv_bytes, file_name="chant_eq_points.csv", mime="text/csv")
    st.download_button("Download PDF Report", data=pdf_bytes, file_name="chant_fft_report.pdf", mime="application/pdf")
    st.markdown("</div>", unsafe_allow_html=True)

with tab_howto:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧠 How to apply in Audacity (workflow)")
    st.write(
        st.write("""
**1) Backup του RAW** (Save Project As…)

**2) High-Pass Filter:** ~70–90 Hz

**3) EQ:** Filter Curve EQ → βάλε κουκίδες στις συχνότητες του CSV και κάνε μικρά cuts (1–3 dB)

**4) Compressor (ψαλτικός):** threshold ~-18 dB, ratio 2:1, attack 0.15–0.20s, release 1.5–2.0s

**5) Cathedral space:** μετά τα παραπάνω (pre-delay + decay + low wet + filtering)

**6) Normalize τελευταίο:** peak -1.0 dB
""")

    st.markdown("</div>", unsafe_allow_html=True)

st.success("Analysis complete ✅  Developed by Nikolaos Sampanis")
