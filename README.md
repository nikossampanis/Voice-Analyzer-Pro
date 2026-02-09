# Chant FFT Analyzer PRO (Psaltic Edition)
**Developed by Nikolaos Sampanis**

A Streamlit application for **psaltic (Byzantine chant)** recordings.

## What the user does
1. Opens the app in a browser.
2. Uploads a **RAW** recording (WAV/FLAC/AIFF).
3. Chooses **Voice Type** (Baritone/Bass/Tenor) and **Target Space** (Hagia Sophia / Big Cathedral / Dry church).
4. Views:
   - FFT spectrum (frequency vs dB)
   - Spectrogram (time–frequency map)
   - Estimated f₀ (fundamental) for guidance
5. Downloads:
   - FFT PNG (printable)
   - Spectrogram PNG
   - EQ points CSV (Hz, dB, labels)
   - PDF report (printable)

## What the app outputs
- **On-screen**
  - Audio metrics: sample rate, duration, peak, RMS
  - FFT plot
  - Spectrogram plot
  - EQ suggestion list + explanations
- **Exports**
  - `chant_fft_spectrum.png`
  - `chant_spectrogram.png`
  - `chant_eq_points.csv`
  - `chant_fft_report.pdf`

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Notes / limitations
- EQ suggestions are **heuristics**: small moves (1–3 dB) and A/B listening are required.
- This is not a “perfect Hagia Sophia simulator”. For true acoustic matching you’d need **convolution reverb + a real IR** from the space.
