import streamlit as st
import numpy as np
import librosa
import soundfile as sf
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, Reverb, Limiter
import io

st.set_page_config(page_title="Sahil AI Music Studio", page_icon="🎙️", layout="centered")

st.title("🎙️ मेरा पर्सनल AI म्यूज़िक स्टूडियो")
st.write("अपनी सादी आवाज़ अपलोड करें—बिना क्रैश हुए, चंद सेकंड में प्रोफ़ेशनल मिक्स-मास्टर्ड गाना तैयार करें!")

uploaded_file = st.file_uploader("1. अपनी आवाज़ अपलोड करें (MP3, WAV, M4A, OGG)", type=["mp3", "wav", "m4a", "ogg"])
mood = st.selectbox("2. बीट और म्यूज़िक का मूड चुनें", [
    "Sad Hip-Hop (Emotional Lo-Fi & Heavy Bass)",
    "Haryanvi Desi Groove (Punchy Kick & Dholki Vibe)",
    "Chill Lo-Fi Sad Beat (Soft Piano & Ambient)"
])

def generate_fast_beat(duration_sec, sr, bpm, mood_type):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    beat = np.zeros_like(t)
    beat_interval = 60.0 / bpm
    step_time = beat_interval / 2.0
    total_steps = int(duration_sec / step_time)
    
    for step in range(total_steps):
        pos = int(step * step_time * sr)
        
        # Kick Pattern
        if step % 8 in [0, 3, 4] or ("Haryanvi" in mood_type and step % 8 == 6):
            k_len = min(int(0.25 * sr), len(t) - pos)
            kt = np.linspace(0, 0.25, k_len, endpoint=False)
            freq_env = 100 * np.exp(-30 * kt) + 40
            kick = np.sin(2 * np.pi * freq_env * kt) * np.exp(-14 * kt)
            beat[pos:pos+k_len] += kick * 0.9

        # Snare / Clap
        if step % 8 in [2, 6]:
            s_len = min(int(0.18 * sr), len(t) - pos)
            st_arr = np.linspace(0, 0.18, s_len, endpoint=False)
            noise = np.random.uniform(-1, 1, s_len) * np.exp(-25 * st_arr)
            tone = np.sin(2 * np.pi * 180 * st_arr) * np.exp(-30 * st_arr)
            beat[pos:pos+s_len] += (noise * 0.5 + tone * 0.4)

        # Hi-Hat
        h_len = min(int(0.04 * sr), len(t) - pos)
        ht = np.linspace(0, 0.04, h_len, endpoint=False)
        hat = np.random.uniform(-1, 1, h_len) * np.exp(-90 * ht)
        beat[pos:pos+h_len] += hat * 0.15

    # Sub-Bass & Chord Pads
    chords = [130.81, 103.83, 155.56, 116.54]
    bar_duration = beat_interval * 4
    for i, freq in enumerate(chords):
        start_idx = int((i * bar_duration) * sr)
        while start_idx < len(t):
            end_idx = min(start_idx + int(bar_duration * sr), len(t))
            seg_t = t[start_idx:end_idx] - t[start_idx]
            env = np.minimum(seg_t / 0.1, 1.0) * np.exp(-0.3 * seg_t)
            sub_bass = np.sin(2 * np.pi * (freq / 2) * seg_t) * 0.4
            pad = (np.sin(2 * np.pi * freq * seg_t) + 0.5 * np.sin(2 * np.pi * (freq * 1.5) * seg_t)) * 0.1 * env
            beat[start_idx:end_idx] += (sub_bass + pad)
            start_idx += int(bar_duration * 4 * sr)

    return beat

if uploaded_file is not None:
    st.audio(uploaded_file)
    if st.button("🎛️ तुरंत गाना बनाएं (Mix & Master)", type="primary"):
        with st.spinner("आवाज़ प्रोसेस हो रही है और बीट तैयार हो रही है... (बस कुछ सेकंड)"):
            try:
                y, sr = librosa.load(uploaded_file, sr=44100, mono=True)
                y, _ = librosa.effects.trim(y, top_db=25)
                
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(np.atleast_1d(tempo)[0])
                if bpm < 70 or bpm > 140:
                    bpm = 88.0

                # Studio Vocal Effects (Clean & Professional)
                vocal_board = Pedalboard([
                    HighpassFilter(cutoff_frequency_hz=80),
                    LowShelfFilter(cutoff_frequency_hz=200, gain_db=-2.0),
                    HighShelfFilter(cutoff_frequency_hz=5000, gain_db=3.5),
                    Compressor(threshold_db=-16, ratio=3.0, attack_ms=8, release_ms=100),
                    Reverb(room_size=0.3, damping=0.5, wet_level=0.15, dry_level=0.9),
                ])
                clean_vocal = vocal_board(y, sr)
                clean_vocal = clean_vocal / (np.max(np.abs(clean_vocal)) + 1e-6) * 0.85

                # Generate Beat matching vocal length
                duration = len(clean_vocal) / sr
                beat = generate_fast_beat(duration, sr, bpm, mood)
                beat = beat / (np.max(np.abs(beat)) + 1e-6) * 0.5

                # Final Mix & Master Limiter
                mixed = clean_vocal + beat
                master_board = Pedalboard([
                    Compressor(threshold_db=-8, ratio=2.0),
                    Limiter(threshold_db=-0.8)
                ])
                final_song = master_board(mixed, sr)

                buf = io.BytesIO()
                sf.write(buf, final_song, sr, format="WAV")
                buf.seek(0)

                st.success(f"✅ गाना सफलताਪूर्वक बन गया! (Tempo: {int(bpm)} BPM)")
                st.audio(buf, format="audio/wav")
                st.download_button(
                    label="⬇️ फाइनल गाना डाउनलोड करें (WAV)",
                    data=buf,
                    file_name="sahil_fast_mastered_song.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"त्रुटि: {e}")
