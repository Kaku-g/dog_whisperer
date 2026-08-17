"""
Dog Whisperer — Streamlit UI, all-Google-AI version
-----------------------------------------------------
Same UI/flow as the original app.py, but uses Gemini for BOTH steps
instead of Gemini + ElevenLabs:
  1. Vision (gemini-2.5-flash): photo -> mood label + first-person
     "inner monologue", returned together as JSON in one call.
  2. Native TTS (gemini-3.1-flash-tts-preview): monologue -> speech.
     The voice + delivery style are picked automatically from the
     inferred mood (with a manual override available in the UI).

Only one API key needed: GEMINI_API_KEY.

Run:
  pip install -r requirements.txt
  cp .env.example .env   # fill in GEMINI_API_KEY
  streamlit run streamlit_app.py
"""

import os
import json
import wave
import io
import sys

from dotenv import load_dotenv
import streamlit as st

from google import genai
from google.genai import types

# Import utils from the utils folder
from utils import snowflake_log as sfl

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

VISION_MODEL = "gemini-3.5-flash-lite"
TTS_MODEL = "gemini-3.1-flash-tts-preview"

# Mood -> (Gemini prebuilt voice, natural-language delivery instruction).
# Gemini TTS is "controllable", so we steer tone with plain English
# instead of juggling separate voice_ids per service.
MOOD_PRESETS = {
    "happy": {"voice": "Puck", "delivery": "Say brightly and happily"},
    "excited": {"voice": "Puck", "delivery": "Say excitedly and a little goofily"},
    "sleepy": {"voice": "Charon", "delivery": "Say in a slow, drowsy, contented voice"},
    "calm": {"voice": "Charon", "delivery": "Say calmly and thoughtfully, like a wise old dog"},
    "grumpy": {"voice": "Kore", "delivery": "Say grumpily and a bit annoyed"},
    "dramatic": {"voice": "Kore", "delivery": "Say dramatically, as if deeply offended"},
    "scared": {"voice": "Kore", "delivery": "Say nervously and a little anxiously"},
    "curious": {"voice": "Puck", "delivery": "Say with playful curiosity"},
}
DEFAULT_MOOD = "happy"

MONOLOGUE_PROMPT = f"""You are inside the mind of the animal in this photo.
Look at its breed/species, posture, facial expression, and surroundings.

Return ONLY a JSON object (no markdown fences, no extra text) with exactly
these two keys:
  "mood": one word from this list: {list(MOOD_PRESETS.keys())}
          — pick whichever best matches the animal's apparent mood.
  "monologue": a short first-person "inner monologue" (2-4 sentences,
          funny, in-character, PG-rated) of what it is thinking RIGHT NOW.
          Do not describe the image. Do not break character. No hashtags.
"""


@st.cache_resource
def get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def generate_monologue(image_bytes: bytes, mime_type: str, extra_context: str) -> dict:
    """Returns {"mood": str, "monologue": str}. Falls back to DEFAULT_MOOD
    if the model's mood pick isn't one we have a voice preset for."""
    client = get_client()
    prompt = MONOLOGUE_PROMPT
    if extra_context:
        prompt += f"\nExtra context from the owner: {extra_context}"

    response = client.models.generate_content(
        model=VISION_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    try:
        data = json.loads(response.text)
        mood = str(data.get("mood", "")).strip().lower()
        monologue = str(data.get("monologue", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        # Model didn't return clean JSON — treat the whole thing as the
        # monologue and fall back to a default mood rather than crashing.
        mood, monologue = "", response.text.strip()

    if mood not in MOOD_PRESETS:
        mood = DEFAULT_MOOD
    if not monologue:
        monologue = "..."

    return {"mood": mood, "monologue": monologue}


def synthesize_speech(text: str, voice_name: str, delivery: str) -> bytes:
    client = get_client()
    scripted_line = f"{delivery}: {text}"

    response = client.models.generate_content(
        model=TTS_MODEL,
        contents=scripted_line,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        ),
    )

    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    return _pcm_to_wav_bytes(pcm_data)


def _pcm_to_wav_bytes(pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Look & feel — minimal palette, two deliberate typefaces, small custom
# components. See .streamlit/config.toml for the base color theme; this CSS
# layers in fonts + a couple of signature touches (the mood pill, the
# "leash-line" divider) that Streamlit's theme system can't do on its own.
# ---------------------------------------------------------------------------

MOOD_COLORS = {
    "happy": "#3D5A6C", "excited": "#C97B3B", "sleepy": "#8A8F84",
    "calm": "#3D5A6C", "grumpy": "#A6462B", "dramatic": "#A6462B",
    "scared": "#8A8F84", "curious": "#C97B3B",
}


def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 600 !important; letter-spacing: -0.01em; }
        .stButton>button, .stDownloadButton>button {
            border-radius: 8px; font-weight: 500; font-family: 'Inter', sans-serif;
        }
        [data-testid="stMetricValue"] { font-family: 'Fraunces', serif; }
        .dw-eyebrow {
            font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; letter-spacing: 0.14em;
            text-transform: uppercase; color: #8A8F84; margin-bottom: 0.3rem;
        }
        .dw-hero-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 2.3rem; color: #1E2320; line-height: 1.05; }
        .dw-hero-sub { color: #6B7268; font-size: 0.98rem; margin-top: 0.35rem; }
        .dw-pill {
            display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.32rem 0.8rem;
            border-radius: 999px; background: #F0EDE4; font-family: 'IBM Plex Mono', monospace;
            font-size: 0.78rem; letter-spacing: 0.02em; color: #1E2320;
        }
        .dw-dot { width: 0.5rem; height: 0.5rem; border-radius: 50%; flex-shrink: 0; }
        .dw-divider { display: flex; align-items: center; gap: 0.7rem; margin: 1.6rem 0; }
        .dw-divider-line { flex: 1; border-top: 1px dashed #D8D4C8; }
        .dw-divider-dot { width: 0.4rem; height: 0.4rem; border-radius: 50%; background: #3D5A6C; flex-shrink: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, eyebrow: str, subtitle: str):
    st.markdown(
        f"""
        <div style="padding: 0.25rem 0 0.75rem 0;">
            <div class="dw-eyebrow">{eyebrow}</div>
            <div class="dw-hero-title">{title}</div>
            <div class="dw-hero-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def leash_divider():
    st.markdown(
        """
        <div class="dw-divider">
            <div class="dw-divider-line"></div>
            <div class="dw-divider-dot"></div>
            <div class="dw-divider-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def mood_pill(mood: str, voice: str):
    color = MOOD_COLORS.get(mood, "#3D5A6C")
    st.markdown(
        f"""
        <div class="dw-pill">
            <span class="dw-dot" style="background:{color};"></span>
            {mood} &middot; voice {voice}
        </div>
        """,
        unsafe_allow_html=True,
    )


def whisperer_tab():
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
        with col2:
            camera_shot = st.camera_input("...or take one live")

        image_file = uploaded or camera_shot

        extra_context = st.text_input(
            "Optional context (e.g. 'just got back from the vet', 'staring at the mailman')"
        )

        auto_mood = st.checkbox("Auto-pick voice from the photo's mood", value=True)
        manual_mood = None
        if not auto_mood:
            manual_mood = st.selectbox("Voice personality", list(MOOD_PRESETS.keys()))

        go = st.button("Translate thoughts 🎙️", use_container_width=True)

    if image_file and go:
        image_bytes = image_file.getvalue()
        mime_type = image_file.type or "image/jpeg"

        with st.spinner("Reading the room (and the ears)..."):
            result = generate_monologue(image_bytes, mime_type, extra_context)

        mood = result["mood"] if auto_mood else manual_mood
        monologue = result["monologue"]
        preset = MOOD_PRESETS[mood]

        leash_divider()

        with st.container(border=True):
            img_col, text_col = st.columns([1, 1.4])
            with img_col:
                st.image(image_bytes, use_container_width=True)
            with text_col:
                if auto_mood:
                    mood_pill(mood, preset["voice"])
                st.markdown(f"#### 💭 {monologue}")

                with st.spinner("Giving it a voice..."):
                    wav_bytes = synthesize_speech(monologue, preset["voice"], preset["delivery"])

                st.audio(wav_bytes, format="audio/wav")
                st.download_button("Download audio", wav_bytes, file_name="dog_thoughts.wav")


def pet_log_tab():
    st.caption("Meals, weight, and walks — logged to Snowflake so you can spot trends over time.")

    if "sf_ready" not in st.session_state:
        try:
            sfl.init_schema()
            st.session_state["sf_ready"] = True
        except Exception as e:
            st.error(f"Couldn't connect to Snowflake: {e}")
            st.info("Check SNOWFLAKE_* values in your .env file.")
            return

    # --- lightweight owner identity ---
    # NOTE: this is a hackathon-speed stand-in for real auth. It scopes each
    # person's pets/logs to whatever name they type — no password, so it's
    # only as private as the name is hard to guess. Swap for st.login() +
    # st.user.email if you want real per-user login before a public deploy.
    owner_id = st.text_input("Your username (used to keep your pets separate from others)", key="owner_id")
    if not owner_id:
        st.info("Enter your username above to see or add your pets.")
        return

    # --- pet selection ---
    # Use session state to track newly added pets without full rerun
    if "pet_added" not in st.session_state:
        st.session_state.pet_added = False

    with st.expander("➕ Add a pet"):
        new_name = st.text_input("Name", key="new_pet_name")
        new_species = st.selectbox("Species", ["dog", "cat", "lizard", "ferret", "other"], key="new_pet_species")
        new_breed = st.text_input("Breed (optional)", key="new_pet_breed")
        if st.button("Add pet") and new_name:
            sfl.add_pet(owner_id, new_name, new_species, new_breed)
            # Clear the cache for list_pets to fetch updated list
            st.cache_data.clear()
            st.session_state.pet_added = True
            st.success(f"Added {new_name}!")

    # Fetch pets list (uses cache unless cleared above)
    pets = sfl.list_pets(owner_id)
    pet_names = {p["name"]: p["pet_id"] for p in pets}

    if not pet_names:
        st.info("No pets logged yet — add one above to start logging.")
        return

    selected_name = st.selectbox("Pet", list(pet_names.keys()))
    pet_id = pet_names[selected_name]

    with st.container(border=True):
        log_type = st.radio("Log", ["Meal", "Weight", "Walk"], horizontal=True)

        if log_type == "Meal":
            food_item = st.text_input("Food item")
            amount = st.number_input("Amount (grams)", min_value=0.0, step=10.0)
            notes = st.text_input("Notes (optional)", key="meal_notes")
            if st.button("Log meal") and food_item:
                sfl.log_meal(pet_id, food_item, amount or None, notes)
                st.cache_data.clear()
                st.success("Meal logged. Refreshing data...")
                st.rerun()

        elif log_type == "Weight":
            weight = st.number_input("Weight (kg)", min_value=0.0, step=0.1)
            if st.button("Log weight") and weight:
                sfl.log_weight(pet_id, weight)
                st.cache_data.clear()
                st.success("Weight logged. Refreshing data...")
                st.rerun()

        else:
            duration = st.number_input("Duration (minutes)", min_value=0.0, step=5.0)
            notes = st.text_input("Notes (optional)", key="walk_notes")
            if st.button("Log walk") and duration:
                sfl.log_walk(pet_id, duration, notes)
                st.cache_data.clear()
                st.success("Walk logged. Refreshing data...")
                st.rerun()

    leash_divider()
    st.markdown("#### 📈 Trends")

    weight_trend = sfl.get_weight_trend(pet_id)
    food_trend = sfl.daily_food_total(pet_id)
    walk_trend = sfl.daily_walk_total(pet_id)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest weight", f"{weight_trend[-1]['weight_kg']:.1f} kg" if weight_trend else "—")
    if food_trend:
        avg_food = sum(r["total_grams"] or 0 for r in food_trend) / len(food_trend)
        m2.metric("Avg. daily food", f"{avg_food:.0f} g")
    else:
        m2.metric("Avg. daily food", "—")
    m3.metric("Meals logged (7d)", len(sfl.get_meal_logs(pet_id, days=7)))
    if walk_trend:
        total_walks = sum(r["walk_count"] or 0 for r in walk_trend)
        m4.metric("Walks logged (7d)", int(total_walks))
    else:
        m4.metric("Walks logged (7d)", 0)

    with st.container(border=True):
        if weight_trend:
            st.caption("Weight over time")
            st.line_chart({row["logged_at"]: row["weight_kg"] for row in weight_trend})
        else:
            st.caption("No weight entries yet.")

    with st.container(border=True):
        if food_trend:
            st.caption("Daily food total (grams)")
            st.bar_chart({str(row["log_date"]): row["total_grams"] for row in food_trend})
        else:
            st.caption("No meal entries yet.")

    with st.container(border=True):
        if walk_trend:
            st.caption("Daily walk duration (minutes)")
            st.bar_chart({str(row["log_date"]): row["total_duration"] for row in walk_trend})
        else:
            st.caption("No walk entries yet.")

    recent_meals = sfl.get_meal_logs(pet_id, days=7)
    if recent_meals:
        with st.container(border=True):
            st.caption("Last 7 days of meals")
            st.table(recent_meals)

    recent_walks = sfl.get_walk_logs(pet_id, days=7)
    if recent_walks:
        with st.container(border=True):
            st.caption("Last 7 days of walks")
            st.table(recent_walks)


def main():
    st.set_page_config(page_title="Dog Whisperer", page_icon="🐾", layout="centered")
    inject_theme()
    hero(
        "🐾 Dog Whisperer",
        "for dogs (and cats, lizards, ferrets)",
        "Gemini reads the room and says it out loud. Snowflake remembers everything else.",
    )

    if not GEMINI_API_KEY:
        st.warning("Add GEMINI_API_KEY to a .env file (see .env.example) before running this for real.")

    tab1, tab2 = st.tabs(["🎙️ Whisperer", "📋 Pet Log"])
    with tab1:
        whisperer_tab()
    with tab2:
        pet_log_tab()


if __name__ == "__main__":
    main()
