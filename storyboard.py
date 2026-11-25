"""Prints the real engine output for the three storyboard scenarios (does not change playlist_core.py).
Run:  python storyboard_data.py
Copy the notes and song lists into your storyboard panels."""
from machine import *

df = load_data(CSV_PATH)
profiles = build_intent_profiles(df)
history = make_history(df)

def run(label, title, w, purpose=None, vibe=None):
    p = plan(df, profiles, title=title, purpose=purpose, vibe=vibe)
    target, pace, genres, note = p["target"], p["pace"], p["genres"], p["note"]
    pl, used_w = build_playlist(df, history, target, w, n=8, pace=pace, genre_filter=genres)
    print(f"\n=== {label} | title='{title}' | history weight={w:.0%} (used {used_w:.0%}) ===")
    print("NOTE:", note)
    cols = [c for c in ["track_name", "artists", GENRE_COL, "tempo", "intent_fit", "history_fit"] if c in pl.columns]
    print(pl[cols].round(2).to_string(index=False))

run("1 Region", "korean songs", 0.3)
run("2 Study + history, 30%", "study session", 0.3, vibe="calm")
run("2b Study + history, 100%", "study session", 1.0, vibe="calm")
run("3 Name only, history weight 0%", "gym workout", 0.0)
