

"""Data prep + scoring for the intent-aware playlist prototype.
Real data: Spotify tracks dataset (genre column). History is SIMULATED from chosen genres.
"""
import re
import pandas as pd

CSV_PATH = r"C:\Users\alish\OneDrive\Documents\YEAR 3\IIS\New folder\spotify-tracks-dataset.csv"  
GENRE_COL = "track_genre"   

RAW_FEATURES = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "tempo"]
# Scoring uses tempo_n (BPM/200, 0-1) so BPM does not swamp the 0-1 features. Raw BPM stays in the data.
FEATURES = ["energy", "valence", "danceability", "acousticness", "instrumentalness", "tempo_n"]
WEIGHTS = {"energy": 1.0, "valence": 1.0, "danceability": 1.0,
           "acousticness": 0.5,   # half weight: r = -0.79 with energy
           "instrumentalness": 1.0, "tempo_n": 1.0}
 
# Which genres stand for each intent. 
INTENT_GENRES = {
    "study":   ["study", "classical", "ambient", "piano", "sleep"],
    "workout": ["edm", "funk", "drum-and-bass", "breakbeat", "party"],
    "party":   ["party", "dance", "club", "reggaeton"],
    "relax":   ["chill", "acoustic", "sleep", "new-age"],
    "drive":   ["rock", "indie", "alt-rock", "hard-rock"],
}


GENRE_GROUPS = {
    "Rock": ["alt-rock", "alternative", "emo", "goth", "grunge", "hard-rock", "indie", "power-pop", "psych-rock",
             "punk", "punk-rock", "rock", "rock-n-roll", "rockabilly", "ska"],
    "Metal & hardcore": ["black-metal", "death-metal", "grindcore", "hardcore", "heavy-metal", "industrial", "metal", "metalcore"],
    "Electronic": ["breakbeat", "chicago-house", "deep-house", "detroit-techno", "drum-and-bass", "dub", "dubstep", "edm",
                   "electro", "electronic", "garage", "hardstyle", "house", "idm", "minimal-techno", "progressive-house",
                   "techno", "trance", "trip-hop"],
    "Dance & party": ["club", "dance", "disco", "groove", "j-dance", "party"],
    "Pop": ["indie-pop", "pop", "pop-film", "synth-pop"],
    "Hip-hop, R&B & soul": ["funk", "gospel", "hip-hop", "r-n-b", "soul"],
    "Acoustic, folk & jazz": ["acoustic", "bluegrass", "blues", "country", "folk", "guitar", "honky-tonk", "jazz",
                              "singer-songwriter", "songwriter"],
    "Calm & instrumental": ["ambient", "chill", "classical", "new-age", "opera", "piano", "sleep", "study"],
    "Kids, mood & novelty": ["children", "comedy", "disney", "happy", "kids", "romance", "sad", "show-tunes"],
    "Latin & Brazilian": ["brazil", "dancehall", "forro", "latin", "latino", "mpb", "pagode", "reggae", "reggaeton",
                          "salsa", "samba", "sertanejo", "spanish", "tango"],
    "Asia & Middle East": ["anime", "cantopop", "indian", "iranian", "j-idol", "j-pop", "j-rock", "k-pop", "malay",
                           "mandopop", "turkish"],
    "Europe, Africa & world": ["afrobeat", "british", "french", "german", "swedish", "world-music"],
}
# 'International' = the three regional groups together
GENRE_GROUPS_ALIASES = {"International": ["Latin & Brazilian", "Asia & Middle East", "Europe, Africa & world"]}


def genres_for(group_names):
    """Group names (or the alias 'International') -> flat list of genre names."""
    out = []
    for g in group_names:
        for name in GENRE_GROUPS_ALIASES.get(g, [g]):
            out += GENRE_GROUPS[name]
    return out

# Words a user might type  
INTENT_KEYWORDS = {
    "study": ["study", "revision", "revise", "focus", "work", "read", "concentrate"],
    "workout": ["gym", "workout", "run", "running", "exercise", "lift"],
    "party": ["party", "dance", "celebrate", "pre-drinks"],
    "relax": ["relax", "chill", "calm", "sleep", "unwind"],
    "drive": ["drive", "driving", "road", "commute"],
}


def load_data(path=CSV_PATH):
    df = pd.read_csv(path)
    keep = [c for c in ["track_id", "track_name", "artists", GENRE_COL] if c in df.columns] + RAW_FEATURES
    id_cols = ["track_id"] if "track_id" in df.columns else [c for c in ["track_name", "artists"] if c in df.columns]
    # Keep a track in every genre it appears in. The same song listed
    df = df[keep].dropna().drop_duplicates(subset=id_cols + [GENRE_COL])
    df = df[df["tempo"] > 0]                           # tempo 0 = beat not detected  
    df["tempo_n"] = (df["tempo"] / 200).clip(0, 1)   
    return df.reset_index(drop=True)

 
PROFILE_MODE = "prototypes"


def build_intent_profiles(df):
    """Target profile(s) for each intent, derived from the genre means in the data.
    Value is a dict (centroid mode) or a list of dicts (prototypes mode)."""
    profiles = {}
    for intent, genres in INTENT_GENRES.items():
        present = [g for g in genres if g in set(df[GENRE_COL])]
        if present and PROFILE_MODE == "prototypes":
            profiles[intent] = [df[df[GENRE_COL] == g][FEATURES].mean().to_dict() for g in present]
        elif present:
            profiles[intent] = df[df[GENRE_COL].isin(present)][FEATURES].mean().to_dict()
        else:
            print(f"WARNING: none of {genres} found in genre column for intent '{intent}'")
    return profiles


def detect_intent(text):
    words = text.lower().replace(",", " ").split()
    for intent, keys in INTENT_KEYWORDS.items():
        if any(k in words for k in keys):
            return intent
    return None   # -> caller falls back to history only and tells the user


def make_history(df, genres=("pop", "dance", "hip-hop", "rock"), n=25, seed=0):
    """SIMULATED listening history sampled from the chosen genres."""
    pool = df[df[GENRE_COL].isin(genres)]
    return pool.sample(n=min(n, len(pool)), random_state=seed)


def similarity(df, target, weights=WEIGHTS):
    """Weighted similarity in [0,1]: 1 - weighted mean absolute difference."""
    total = sum(weights.values())
    diff = sum(weights[f] * (df[f] - target[f]).abs() for f in FEATURES) / total
    return 1 - diff


def build_playlist(df, history, target, w, n=10, exclude_history=True, pace=None, genre_filter=None):
    """score = (1-w) * intent_fit + w * history_fit
    genre_filter: optional list of genres to restrict the pool (see genres_for()).
    pace (0-1, slow..fast): optional user-set beat. It overrides the intent's tempo and
    doubles tempo's weight, so e.g. a 'happy but slow' mood can be asked for explicitly."""
    weights = WEIGHTS
    if pace is not None and target is not None:
        target = with_overrides(target, {"tempo_n": pace})
        weights = {**WEIGHTS, "tempo_n": 2.0}
    hist_profile = history[FEATURES].mean().to_dict()
    out = df if genre_filter is None else df[df[GENRE_COL].isin(genre_filter)]    
    out = out.copy()
    if "track_id" in out.columns:
        out = out.drop_duplicates("track_id")   # one song listed under several genres appears once
    out["history_fit"] = similarity(out, hist_profile)
    if target is None:                    # no intent recognised -> history only
        out["intent_fit"], w = out["history_fit"], 1.0
    else:
        if isinstance(target, list):      # best match to any prototype
            out["intent_fit"] = pd.concat([similarity(out, t, weights) for t in target], axis=1).max(axis=1)
        else:
            out["intent_fit"] = similarity(out, target, weights)
    out["score"] = (1 - w) * out["intent_fit"] + w * out["history_fit"]
    if exclude_history and "track_id" in out.columns:
        out = out[~out["track_id"].isin(history["track_id"])]
    return out.sort_values("score", ascending=False).head(n), w

 # Q1 "What is it for?" -> activity purposes (same keys as INTENT_GENRES)
PURPOSES = ["study", "workout", "party", "relax", "drive"]
# Q2 "What should it feel like?" -> mood targets in valence-energy space  
# starting assumptions (tempo_n = BPM/200), not fitted to data.
MOODS = {
    "calm":      {"energy": 0.20, "valence": 0.50, "tempo_n": 0.35},
    "joyful":    {"energy": 0.70, "valence": 0.85, "tempo_n": 0.60, "danceability": 0.70},
    "sad":       {"energy": 0.25, "valence": 0.15, "tempo_n": 0.35, "danceability": 0.30},
    "energetic": {"energy": 0.85, "valence": 0.65, "tempo_n": 0.70},
    "aggressive": {"energy": 0.90, "valence": 0.20, "tempo_n": 0.70},
}
VIBES = MOODS   # resolve_target() reads VIBES; kept as an alias


MOOD_KEYWORDS = {
    "calm": ["calm", "peaceful", "gentle", "soft"],
    "joyful": ["happy", "joy", "joyful", "cheerful", "upbeat"],
    "sad": ["sad", "melancholy", "rainy", "rain", "cry"],
    "energetic": ["energetic", "hype", "pumped", "energy"],
    "aggressive": ["angry", "aggressive", "rage"],
}


def detect_mood(text):
    words = text.lower().replace(",", " ").split()
    for mood, keys in MOOD_KEYWORDS.items():
        if any(k in words for k in keys):
            return mood
    return None


def with_overrides(target, mods):
    """Apply feature overrides (mood / pace) to one target or to every prototype."""
    if isinstance(target, list):
        return [{**t, **mods} for t in target]
    return {**target, **mods}


def resolve_target(df, profiles, title="", purpose=None, vibe=None):
    """Combine title text, Q1 purpose and Q2 vibe into one target profile.
    Priority: tapped purpose > purpose detected in the title > none.
    Returns (target or None, pace or None, plain-language note shown to the user)."""
    detected = detect_intent(title) if title.strip() else None
    vibe = vibe or (detect_mood(title) if title.strip() else None)    
    intent = purpose or detected
    notes = []
    if intent in profiles:
        target = with_overrides(profiles[intent], {})
        if purpose and detected and purpose != detected:
            notes.append(f"Your title suggested '{detected}', but you chose '{purpose}', so I used '{purpose}'.")
        else:
            notes.append(f"Purpose: {intent}" + (" (from your title)" if not purpose else ""))
    elif vibe:
        target = df[FEATURES].mean().to_dict()          # neutral start; the vibe shapes it
        notes.append("No purpose given, so I used the vibe on its own.")
    else:
        return None, None, "I couldn't tell what this playlist is for, so I used your history only. Pick a purpose or vibe."
    pace = None
    if vibe:
        mods = VIBES[vibe]
        target = with_overrides(target, mods)
        pace = mods.get("tempo_n")
        notes.append(f"Vibe: {vibe}")
    return target, pace, " · ".join(notes)


 
# Regions are a different lens from GENRE_GROUPS and may overlap with it. They are built from GENRE LABELS only:
 
REGIONS = {
    "Latin": ["latin", "latino", "salsa", "reggaeton", "spanish", "tango", "brazil", "forro", "mpb", "pagode",
              "samba", "sertanejo"],
    "East Asian": ["k-pop", "j-pop", "j-rock", "j-idol", "j-dance", "anime", "cantopop", "mandopop"],  # Korean, Japanese, Chinese
    "Middle Eastern": ["turkish", "iranian"],
    "South & Southeast Asian": ["indian", "malay"],
    "European": ["british", "french", "german", "swedish"],
    "African": ["afrobeat"],
    "Caribbean": ["reggae", "dancehall"],
    "World (mixed)": ["world-music"],
}
REGION_CAVEATS = {
    "Middle Eastern": "There is no Arabic genre in this dataset, so I used Turkish and Iranian tracks.",
}
REGION_KEYWORDS = {
    "Latin": ["latin", "latino", "latina", "spanish", "salsa", "reggaeton", "brazilian", "samba", "tango", "hispanic"],
    "East Asian": ["korean", "japanese", "kpop", "k-pop", "jpop", "j-pop", "anime", "chinese", "cantonese", "mandarin", "asian"],
    "Middle Eastern": ["arabic", "arab", "middle-eastern", "turkish", "persian", "iranian"],
    "South & Southeast Asian": ["indian", "bollywood", "malay", "malaysian"],
    "European": ["european", "french", "german", "swedish", "british"],
    "African": ["african", "afrobeat", "afrobeats"],
    "Caribbean": ["caribbean", "reggae", "dancehall"],
    "World (mixed)": ["world"],
}
INTERNATIONAL_WORDS = ["international", "countries", "global", "foreign"]   # -> every region


def detect_regions(text):
    tokens = re.findall(r"[a-z][a-z\-]*", text.lower())
    if any(w in tokens for w in INTERNATIONAL_WORDS):
        return list(REGIONS)
    return [r for r, keys in REGION_KEYWORDS.items() if any(k in tokens for k in keys)]


def resolve_genres(df, title="", regions=None):
    """Regions tapped by the user plus regions named in the title -> (genre list or None, region names, caveat text)."""
    chosen = list(dict.fromkeys((regions or []) + (detect_regions(title) if title.strip() else [])))
    if "International" in chosen:
        chosen = list(REGIONS)
    if not chosen:
        return None, [], ""
    present = set(df[GENRE_COL])
    genres = sorted({g for r in chosen for g in REGIONS[r] if g in present})
    caveat = " ".join(REGION_CAVEATS[r] for r in chosen if r in REGION_CAVEATS)
    return (genres or None), chosen, caveat


def plan(df, profiles, title="", purpose=None, vibe=None, regions=None):
    """One call for the app and tests: purpose + mood + regions -> target, pace, genre filter and a plain-language note."""
    target, pace, note = resolve_target(df, profiles, title, purpose, vibe)
    genres, chosen, caveat = resolve_genres(df, title, regions)
    if chosen:
        if target is None:
            note = "Regions: " + ", ".join(chosen) + ". No purpose or mood given, so I ranked by your history within them."
        else:
            note += " · Regions: " + ", ".join(chosen)
        if genres is None:
            note += " (no tracks found for those regions, so I used everything)"
        if caveat:
            note += " · " + caveat
    return dict(target=target, pace=pace, genres=genres, note=note, regions=chosen)


if __name__ == "__main__":
    import sys
    df = load_data(sys.argv[1] if len(sys.argv) > 1 else CSV_PATH)
    profiles = build_intent_profiles(df)
    history = make_history(df)
    for intent in ["study", "workout"]:
        for w in (0.0, 0.5, 1.0):
            pl, _ = build_playlist(df, history, profiles[intent], w, n=5)
            print(f"\n{intent} @ {w:.0%} history -> genres: {pl[GENRE_COL].value_counts().to_dict()}")
