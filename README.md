# Intent-aware playlist generator (IIS Assignment 2)

**Module:** Intelligent Interactive Systems, University of Birmingham
**Author:** Alisha Ali
**Submitted version:**  v1.0-a2-submission, or commit hash

A playlist generator that treats the **playlist title as the user's stated intent** (for example "study session" or "gym workout"), with an optional **vibe** for example ("calm" or "energetic"), an optional **region**, and a **history weight** the user sets.

There is no trained model. The system ranks songs by how close their audio features are to a target, using a weighted similarity score:

`score = (1 - w) x intent_fit + w x history_fit`

`w` is the history weight, that scores between 0 and 1 (0 = ignore history, 1 = history only). If nothing in the title or vibe is recognised, the system ranks by history only and tells the user.

## Status

| Part | Status | Improvements
|---|---|
| Playlist title read for purpose (study, workout, party, relax, drive) | Implemented | More purposes are required
| Vibe words (calm, joyful, sad, energetic, aggressive) read from the title | Implemented | More vibes/emotion are required
| Region words (8 regions, 32 genres) read from the title | Implemented | Instead of region prehaps language
| History weight `w` as a value in code | Implemented |
| Fallback to history only, with a message | Implemented |
| Behaviour tests (35 checks, all passing on the real data) | Implemented | Need simulation with real users and datasets
| Listening history | Simulated (25 songs sampled from pop, dance, hip-hop and rock in `make_history`) |
| Vibe menu, region buttons, history slider, Streamlit interface | Planned |
| Default vibe per purpose, song-to-song mode, multi-genre weighting, ratings log, mid-session mood updates, user study | Planned |

## Tech stack

Python 3, pandas,

## Data

Kaggle Spotify Tracks Dataset, 114 genres: (https://www.kaggle.com/datasets/yashdev01/spotify-tracks-dataset). The CSV is included in this repository (size and licence). Download it and save it anywhere. Ensure to update filepath in music.py file

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Download the dataset and set `CSV_PATH` at the top of `music.py` to where you saved it.
3. Quick demo of the engine: `music.py`
4. Run the behaviour tests: `python tests.py "path/to/spotify-tracks-dataset.csv"` (writes to`test_results.csv` and `scenario_table.csv`)
5. Print the example outputs used in the storyboards: `python storyboard_data.py`
6. Recreate the analysis figures: `python analysis.py` - shows many different correlations important to the understanding of the logic  


## Files

- `music.py`: the engine (data prep, keyword reader, target builder, region filter, scorer, fallback)
- `tests.py`: behaviour tests
- `analysis.py`: correlation and eta-squared analysis behind the feature and genre choices

## Development record

| Stage | What I did and decided | 
|---|---| 
| 1. Analysis | Looked at correlations and eta-squared (how much genre explains each feature). Kept energy, valence, danceability, acousticness (half weight), instrumentalness and tempo. Dropped popularity (|r| < 0.2) |  
| 2. First prototype | Streamlit app with synthetic songs. Superseded once I moved to the real dataset | 
| 3. Real-data engine | Built one profile per intent from example genres. Tempo scaled by /200 for scoring only, rows with tempo 0 dropped |  
| 4. Profiles fix | Averaging genres into one profile gave targets "between" genres (study returned Disney and tango). Switched to one prototype per genre and take the best match | 
| 5. Moods | Added hand-set mood targets from the tempo and mood literature. Values are my own choices and untested with users | 
| 6. Regions | Added 8 regions by genre label (no language column in the data) | 
| 7. Tests | Wrote behaviour tests on audio direction, not genre labels |  
| 8. Workout fix | Workout test failed (tempo gain too small). Changed workout genres to edm, funk, drum-and-bass, breakbeat, party. All 35 checks then passed | 

**What went wrong and what I did about it:** genre-label hit rates stay at 0 to 10% because playlists match audio character, not labels, so I test audio direction instead. The workout genres were changed after a test failed, so they are partly tuned to my own test and the tempo gain is small (about 129 vs 122 BPM). I have not tested any of this with users.

## Known limitations

- Listening history is simulated, not real.
- Mood values and the genre-to-intent mapping are my own assumptions.
- Regions use genre labels only, so "korean" returns the whole East Asian region.
- With an empty history the scoring breaks (NaN scores). Not yet handled. 
- 18.2% of tracks appear under more than one genre. Songs are collapsed only when ranking. 