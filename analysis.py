import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.contingency import association

 
df = pd.read_csv(
    r'C:\Users\alish\OneDrive\Documents\YEAR 3\IIS\New folder\spotify-tracks-dataset.csv'
)
tracks_df = df   

GENRE_COL = "track_genre"
RAW_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

n = tracks_df.groupby("track_id")["track_genre"].nunique()
print((n > 1).mean(), n.max())

 
INTENT_GENRES = {
    "workout": ["drum-and-bass", "hardstyle", "dubstep", "edm", "rock"],
    "study_sleep": ["study", "sleep", "ambient", "classical", "piano"],
    "party": ["dance", "house", "pop", "disco", "reggaeton"],
    "chill": ["chill", "acoustic", "jazz", "soul", "indie"]
}

def eta_squared(data, col, by):
    """Correlation ratio: share of the feature's variance explained by genre (0 = none, 1 = all)."""
    grand = data[col].mean()
    g = data.groupby(by)[col]
    ss_between = (g.count() * (g.mean() - grand) ** 2).sum()
    ss_total = ((data[col] - grand) ** 2).sum()
    return ss_between / ss_total

# 1) How much does genre explain each feature?
eta = pd.Series({f: eta_squared(df, f, GENRE_COL) for f in RAW_FEATURES}).sort_values()
print("Eta-squared (variance explained by genre):\n", eta.round(3).to_string())
plt.figure(figsize=(6, 3.5))
eta.plot.barh(color="steelblue")
plt.xlabel("Eta squared (share of variance explained by genre)")
plt.title("How strongly genre separates each feature")
plt.tight_layout()
plt.savefig("fig_eta_squared.png", dpi=200)
plt.close()
 
genres = sorted({g for gl in INTENT_GENRES.values() for g in gl if g in set(df[GENRE_COL])})
means = df[df[GENRE_COL].isin(genres)].groupby(GENRE_COL)[RAW_FEATURES].mean()
z = (means - df[RAW_FEATURES].mean()) / df[RAW_FEATURES].std()

plt.figure(figsize=(8, 0.35 * len(genres) + 2))
sns.heatmap(z, annot=means.round(2), fmt="", cmap="coolwarm", center=0, cbar_kws={"label": "z-score vs all tracks"})
plt.title("Mean features by genre (colour = z-score, number = raw mean; tempo in BPM)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("fig_genre_feature_heatmap.png", dpi=200)
plt.close()

 
order = df[df[GENRE_COL].isin(genres)].groupby(GENRE_COL)["tempo"].median().sort_values().index
plt.figure(figsize=(8, 0.3 * len(genres) + 2))
sns.boxplot(data=df[df[GENRE_COL].isin(genres)], y=GENRE_COL, x="tempo", order=order, showfliers=False, color="lightsteelblue")
plt.xlabel("Tempo (BPM)")
plt.ylabel("")
plt.title("Tempo by genre (sorted by median)")
plt.tight_layout()
plt.savefig("fig_tempo_by_genre.png", dpi=200)
plt.close()
print("Saved 3 figures.")
 
genre_avg_tempo = tracks_df.groupby("track_genre")["tempo"].median().sort_values(ascending=False)
total = list(zip(genre_avg_tempo.index.tolist(), genre_avg_tempo.tolist()))
print("Genres sorted by median tempo:", total)
 
bins = [0, 90, 120, 150, 245]
labels = ["Slow (<90)", "Moderate (90-120)", "Fast (120-150)", "Very Fast (150+)"]

tracks_df["tempo_category"] = pd.cut(
    tracks_df["tempo"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

 
contingency_table = pd.crosstab(tracks_df["track_genre"], tracks_df["tempo_category"])
cramers_v = association(contingency_table, method="cramer")
print(f"Cramér's V: {cramers_v:.3f}")
 
genre_tempo = pd.crosstab(
    tracks_df["track_genre"],
    tracks_df["tempo_category"],
    normalize="index"
) * 100

genre_tempo = genre_tempo.loc[genre_avg_tempo.index]

#  Plot heatmap
plt.figure(figsize=(10, 20))
sns.heatmap(
    genre_tempo,
    annot=False,
    cmap="coolwarm",
    linewidths=0.3,
    cbar_kws={"label": "Percentage of tracks (%)"}
)

plt.title("Distribution of Tempo Across Music Genres")
plt.xlabel("Tempo Category")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()


COLUMNS_NAME = list(tracks_df.columns.values)

print(COLUMNS_NAME)
tempo_data = tracks_df['tempo'].unique()
unique_tempo_data = tracks_df["tempo"].drop_duplicates()
print("maxm")
print(max(unique_tempo_data))
print(min(unique_tempo_data))
      

column_data = tracks_df['track_genre'].unique()
unique_column_data = tracks_df["track_genre"].drop_duplicates()
print(tracks_df['track_genre'].unique().tolist())

 
print(unique_column_data)

reduced_df = tracks_df[:1000].copy()

# Convert explicit from Boolean to integer
reduced_df['explicit'] = reduced_df['explicit'].astype(int)

# Create scatter matrix
pd.plotting.scatter_matrix(
    reduced_df[[
        'popularity',
        'duration_ms',
        'explicit',
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'time_signature'
    ]],
    figsize=(20, 12)
)

 

numeric_df = reduced_df.select_dtypes(include='number')

numeric_df = numeric_df.drop(
    columns=['Unnamed: 0', 'Unnamed: 0.1'],
    errors='ignore'
)

from itertools import combinations

for x, y in combinations(numeric_df.columns, 2):

    plt.figure(figsize=(6, 4))
    plt.scatter(numeric_df[x], numeric_df[y], alpha=0.4)
    correlation = numeric_df[x].corr(numeric_df[y])

    #print(f'{x} vs {y}: {correlation:.3f}')
 


  
