import sys
import subprocess


required_libraries = ["matplotlib", "seaborn", "pandas", "numpy", "scikit-learn", "scipy"]
for lib in required_libraries:
    try:
        import_name = "sklearn" if lib == "scikit-learn" else lib
        __import__(import_name)
    except ImportError:
        print(f"\n[SİSTEM]: {lib} bulunamadı. Otomatik yükleniyor, lütfen bekleyin...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
        print(f"[SİSTEM]: {lib} başarıyla yüklendi!\n")


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack

warnings.filterwarnings('ignore')


df = pd.read_csv("data/dataset.csv")
df = df.sample(n=min(50000, len(df)), random_state=42) 

df["text"] = (
    df["artists"].fillna("") + " " + 
    df["album_name"].fillna("") + " " + 
    df["track_name"].fillna("")
).str.lower()


df["is_heavy"] = (df["energy"] * (df["loudness"] + 60)) / 60
df["tempo_log"] = np.log1p(df["tempo"])
df["energy_dance"] = df["energy"] * df["danceability"]
df["acoustic_instr"] = df["acousticness"] * df["instrumentalness"]

numeric_features = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo_log", "energy_dance", "acoustic_instr", "is_heavy"
]


sub_genre_cleaner = {
    "black-metal": "Heavy-Metal", "death-metal": "Heavy-Metal", "heavy-metal": "Heavy-Metal", "grindcore": "Heavy-Metal", "metalcore": "Heavy-Metal", "metal": "Heavy-Metal",
    "alt-rock": "Alternative-Rock", "alternative": "Alternative-Rock", "grunge": "Alternative-Rock", "psych-rock": "Alternative-Rock", "indie": "Alternative-Rock",
    "hard-rock": "Classic-Rock", "rock": "Classic-Rock", "rock-n-roll": "Classic-Rock", "rockabilly": "Classic-Rock",
    "punk": "Punk", "punk-rock": "Punk", "hardcore": "Punk",
    
    "house": "House", "deep-house": "House", "chicago-house": "House", "progressive-house": "House",
    "techno": "Techno", "detroit-techno": "Techno", "minimal-techno": "Techno",
    "trance": "Trance", "edm": "Dance-EDM", "electro": "Dance-EDM", "club": "Dance-EDM", "dance": "Dance-EDM", "hardstyle": "Dance-EDM",
    "breakbeat": "Bass-Music", "drum-and-bass": "Bass-Music", "dubstep": "Bass-Music",
    
    "pop": "Pop-General", "pop-film": "Pop-General", "synth-pop": "Pop-General", "power-pop": "Pop-General",
    "k-pop": "Asian-Pop", "j-pop": "Asian-Pop", "j-rock": "Asian-Pop", "j-idol": "Asian-Pop", "cantopop": "Asian-Pop", "mandopop": "Asian-Pop"
}

df["track_genre"] = df["track_genre"].map(sub_genre_cleaner).fillna(df["track_genre"])

genre_map = {
    "Heavy-Metal": "rock_metal", "Alternative-Rock": "rock_metal", "Classic-Rock": "rock_metal", "Punk": "rock_metal",
    "House": "electronic", "Techno": "electronic", "Trance": "electronic", "Dance-EDM": "electronic", "Bass-Music": "electronic",
    "Pop-General": "pop", "Asian-Pop": "pop",
    "jazz": "jazz", "blues": "jazz", "funk": "jazz", "soul": "jazz", "groove": "jazz",
    "afrobeat": "world", "brazil": "world", "forro": "world", "latin": "world", "latino": "world", "samba": "world", "salsa": "world", "tango": "world", "world-music": "world", "indian": "world", "iranian": "world", "turkish": "world", "malay": "world", "sertanejo": "world", "mpb": "world",
    "classical": "classical", "opera": "classical", "piano": "classical", "new-age": "classical",
    "indie-pop": "other", "emo": "other", "goth": "other", "industrial": "other", "trip-hop": "other", "idm": "other", "ska": "other", "reggae": "other", "dancehall": "other", "sleep": "other", "study": "other", "kids": "other", "romance": "other", "party": "other", "happy": "other", "sad": "other"
}

df["group"] = df["track_genre"].map(genre_map)
df = df.dropna(subset=["group"])


X_train_data, X_test_data, y_train_group, y_test_group = train_test_split(
    df, df["group"], test_size=0.2, random_state=42, stratify=df["group"]
)

tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)
X_train_text = tfidf.fit_transform(X_train_data["text"])
X_test_text = tfidf.transform(X_test_data["text"])

scaler = MaxAbsScaler()
X_train_num = scaler.fit_transform(X_train_data[numeric_features])
X_test_num = scaler.transform(X_test_data[numeric_features])

X_train_final = hstack([X_train_text, X_train_num])
X_test_final = hstack([X_test_text, X_test_num])

print("\n=== MODEL 1 (VOTING ENSEMBLE) ===")
m1 = LinearSVC(C=0.15, class_weight="balanced", random_state=42)
m2 = LogisticRegression(C=0.6, class_weight="balanced", max_iter=1000, random_state=42)

voter = VotingClassifier(estimators=[('svc', m1), ('lr', m2)], voting='hard')
voter.fit(X_train_final, y_train_group)

group_acc = accuracy_score(y_test_group, voter.predict(X_test_final))
print("Ana Grup Doğruluk Oranı (Group Accuracy):", group_acc)


saved_metrics = {
    "overall_accuracy": f"%{group_acc * 100:.2f}"
}


print("\n=== MODEL 2 (ALT TÜRLER - REALISTIC OPTIMIZED) ===")

for group in df["group"].unique():
    df_group = df[df["group"] == group]
    if len(df_group) < 100: continue

    tfidf_g = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X_t_g = tfidf_g.fit_transform(df_group["text"])
    
    X_n_values = df_group[numeric_features].values
    min_val = X_n_values.min(axis=0)
    max_val = X_n_values.max(axis=0)
    denom = np.where(max_val - min_val == 0, 1, max_val - min_val)
    X_n_g = (X_n_values - min_val) / denom
    
    X_f_g = hstack([X_t_g, X_n_g])
    y_g = df_group["track_genre"]

    if len(y_g.unique()) <= 1:
        continue

    selector = SelectKBest(chi2, k=min(4000, X_f_g.shape[1]))
    X_f_g_selected = selector.fit_transform(X_f_g, y_g)

    Xg_train, Xg_test, yg_train, yg_test = train_test_split(
        X_f_g_selected, y_g, test_size=0.2, random_state=42, stratify=y_g
    )

    model2 = LinearSVC(C=0.1, class_weight="balanced", max_iter=10000, random_state=42)
    model2.fit(Xg_train, yg_train)

    acc = accuracy_score(yg_test, model2.predict(Xg_test))
    print(f"{group.upper():<12} alt tür doğruluk oranı: {acc:.4f}")
    
    
    saved_metrics[group] = f"%{acc * 100:.2f}"


print("\n=== GRAFİKLER OLUŞTURULUYOR ===")
y_pred_group = voter.predict(X_test_final)
cm_group = confusion_matrix(y_test_group, y_pred_group)
group_labels = sorted(df["group"].unique())

plt.figure(figsize=(10, 7))
sns.heatmap(cm_group, annot=True, fmt='d', cmap='Blues', 
            xticklabels=group_labels, yticklabels=group_labels)
plt.title('Model 1: Ana Grup Tahmin Başarısı (Confusion Matrix)')
plt.ylabel('Gerçek Grup')
plt.xlabel('Modelin Tahmini')
plt.tight_layout()

plt.savefig('ana_grup_confusion_matrix.png') 
print("- Başarı grafiği 'ana_grup_confusion_matrix.png' adıyla kaydedildi.")


print("\n=== MODELLER KAYDEDİLİYOR ===")
with open('voter_model.pkl', 'wb') as f:
    pickle.dump(voter, f)
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('metrics.pkl', 'wb') as f:
    pickle.dump(saved_metrics, f) 
print("- 'voter_model.pkl', 'scaler.pkl', 'tfidf_vectorizer.pkl' ve 'metrics.pkl' dosyaları başarıyla oluşturuldu!")

plt.show()