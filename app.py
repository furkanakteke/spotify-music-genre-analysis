import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from scipy.sparse import hstack

# Sayfayı geniş modda açıyoruz
st.set_page_config(page_title="Müzik Türü Sınıflandırıcı", page_icon="🎵", layout="centered")

# --- CSS İLE GÖRSEL DÜZENLEME (SPOTIFY TEMASI) ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input { background-color: #262730; color: white; border-radius: 10px; }
    .stSelectbox > div > div > div { background-color: #262730; color: white; border-radius: 10px; }
    .stButton>button {
        width: 100%;
        background-color: #1DB954;
        color: white;
        border-radius: 20px;
        height: 3em;
        font-weight: bold;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover { background-color: #1ed760; border: none; color: white; }
    .result-card {
        background-color: #1e293b;
        padding: 30px;
        border-radius: 15px;
        border-left: 10px solid #1DB954;
        margin-top: 20px;
    }
    .rec-box {
        background-color: #0f172a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid #334155;
        height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .info-box {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 10px;
        border: 1px dashed #475569;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .metric-badge-acc {
        background-color: #1e293b;
        color: #22d3ee;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        border: 1px solid #22d3ee;
        display: inline-block;
        margin-top: 5px;
    }
    .metric-badge-f1 {
        background-color: #1e293b;
        color: #bef264;
        padding: 8px 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 16px;
        border: 1px solid #bef264;
        display: inline-block;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎵 Yapay Zeka ile Müzik Türü Analizi")
st.write("Şarkı bilgilerini girin, sistemin çalışacağı akustik profili yapılandırın ve yapay zekayı tetikleyin.")
st.write("---")

# Modelleri ve Veri Setini Yükleme Fonksiyonu
@st.cache_resource
def load_assets():
    try:
        with open('voter_model.pkl', 'rb') as f: voter_model = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f: tfidf_model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f: scaler_model = pickle.load(f)
        with open('metrics.pkl', 'rb') as f: dynamic_metrics = pickle.load(f)
        
        # Veri setini yüklüyoruz
        music_df = pd.read_csv("data/dataset.csv")
        
        # Hiyerarşik temizlik adımları (main.py ile birebir aynı)
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
        music_df["track_genre"] = music_df["track_genre"].map(sub_genre_cleaner).fillna(music_df["track_genre"])
        
        genre_map = {
            "Heavy-Metal": "rock_metal", "Alternative-Rock": "rock_metal", "Classic-Rock": "rock_metal", "Punk": "rock_metal",
            "House": "electronic", "Techno": "electronic", "Trance": "electronic", "Dance-EDM": "electronic", "Bass-Music": "electronic",
            "Pop-General": "pop", "Asian-Pop": "pop",
            "jazz": "jazz", "blues": "jazz", "funk": "jazz", "soul": "jazz", "groove": "jazz",
            "afrobeat": "world", "brazil": "world", "forro": "world", "latin": "world", "latino": "world", "samba": "world", "salsa": "world", "tango": "world", "world-music": "world", "indian": "world", "iranian": "world", "turkish": "world", "malay": "world", "sertanejo": "world", "mpb": "world",
            "classical": "classical", "opera": "classical", "piano": "classical", "new-age": "classical",
            "indie-pop": "other", "emo": "other", "goth": "other", "industrial": "other", "trip-hop": "other", "idm": "other", "ska": "other", "reggae": "other", "dancehall": "other", "sleep": "other", "study": "other", "kids": "other", "romance": "other", "party": "other", "happy": "other", "sad": "other"
        }
        music_df["group"] = music_df["track_genre"].map(genre_map)
        music_df = music_df.dropna(subset=["group"])
        
        # Alfabetik benzersiz sanatçı listesi (Başına boş değer koyuyoruz ki ilk açılışta boş dursun)
        artist_list = sorted(music_df["artists"].dropna().unique().tolist())
        
        return voter_model, tfidf_model, scaler_model, music_df, artist_list, dynamic_metrics
    except Exception as e:
        return None, None, None, None, None, None

voter, tfidf, scaler, music_db, all_artists, metrics_db = load_assets()

if voter is None or music_db is None or metrics_db is None:
    st.error("⚠️ Gerekli dosyalar bulunamadı! Lütfen 'data/dataset.csv' konumunu ve '.pkl' dosyalarını kontrol edin.")
else:
    # 1. METİN KUTULARI (TAM ANLAMIYLA YAZDIKÇA ALTINDA ÇIKAN TEK BİLEŞENLİ YAPI)
    st.markdown("### 📝 1. Şarkı Kimliği")
    col_a, col_b = st.columns(2)
    
    with col_a:
        # placeholder ekleyerek kullanıcının yazmasını teşvik ediyoruz
        artist_input = st.selectbox(
            "Sanatçı / Grup Adı",
            options=[""] + all_artists,
            index=0,
            placeholder="Yazarak arayın... (Örn: Duman, Metallica)",
            help="Kutuya tıkladıktan sonra harf yazmaya başladığınızda gerçek sanatçılar anlık olarak listelenir."
        )
    
    with col_b:
        # Sanatçı seçimine göre şarkı havuzunu dinamik daraltıyoruz
        if artist_input != "":
            filtered_tracks = sorted(music_db[music_db["artists"] == artist_input]["track_name"].dropna().unique().tolist())
            track_options = [""] + filtered_tracks
        else:
            # Henüz sanatçı girilmediyse arayüz boğulmasın diye boş bırakıyoruz, yazmaya başlayınca genel listeden arayacak
            track_options = [""] + sorted(music_db["track_name"].dropna().unique().tolist())
            
        track_input = st.selectbox(
            "Şarkı Adı",
            options=track_options,
            index=0,
            placeholder="Şarkı adını yazarak arayın...",
            help="Seçtiğiniz sanatçıya ait veya genel havuzdaki şarkıları harf girdikçe listeler."
        )
    
    st.write("---")

    # 2. HAZIR MOD / TARZ SEÇİMİ (BİLMİYORUM SEÇENEĞİ İLE BERABER)
    st.markdown("### 🎛️ 2. Müzikal Altyapı Yapılandırması")
    mood_selection = st.selectbox(
        "Şarkının altyapı karakterini en iyi hangisi tarif ediyor?",
        [
            "🤷‍♂️ Fikrim Yok / Şarkının Orijinal Verilerini Kullan",
            "⚡ Standart Pop / Hareketli / Radyo Dostu",
            "🎸 Sert / Agresif / Yoğun Elektro Gitar & Davul (Rock-Metal)",
            "☕ Sakin / Saf Akustik / Enstrümantal (Klasik Müzik)",
            "🕺 Yoğun Elektronik / Dijital Bas & Synthesizer",
            "🎷 Caz / Blues Altyapılı / Üflemeli ve Yürüyen Baslar",
            "🌍 Kültürel / Yerel Enstrümanlar ve Etnik Ritimseller (World)"
        ]
    )

    # Varsayılan dengeli mod değerleri
    danceability, energy, loudness, speechiness = 0.5, 0.5, -10.0, 0.1
    acousticness, instrumentalness, liveness, valence, tempo = 0.5, 0.2, 0.15, 0.5, 120
    is_database_song = False

    # Veri tabanı kontrolü (Eğer şarkı kayıtlıysa her halükarda orijinal veriyi çeker)
    if artist_input != "" and track_input != "":
        db_match = music_db[(music_db["artists"] == artist_input) & (music_db["track_name"] == track_input)]
        if not db_match.empty:
            row = db_match.iloc[0]
            danceability = float(row.get("danceability", danceability))
            energy = float(row.get("energy", energy))
            loudness = float(row.get("loudness", loudness))
            speechiness = float(row.get("speechiness", speechiness))
            acousticness = float(row.get("acousticness", acousticness))
            instrumentalness = float(row.get("instrumentalness", instrumentalness))
            liveness = float(row.get("liveness", liveness))
            valence = float(row.get("valence", valence))
            tempo = float(row.get("tempo", tempo))
            is_database_song = True

    # Eğer şarkı veri tabanında yoksa ve kullanıcı bir mod seçtiyse değerleri güncelle
    if not is_database_song:
        if mood_selection == "⚡ Standart Pop / Hareketli / Radyo Dostu":
            danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = 0.65, 0.60, -6.0, 0.05, 0.0, 0.60, 105
        elif mood_selection == "🎸 Sert / Agresif / Yoğun Elektro Gitar & Davul (Rock-Metal)":
            danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = 0.35, 0.85, -5.0, 0.06, 0.0, 0.30, 130
        elif mood_selection == "☕ Sakin / Saf Akustik / Enstrümantal (Klasik Müzik)":
            danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = 0.30, 0.15, -22.0, 0.04, 0.85, 0.25, 90
        elif mood_selection == "🕺 Yoğun Elektronik / Dijital Bas & Synthesizer":
            danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = 0.75, 0.75, -6.0, 0.08, 0.30, 0.65, 125
        elif mood_selection == "🎷 Caz / Blues Altyapılı / Üflemeli ve Yürüyen Baslar":
            danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = 0.55, 0.45, -12.0, 0.05, 0.20, 0.50, 110
        elif mood_selection == "🌍 Kültürel / Yerel Enstrümanlar ve Etnik Ritimseller (World)":
            danceability, energy, loudness, acousticness, instrumentalness, valence, tempo = 0.60, 0.55, -9.0, 0.07, 0.10, 0.55, 100

    # Özellik mühendisliği hesaplamaları
    is_heavy = (energy * (loudness + 60)) / 60
    tempo_log = np.log1p(tempo)
    energy_dance = energy * danceability
    acoustic_instr = acousticness * instrumentalness

    # --- TEKNİK ALTYAPI ÖZETİ KARTI ---
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    if is_database_song:
        st.markdown(f"⚙️ **Veri Yönetim Durumu:** Bu şarkı veri kümesinde doğrulandı! Sistem otomatik olarak şarkının **orijinal Spotify sinyallerini** doğrudan veri tabanından çekti.")
    elif mood_selection.startswith("🤷‍♂️"):
        st.markdown(f"⚙️ **Veri Yönetim Durumu:** Fikrim Yok / Bilmiyorum seçildi. Model, serbest metin girdisi için **matematiksel nötr taban matrisi (Dengeli Mod)** değerlerini simüle edecektir.")
    else:
        st.markdown(f"⚙️ **Veri Yönetim Durumu:** Yapay zeka modeli seçtiğiniz **{mood_selection.split('/')[0]}** altyapı profili simülasyonu ile beslenecektir.")
    
    st.markdown(f"""
    <span style='color:#94a3b8; font-size:13px;'>
    📊 <b>Yapay Zekaya Gönderilen Akustik Sinyal Özeti:</b><br>
    • Enerji: {energy:.2f} | • Dans Edilebilirlik: {danceability:.2f} | • Ses Şiddeti (dB): {loudness:.1f} | • Akustiklik: {acousticness:.2f}<br>
    • Enstrümantallik: {instrumentalness:.2f} | • Tempo (BPM): {tempo:.0f} | • Özel Metrik (is_heavy): {is_heavy:.2f}
    </span>
    </div>
    """, unsafe_allow_html=True)

    # --- TAHMİN BUTONU ---
    predict_button = st.button("🚀 TÜRE GÖRE ANALİZ ET VE TAHMİN ET", use_container_width=True)

    if predict_button:
        combined_text = f"{artist_input} {track_input}".lower().strip()
        text_vector = tfidf.transform([combined_text])
        
        num_features = np.array([[
            danceability, energy, loudness, speechiness, acousticness, 
            instrumentalness, liveness, valence, tempo_log, energy_dance, 
            acoustic_instr, is_heavy
        ]])
        num_scaled = scaler.transform(num_features)
        final_input = hstack([text_vector, num_scaled])
        
        prediction = voter.predict(final_input)[0]
        
        desc_dict = {
            "rock_metal": "Baskın elektro gitar riffleri, yüksek distorsiyon ve agresif davul ritimleri mevcuttur.",
            "electronic": "Sentezleyiciler, dijital ses döngüleri ve 4/4'lük elektronik vuruşlar (kick) barındırır.",
            "pop": "Radyo dostu prodüksiyon, ön planda temiz vokaller ve akıcı melodi geçişleri içerir.",
            "classical": "Akustik enstrümanlar (piyano, yaylılar), tamamen enstrümantal kompozisyon sunar.",
            "jazz": "Yürüyen bas hatları, üflemeli sololar (saksofon) ve zengin akor geçişleri barındırır.",
            "world": "Etnik enstrümanlar, yöresel perküsyon kalıpları ve kültürel ritim yapısı içerir.",
            "other": "Belirli bir kalıba uymayan deneysel altyapı veya spesifik tematik frekanslar içerir."
        }

        # Şık Sonuç Kartı
        st.markdown(f"""
            <div class="result-card">
                <h2 style='color: #1DB954; margin:0;'>🔍 Analiz Sonucu: {prediction.upper()}</h2>
                <p style='color: white; font-size: 18px; margin-top:10px;'>
                    Yapılandırılan müzikal örüntü başarıyla işlendi ve yapay zeka tarafından <b>{prediction}</b> olarak sınıflandırıldı.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        overall_acc_value = metrics_db.get("overall_accuracy", "%83.00")
        specific_acc_value = metrics_db.get(prediction, "%75.00")
        
        st.write("")
        col_m1, col_m2, col_m3 = st.columns([2, 1, 1])
        with col_m1:
            st.success(f"🧬 **Altyapı Karakteristiği:** {desc_dict.get(prediction, 'Dengeli altyapı.')}")
        with col_m2:
            st.markdown(f"<div style='text-align:center; font-size:14px; color:#94a3b8;'>Model 1 Doğruluğu<br><span class='metric-badge-acc'>{overall_acc_value}</span></div>", unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"<div style='text-align:center; font-size:14px; color:#94a3b8;'>Model 2 Doğruluğu<br><span class='metric-badge-f1'>{specific_acc_value}</span></div>", unsafe_allow_html=True)
        
        # --- VERİ KÜMESİNDEN DİNAMİK ŞARKI ÖNERİLERİ ---
        st.write("---")
        st.markdown("### 🎯 Veri Kümesinden Dinamik Şarkı Önerileri")
        filtered_songs = music_db[music_db["group"] == prediction]
        
        if not filtered_songs.empty:
            top_songs = filtered_songs.sort_values(by="popularity", ascending=False).drop_duplicates(subset=["track_name"]).head(15)
            sampled_rec = top_songs.sample(n=min(3, len(top_songs)), random_state=random.getrandbits(16)).to_dict(orient="records")
            
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            if len(sampled_rec) >= 1:
                with rec_col1:
                    st.markdown(f"<div class='rec-box'>🎵 <b>{sampled_rec[0]['track_name']}</b><br><span style='color:#94a3b8; font-size:14px;'>{sampled_rec[0]['artists']}</span></div>", unsafe_allow_html=True)
            if len(sampled_rec) >= 2:
                with rec_col2:
                    st.markdown(f"<div class='rec-box'>🎵 <b>{sampled_rec[1]['track_name']}</b><br><span style='color:#94a3b8; font-size:14px;'>{sampled_rec[1]['artists']}</span></div>", unsafe_allow_html=True)
            if len(sampled_rec) >= 3:
                with rec_col3:
                    st.markdown(f"<div class='rec-box'>🎵 <b>{sampled_rec[2]['track_name']}</b><br><span style='color:#94a3b8; font-size:14px;'>{sampled_rec[2]['artists']}</span></div>", unsafe_allow_html=True)
        else:
            st.warning("Bu türe ait veri kümesinde öneri bulunamadı.")

        st.write("")
        st.markdown(f"ℹ *Analiz edilen metin:* **\"{combined_text if combined_text != '' else 'Boş Metin'}\"**")