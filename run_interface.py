import sys
import subprocess

# venv içindeki Python'ın tam yolunu alıyoruz
python_exe = sys.executable

print("[SİSTEM]: Sanal ortam (venv) tespit edildi.")
print("[SİSTEM]: Eksik kütüphaneler venv içine yükleniyor, lütfen bekleyin...")

# venv içine zorunlu kütüphaneleri kuruyoruz
try:
    subprocess.check_call([python_exe, "-m", "pip", "install", "streamlit", "pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "seaborn"])
    print("[SİSTEM]: Tüm kütüphaneler venv içine başarıyla yüklendi!")
    print("[SİSTEM]: Şimdi arayüz başlatılıyor...\n")
    
    # Kütüphaneler yüklenirse arayüzü otomatik tetikle
    subprocess.run([python_exe, "-m", "streamlit", "run", "app.py"], check=True)

except Exception as e:
    print(f"\n[HATA]: Bir sorun oluştu: {e}")