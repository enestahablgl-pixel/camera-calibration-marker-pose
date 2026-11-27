import cv2
import numpy as np
import stag
import glob
import os

# ==========================================
# --- AYARLAR ---
# ==========================================
RESIM_KLASORU = os.path.join("task4", "saf_goruntuler")
RESIM_FORMATI = "*.jpg" 

STAG_AILESI = 15
MARKER_BOYUTU = 14.0  # mm
MERKEZ_MESAFESI = 18.0  # mm
SUTUN_SAYISI = 15

# ==========================================

objpoints = [] # Gerçek dünyadaki 3D noktalar
imgpoints = [] # Resimdeki 2D pikseller

arama_yolu = os.path.join(RESIM_KLASORU, RESIM_FORMATI)
resim_dosyalari = glob.glob(arama_yolu)

if not resim_dosyalari:
    print(f"HATA: '{RESIM_KLASORU}' klasöründe hiç resim bulunamadı!")
    exit()

print(f"Toplam {len(resim_dosyalari)} adet STag görüntüsü işleniyor...\n")

basarili_resim_sayisi = 0
ornek_shape = None

for dosya_yolu in resim_dosyalari:
    img = cv2.imread(dosya_yolu)
    if img is None:
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if ornek_shape is None:
        ornek_shape = gray.shape[::-1]

    # STag Marker'ları Tespit Et
    corners, ids, rejected = stag.detectMarkers(gray, STAG_AILESI)

    if ids is not None and len(ids) > 10: # En az 10 marker bulduysa geçerli sayalım
        resim_obj_pts = []
        resim_img_pts = []
        
        for i, m_id in enumerate(ids.flatten()):
            if m_id <= 164: # Bizim tahtamızdaki ID'ler
                # Tahta üzerindeki gerçek X, Y hesaplaması (gercek_konumlar.txt ile aynı mantık)
                row = m_id // SUTUN_SAYISI
                col = m_id % SUTUN_SAYISI
                
                X = col * MERKEZ_MESAFESI
                Y = row * MERKEZ_MESAFESI
                
                # Marker'ın 4 köşesinin 3D koordinatları (Sol Üst, Sağ Üst, Sağ Alt, Sol Alt)
                pts_3d = np.array([
                    [X, Y, 0],
                    [X + MARKER_BOYUTU, Y, 0],
                    [X + MARKER_BOYUTU, Y + MARKER_BOYUTU, 0],
                    [X, Y + MARKER_BOYUTU, 0]
                ], dtype=np.float32)
                
                resim_obj_pts.extend(pts_3d)
                resim_img_pts.extend(corners[i].reshape(4, 2))
        
        if len(resim_obj_pts) > 0:
            objpoints.append(np.array(resim_obj_pts, dtype=np.float32))
            imgpoints.append(np.array(resim_img_pts, dtype=np.float32))
            basarili_resim_sayisi += 1
            print(f"✅ {os.path.basename(dosya_yolu)} - {len(ids)} marker bulundu ve eklendi.")
    else:
        print(f"❌ {os.path.basename(dosya_yolu)} - Yeterli marker BULUNAMADI (Atlanıyor).")

print(f"\n--- Kalibrasyon Hesaplanıyor ({basarili_resim_sayisi} başarılı görüntü) ---")

if basarili_resim_sayisi > 0:
    # Kamera matrisi ve distorsiyon katsayılarını hesapla
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, ornek_shape, None, None)
    
    print("\n✅ Yeni STag Kalibrasyonu Başarılı!")
    print(f"RMS Hatası (Ne kadar düşükse o kadar iyi, 1.0'ın altı harikadır): {ret:.4f}")
    
    np.savez("kamera_parametreleri.npz", mtx=mtx, dist=dist)
    print("\nYeni ve daha temiz 'kamera_parametreleri.npz' dosyası başarıyla kaydedildi.")
    
else:
    print("\nHATA: Hiçbir resimde yeterli marker tespit edilemedi.")