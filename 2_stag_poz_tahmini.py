import cv2
import numpy as np
import stag
import csv
import os
import time

# ==========================================
# --- AYARLAR ---
# ==========================================
MARKER_BOYUTU = 14.0  
STAG_AILESI = 15      
TXT_DOSYASI = "gercek_konumlar.txt"

MARKER_CSV = "marker_verileri.csv"
STABILITE_CSV = "stabilite_analizi.csv"
# --- HARİTAYI YÜKLEME ---
bagil_konumlar = {}
current_origin = None

with open("marker_haritasi.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("# Origin:"):
            current_origin = int(line.split(":")[1].strip())
            bagil_konumlar[current_origin] = {}
        elif ":" in line and "np.array" in line:
            parts = line.split(":")
            target_id = int(parts[0].strip())
            # Satırın içinden [x, y, z] değerlerini çekiyoruz
            val_str = parts[1].split("[")[1].split("]")[0]
            vals = [float(x) for x in val_str.split(",")]
            bagil_konumlar[current_origin][target_id] = np.array(vals, dtype=np.float32)

# --- 1. KALİBRASYON VERİSİNİ YÜKLE ---
try:
    with np.load("kamera_parametreleri.npz") as data:
        mtx = data['mtx']
        dist = data['dist']
except FileNotFoundError:
    print("HATA: 'kamera_parametreleri.npz' bulunamadi!")
    exit()

# --- 2. GERÇEK KONUMLARI (.txt) YÜKLE ---
marker_konumlari = {}
try:
    with open(TXT_DOSYASI, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:] 
        for line in lines:
            parts = line.split(',')
            if len(parts) >= 3:
                m_id = int(parts[0].strip())
                x = float(parts[1].strip())
                y = float(parts[2].strip())
                marker_konumlari[m_id] = (x, y)
    print(f"✅ {len(marker_konumlari)} adet markerin gercek konumu sisteme yuklendi.")
except FileNotFoundError:
    print(f"HATA: '{TXT_DOSYASI}' bulunamadi!")
    exit()

varyans_hafizasi = []

# --- 5. ADIM: CSV DOSYALARI HAZIRLIĞI (Noktalı Virgül Ayarı Eklendi) ---
if not os.path.exists(MARKER_CSV):
    with open(MARKER_CSV, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';') # <-- BURASI DEĞİŞTİ
        writer.writerow(["Zaman Damgasi", "Marker ID", "rvecX", "rvecY", "rvecZ", "tvecX", "tvecY", "tvecZ", "pixelX", "pixelY"])

if not os.path.exists(STABILITE_CSV):
    with open(STABILITE_CSV, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=';') # <-- BURASI DEĞİŞTİ
        writer.writerow(["Zaman Damgasi", "VarX", "VarY", "VarZ", "VarMag", "Durum"])

# --- KAMERA AYARLARI ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

PENCERE_ADI = "Adim 2: Multi-Marker Poz Tahmini (Ayna Modu)"
cv2.namedWindow(PENCERE_ADI, cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret: break

    corners, ids, rejected = stag.detectMarkers(frame, STAG_AILESI)
    
    frame = cv2.flip(frame, 1) 
    GENISLIK = frame.shape[1] 
    
    anlik_marker_verileri = []
    anlik_varyans_verisi = None
    
    if ids is not None and len(ids) > 0:
        all_obj_points = []
        all_img_points = []
        
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = 0, 0

        for i, m_id in enumerate(ids.flatten()):
            if m_id in marker_konumlari:
                img_pts = corners[i].reshape(4, 2)
                
                for pt in img_pts:
                    if pt[0] < min_x: min_x = pt[0]
                    if pt[0] > max_x: max_x = pt[0]
                    if pt[1] < min_y: min_y = pt[1]
                    if pt[1] > max_y: max_y = pt[1]
                
                cizim_img_pts = img_pts.copy()
                cizim_img_pts[:, 0] = GENISLIK - cizim_img_pts[:, 0]
                cv2.polylines(frame, [np.int32(cizim_img_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
                
                X, Y = marker_konumlari[m_id]
                pts_3d = np.array([
                    [X, Y, 0],
                    [X + MARKER_BOYUTU, Y, 0],
                    [X + MARKER_BOYUTU, Y + MARKER_BOYUTU, 0],
                    [X, Y + MARKER_BOYUTU, 0]
                ], dtype=np.float32)
                
                all_obj_points.extend(pts_3d)
                all_img_points.extend(img_pts)

        if len(all_obj_points) >= 4:
            all_obj_points = np.array(all_obj_points, dtype=np.float32)
            all_img_points = np.array(all_img_points, dtype=np.float32)
            
            ret_pnp, board_rvec, board_tvec = cv2.solvePnP(all_obj_points, all_img_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            
            if ret_pnp:
                varyans_hafizasi.append(board_tvec.flatten())
                if len(varyans_hafizasi) > 30: varyans_hafizasi.pop(0)
                
                R_board, _ = cv2.Rodrigues(board_rvec)
                zaman_damgasi = time.strftime("%H:%M:%S")
                
                for i, m_id in enumerate(ids.flatten()):
                    if m_id not in marker_konumlari: continue
                    
                    X, Y = marker_konumlari[m_id]
                    img_pts = corners[i].reshape(-1, 2)
                    
                    pixel_x_gercek = int(np.mean(img_pts[:, 0]))
                    pixel_y = int(np.mean(img_pts[:, 1]))
                    pixel_x_cizim = GENISLIK - pixel_x_gercek
                    
                    center_x = X + (MARKER_BOYUTU / 2.0)
                    center_y = Y + (MARKER_BOYUTU / 2.0)
                    P_center = np.array([[center_x], [center_y], [0.0]])
                    
                    marker_tvec = np.dot(R_board, P_center) + board_tvec
                    marker_rvec = board_rvec 
                    
                    z_mesafesi = marker_tvec[2][0]
                    
                    cv2.putText(frame, f"ID:{m_id}", (pixel_x_cizim - 30, pixel_y - 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    cv2.putText(frame, f"Z:{z_mesafesi:.0f}", (pixel_x_cizim + 5, pixel_y + 15), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    eksen_uzunlugu = MARKER_BOYUTU / 1.5
                    eksen_noktalari = np.float32([
                        [0, 0, 0], 
                        [eksen_uzunlugu, 0, 0], 
                        [0, eksen_uzunlugu, 0], 
                        [0, 0, -eksen_uzunlugu] 
                    ])
                    
                    img_pts_eksen, _ = cv2.projectPoints(eksen_noktalari, marker_rvec, marker_tvec, mtx, dist)
                    cizim_eksen = np.int32(img_pts_eksen).reshape(-1, 2)
                    cizim_eksen[:, 0] = GENISLIK - cizim_eksen[:, 0] 
                    
                    orijin = tuple(cizim_eksen[0])
                    cv2.line(frame, orijin, tuple(cizim_eksen[1]), (0, 0, 255), 2)  
                    cv2.line(frame, orijin, tuple(cizim_eksen[2]), (0, 255, 0), 2)  
                    cv2.line(frame, orijin, tuple(cizim_eksen[3]), (255, 0, 0), 2)  
                    
                    # --- YUVARLAMA (ROUND) İŞLEMİ BURADA YAPILDI (4 BASAMAK) ---
                    anlik_marker_verileri.append([
                        zaman_damgasi, m_id, 
                        round(marker_rvec[0][0], 4), round(marker_rvec[1][0], 4), round(marker_rvec[2][0], 4),
                        round(marker_tvec[0][0], 4), round(marker_tvec[1][0], 4), round(marker_tvec[2][0], 4), 
                        pixel_x_gercek, pixel_y 
                    ])
                
                if min_x != float('inf'):
                    cizim_min_x = int(GENISLIK - max_x)
                    cizim_max_x = int(GENISLIK - min_x)
                    cv2.rectangle(frame, (cizim_min_x, int(min_y)), (cizim_max_x, int(max_y)), (0, 255, 0), 2)
                
                if len(varyans_hafizasi) == 30:
                    tvec_array = np.array(varyans_hafizasi)
                    var_x = np.var(tvec_array[:, 0])
                    var_y = np.var(tvec_array[:, 1])
                    var_z = np.var(tvec_array[:, 2])
                    var_mag = np.sqrt(var_x**2 + var_y**2 + var_z**2)
                    
                    pano_x, pano_y = 30, 40
                    cv2.putText(frame, f"VarX: {var_x:.2e}", (pano_x, pano_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"VarY: {var_y:.2e}", (pano_x, pano_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"VarZ: {var_z:.2e}", (pano_x, pano_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"VarMag: {var_mag:.2e}", (pano_x, pano_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    durum = "Stabil" if var_mag < 1000 else "Titrek"
                    
                    # --- YUVARLAMA İŞLEMİ BURADA DA YAPILDI (4 BASAMAK) ---
                    anlik_varyans_verisi = [zaman_damgasi, round(var_x, 4), round(var_y, 4), round(var_z, 4), round(var_mag, 4), durum]

    ekran_goruntusu = cv2.resize(frame, (1440, 810), interpolation=cv2.INTER_AREA)
    cv2.imshow(PENCERE_ADI, ekran_goruntusu)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        if len(anlik_marker_verileri) > 0:
            with open(MARKER_CSV, mode='a', newline='') as file:
                # --- KAYDEDERKEN NOKTALI VİRGÜL AYARI ---
                writer = csv.writer(file, delimiter=';')
                writer.writerows(anlik_marker_verileri)
            print(f"✅ {len(anlik_marker_verileri)} marker konumu '{MARKER_CSV}' dosyasina kaydedildi.")
        
        if anlik_varyans_verisi is not None:
            with open(STABILITE_CSV, mode='a', newline='') as file:
                # --- KAYDEDERKEN NOKTALI VİRGÜL AYARI ---
                writer = csv.writer(file, delimiter=';')
                writer.writerow(anlik_varyans_verisi)
            print(f"✅ O anki sistem stabilitesi '{STABILITE_CSV}' dosyasina kaydedildi.\n")

cap.release()
cv2.destroyAllWindows()