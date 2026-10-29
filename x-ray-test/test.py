from flask import Flask, render_template, request, jsonify
import torch
import torchvision.transforms as T
from PIL import Image
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Model yolu
MODEL_PATH = "../TRAINED_MODEL_FOLDER/densenet121.pth"

# İzin verilen uzantılar
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.dcm'}

# Model yükle
print("Model yükleniyor...")
model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.eval()
print("Model hazır!")

# Hastalık isimleri
hastaliklar = {
    1: "Opasite",
    2: "Normal",
    3: "Akciğer",
    4: "Pnömoni",
    5: "Konsolidasyon",
    6: "Atelektazi",
    7: "Ödem",
    8: "Nodül"
}

# Renkler
renkler = {
    1: (255,255,255),  # Beyaz
    4: (255,0,0),      # Kırmızı
    5: (0,255,0),      # Yeşil
    6: (0,0,255),      # Mavi
    7: (255,255,0),    # Sarı
    8: (255,0,255)     # Mor
}

@app.route('/')
def ana_sayfa():
    return render_template('test.html')

@app.route('/tahmin', methods=['POST'])
def tahmin_yap():
    # Dosyayı al
    file = request.files['resim']
    if file.filename == '':
        return jsonify({'hata': 'Dosya seçilmedi'})
    
    # Dosyayı kaydet
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    
    # Resmi oku
    try:
        # DICOM kontrolü
        if filename.endswith('.dcm'):
            import pydicom
            ds = pydicom.dcmread(filepath)
            img = ds.pixel_array
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            img = Image.fromarray(img).convert('RGB')
        else:
            img = Image.open(filepath).convert('RGB')
    except:
        return jsonify({'hata': 'Resim okunamadı'})
    
    # Resmi hazırla
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)
    
    # Tahmin yap
    with torch.no_grad():
        tahmin = model([img_tensor.squeeze(0)])
    
    # Sonuçları al
    kutular = tahmin[0]['boxes'].cpu().numpy()
    etiketler = tahmin[0]['labels'].cpu().numpy()
    skorlar = tahmin[0]['scores'].cpu().numpy()
    
    # Yüksek skorluları filtrele (0.5 üstü)
    sonuc = []
    for i in range(len(skorlar)):
        if skorlar[i] > 0.5 and etiketler[i] not in [2,3]:  # Normal ve Akciğer hariç
            sonuc.append({
                'hastalik': hastaliklar.get(etiketler[i], 'Bilinmiyor'),
                'skor': float(skorlar[i]),
                'kutu': kutular[i].tolist()
            })
    
    # Resme kutuları çiz
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    for i, s in enumerate(sonuc):
        x1, y1, x2, y2 = map(int, s['kutu'])
        renk = renkler.get(etiketler[i], (255,255,255))
        cv2.rectangle(img_cv, (x1,y1), (x2,y2), renk, 3)
        cv2.putText(img_cv, f"{s['hastalik']} %{int(s['skor']*100)}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, renk, 2)
    
    # Sonuç resmini kaydet
    sonuc_path = os.path.join('static', 'sonuc.png')
    cv2.imwrite(sonuc_path, img_cv)
    
    # Cevap hazırla
    if len(sonuc) == 0:
        mesaj = "Hastalık tespit edilmedi"
    else:
        mesaj = f"{len(sonuc)} opasite bulundu"
        for s in sonuc:
            mesaj += f"\n- {s['hastalik']}: %{int(s['skor']*100)}"
    
    return jsonify({
        'sonuc': mesaj,
        'resim': '/static/sonuc.png?t=' + str(np.random.random())
    })

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)