# YOLOv11n-MobileNetV3 (Ultra-Lightweight Detection for Mobile)

Ce projet implémente une architecture de détection d'objets ultra-légère combinant le **Backbone MobileNetV3-Small** et le **Neck/Head de YOLOv11n (Nano)**. Le modèle est optimisé pour une inférence rapide sur les processeurs de smartphones (Android/iOS).

## 📊 Dataset
Vous pouvez télécharger le dataset nécessaire à l'entraînement ici : [Télécharger le Dataset (Kaggle/Google Storage)](https://storage.googleapis.com/kaggle-data-sets/6709331/10808746/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260316%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260316T115829Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=9228ca4bcae9a9487c3b1bb3e9d6f457498c386a58e934ef29efb4b2efcc814f6eee93aaff4cd31636ad28a07479a3eadcfea9470b8b75df8ef5fbec08304aa9d209dedaaad38823624569354b22a539f0d14557a1993fdf169bfadbf71d4f6b7a9aae65759a045d46db8a3a558ee1a6390e60f2e4a8f6a4373c4c027a3ed34458fdb376c645d4029912f5485c184d10ef405d9f1c5d8b588c45485fa55e749010108110b103ee879d4ba3efdc5f998d511ba2f67a4bb6ab47aaa3c34aa73f1dea93be17cd37a048be4d12da8e00bd8f58ae6e7f552f1b856a378a056a9bfd7f4132d1c96267633eac98b17345422c9e4ac96ced6e95c7f50d035d8b7588915e)

## 🚀 Guide d'installation et d'utilisation (De A à Z)

### 1. Prérequis
- Python 3.8 ou plus récent.
- Accès à Internet pour télécharger les poids pré-entraînés du backbone.

### 2. Configuration de l'environnement
Ouvrez votre terminal dans le dossier racine du projet et suivez ces étapes :

```bash
# 1. Créer l'environnement virtuel (venv)
python -m venv venv

# 2. Activer l'environnement virtuel
# Sur Windows :
.\venv\Scripts\activate
# Sur Linux/macOS :
source venv/bin/activate

# 3. Installer les dépendances
pip install torch torchvision onnx onnxscript pillow
```

### 3. Entraînement du modèle
Le script `train.py` permet d'entraîner le modèle sur vos données situées dans le dossier `data/`.

**Structure attendue des données :**
```
data/
├── train/
│   ├── images/  # Fichiers .jpg ou .png
│   └── labels/  # Fichiers .txt (format YOLO: class x_c y_c w h)
├── valid/
│   ├── images/
│   └── labels/
```

**Lancer l'entraînement :**
```bash
python train.py
```
*Le script sauvegardera un fichier `.pth` à chaque époque (ex: `yolov11n_mobile_epoch_1.pth`).*

### 4. Test et Exportation ONNX
Une fois entraîné (ou pour tester l'architecture vide), lancez le script principal :

```bash
python model.py
```
- **Vérifie** le nombre de paramètres (doit être < 4M).
- **Exporte** le modèle au format `yolov11n_mobilenet_v3.onnx`.

## 📱 Déploiement Mobile

Une fois le fichier `.onnx` généré, vous pouvez le convertir pour votre plateforme cible :

### Pour Android (TFLite)
Utilisez `onnx2tf` pour une conversion optimisée :
```bash
pip install onnx2tf
onnx2tf -i yolov11n_mobilenet_v3.onnx -o saved_model
```

### Pour iOS (CoreML)
Utilisez `coremltools` :
```python
import coremltools as ct
model = ct.converters.onnx.convert(model='yolov11n_mobilenet_v3.onnx')
model.save('yolov11n_mobilenet_v3.mlmodel')
```

---
*Note : Pour un entraînement de production avec Target Assignment (TAL) complet, il est recommandé d'intégrer ce modèle dans la suite Ultralytics.*
