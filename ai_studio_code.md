# PROMPT : Création d'un modèle ultra-léger pour Mobile (YOLOv11-Nano + MobileNetV3-Small) sous PyTorch

**Contexte pour l'IA :**
Agis en tant qu'ingénieur expert en Vision par Ordinateur (Computer Vision), en Deep Learning et en Edge AI (déploiement sur mobile Android/iOS), avec une expertise pointue sur **PyTorch**, `torchvision`, et l'architecture interne des modèles YOLO d'Ultralytics (spécifiquement la dernière version, YOLOv11).

**Objectif de la tâche :**
Je souhaite construire "from scratch" l'architecture d'un modèle de détection d'objets extrêmement léger et rapide, destiné à tourner en temps réel sur un CPU de smartphone. 
Le modèle doit utiliser **MobileNetV3-Small** comme *Backbone* et l'architecture spécifique de **YOLOv11n (version Nano)** pour le *Neck* et la *Head*.

**Voici les contraintes techniques et les étapes que tu dois respecter dans ton code :**

### 1. Le Backbone (MobileNetV3-Small)
- Utilise strictement `torchvision.models.mobilenet_v3_small` avec les poids pré-entraînés sur ImageNet (`weights='DEFAULT'`).
- Utilise les hooks ou `IntermediateLayerGetter` de PyTorch pour extraire les tenseurs à trois niveaux de résolution :
  - **P3** : Stride 8
  - **P4** : Stride 16
  - **P5** : Stride 32
- Projette les canaux de sortie de ce MobileNetV3-Small via des convolutions 1x1 pour qu'ils correspondent aux faibles dimensions attendues par le Neck d'un modèle YOLO de taille "Nano" (ex: 64, 128 et 256 canaux maximum).

### 2. Le Neck (Architecture YOLOv11n - Nano)
- Implémente le Neck (PANet / Feature Pyramid Network) en respectant le "scaling" de la version **Nano** de YOLOv11 (faible profondeur et faible largeur de canaux pour garantir moins de 3 millions de paramètres au total).
- Implémente les blocs spécifiques de YOLOv11 de manière optimisée pour le mobile : 
  - **C3k2** (CSP Bottleneck).
  - **C2PSA** (Cross-Stage Partial avec Spatial Attention) - *Note : garde ce module très léger*.
  - Les blocs de base **Conv-BN-SiLU**.
- Connecte les features P3, P4 et P5 via Up-sampling (`F.interpolate` en mode 'nearest' pour la rapidité) et concaténation.

### 3. La Head (Tête découplée YOLOv11n)
- Implémente la *Decoupled Head* de YOLOv11, mais garde les canaux internes très fins (ex: 64 canaux) pour économiser la batterie et le calcul sur mobile.
- Les deux branches :
  1. Régression des boîtes (DFL - Distribution Focal Loss) : sortie de dimension `4 * reg_max` (utilise `reg_max=16` par défaut).
  2. Classification : probabilités pour `num_classes`.
- Format "Anchor-free" basé sur les centres.

### 4. Code attendu
Fournis un script Python de qualité production, modulaire et orienté objet :
1. Les classes de base `nn.Module` (Conv, C3k2, C2PSA, DFL).
2. Une classe `MobileNetV3SmallBackbone`.
3. Une classe globale `YOLOv11nMobileNet(nn.Module)`.
4. Un script de test exécutable avec `num_classes=80`, un tenseur factice `torch.randn(1, 3, 320, 320)` (résolution typique pour mobile) et une passe *forward*.
5. **Affiche le nombre total de paramètres du modèle** à la fin du script de test (il doit idéalement être inférieur à 3 ou 4 millions).

### 5. Exportation ciblée Mobile (ONNX -> CoreML / TFLite)
- Fournis le code Python permettant d'exporter ce modèle PyTorch au format **ONNX** (`torch.onnx.export`) avec l'opset 11 ou 12 (meilleure compatibilité mobile).
- Ajoute des commentaires expliquant brièvement comment passer de cet ONNX à **TensorFlow Lite (Android)** ou **CoreML (iOS)**.

Rédige un code PyTorch robuste, sans erreur de dimension (shape mismatch), parfaitement taillé pour l'inférence temps réel sur mobile.