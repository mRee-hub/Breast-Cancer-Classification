# Gerekli kütüphanelerin yüklenmesi
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# Görselleri yüklemek için gerekli fonksiyon


def load_image(filename):
    img_path = os.path.join('E:\\MIAS_project\\all-mias', f"{filename}.pgm")
    img = tf.keras.preprocessing.image.load_img(
        img_path, color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    return tf.image.resize(img_array, (512, 512))


# Etiketleri yükle ve tekrar edenleri kaldır
info_file_path = 'E:\\MIAS_project\\Info.txt'
labels_df = pd.read_csv(info_file_path, delimiter=' ', header=0)
labels_df.columns = ['REFNUM', 'BG', 'CLASS',
                     'SEVERITY', 'X', 'Y', 'RADIUS', ' ']
labels_df = labels_df.drop_duplicates(subset='REFNUM')

# Sınıfları eşleştir
class_map = {'CALC': 0, 'CIRC': 1, 'SPIC': 2,
             'MISC': 3, 'ARCH': 4, 'ASYM': 5, 'NORM': 6}
y = labels_df['CLASS'].map(class_map).values

# Görselleri yükle ve 3 kanala dönüştür
X = np.array([tf.image.grayscale_to_rgb(load_image(img_id))
             for img_id in labels_df['REFNUM']])

# Verileri böl (80% eğitim, 20% test) ve normalizasyon yap
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# VGG16 modelini yükle
vggmodel = VGG16(weights="imagenet", include_top=False,
                 input_shape=(512, 512, 3))
for layers in (vggmodel.layers):
    layers.trainable = False

# Fully Connected Katmanlar
X = Flatten()(vggmodel.output)
X = Dense(4096, name='fc1', activation='relu')(X)
X = Dense(4096, name='fc2', activation='relu')(X)
predictions = Dense(7, activation="softmax")(X)
model_final = Model(vggmodel.input, predictions)

# Modeli Derleme
opt = Adam(learning_rate=0.0001)
model_final.compile(loss='sparse_categorical_crossentropy',
                    optimizer=opt, metrics=["accuracy"])

# Modelin Özeti
model_final.summary()

# Modeli eğitme
history = model_final.fit(X_train, y_train, epochs=20,
                          batch_size=32, validation_data=(X_test, y_test))

# Eğitim ve doğrulama kayıplarını görselleştirme
plt.figure(figsize=(12, 6))

# Accuracy grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# Loss grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

# Grafikleri kaydet ve göster
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'training_validation_graphs.jpeg'))
plt.show()

# Modeli test setinde değerlendir
test_loss, test_acc = model_final.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Eğitim setinde tahminler yapma
y_train_pred = model_final.predict(X_train)
y_train_pred_classes = np.argmax(
    y_train_pred, axis=1)  # Tahmin edilen sınıflar

# Eğitim seti için Confusion Matrix oluşturma
cm_train = confusion_matrix(y_train, y_train_pred_classes)

# Eğitim seti için Confusion Matrix'i görselleştirme
disp_train = ConfusionMatrixDisplay(
    confusion_matrix=cm_train, display_labels=class_map.keys())
disp_train.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix (Train Set)')
# Test Confusion Matrix'ini kaydet
plt.savefig(os.path.join(output_dir, 'train_confusion_matrix.jpeg'))
plt.show()

# Test setinde tahminler yapma
y_pred = model_final.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix oluşturma
cm = confusion_matrix(y_test, y_pred_classes)

# Confusion Matrix'i görselleştirme
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=class_map.keys())
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set)')
plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.jpeg'))
plt.show()
