from IPython.display import display
import pandas as pd
import numpy as np
import os
import tensorflow.keras.preprocessing.image as kimage
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt

# Veri setindeki yolları göster
# for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

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

# Görselleri yüklemek için gerekli fonksiyon


def load_image(filename):
    img_path = os.path.join(
        'E:\\MIAS_project\\all-mias', f"{filename}.pgm")
    img = kimage.load_img(img_path, color_mode="grayscale")
    img_array = kimage.img_to_array(img)
    return tf.image.resize(img_array, (512, 512))


# Görselleri bir diziye yükle
X = np.array([load_image(img_id) for img_id in labels_df['REFNUM']])

# Verileri böl (yüzde 80 eğitim, yüzde 20 test) ve normalizasyon yap
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Verileri doğrula
print(f"İlk 5 etiket: {y_train[:5]}")
print(f"Eğitim veri setinin şekli: {X_train.shape}")

# İlk 5 görseli ve etiketlerini göster
for i in range(5):
    imagen_id = labels_df['REFNUM'].iloc[i]  # Görselin REFNUM'ını al
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    clase_nombre = [k for k, v in class_map.items() if v == y_train[i]][0]
    plt.title(f"Görsel: {imagen_id} - Etiket: {clase_nombre}")
    plt.show()

# Data Augmentation tanımlama
data_augmentation = tf.keras.Sequential([
    # Görselleri yatay ve dikey çevir
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),  # Görselleri rastgele döndür
    layers.RandomZoom(0.2),      # Rastgele zoom uygula
    layers.RandomContrast(0.2),  # Kontrastı rastgele değiştir
])

# Augmented veri setini görselleştirme (isteğe bağlı olarak ekleyebilirsiniz)
plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(X_train[i:i+1])
    plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0].numpy().squeeze(), cmap='gray')
    plt.axis("off")
plt.show()

input_layer = Input(shape=(512, 512, 1))
augmented_input = data_augmentation(input_layer)

# İlk konvolüsyon katmanı
x = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x = layers.MaxPooling2D((2, 2))(x)

# İkinci konvolüsyon katmanı
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Üçüncü konvolüsyon katmanı
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)

# Tam bağlantılı katmanlar
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)

# Çıkış katmanı
output_layer = layers.Dense(7, activation='softmax')(x)

# Model oluşturma
model = models.Model(inputs=input_layer, outputs=output_layer)

# Modeli derle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Model özetini göster
model.summary()

# Modeli eğit
history = model.fit(X_train, y_train, epochs=15,
                    batch_size=32, validation_data=(X_test, y_test))

# Eğitim ve doğrulama kayıplarını görselleştirme
plt.figure(figsize=(12, 6))

# Doğruluk (accuracy) grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# Kayıp (loss) grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)

# Grafikleri göster
plt.tight_layout()
plt.show()

# Eğitim ve doğrulama grafiğini kaydet
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(output_dir, 'training_validation_graphs.jpeg'))

# Modeli test setinde değerlendir
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# Eğitim setinde tahminler yapma
y_train_pred = model.predict(X_train)
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
plt.savefig(os.path.join(output_dir, 'train_confusion_matrix_cnn.jpeg'))
plt.show()

# Test setinde tahminler yapma
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Tahmin edilen sınıflar

# Confusion Matrix oluşturma
cm = confusion_matrix(y_test, y_pred_classes)

# Confusion Matrix'i görselleştirme
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=class_map.keys())
disp.plot(cmap='Blues', values_format='d')

plt.title('Confusion Matrix (Test Set)')
# Eğitim Confusion Matrix'ini kaydet
plt.savefig(os.path.join(output_dir, 'test_confusion_matrix_cnn.jpeg'))
plt.show()

# Eğitim sonrası türlerin sayısını ve görsel sınıflarını kaydetme
# Sınıf isimleri ve açıklamalar
class_descriptions = {
    0: "CALC - Calcification",
    1: "CIRC - Well-defined/circumscribed masses",
    2: "SPIC - Spiculated masses",
    3: "MISC - Other, ill-defined masses",
    4: "ARCH - Architectural distortion",
    5: "ASYM - Asymmetry",
    6: "NORM - Normal",
}

# Eğitim ve test setlerindeki örneklerin sınıf dağılımı
train_class_counts = pd.Series(y_train).value_counts().sort_index()
test_class_counts = pd.Series(y_test).value_counts().sort_index()

# Görsel sınıflarını içeren bir DataFrame oluşturma
results = pd.DataFrame({
    "Image ID": labels_df['REFNUM'],
    "Class ID": y,
    "Class Description": [class_descriptions[class_id] for class_id in y]
})

# Eğitim setindeki görsellerin sınıf bilgileri
train_results = pd.DataFrame({
    "Image ID": labels_df['REFNUM'][0:len(y_train)],
    "Class ID": y_train,
    "Class Description": [class_descriptions[class_id] for class_id in y_train]
})

# Test setindeki görsellerin sınıf bilgileri
test_results = pd.DataFrame({
    "Image ID": labels_df['REFNUM'][len(y_train):],
    "Class ID": y_test,
    "Class Description": [class_descriptions[class_id] for class_id in y_test]
})

# Excel dosyasına yazma
excel_path = os.path.join(output_dir, "classification_results_detailed.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    # Eğitim setindeki sınıf dağılımını ekleme
    train_class_counts_df = pd.DataFrame({
        "Class ID": train_class_counts.index,
        "Count": train_class_counts.values,
        "Class Description": [class_descriptions[class_id] for class_id in train_class_counts.index]
    })
    train_class_counts_df.to_excel(
        writer, sheet_name="Train Class Counts", index=False)

    # Test setindeki sınıf dağılımını ekleme
    test_class_counts_df = pd.DataFrame({
        "Class ID": test_class_counts.index,
        "Count": test_class_counts.values,
        "Class Description": [class_descriptions[class_id] for class_id in test_class_counts.index]
    })
    test_class_counts_df.to_excel(
        writer, sheet_name="Test Class Counts", index=False)

    # Eğitim setindeki görsel sınıf bilgilerini ekleme
    train_results.to_excel(
        writer, sheet_name="Train Image Classifications", index=False)

    # Test setindeki görsel sınıf bilgilerini ekleme
    test_results.to_excel(
        writer, sheet_name="Test Image Classifications", index=False)

    # Tüm görsellerin sınıf bilgilerini ekleme
    results.to_excel(
        writer, sheet_name="All Image Classifications", index=False)

print(f"Sonuçlar '{excel_path}' dosyasına kaydedildi.")
