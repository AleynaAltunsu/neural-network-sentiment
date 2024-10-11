# Gerekli kütüphaneleri yükleyelim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten

# 1. Veri Yükleme ve İşleme
# Amazon Review datasetini yükleyelim (örnek veri: Cell_Phones_and_Accessories_5.json)
data = pd.read_json("/kaggle/input/cell-phones-andaccessories-5/Cell_Phones_and_Accessories_5.json", lines=True)

# Veri setindeki gerekli sütunları inceleyelim
print(data.head())

# Gerekli sütunları seçelim, 'reviewText' ve 'sentiment' olarak güncelleyin.
# Bu örnekte 'sentiment' veya uygun bir etiket kolonu olup olmadığını kontrol edin.
# Örneğin, 1 ile 5 arası değerlendirmeleri olumlu ve 1 ile 2 arası olumsuz olarak işleyebilirsiniz.

# Sentiment (etiket) verisini binary forma çevirelim
data['sentiment'] = np.where(data['overall'] >= 3, 1, 0)  # 3 ve üzeri olumlu, altı olumsuz

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['reviewText'], data['sentiment'], test_size=0.2, random_state=42)

# 2. Metinlerin Vektörize Edilmesi
max_words = 5000  # Maksimum kelime sayısı
max_len = 200     # Maksimum inceleme uzunluğu

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# 3. 4 Katmanlı Nöral Ağ Modeli Oluşturma
def create_amazon_review_model():
    model = Sequential()
    
    # 1. Katman: Embedding
    model.add(Embedding(input_dim=max_words, output_dim=128, input_length=max_len))
    
    # 2. Katman: Ara Katman 1
    model.add(Flatten())
    
    # 3. Katman: Ara Katman 2
    model.add(Dense(64, activation='relu'))
    
    # 4. Katman: Ara Katman 3
    model.add(Dense(32, activation='relu'))
    
    # 5. Katman: Çıkış Katmanı
    model.add(Dense(1, activation='sigmoid'))  # Sentiment analizi için sigmoid
    
    # Modeli derleyelim
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Modeli oluşturalım
model = create_amazon_review_model()

# 4. Modeli Eğitme ve Test Etme
# Modeli eğitelim
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Test seti ile tahmin yapalım
y_pred = (model.predict(X_test_pad) > 0.5).astype('int32')

# Modelin doğruluğunu ölçelim
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluğu: {accuracy:.2f}")

# Detaylı sınıflandırma raporu
print(classification_report(y_test, y_pred))
