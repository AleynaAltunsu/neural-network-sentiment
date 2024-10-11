from sklearn.feature_extraction.text import TfidfVectorizer

# Maksimum kelime sayısını belirleyelim
max_words = 5000  # TF-IDF için kullanılacak maksimum kelime sayısı

# TF-IDF ile metinleri vektörleştirme
tfidf = TfidfVectorizer(max_features=max_words)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()  # Train veri setini dönüştür
X_test_tfidf = tfidf.transform(X_test).toarray()  # Test veri setini dönüştür

print("X_train TF-IDF shape:", X_train_tfidf.shape)
print("X_test TF-IDF shape:", X_test_tfidf.shape)


# Modelin girişini değiştirelim
def create_amazon_review_model_tfidf():
    model = Sequential()
    
    # 1. Katman: TF-IDF ile vektörleştirilmiş veriyi kullanıyoruz, embedding yok.
    
    # 2. Katman: Ara Katman 1
    model.add(Dense(64, activation='relu', input_dim=X_train_tfidf.shape[1]))
    
    # 3. Katman: Ara Katman 2
    model.add(Dense(32, activation='relu'))
    
    # 4. Katman: Çıkış Katmanı
    model.add(Dense(1, activation='sigmoid'))  # Sentiment analizi için sigmoid
    
    # Modeli derleyelim
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Modeli oluşturalım
model_tfidf = create_amazon_review_model_tfidf()

# Modeli eğitelim
history_tfidf = model_tfidf.fit(X_train_tfidf, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Test seti ile tahmin yapalım
y_pred_tfidf = (model_tfidf.predict(X_test_tfidf) > 0.5).astype('int32')

# Doğruluğu ölçelim
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print(f"TF-IDF ile model doğruluğu: {accuracy_tfidf:.2f}")
