from sklearn.feature_extraction.text import CountVectorizer

max_words = 5000

# BoW ile metinleri vektörleştirme
bow = CountVectorizer(max_features=max_words)
X_train_bow = bow.fit_transform(X_train).toarray()
X_test_bow = bow.transform(X_test).toarray()

# Modelin girişini değiştirelim
def create_amazon_review_model_bow():
    model = Sequential()
    
    # 1. Katman: BoW ile vektörleştirilmiş veriyi kullanıyoruz, embedding yok.
    
    # 2. Katman: Ara Katman 1
    model.add(Dense(64, activation='relu', input_dim=X_train_bow.shape[1]))
    
    # 3. Katman: Ara Katman 2
    model.add(Dense(32, activation='relu'))
    
    # 4. Katman: Çıkış Katmanı
    model.add(Dense(1, activation='sigmoid'))  # Sentiment analizi için sigmoid
    
    # Modeli derleyelim
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Modeli oluşturalım
model_bow = create_amazon_review_model_bow()

# Modeli eğitelim
history_bow = model_bow.fit(X_train_bow, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Test seti ile tahmin yapalım
y_pred_bow = (model_bow.predict(X_test_bow) > 0.5).astype('int32')

# Doğruluğu ölçelim
accuracy_bow = accuracy_score(y_test, y_pred_bow)
print(f"BoW ile model doğruluğu: {accuracy_bow:.2f}")
