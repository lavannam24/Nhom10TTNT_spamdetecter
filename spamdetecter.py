#Đọc dữ liệu và tiền xử lý
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# Đọc dữ liệu từ tệp CSV
df = pd.read_csv("testTraining_spam.csv", encoding='latin-1')
# Chọn cột chứa nhãn (spam/ham) và nội dung tin nhắn
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Chuyển nhãn thành dạng số: 'spam' -> 1, 'ham' -> 0
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

# Chia tập dữ liệu thành huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Biến đổi văn bản thành vector TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

#Huấn luyện mô hình Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Khởi tạo mô hình Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_tfidf)

# Đánh giá độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f'Độ chính xác của mô hình: {accuracy:.2f}')

#Kiểm tra với tin nhắn mới
def classify_message(msg):
    msg_tfidf = vectorizer.transform([msg])
    prediction = model.predict(msg_tfidf)
    return "spam" if prediction[0] == 1 else "ham"

message = ("dsafds")
print(f'Kết quả: {classify_message(message)}')

#luu mo hinh huan luyen
import joblib
joblib.dump(model, "luutientrinhtrainning.pkl") # Lưu mô hình
model = joblib.load("luutientrinhtrainning.pkl")  # Load mô hình đã lưu