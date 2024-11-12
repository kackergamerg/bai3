import os
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.datasets import cifar10
from sklearn.decomposition import PCA

# Tắt các log thông báo không cần thiết từ TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tải dữ liệu CIFAR-10
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test_full = y_test_full.flatten()

# Danh sách nhãn muốn chọn, ví dụ: "Chim", "Mèo" và "Chó" (nhãn 2, 3, 5)
selected_labels = [2, 3, 5]

# Lọc dữ liệu chỉ với các nhãn đã chọn
train_mask = np.isin(y_train_full, selected_labels)
test_mask = np.isin(y_test_full, selected_labels)

x_train_selected = x_train_full[train_mask]
y_train_selected = y_train_full[train_mask]
x_test_selected = x_test_full[test_mask]
y_test_selected = y_test_full[test_mask]

# Chọn ngẫu nhiên 600 ảnh từ tập huấn luyện đã lọc và 300 ảnh từ tập kiểm tra đã lọc
np.random.seed(42)
train_indices = np.random.choice(len(x_train_selected), 600, replace=False)
test_indices = np.random.choice(len(x_test_selected), 300, replace=False)

x_train, y_train = x_train_selected[train_indices], y_train_selected[train_indices]
x_test, y_test = x_test_selected[test_indices], y_test_selected[test_indices]

# Chuẩn hóa dữ liệu (từ 0-255 thành 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Giảm kích thước ảnh và trích xuất đặc trưng bằng PCA
x_train = x_train.reshape(x_train.shape[0], -1)  # Flatten hình ảnh
x_test = x_test.reshape(x_test.shape[0], -1)

pca = PCA(n_components=50)  # Giảm số chiều dữ liệu
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Hàm huấn luyện và đánh giá mô hình
def train_and_evaluate_model(model, model_name):
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(x_train_pca, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(x_test_pca)
    prediction_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"{model_name} Results:")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Prediction Time: {prediction_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print("-" * 30)

svm_model = SVC(kernel='linear')
train_and_evaluate_model(svm_model, "SVM")

knn_model = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate_model(knn_model, "KNN")

dt_model = DecisionTreeClassifier(max_depth=10)
train_and_evaluate_model(dt_model, "Decision Tree")