import joblib
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report


X = joblib.load("X_data.pkl")
y = joblib.load("y_labels.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)


model = svm.SVC(kernel='linear') 
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

joblib.dump(model, "svm_cat_dog_model.pkl")
print("Model saved as svm_cat_dog_model.pkl")
