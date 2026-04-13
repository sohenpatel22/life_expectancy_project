import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Import the functions we wrote in the other files
from data_loader import load_and_split_data
from preprocess import preprocess_data

def main():
    # 1. Load Data (Now passing the URL)
    data_url = 'https://raw.githubusercontent.com/Sabaae/Dataset/main/LifeExpectancy.csv'
    X_train, X_test, y_train, y_test = load_and_split_data(url=data_url)
    
    # 2. Preprocess
    X_train_scaled, X_test_scaled, imputer, scaler = preprocess_data(X_train, X_test)
    
    # 3. Train Model 
    print("Training KNN model")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    
    # 4. Evaluate
    print("Evaluating model")
    y_pred = knn.predict(X_test_scaled)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # 5. Save Artifacts
    print("Saving model and scaler")
    joblib.dump(knn, 'knn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Pipeline complete!")

if __name__ == "__main__":
    main()