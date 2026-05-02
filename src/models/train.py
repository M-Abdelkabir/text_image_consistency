import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from src.features.text_features import extract_text_features
from src.features.image_features import extract_image_features
from src.features.fusion import fuse_features
import os

def load_data(data_dir):
    # Implémentation simplifiée : lit les paires depuis les sous-dossiers
    texts, images, labels = [], [], []
    for label, category in enumerate(['incoherent', 'coherent']):
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            continue
        for fname in os.listdir(path):
            if fname.endswith('.txt'):
                with open(os.path.join(path, fname), 'r') as f:
                    texts.append(f.read())
                img_name = fname.replace('.txt', '.jpg')
                img_path = os.path.join(path, img_name)
                if os.path.exists(img_path):
                    images.append(img_path)
                    labels.append(label)
    return texts, images, labels

def main():
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/validation'
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Les répertoires de données n'existent pas. Veuillez générer les paires d'abord.")
        return

    texts_train, images_train, y_train = load_data(train_dir)
    texts_val, images_val, y_val = load_data(val_dir)
    
    if not texts_train or not texts_val:
        print("Pas de données trouvées dans les répertoires.")
        return

    # Features texte
    X_train_tfidf, X_train_extra, vectorizer = extract_text_features(texts_train)
    X_val_tfidf, X_val_extra, _ = extract_text_features(texts_val, vectorizer=vectorizer)
    
    # Features image
    X_train_img = np.array([extract_image_features(p) for p in images_train])
    X_val_img = np.array([extract_image_features(p) for p in images_val])
    
    # Fusion
    X_train = np.array([fuse_features(X_train_tfidf[i], X_train_extra[i], X_train_img[i]) for i in range(len(y_train))])
    X_val = np.array([fuse_features(X_val_tfidf[i], X_val_extra[i], X_val_img[i]) for i in range(len(y_val))])
    
    # GridSearch
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2,5,10]}
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    print("Meilleurs paramètres :", grid.best_params_)
    print("F1 score moyen :", grid.best_score_)
    
    # Sauvegarde
    os.makedirs('api', exist_ok=True)
    joblib.dump(grid.best_estimator_, 'api/model.pkl')
    joblib.dump(vectorizer, 'api/vectorizer.pkl')

if __name__ == '__main__':
    main()
