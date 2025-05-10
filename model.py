from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                           roc_curve, auc, precision_recall_curve,
                           classification_report)
import pandas as pd
import numpy as np
from data_cleaning import clean_data

# Load data once
df, df_reduced = clean_data()

def train_model(model_type='rf', n_estimators=100, learning_rate=0.1, max_depth=6, 
               min_samples_split=10, min_samples_leaf=4, hidden_layer_sizes=(100,)):
    try:
        X = df_reduced.drop('satisfaction_binary', axis=1)
        y = df_reduced['satisfaction_binary']

        # Drop specific features for tree-based models
        if model_type in ['rf', 'xgb', 'gbc']:
            X = X.drop(columns=[
                'online_boarding',
                'type_of_travel_Personal',
                'class_Economy',
                'in_flight_entertainment'
            ], errors='ignore')
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        # Model initialization with proper parameters
        if model_type == 'logreg':
            model = LogisticRegression(
                max_iter=n_estimators,
                solver='liblinear',
                random_state=42
            )
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight='balanced',
                random_state=42
            )
        elif model_type == 'xgb':
            model = XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=n_estimators,
                learning_rate_init=learning_rate,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cross-validation (skip for MLP)
        cv_score = None
        if model_type != 'mlp':
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
            cv_score = np.mean(cv_scores)
            print(f"✅ {model_type.upper()} CV Accuracy: {cv_score:.3f}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"🔸 {model_type.upper()} Results:")
        print(f"Test Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))

        # Calculate metrics if probabilities are available
        roc_auc, pr_auc = None, None
        fpr, tpr, precision, recall = None, None, None, None
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)

        # Feature importance/coefficients
        importance = None
        if model_type == 'logreg':
            importance = model.coef_[0]
        elif model_type in ['rf', 'xgb', 'gbc']:
            importance = model.feature_importances_

        return {
            'model': model,
            'accuracy': accuracy,
            'cm': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'pr_auc': pr_auc,
            'importance': importance,
            'feature_names': X.columns.tolist(),
            'cv_score': cv_score
        }

    except Exception as e:
        print(f"❌ Error training {model_type} model:")
        print(f"Error details: {str(e)}")
        raise