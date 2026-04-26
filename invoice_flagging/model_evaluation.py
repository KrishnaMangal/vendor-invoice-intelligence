from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer

def train_random_forest_classifier(X_train_scaled, y_train):
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    
    # Smaller parameter grid to reduce runtime and avoid terminal timeouts.
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [None, 10],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [1,2],
        'max_features': ['sqrt'],
        'criterion': ['gini', 'entropy']
    }
    
    # Using F1 Score as the metric for class imbalance
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        scoring=scorer, 
        cv=5, 
        n_jobs=1, 
        verbose=0
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Return the best tuned model
    return grid_search.best_estimator_

def evaluate_classifier(model, X_test_scaled, y_test):
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"\n{model} Performance")
    print(f"Accuracy: {accuracy:.2f}")
    print(report)
