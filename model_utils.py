"""
Модуль для обучения, оценки и сохранения моделей регрессии
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import pickle
import json
from datetime import datetime
import os

RESULTS_FOLDER = "results"
MODELS_FOLDER = "models"


class ModelTrainer:
    """Класс для обучения и оценки моделей регрессии"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def train_baseline(self, X_train, y_train):
        """Обучить baseline модель (DummyRegressor)"""
        print("\n--- Baseline: DummyRegressor (mean) ---")
        
        model = DummyRegressor(strategy='mean')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.4f}")
        
        self.models['Baseline'] = model
        self.results['Baseline'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'train_time': 0.0
        }
        
        return model
    
    def train_linear_regression(self, X_train, y_train):
        """Обучить Linear Regression"""
        print("\n--- Linear Regression ---")
        
        start_time = datetime.now()
        model = LinearRegression()
        model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = model.predict(X_train)
        
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.4f}")
        print(f"Время обучения: {train_time:.2f}s")
        
        self.models['LinearRegression'] = model
        self.results['LinearRegression'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'train_time': train_time
        }
        
        return model
    
    def train_random_forest(self, X_train, y_train, use_grid_search=False):
        """Обучить Random Forest с опциональным GridSearch"""
        print("\n--- Random Forest Regressor ---")
        
        if use_grid_search:
            print("Выполняется GridSearchCV (это может занять время)...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf, param_grid, cv=3, 
                scoring='neg_mean_squared_error',
                verbose=1, n_jobs=-1
            )
            
            start_time = datetime.now()
            grid_search.fit(X_train, y_train)
            train_time = (datetime.now() - start_time).total_seconds()
            
            model = grid_search.best_estimator_
            print(f"Лучшие параметры: {grid_search.best_params_}")
        else:
            start_time = datetime.now()
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            train_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = model.predict(X_train)
        
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.4f}")
        print(f"Время обучения: {train_time:.2f}s")
        
        self.models['RandomForest'] = model
        self.results['RandomForest'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'train_time': train_time
        }
        
        return model
    
    def train_xgboost(self, X_train, y_train, use_random_search=False):
        """Обучить XGBoost с опциональным RandomizedSearch"""
        print("\n--- XGBoost Regressor ---")
        
        if use_random_search:
            print("Выполняется RandomizedSearchCV (это может занять время)...")
            
            param_distributions = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            random_search = RandomizedSearchCV(
                xgb_model, param_distributions,
                n_iter=10, cv=3,
                scoring='neg_mean_squared_error',
                verbose=1, random_state=42, n_jobs=-1
            )
            
            start_time = datetime.now()
            random_search.fit(X_train, y_train)
            train_time = (datetime.now() - start_time).total_seconds()
            
            model = random_search.best_estimator_
            print(f"Лучшие параметры: {random_search.best_params_}")
        else:
            start_time = datetime.now()
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            train_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = model.predict(X_train)
        
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.4f}")
        print(f"Время обучения: {train_time:.2f}s")
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'train_time': train_time
        }
        
        return model
    
    def train_svr(self, X_train, y_train):
        """Обучить Support Vector Regressor"""
        print("\n--- Support Vector Regressor ---")
        
        start_time = datetime.now()
        model = SVR(kernel='rbf', C=100, epsilon=0.1)
        model.fit(X_train, y_train)
        train_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = model.predict(X_train)
        
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)
        
        print(f"MAE:  ${mae:,.2f}")
        print(f"RMSE: ${rmse:,.2f}")
        print(f"R²:   {r2:.4f}")
        print(f"Время обучения: {train_time:.2f}s")
        
        self.models['SVR'] = model
        self.results['SVR'] = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'train_time': train_time
        }
        
        return model
    
    def train_all_models(self, X_train, y_train, feature_names=None):
        """Обучить все модели"""
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        self.feature_names = feature_names
        
        # Baseline
        self.train_baseline(X_train, y_train)
        
        # Linear Regression
        self.train_linear_regression(X_train, y_train)
        
        # Random Forest
        self.train_random_forest(X_train, y_train, use_grid_search=False)
        
        # XGBoost
        self.train_xgboost(X_train, y_train, use_random_search=False)
        
        # SVR (может быть медленным на больших данных)
        if X_train.shape[0] < 5000:
            self.train_svr(X_train, y_train)
        else:
            print("\n--- SVR пропущен (слишком много данных) ---")
        
        return self.models
    
    def compare_models(self):
        """Сравнить все обученные модели"""
        print("\n" + "="*60)
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('RMSE')
        
        print(results_df.to_string())
        
        # Выбираем лучшую модель по RMSE
        best_name = results_df.index[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n✓ Лучшая модель: {best_name}")
        print(f"  RMSE: ${results_df.loc[best_name, 'RMSE']:,.2f}")
        print(f"  R²:   {results_df.loc[best_name, 'R2']:.4f}")
        
        return results_df
    
    def error_analysis(self, X_train, y_train, top_n=10):
        """Анализ ошибок: показать объекты с наибольшей ошибкой"""
        if self.best_model is None:
            print("✗ Сначала обучите модели и выберите лучшую!")
            return
        
        print(f"\n=== АНАЛИЗ ОШИБОК ({self.best_model_name}) ===\n")
        
        y_pred = self.best_model.predict(X_train)
        errors = np.abs(y_train - y_pred)
        
        # Находим индексы с наибольшими ошибками
        worst_indices = np.argsort(errors)[-top_n:][::-1]
        
        print(f"Топ-{top_n} объектов с наибольшей ошибкой:\n")
        print(f"{'№':<4} {'Реальная цена':<15} {'Предсказание':<15} {'Ошибка':<15}")
        print("-" * 50)
        
        for i, idx in enumerate(worst_indices, 1):
            real = y_train[idx]
            pred = y_pred[idx]
            error = errors[idx]
            print(f"{i:<4} ${real:<14,.0f} ${pred:<14,.0f} ${error:<14,.0f}")
        
        print(f"\nСредняя ошибка: ${np.mean(errors):,.2f}")
        print(f"Медианная ошибка: ${np.median(errors):,.2f}")
    
    def get_feature_importance(self, top_n=20):
        """Получить важность признаков (для древесных моделей)"""
        if self.best_model is None:
            print("✗ Сначала обучите модели!")
            return None
        
        if not hasattr(self.best_model, 'feature_importances_'):
            print(f"✗ Модель {self.best_model_name} не поддерживает feature_importances_")
            return None
        
        if self.feature_names is None:
            print("✗ Названия признаков не заданы")
            return None
        
        print(f"\n=== ВАЖНОСТЬ ПРИЗНАКОВ ({self.best_model_name}) ===\n")
        
        importances = self.best_model.feature_importances_
        feature_importance = list(zip(self.feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Топ-{top_n} самых важных признаков:\n")
        for i, (name, importance) in enumerate(feature_importance[:top_n], 1):
            print(f"{i:2d}. {name:25s} - {importance:.4f}")
        
        return feature_importance
    
    def save_model(self, filepath='best_model.pkl'):
        """Сохранить лучшую модель"""
        if self.best_model is None:
            print("✗ Нет обученной модели для сохранения!")
            return False
        
        try:
            
            if not filepath.endswith(".pk1"):
                filepath += ".pk1"

            filepath = os.path.join(MODELS_FOLDER, filepath)

            with open(filepath, 'wb') as f:
                pickle.dump({
                    'model': self.best_model,
                    'model_name': self.best_model_name,
                    'feature_names': self.feature_names,
                    'results': self.results
                }, f)
            print(f"✓ Модель сохранена в {filepath}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при сохранении: {e}")
            return False
    
    def load_model(self, filepath='best_model.pkl'):
        """Загрузить модель"""
        try:
            if not filepath.endswith(".pkl"):
                filepath += ".pkl"

            filepath = os.path.join(MODELS_FOLDER, filepath)

            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.best_model = data['model']
            self.best_model_name = data['model_name']
            self.feature_names = data.get('feature_names')
            self.results = data.get('results', {})
            
            print(f"✓ Модель загружена из {filepath}")
            print(f"  Модель: {self.best_model_name}")
            return True
        except FileNotFoundError:
            print(f"✗ Файл {filepath} не найден")
            return False
        except Exception as e:
            print(f"✗ Ошибка при загрузке: {e}")
            return False
    
    def predict(self, X):
        """Сделать предсказание"""
        if self.best_model is None:
            print("✗ Модель не загружена!")
            return None
        
        return self.best_model.predict(X)
    
    def save_results_log(self, filepath='training_log.txt'):
        """Сохранить лог результатов"""
        try:
            filepath = os.path.join(RESULTS_FOLDER, filepath)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ\n")
                f.write("="*60 + "\n\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for model_name, metrics in self.results.items():
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  MAE:  ${metrics['MAE']:,.2f}\n")
                    f.write(f"  RMSE: ${metrics['RMSE']:,.2f}\n")
                    f.write(f"  R²:   {metrics['R2']:.4f}\n")
                    f.write(f"  Время: {metrics['train_time']:.2f}s\n")
                
                if self.best_model_name:
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Лучшая модель: {self.best_model_name}\n")
                    f.write(f"{'='*60}\n")
            
            print(f"✓ Лог сохранен в {filepath}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при сохранении лога: {e}")
            return False