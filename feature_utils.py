"""
Модуль для отбора признаков (Feature Selection)
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import json
from sklearn.linear_model import LinearRegression as _LR
from sklearn.metrics import r2_score

import os
RESULTS_FOLDER = "results"
MODELS_FOLDER = "models"

class FeatureSelector:
    """Класс для отбора наиболее важных признаков"""
    
    def __init__(self):
        self.selected_features_indices = None
        self.selected_features_names = None
        self.selection_method = None
        self.selection_info = {}
        
    def select_k_best(self, X, y, feature_names, k=20):
        """Отбор K лучших признаков по F-статистике"""
        print(f"\n=== SelectKBest: отбор {k} лучших признаков ===\n")
        
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Получаем индексы выбранных признаков
        self.selected_features_indices = selector.get_support(indices=True)
        self.selected_features_names = [feature_names[i] for i in self.selected_features_indices]
        
        # Получаем оценки (scores) для всех признаков
        scores = selector.scores_
        feature_scores = list(zip(feature_names, scores))
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("Топ-20 признаков по F-статистике:")
        for i, (name, score) in enumerate(feature_scores[:20], 1):
            print(f"{i:2d}. {name:25s} - {score:12.2f}")
        
        self.selection_method = 'SelectKBest'
        self.selection_info = {
            'method': 'SelectKBest',
            'k': k,
            'selected_features': self.selected_features_names
        }
        
        return X_selected, self.selected_features_names
    
    def select_rfe(self, X, y, feature_names, n_features=20):
        """Рекурсивное исключение признаков (RFE)"""
        print(f"\n=== RFE: отбор {n_features} признаков ===\n")
        
        estimator = LinearRegression()
        selector = RFE(estimator, n_features_to_select=n_features, step=5)
        X_selected = selector.fit_transform(X, y)
        
        # Получаем индексы выбранных признаков
        self.selected_features_indices = selector.get_support(indices=True)
        self.selected_features_names = [feature_names[i] for i in self.selected_features_indices]
        
        # Получаем ранги признаков
        rankings = selector.ranking_
        feature_rankings = list(zip(feature_names, rankings))
        feature_rankings.sort(key=lambda x: x[1])
        
        print("Топ-20 признаков по RFE (ранг 1 = лучший):")
        for i, (name, rank) in enumerate(feature_rankings[:20], 1):
            print(f"{i:2d}. {name:25s} - Ранг: {rank}")
        
        self.selection_method = 'RFE'
        self.selection_info = {
            'method': 'RFE',
            'n_features': n_features,
            'selected_features': self.selected_features_names
        }
        
        return X_selected, self.selected_features_names
    
    def select_pca(self, X, feature_names, n_components=20):
        """Отбор главных компонент (PCA)"""
        print(f"\n=== PCA: отбор {n_components} главных компонент ===\n")
        
        pca = PCA(n_components=n_components)
        X_selected = pca.fit_transform(X)
        
        # Для PCA мы не храним конкретные признаки, т.к. это линейные комбинации
        self.selected_features_indices = None
        self.selected_features_names = [f'PC{i+1}' for i in range(n_components)]
        
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print("Объясненная дисперсия по компонентам:")
        for i in range(min(10, n_components)):
            print(f"PC{i+1:2d}: {explained_variance[i]:6.2%} (накопл.: {cumulative_variance[i]:6.2%})")
        
        print(f"\nВсего объяснено дисперсии: {cumulative_variance[-1]:.2%}")
        
        self.selection_method = 'PCA'
        self.selection_info = {
            'method': 'PCA',
            'n_components': n_components,
            'explained_variance': float(cumulative_variance[-1]),
            'selected_features': self.selected_features_names
        }
        
        return X_selected, self.selected_features_names
    
    def select_vif(self, X, feature_names, threshold=10.0, max_features=30):
        """Отбор признаков по VIF (мультиколлинеарность)"""
        print(f"\n=== VIF: удаление признаков с VIF > {threshold} ===\n")
        
        X_df = pd.DataFrame(X, columns=feature_names)
        selected_features = feature_names.copy()
        
        iteration = 1
        while True:
            if len(selected_features) <= max_features:
                print(f"Достигнуто максимальное количество признаков: {max_features}")
                break
                
            vif_data = pd.DataFrame()
            vif_data["Feature"] = selected_features
            
            try:
                # Compute VIF manually without requiring statsmodels
                vif_values = []
                X_sub = X_df[selected_features]

                for i, col in enumerate(selected_features):
                    y_col = X_sub[col].values
                    X_others = X_sub.drop(columns=[col]).values

                    # If there are no other features, VIF is 0 (or 1)
                    if X_others.shape[1] == 0:
                        vif = 0.0
                    else:
                        lr = _LR()
                        lr.fit(X_others, y_col)
                        y_pred = lr.predict(X_others)
                        r2 = r2_score(y_col, y_pred)
                        # Protect against division by zero / perfect multicollinearity
                        if r2 >= 0.9999:
                            vif = float('inf')
                        else:
                            vif = 1.0 / (1.0 - r2)

                    vif_values.append(vif)

                vif_data["VIF"] = vif_values

                max_vif = vif_data["VIF"].replace([np.inf], np.nan).max()

                if np.isnan(max_vif) or max_vif <= threshold:
                    print(f"Все признаки имеют VIF <= {threshold}")
                    break

                # Удаляем признак с максимальным VIF
                feature_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Feature"]
                print(f"Итерация {iteration}: Удаляем '{feature_to_remove}' (VIF={vif_data['VIF'].max():.2f})")
                selected_features.remove(feature_to_remove)
                iteration += 1

            except Exception as e:
                print(f"Ошибка при расчете VIF: {e}")
                break
        
        print(f"\nОсталось признаков: {len(selected_features)}")
        
        # Получаем индексы выбранных признаков
        self.selected_features_indices = [i for i, name in enumerate(feature_names) 
                                          if name in selected_features]
        self.selected_features_names = selected_features
        
        X_selected = X[:, self.selected_features_indices]
        
        self.selection_method = 'VIF'
        self.selection_info = {
            'method': 'VIF',
            'threshold': threshold,
            'max_features': max_features,
            'n_features': len(selected_features),
            'selected_features': self.selected_features_names
        }
        
        return X_selected, self.selected_features_names
    
    def transform_test(self, X_test):
        """Применить отбор признаков к тестовым данным"""
        if self.selected_features_indices is None:
            if self.selection_method == 'PCA':
                print("✗ Для PCA нужно сохранить и загрузить трансформер отдельно")
                return X_test
            else:
                print("✗ Сначала выполните отбор признаков на train данных")
                return X_test
        
        return X_test[:, self.selected_features_indices]
    
    def save_selection_info(self, filepath='feature_selection_info.json'):
        """Сохранить информацию об отборе признаков"""
        try:
            filepath = os.path.join(RESULTS_FOLDER, filepath)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.selection_info, f, indent=4, ensure_ascii=False)
            print(f"✓ Информация об отборе признаков сохранена в {filepath}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при сохранении: {e}")
            return False
    
    def get_selected_features(self):
        """Получить список выбранных признаков"""
        return self.selected_features_names


def compare_feature_selection_methods(X, y, feature_names, k=20):
    """Сравнить различные методы отбора признаков"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТОДОВ ОТБОРА ПРИЗНАКОВ")
    print("="*60)
    
    results = {}
    
    # SelectKBest
    selector1 = FeatureSelector()
    X_kb, features_kb = selector1.select_k_best(X, y, feature_names, k=k)
    results['SelectKBest'] = features_kb
    
    # RFE
    selector2 = FeatureSelector()
    X_rfe, features_rfe = selector2.select_rfe(X, y, feature_names, n_features=k)
    results['RFE'] = features_rfe
    
    # PCA
    selector3 = FeatureSelector()
    X_pca, features_pca = selector3.select_pca(X, feature_names, n_components=k)
    results['PCA'] = features_pca
    
    print("\n" + "="*60)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*60)
    print(f"SelectKBest: {len(features_kb)} признаков")
    print(f"RFE:         {len(features_rfe)} признаков")
    print(f"PCA:         {len(features_pca)} компонент")
    
    # Пересечение между SelectKBest и RFE
    common_features = set(features_kb) & set(features_rfe)
    print(f"\nОбщих признаков между SelectKBest и RFE: {len(common_features)}")
    if len(common_features) > 0:
        print("Общие признаки:", list(common_features)[:10])
    
    return results