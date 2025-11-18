"""
Модуль для загрузки и предобработки данных датасета House Prices
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import json
import os


class DataProcessor:
    """Класс для обработки данных House Prices"""
    
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.test_ids = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.preprocessing_info = {}
        
    def load_datasets(self):
        """Загрузка train.csv и test.csv"""
        try:
            train_path = os.path.join(self.data_folder, 'train.csv')
            test_path = os.path.join(self.data_folder, 'test.csv')
            
            print(f"Загрузка {train_path}...")
            self.train_df = pd.read_csv(train_path)
            print(f"✓ Train загружен: {self.train_df.shape}")
            
            print(f"Загрузка {test_path}...")
            self.test_df = pd.read_csv(test_path)
            print(f"✓ Test загружен: {self.test_df.shape}")
            
            # Сохраняем ID для submission
            self.test_ids = self.test_df['Id'].copy()
            
            print(f"\nЦелевая переменная (SalePrice) статистика:")
            print(self.train_df['SalePrice'].describe())
            
            return True
            
        except FileNotFoundError as e:
            print(f"✗ Ошибка: файл не найден - {e}")
            return False
        except Exception as e:
            print(f"✗ Ошибка при загрузке: {e}")
            return False
    
    def analyze_missing_values(self, df):
        """Анализ пропущенных значений"""
        missing = df.isnull().sum()
        missing_percent = 100 * missing / len(df)
        missing_table = pd.DataFrame({
            'Пропусков': missing,
            'Процент': missing_percent
        })
        missing_table = missing_table[missing_table['Пропусков'] > 0].sort_values(
            'Процент', ascending=False
        )
        return missing_table
    
    def preprocess_data(self):
        """Комплексная предобработка данных"""
        if self.train_df is None or self.test_df is None:
            print("✗ Сначала загрузите датасеты!")
            return False
        
        print("\n=== ПРЕДОБРАБОТКА ДАННЫХ ===\n")
        
        # Анализ пропусков в train
        print("Анализ пропущенных значений в train:")
        missing_train = self.analyze_missing_values(self.train_df)
        if len(missing_train) > 0:
            print(missing_train.head(10))
        else:
            print("Пропусков нет")
        
        # Отделяем целевую переменную
        y = self.train_df['SalePrice'].copy()
        
        # Удаляем Id и SalePrice из train
        train_features = self.train_df.drop(['Id', 'SalePrice'], axis=1)
        test_features = self.test_df.drop(['Id'], axis=1)
        
        # Объединяем для единой обработки
        all_data = pd.concat([train_features, test_features], axis=0, ignore_index=True)
        
        print(f"\nОбъединенный датасет: {all_data.shape}")
        
        # Разделяем на числовые и категориальные признаки
        numeric_features = all_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = all_data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Числовых признаков: {len(numeric_features)}")
        print(f"Категориальных признаков: {len(categorical_features)}")
        
        # Обработка числовых признаков
        print("\n--- Обработка числовых признаков ---")
        numeric_imputer = SimpleImputer(strategy='median')
        all_data[numeric_features] = numeric_imputer.fit_transform(all_data[numeric_features])
        
        # Обработка категориальных признаков
        print("--- Обработка категориальных признаков ---")
        for col in categorical_features:
            # Заполняем пропуски наиболее частым значением или 'None'
            if all_data[col].isnull().sum() > 0:
                most_frequent = all_data[col].mode()
                if len(most_frequent) > 0:
                    all_data[col].fillna(most_frequent[0], inplace=True)
                else:
                    all_data[col].fillna('None', inplace=True)
            
            # Label Encoding для категориальных признаков
            le = LabelEncoder()
            all_data[col] = le.fit_transform(all_data[col].astype(str))
            self.label_encoders[col] = le
        
        print(f"✓ Закодировано {len(categorical_features)} категориальных признаков")
        
        # Разделяем обратно на train и test
        train_size = len(train_features)
        processed_train = all_data.iloc[:train_size].copy()
        processed_test = all_data.iloc[train_size:].copy()
        
        # Масштабирование
        print("\n--- Масштабирование признаков ---")
        self.X_train = self.scaler.fit_transform(processed_train)
        self.X_test = self.scaler.transform(processed_test)
        self.y_train = y.values
        self.feature_names = processed_train.columns.tolist()
        
        print(f"✓ X_train: {self.X_train.shape}")
        print(f"✓ X_test: {self.X_test.shape}")
        print(f"✓ y_train: {self.y_train.shape}")
        
        # Сохраняем информацию о предобработке
        self.preprocessing_info = {
            'n_features': len(self.feature_names),
            'n_numeric': len(numeric_features),
            'n_categorical': len(categorical_features),
            'n_train_samples': self.X_train.shape[0],
            'n_test_samples': self.X_test.shape[0],
            'target_mean': float(np.mean(self.y_train)),
            'target_std': float(np.std(self.y_train))
        }
        
        return True
    
    def get_preprocessed_data(self):
        """Получить предобработанные данные"""
        return self.X_train, self.X_test, self.y_train, self.test_ids
    
    def get_feature_names(self):
        """Получить названия признаков"""
        return self.feature_names
    
    def save_preprocessing_info(self, filepath='preprocessing_info.json'):
        """Сохранить информацию о предобработке"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.preprocessing_info, f, indent=4, ensure_ascii=False)
            print(f"✓ Информация о предобработке сохранена в {filepath}")
            return True
        except Exception as e:
            print(f"✗ Ошибка при сохранении: {e}")
            return False


def create_manual_prediction_sample():
    """Создать пример для ручного предсказания"""
    sample = {
        'MSSubClass': 60,
        'MSZoning': 'RL',
        'LotFrontage': 65.0,
        'LotArea': 8450,
        'Street': 'Pave',
        'LotShape': 'Reg',
        'LandContour': 'Lvl',
        'Utilities': 'AllPub',
        'LotConfig': 'Inside',
        'LandSlope': 'Gtl',
        'Neighborhood': 'CollgCr',
        'OverallQual': 7,
        'OverallCond': 5,
        'YearBuilt': 2003,
        'YearRemodAdd': 2003,
        'GrLivArea': 1710,
        'FullBath': 2,
        'BedroomAbvGr': 3,
        'TotRmsAbvGrd': 8,
        'GarageCars': 2,
        'GarageArea': 548
    }
    return sample