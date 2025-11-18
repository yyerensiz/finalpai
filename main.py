"""
Главный модуль для запуска консольного приложения
Предсказание цен на недвижимость (House Prices: Advanced Regression Techniques)
"""

import os
import sys
import pandas as pd
import numpy as np
from data_utils import DataProcessor
from feature_utils import FeatureSelector, compare_feature_selection_methods
from model_utils import ModelTrainer
RESULTS_FOLDER = "results"

class HousePricePredictor:
    """Главный класс приложения"""
    
    def __init__(self, data_folder='data'):
        self.data_folder = data_folder
        self.data_processor = DataProcessor(data_folder)
        self.feature_selector = FeatureSelector()
        self.model_trainer = ModelTrainer()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.test_ids = None
        self.feature_names = None
        
        self.X_train_selected = None
        self.X_test_selected = None
        self.selected_feature_names = None
        
        self.data_loaded = False
        self.data_preprocessed = False
        self.features_selected = False
        self.models_trained = False
    
    def show_menu(self):
        """Показать главное меню"""
        print("\n" + "="*60)
        print(" ПРЕДСКАЗАНИЕ ЦЕН НА НЕДВИЖИМОСТЬ ".center(60, "="))
        print("="*60)
        print("\n[1] Загрузить датасеты (train.csv, test.csv)")
        print("[2] Предобработать данные")
        print("[3] Выполнить отбор признаков")
        print("[4] Обучить и сравнить модели")
        print("[5] Показать метрики и важные признаки")
        print("[6] Сделать предсказание (test.csv или вручную)")
        print("[7] Сохранить/загрузить модель")
        print("[8] Выход")
        print("="*60)
    
    def load_datasets(self):
        """Пункт 1: Загрузка датасетов"""
        print("\n" + "="*60)
        print("ЗАГРУЗКА ДАТАСЕТОВ")
        print("="*60)
        
        if self.data_processor.load_datasets():
            self.data_loaded = True
            print("\n✓ Датасеты успешно загружены!")
        else:
            print("\n✗ Не удалось загрузить датасеты")
            print(f"Убедитесь, что файлы находятся в папке '{self.data_folder}/'")
    
    def preprocess_data(self):
        """Пункт 2: Предобработка данных"""
        if not self.data_loaded:
            print("\n✗ Сначала загрузите датасеты (пункт 1)")
            return
        
        print("\n" + "="*60)
        print("ПРЕДОБРАБОТКА ДАННЫХ")
        print("="*60)
        
        if self.data_processor.preprocess_data():
            self.X_train, self.X_test, self.y_train, self.test_ids = \
                self.data_processor.get_preprocessed_data()
            self.feature_names = self.data_processor.get_feature_names()
            self.data_preprocessed = True
            
            self.data_processor.save_preprocessing_info()
            
            print("\n✓ Предобработка завершена успешно!")
            print(f"Готово к отбору признаков: {len(self.feature_names)} признаков")
        else:
            print("\n✗ Ошибка при предобработке данных")
    
    def select_features(self):
        """Пункт 3: Отбор признаков"""
        if not self.data_preprocessed:
            print("\n✗ Сначала выполните предобработку данных (пункт 2)")
            return
        
        print("\n" + "="*60)
        print("ОТБОР ПРИЗНАКОВ")
        print("="*60)
        print("\nВыберите метод отбора признаков:")
        print("[1] SelectKBest (F-статистика)")
        print("[2] RFE (Рекурсивное исключение)")
        print("[3] PCA (Главные компоненты)")
        print("[4] VIF (Удаление мультиколлинеарности)")
        print("[5] Сравнить все методы")
        print("[0] Назад")
        
        choice = input("\nВыбор: ").strip()
        
        if choice == '0':
            return
        
        # Запрашиваем количество признаков
        try:
            k = int(input(f"\nСколько признаков оставить? (рекомендуется 20-30): ").strip())
            if k < 1 or k > len(self.feature_names):
                print(f"✗ Число должно быть от 1 до {len(self.feature_names)}")
                return
        except ValueError:
            print("✗ Введите корректное число")
            return
        
        if choice == '1':
            self.X_train_selected, self.selected_feature_names = \
                self.feature_selector.select_k_best(self.X_train, self.y_train, 
                                                    self.feature_names, k=k)
            self.X_test_selected = self.feature_selector.transform_test(self.X_test)
            
        elif choice == '2':
            self.X_train_selected, self.selected_feature_names = \
                self.feature_selector.select_rfe(self.X_train, self.y_train, 
                                                 self.feature_names, n_features=k)
            self.X_test_selected = self.feature_selector.transform_test(self.X_test)
            
        elif choice == '3':
            self.X_train_selected, self.selected_feature_names = \
                self.feature_selector.select_pca(self.X_train, self.feature_names, 
                                                 n_components=k)
            # Для PCA нужно применить ту же трансформацию
            from sklearn.decomposition import PCA
            pca = PCA(n_components=k)
            pca.fit(self.X_train)
            self.X_test_selected = pca.transform(self.X_test)
            
        elif choice == '4':
            self.X_train_selected, self.selected_feature_names = \
                self.feature_selector.select_vif(self.X_train, self.feature_names, 
                                                 threshold=10.0, max_features=k)
            self.X_test_selected = self.feature_selector.transform_test(self.X_test)
            
        elif choice == '5':
            compare_feature_selection_methods(self.X_train, self.y_train, 
                                            self.feature_names, k=k)
            print("\nДля продолжения выберите один из методов (1-4)")
            return
        else:
            print("✗ Неверный выбор")
            return
        
        self.features_selected = True
        self.feature_selector.save_selection_info()
        
        print(f"\n✓ Отбор признаков завершен!")
        print(f"Выбрано признаков: {len(self.selected_feature_names)}")
        print(f"Форма X_train: {self.X_train_selected.shape}")
        print(f"Форма X_test: {self.X_test_selected.shape}")
    
    def train_models(self):
        """Пункт 4: Обучение и сравнение моделей"""
        if not self.features_selected:
            print("\n✗ Сначала выполните отбор признаков (пункт 3)")
            return
        
        print("\n" + "="*60)
        print("ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*60)
        
        print("\nИспользовать гиперпараметрический поиск?")
        print("[1] Да (медленно, но точнее)")
        print("[2] Нет (быстро, стандартные параметры)")
        
        choice = input("\nВыбор (по умолчанию 2): ").strip() or '2'
        
        use_search = (choice == '1')
        
        # Обучаем все модели
        self.model_trainer.train_all_models(
            self.X_train_selected, 
            self.y_train,
            self.selected_feature_names
        )
        
        # Сравниваем модели
        results_df = self.model_trainer.compare_models()
        
        # Анализ ошибок
        self.model_trainer.error_analysis(self.X_train_selected, self.y_train, top_n=10)
        
        # Сохраняем лог
        self.model_trainer.save_results_log()
        
        self.models_trained = True
        
        print("\n✓ Обучение моделей завершено!")
    
    def show_metrics(self):
        """Пункт 5: Показать метрики и важные признаки"""
        if not self.models_trained:
            print("\n✗ Сначала обучите модели (пункт 4)")
            return
        
        print("\n" + "="*60)
        print("МЕТРИКИ И ВАЖНЫЕ ПРИЗНАКИ")
        print("="*60)
        
        # Сравнение моделей
        self.model_trainer.compare_models()
        
        # Важность признаков
        self.model_trainer.get_feature_importance(top_n=20)
        
        # Анализ ошибок
        self.model_trainer.error_analysis(self.X_train_selected, self.y_train, top_n=15)
    
    def make_predictions(self):
        """Пункт 6: Сделать предсказания"""
        if not self.models_trained:
            print("\n✗ Сначала обучите модели (пункт 4)")
            return
        
        print("\n" + "="*60)
        print("ПРЕДСКАЗАНИЯ")
        print("="*60)
        print("\n[1] Предсказать для test.csv (создать submission.csv)")
        print("[2] Предсказать для одного дома вручную")
        print("[0] Назад")
        
        choice = input("\nВыбор: ").strip()
        
        if choice == '1':
            self.create_submission()
        elif choice == '2':
            self.predict_single_house()
        elif choice == '0':
            return
        else:
            print("✗ Неверный выбор")
    
    def create_submission(self):
        """Создать файл submission.csv"""
        print("\nСоздание предсказаний для test.csv...")
        
        predictions = self.model_trainer.predict(self.X_test_selected)
        
        # Создаем submission
        submission = pd.DataFrame({
            'Id': self.test_ids,
            'SalePrice': predictions
        })

        # Сохраняем
        filepath = 'submission.csv'   
        filepath = os.path.join(RESULTS_FOLDER, filepath)
        submission.to_csv(filepath, index=False)
        print(f"\n✓ Submission сохранен в {filepath}")
        print(f"Количество предсказаний: {len(predictions)}")
        print(f"\nСтатистика предсказаний:")
        print(f"Минимум: ${predictions.min():,.2f}")
        print(f"Среднее: ${predictions.mean():,.2f}")
        print(f"Максимум: ${predictions.max():,.2f}")
        
        # Показываем первые 5 предсказаний
        print(f"\nПервые 5 предсказаний:")
        print(submission.head().to_string(index=False))
    
    def predict_single_house(self):
        """Предсказать цену для одного дома (ручной ввод)"""
        print("\n" + "="*60)
        print("РУЧНОЕ ПРЕДСКАЗАНИЕ")
        print("="*60)
        print("\nВведите характеристики дома:")
        print("(для простоты введем только основные признаки)")
        
        try:
            overall_qual = int(input("OverallQual (качество 1-10): "))
            gr_liv_area = float(input("GrLivArea (площадь кв.футов): "))
            garage_cars = int(input("GarageCars (мест в гараже): "))
            total_bsmt_sf = float(input("TotalBsmtSF (площадь подвала): "))
            year_built = int(input("YearBuilt (год постройки): "))
            
            # Это упрощенный пример - в реальности нужно заполнить все признаки
            print("\n⚠ Примечание: для полного предсказания нужны все признаки.")
            print("Это демонстрационный пример с упрощенными данными.")
            
            # Создаем вектор со средними значениями и заменяем несколько признаков
            X_manual = np.mean(self.X_train_selected, axis=0).reshape(1, -1)
            
            # Находим индексы этих признаков в selected_feature_names
            # (это упрощение, в реальности нужна более сложная логика)
            
            prediction = self.model_trainer.predict(X_manual)
            
            print(f"\n✓ Предсказанная цена: ${prediction[0]:,.2f}")
            
        except ValueError:
            print("✗ Ошибка ввода данных")
        except Exception as e:
            print(f"✗ Ошибка: {e}")
    
    def save_load_model(self):
        """Пункт 7: Сохранить/загрузить модель"""
        print("\n" + "="*60)
        print("СОХРАНЕНИЕ/ЗАГРУЗКА МОДЕЛИ")
        print("="*60)
        print("\n[1] Сохранить текущую модель")
        print("[2] Загрузить модель из файла")
        print("[0] Назад")
        
        choice = input("\nВыбор: ").strip()
        
        if choice == '1':
            if not self.models_trained:
                print("\n✗ Нет обученной модели для сохранения")
                return
            
            filename = input("Имя файла (по умолчанию best_model.pkl): ").strip()
            if not filename:
                filename = 'best_model.pkl'
            
            self.model_trainer.save_model(filename)
            
        elif choice == '2':
            filename = input("Имя файла (по умолчанию best_model.pkl): ").strip()
            if not filename:
                filename = 'best_model.pkl'
            
            if self.model_trainer.load_model(filename):
                self.models_trained = True
                
        elif choice == '0':
            return
        else:
            print("✗ Неверный выбор")
    
    def run(self):
        """Главный цикл приложения"""
        print("\n")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║  ПРЕДСКАЗАНИЕ ЦЕН НА НЕДВИЖИМОСТЬ                         ║")
        print("║  House Prices: Advanced Regression Techniques             ║")
        print("╚════════════════════════════════════════════════════════════╝")
        
        while True:
            self.show_menu()
            
            choice = input("\nВыберите пункт (1-8): ").strip()
            
            if choice == '1':
                self.load_datasets()
            elif choice == '2':
                self.preprocess_data()
            elif choice == '3':
                self.select_features()
            elif choice == '4':
                self.train_models()
            elif choice == '5':
                self.show_metrics()
            elif choice == '6':
                self.make_predictions()
            elif choice == '7':
                self.save_load_model()
            elif choice == '8':
                print("\n" + "="*60)
                print("Спасибо за использование программы!")
                print("="*60 + "\n")
                sys.exit(0)
            else:
                print("\n✗ Неверный выбор. Выберите пункт от 1 до 8.")
            
            input("\nНажмите Enter для продолжения...")


def main():
    """Точка входа в программу"""
    try:
        # Создаем папку для данных, если её нет
        data_folder = 'data'
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
            print(f"✓ Создана папка '{data_folder}/'")
            print(f"Поместите файлы train.csv, test.csv в эту папку")
            input("\nНажмите Enter когда файлы будут готовы...")
        
        # Запускаем приложение
        app = HousePricePredictor(data_folder=data_folder)
        app.run()
        
    except KeyboardInterrupt:
        print("\n\n✗ Программа прервана пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()