import os
import sys
import pytest
import pandas as pd
import numpy as np
import importlib.util
import json
import re

# Добавляем родительскую директорию в sys.path для импорта модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

@pytest.fixture(scope="session")
def titanic_train_data():
    """Загрузка тренировочных данных Titanic."""
    file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../datasets/titanic/train.csv'
    ))
    return pd.read_csv(file_path)

@pytest.fixture(scope="session")
def titanic_test_data():
    """Загрузка тестовых данных Titanic."""
    file_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../../../datasets/titanic/test.csv'
    ))
    return pd.read_csv(file_path)

@pytest.fixture(scope="session")
def notebook_output_parser():
    """Фикстура для извлечения результатов из ячеек Jupyter."""
    
    def _parse_notebook(notebook_path, task_number):
        """Загружает ноутбук и ищет результат выполнения указанного задания.
        
        Args:
            notebook_path (str): Путь к файлу ноутбука
            task_number (int): Номер задания, результат которого нужно извлечь
            
        Returns:
            Объект, полученный в результате выполнения ячейки для задания
        """
        if not os.path.exists(notebook_path):
            pytest.skip(f"Ноутбук {notebook_path} не найден")
            
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
            
        cells = notebook_content.get('cells', [])
        
        # Паттерн для поиска ячеек с заданием
        task_pattern = re.compile(rf"## Задание {task_number}[\s\S]*?```python")
        
        # Ищем ячейку с заданием и следующую за ней ячейку с решением
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'markdown' and task_pattern.search(cell.get('source', '')):
                # Проверяем следующую ячейку, которая должна содержать решение
                if i + 1 < len(cells) and cells[i+1]['cell_type'] == 'code':
                    code_cell = cells[i+1]
                    
                    # Проверяем, есть ли у ячейки вывод
                    if code_cell.get('outputs'):
                        for output in code_cell['outputs']:
                            if 'data' in output and 'text/plain' in output['data']:
                                # Извлекаем вывод
                                result_text = output['data']['text/plain']
                                
                                try:
                                    # Пытаемся преобразовать в объект Python
                                    result = eval(result_text)
                                    return result
                                except:
                                    # Если не удалось преобразовать, возвращаем текст
                                    return result_text
                    
                    # Если вывода нет, но есть исходный код
                    if code_cell.get('source'):
                        code = '\n'.join(code_cell['source'])
                        return code
                        
        # Если не нашли задание или решение
        pytest.skip(f"Решение для задания {task_number} не найдено в ноутбуке {notebook_path}")
        return None
    
    return _parse_notebook

@pytest.fixture(scope="function")
def execute_notebook_task():
    """Фикстура для выполнения кода из ячейки ноутбука."""
    
    def _execute_task(notebook_path, task_number, globals_dict=None):
        """Выполняет код из ячейки для указанного задания.
        
        Args:
            notebook_path (str): Путь к файлу ноутбука
            task_number (int): Номер задания, код которого нужно выполнить
            globals_dict (dict, optional): Словарь глобальных переменных
            
        Returns:
            dict: Локальные переменные после выполнения кода
        """
        if globals_dict is None:
            globals_dict = {
                'pd': pd,
                'np': np,
                'titanic_train': titanic_train_data(),
                'titanic_test': titanic_test_data()
            }
            
        if not os.path.exists(notebook_path):
            pytest.skip(f"Ноутбук {notebook_path} не найден")
            
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = json.load(f)
            
        cells = notebook_content.get('cells', [])
        
        # Паттерн для поиска ячеек с заданием
        task_pattern = re.compile(rf"## Задание {task_number}[\s\S]*?```python")
        
        # Ищем ячейку с заданием и следующую за ней ячейку с решением
        for i, cell in enumerate(cells):
            if cell['cell_type'] == 'markdown' and task_pattern.search(''.join(cell.get('source', ''))):
                # Проверяем следующую ячейку, которая должна содержать решение
                if i + 1 < len(cells) and cells[i+1]['cell_type'] == 'code':
                    code_cell = cells[i+1]
                    
                    # Получаем исходный код
                    if code_cell.get('source'):
                        code = '\n'.join(code_cell['source'])
                        
                        # Выполняем код
                        local_vars = {}
                        try:
                            exec(code, globals_dict, local_vars)
                            return local_vars
                        except Exception as e:
                            pytest.fail(f"Ошибка при выполнении кода для задания {task_number}: {str(e)}")
                        
        # Если не нашли задание или решение
        pytest.skip(f"Решение для задания {task_number} не найдено в ноутбуке {notebook_path}")
        return {}
    
    return _execute_task
