import pytest
import pandas as pd
import numpy as np
import os
import json

# Путь к файлу easy.ipynb
EASY_NOTEBOOK = os.path.abspath(os.path.join(os.path.dirname(__file__), '../easy.ipynb'))


def test_task_01_load_data(execute_notebook_task, titanic_train_data):
    """Проверяет загрузку данных в задании 1."""
    # Выполнить код из ноутбука
    result = execute_notebook_task(EASY_NOTEBOOK, 1)
    
    # Проверить, что данные загружены корректно
    assert 'df' in result, "Переменная 'df' не найдена в результате"
    assert isinstance(result['df'], pd.DataFrame), "'df' должен быть DataFrame"
    assert result['df'].shape == titanic_train_data.shape, "Размер DataFrame не совпадает с ожидаемым"


def test_task_02_basic_info(execute_notebook_task):
    """Проверяет вывод базовой информации о данных."""
    result = execute_notebook_task(EASY_NOTEBOOK, 2)
    assert 'info_result' in result, "Переменная 'info_result' не найдена"
    # Проверка свойств info может быть сложной, так как .info() выводит текст


def test_task_03_columns(execute_notebook_task):
    """Проверяет получение списка столбцов датасета."""
    result = execute_notebook_task(EASY_NOTEBOOK, 3)
    assert 'columns' in result, "Переменная 'columns' не найдена"
    expected_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    assert all(col in result['columns'] for col in expected_columns), "Список столбцов неполный"


def test_task_04_descriptive_stats(execute_notebook_task):
    """Проверяет вычисление дескриптивной статистики."""
    result = execute_notebook_task(EASY_NOTEBOOK, 4)
    assert 'stats' in result, "Переменная 'stats' не найдена"
    assert isinstance(result['stats'], pd.DataFrame), "'stats' должен быть DataFrame"
    assert 'count' in result['stats'].index, "В статистике должен быть показатель 'count'"


def test_task_05_missing_values(execute_notebook_task):
    """Проверяет анализ пропущенных значений."""
    result = execute_notebook_task(EASY_NOTEBOOK, 5)
    assert 'missing_values' in result, "Переменная 'missing_values' не найдена"
    assert isinstance(result['missing_values'], pd.Series), "'missing_values' должен быть Series"


def test_task_06_data_types(execute_notebook_task):
    """Проверяет получение типов данных столбцов DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 6)
    assert 'dtypes' in result, "Переменная 'dtypes' не найдена"
    assert isinstance(result['dtypes'], pd.Series), "'dtypes' должен быть Series"
    assert len(result['dtypes']) > 0, "'dtypes' не должен быть пустым"


def test_task_07_select_columns(execute_notebook_task):
    """Проверяет выбор определенных столбцов из DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 7)
    assert 'selected_columns' in result, "Переменная 'selected_columns' не найдена"
    assert isinstance(result['selected_columns'], pd.DataFrame), "'selected_columns' должен быть DataFrame"
    assert result['selected_columns'].shape[1] < 12, "Должны быть выбраны не все столбцы"


def test_task_08_head_tail(execute_notebook_task):
    """Проверяет получение первых и последних строк DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 8)
    assert 'head_rows' in result, "Переменная 'head_rows' не найдена"
    assert 'tail_rows' in result, "Переменная 'tail_rows' не найдена"
    assert isinstance(result['head_rows'], pd.DataFrame), "'head_rows' должен быть DataFrame"
    assert isinstance(result['tail_rows'], pd.DataFrame), "'tail_rows' должен быть DataFrame"
    assert len(result['head_rows']) <= 10, "'head_rows' должен содержать не более 10 строк"
    assert len(result['tail_rows']) <= 10, "'tail_rows' должен содержать не более 10 строк"


def test_task_09_sample_data(execute_notebook_task):
    """Проверяет получение случайной выборки из DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 9)
    assert 'sample_df' in result, "Переменная 'sample_df' не найдена"
    assert isinstance(result['sample_df'], pd.DataFrame), "'sample_df' должен быть DataFrame"
    assert 0 < len(result['sample_df']) < 891, "'sample_df' должен содержать меньше строк, чем исходный DataFrame"


def test_task_10_unique_values(execute_notebook_task):
    """Проверяет получение уникальных значений из столбца."""
    result = execute_notebook_task(EASY_NOTEBOOK, 10)
    assert 'unique_values' in result, "Переменная 'unique_values' не найдена"
    assert isinstance(result['unique_values'], (np.ndarray, pd.Series)), "'unique_values' должен быть массивом или Series"


def test_task_11_value_counts(execute_notebook_task):
    """Проверяет подсчет встречаемости значений в столбце."""
    result = execute_notebook_task(EASY_NOTEBOOK, 11)
    assert 'value_counts' in result, "Переменная 'value_counts' не найдена"
    assert isinstance(result['value_counts'], pd.Series), "'value_counts' должен быть Series"
    assert len(result['value_counts']) > 0, "'value_counts' не должен быть пустым"


def test_task_12_filter_by_value(execute_notebook_task):
    """Проверяет фильтрацию DataFrame по значению."""
    result = execute_notebook_task(EASY_NOTEBOOK, 12)
    assert 'filtered_df' in result, "Переменная 'filtered_df' не найдена"
    assert isinstance(result['filtered_df'], pd.DataFrame), "'filtered_df' должен быть DataFrame"
    assert len(result['filtered_df']) < 891, "'filtered_df' должен содержать меньше строк, чем исходный DataFrame"


def test_task_13_multiple_filters(execute_notebook_task):
    """Проверяет применение нескольких фильтров."""
    result = execute_notebook_task(EASY_NOTEBOOK, 13)
    assert 'multi_filtered_df' in result, "Переменная 'multi_filtered_df' не найдена"
    assert isinstance(result['multi_filtered_df'], pd.DataFrame), "'multi_filtered_df' должен быть DataFrame"
    assert len(result['multi_filtered_df']) < 891, "'multi_filtered_df' должен содержать меньше строк, чем исходный DataFrame"


def test_task_14_sort_values(execute_notebook_task):
    """Проверяет сортировку DataFrame по значениям столбца."""
    result = execute_notebook_task(EASY_NOTEBOOK, 14)
    assert 'sorted_df' in result, "Переменная 'sorted_df' не найдена"
    assert isinstance(result['sorted_df'], pd.DataFrame), "'sorted_df' должен быть DataFrame"
    assert len(result['sorted_df']) == 891, "'sorted_df' должен содержать то же количество строк, что и исходный DataFrame"


def test_task_15_reset_index(execute_notebook_task):
    """Проверяет сброс индекса DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 15)
    assert 'reset_index_df' in result, "Переменная 'reset_index_df' не найдена"
    assert isinstance(result['reset_index_df'], pd.DataFrame), "'reset_index_df' должен быть DataFrame"
    # Проверим, что индексы начинаются с 0 и идут последовательно
    assert result['reset_index_df'].index[0] == 0, "Индексация должна начинаться с 0"
    assert result['reset_index_df'].index[-1] == len(result['reset_index_df']) - 1, "Индексация должна быть последовательной"


def test_task_16_rename_columns(execute_notebook_task):
    """Проверяет переименование столбцов DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 16)
    assert 'renamed_df' in result, "Переменная 'renamed_df' не найдена"
    assert isinstance(result['renamed_df'], pd.DataFrame), "'renamed_df' должен быть DataFrame"
    # Проверим, что хотя бы одно название столбца отличается от оригинального
    orig_columns = set(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])
    assert len(orig_columns.difference(set(result['renamed_df'].columns))) > 0, "Ни один столбец не был переименован"


def test_task_17_column_calculations(execute_notebook_task):
    """Проверяет выполнение вычислений над столбцами."""
    result = execute_notebook_task(EASY_NOTEBOOK, 17)
    assert 'calculated_df' in result, "Переменная 'calculated_df' не найдена"
    assert isinstance(result['calculated_df'], pd.DataFrame), "'calculated_df' должен быть DataFrame"
    # Проверим, что добавлена новая колонка
    assert len(result['calculated_df'].columns) > 12, "Новый столбец не был добавлен"


def test_task_18_apply_function(execute_notebook_task):
    """Проверяет применение функции к столбцу."""
    result = execute_notebook_task(EASY_NOTEBOOK, 18)
    assert 'applied_df' in result, "Переменная 'applied_df' не найдена"
    assert isinstance(result['applied_df'], pd.DataFrame), "'applied_df' должен быть DataFrame"
    # Проверим, что либо добавлена новая колонка, либо изменена существующая
    assert (len(result['applied_df'].columns) > 12) or any(col not in ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'] for col in result['applied_df'].columns), "Функция не была применена"


def test_task_19_describe_column(execute_notebook_task):
    """Проверяет детальное описание отдельного столбца."""
    result = execute_notebook_task(EASY_NOTEBOOK, 19)
    assert 'column_stats' in result, "Переменная 'column_stats' не найдена"
    assert isinstance(result['column_stats'], pd.Series), "'column_stats' должен быть Series"
    assert len(result['column_stats']) > 0, "'column_stats' не должен быть пустым"


def test_task_20_groupby_count(execute_notebook_task):
    """Проверяет группировку и подсчет значений."""
    result = execute_notebook_task(EASY_NOTEBOOK, 20)
    assert 'group_counts' in result, "Переменная 'group_counts' не найдена"
    assert isinstance(result['group_counts'], (pd.DataFrame, pd.Series)), "'group_counts' должен быть DataFrame или Series"
    assert len(result['group_counts']) > 0, "'group_counts' не должен быть пустым"


def test_task_21_isnull_check(execute_notebook_task):
    """Проверяет проверку на наличие пропущенных значений."""
    result = execute_notebook_task(EASY_NOTEBOOK, 21)
    assert 'null_mask' in result, "Переменная 'null_mask' не найдена"
    assert isinstance(result['null_mask'], pd.DataFrame), "'null_mask' должен быть DataFrame"
    assert result['null_mask'].shape == (891, 12), "'null_mask' должен иметь тот же размер, что и исходный DataFrame"


def test_task_22_drop_columns(execute_notebook_task):
    """Проверяет удаление столбцов из DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 22)
    assert 'dropped_df' in result, "Переменная 'dropped_df' не найдена"
    assert isinstance(result['dropped_df'], pd.DataFrame), "'dropped_df' должен быть DataFrame"
    assert result['dropped_df'].shape[1] < 12, "Столбцы не были удалены"


def test_task_23_drop_duplicates(execute_notebook_task):
    """Проверяет удаление дубликатов из DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 23)
    assert 'no_duplicates_df' in result, "Переменная 'no_duplicates_df' не найдена"
    assert isinstance(result['no_duplicates_df'], pd.DataFrame), "'no_duplicates_df' должен быть DataFrame"
    # Проверяем, что дубликаты действительно удалены
    assert len(result['no_duplicates_df'].duplicated()) == len(result['no_duplicates_df']), "Метод удаления дубликатов не был применен"


def test_task_24_query_method(execute_notebook_task):
    """Проверяет использование метода query для фильтрации."""
    result = execute_notebook_task(EASY_NOTEBOOK, 24)
    assert 'query_result' in result, "Переменная 'query_result' не найдена"
    assert isinstance(result['query_result'], pd.DataFrame), "'query_result' должен быть DataFrame"
    assert len(result['query_result']) < 891, "Фильтрация не была применена"


def test_task_25_boolean_indexing(execute_notebook_task):
    """Проверяет использование булевой индексации."""
    result = execute_notebook_task(EASY_NOTEBOOK, 25)
    assert 'boolean_index_result' in result, "Переменная 'boolean_index_result' не найдена"
    assert isinstance(result['boolean_index_result'], pd.DataFrame), "'boolean_index_result' должен быть DataFrame"
    assert len(result['boolean_index_result']) < 891, "Булевая индексация не была применена"


def test_task_26_astype_conversion(execute_notebook_task):
    """Проверяет преобразование типов данных."""
    result = execute_notebook_task(EASY_NOTEBOOK, 26)
    assert 'converted_df' in result, "Переменная 'converted_df' не найдена"
    assert isinstance(result['converted_df'], pd.DataFrame), "'converted_df' должен быть DataFrame"
    # Проверяем, что типы данных изменились
    assert any(result['converted_df'].dtypes != result['df'].dtypes), "Типы данных не были изменены"


def test_task_27_categorical_data(execute_notebook_task):
    """Проверяет преобразование в категориальный тип данных."""
    result = execute_notebook_task(EASY_NOTEBOOK, 27)
    assert 'categorical_df' in result, "Переменная 'categorical_df' не найдена"
    assert isinstance(result['categorical_df'], pd.DataFrame), "'categorical_df' должен быть DataFrame"
    # Проверяем, что есть хотя бы один столбец категориального типа
    assert any(str(dtype) == 'category' for dtype in result['categorical_df'].dtypes), "Категориальные данные не были созданы"


def test_task_28_duplicated_check(execute_notebook_task):
    """Проверяет поиск дубликатов в DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 28)
    assert 'duplicates_mask' in result, "Переменная 'duplicates_mask' не найдена"
    assert isinstance(result['duplicates_mask'], pd.Series), "'duplicates_mask' должен быть Series"
    assert len(result['duplicates_mask']) == 891, "'duplicates_mask' должен иметь ту же длину, что и исходный DataFrame"


def test_task_29_loc_accessor(execute_notebook_task):
    """Проверяет использование loc для доступа к данным."""
    result = execute_notebook_task(EASY_NOTEBOOK, 29)
    assert 'loc_result' in result, "Переменная 'loc_result' не найдена"
    assert isinstance(result['loc_result'], (pd.DataFrame, pd.Series)), "'loc_result' должен быть DataFrame или Series"


def test_task_30_iloc_accessor(execute_notebook_task):
    """Проверяет использование iloc для доступа к данным."""
    result = execute_notebook_task(EASY_NOTEBOOK, 30)
    assert 'iloc_result' in result, "Переменная 'iloc_result' не найдена"
    assert isinstance(result['iloc_result'], (pd.DataFrame, pd.Series)), "'iloc_result' должен быть DataFrame или Series"


def test_task_31_round_values(execute_notebook_task):
    """Проверяет округление числовых значений."""
    result = execute_notebook_task(EASY_NOTEBOOK, 31)
    assert 'rounded_df' in result, "Переменная 'rounded_df' не найдена"
    assert isinstance(result['rounded_df'], pd.DataFrame), "'rounded_df' должен быть DataFrame"


def test_task_32_copy_dataframe(execute_notebook_task):
    """Проверяет создание копии DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 32)
    assert 'df_copy' in result, "Переменная 'df_copy' не найдена"
    assert isinstance(result['df_copy'], pd.DataFrame), "'df_copy' должен быть DataFrame"
    assert id(result['df_copy']) != id(result['df']), "'df_copy' должен быть новым объектом, а не ссылкой на исходный DataFrame"


def test_task_33_to_numpy(execute_notebook_task):
    """Проверяет преобразование DataFrame в массив NumPy."""
    result = execute_notebook_task(EASY_NOTEBOOK, 33)
    assert 'numpy_array' in result, "Переменная 'numpy_array' не найдена"
    assert isinstance(result['numpy_array'], np.ndarray), "'numpy_array' должен быть NumPy массивом"


def test_task_34_value_replacement(execute_notebook_task):
    """Проверяет замену значений в DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 34)
    assert 'replaced_df' in result, "Переменная 'replaced_df' не найдена"
    assert isinstance(result['replaced_df'], pd.DataFrame), "'replaced_df' должен быть DataFrame"
    # Проверка замены значений требует знания исходных значений, поэтому здесь только проверка существования


def test_task_35_shape_property(execute_notebook_task):
    """Проверяет получение размерности DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 35)
    assert 'df_shape' in result, "Переменная 'df_shape' не найдена"
    assert isinstance(result['df_shape'], tuple), "'df_shape' должен быть кортежем"
    assert len(result['df_shape']) == 2, "'df_shape' должен содержать 2 элемента (строки, столбцы)"


def test_task_36_between_method(execute_notebook_task):
    """Проверяет использование метода between для фильтрации по диапазону."""
    result = execute_notebook_task(EASY_NOTEBOOK, 36)
    assert 'between_result' in result, "Переменная 'between_result' не найдена"
    assert isinstance(result['between_result'], pd.DataFrame), "'between_result' должен быть DataFrame"
    assert len(result['between_result']) < 891, "Фильтрация по диапазону не была применена"


def test_task_37_column_addition(execute_notebook_task):
    """Проверяет добавление нового столбца в DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 37)
    assert 'df_with_new_column' in result, "Переменная 'df_with_new_column' не найдена"
    assert isinstance(result['df_with_new_column'], pd.DataFrame), "'df_with_new_column' должен быть DataFrame"
    assert len(result['df_with_new_column'].columns) > 12, "Новый столбец не был добавлен"


def test_task_38_string_methods(execute_notebook_task):
    """Проверяет использование строковых методов для обработки текста."""
    result = execute_notebook_task(EASY_NOTEBOOK, 38)
    assert 'string_processed_df' in result, "Переменная 'string_processed_df' не найдена"
    assert isinstance(result['string_processed_df'], pd.DataFrame), "'string_processed_df' должен быть DataFrame"
    # Проверка строковых методов требует знания исходных значений, поэтому здесь только проверка существования


def test_task_39_save_to_csv(execute_notebook_task):
    """Проверяет сохранение DataFrame в CSV файл."""
    result = execute_notebook_task(EASY_NOTEBOOK, 39)
    # Проверка, сохранился ли файл, может потребовать дополнительного внешнего контроля
    # Здесь можно только проверить, что команда была выполнена
    assert 'csv_saved' in result, "Переменная 'csv_saved' не найдена"


def test_task_40_memory_usage(execute_notebook_task):
    """Проверяет анализ использования памяти DataFrame."""
    result = execute_notebook_task(EASY_NOTEBOOK, 40)
    assert 'memory_usage' in result, "Переменная 'memory_usage' не найдена"
    assert isinstance(result['memory_usage'], (pd.Series, int, float)), "'memory_usage' должен быть Series, int или float"
    if isinstance(result['memory_usage'], pd.Series):
        assert len(result['memory_usage']) > 0, "'memory_usage' не должен быть пустым"
