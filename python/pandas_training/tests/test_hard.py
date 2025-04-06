import pytest
import pandas as pd
import numpy as np
import os
import json

# Путь к файлу hard.ipynb
HARD_NOTEBOOK = os.path.abspath(os.path.join(os.path.dirname(__file__), '../hard.ipynb'))


def test_task_01_complex_filtering(execute_notebook_task):
    """Проверяет сложную фильтрацию данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 1)
    assert 'filtered_df' in result, "Переменная 'filtered_df' не найдена"
    assert isinstance(result['filtered_df'], pd.DataFrame), "'filtered_df' должен быть DataFrame"
    assert result['filtered_df'].shape[0] < result['df'].shape[0], "Фильтрация не уменьшила количество строк"


def test_task_02_multiindex(execute_notebook_task):
    """Проверяет работу с многоуровневыми индексами."""
    result = execute_notebook_task(HARD_NOTEBOOK, 2)
    assert 'multi_index_df' in result, "Переменная 'multi_index_df' не найдена"
    assert isinstance(result['multi_index_df'], pd.DataFrame), "'multi_index_df' должен быть DataFrame"
    assert isinstance(result['multi_index_df'].index, pd.MultiIndex), "Индекс 'multi_index_df' должен быть MultiIndex"


def test_task_03_custom_aggregation(execute_notebook_task):
    """Проверяет пользовательскую агрегацию."""
    result = execute_notebook_task(HARD_NOTEBOOK, 3)
    assert 'agg_result' in result, "Переменная 'agg_result' не найдена"
    assert isinstance(result['agg_result'], pd.DataFrame), "'agg_result' должен быть DataFrame"
    
    # Проверяем наличие нескольких агрегаций в результате
    if isinstance(result['agg_result'].columns, pd.MultiIndex):
        assert result['agg_result'].columns.nlevels > 1, "В результате должно быть несколько уровней агрегации"


def test_task_04_time_series(execute_notebook_task):
    """Проверяет работу с временными рядами."""
    result = execute_notebook_task(HARD_NOTEBOOK, 4)
    assert 'time_series' in result, "Переменная 'time_series' не найдена"
    assert isinstance(result['time_series'], pd.Series) or isinstance(result['time_series'], pd.DataFrame), \
        "'time_series' должен быть Series или DataFrame"
    
    # Проверяем, что индекс временного ряда является DatetimeIndex
    if isinstance(result['time_series'], pd.Series):
        assert isinstance(result['time_series'].index, pd.DatetimeIndex), "Индекс 'time_series' должен быть DatetimeIndex"
    else:
        assert isinstance(result['time_series'].index, pd.DatetimeIndex), "Индекс 'time_series' должен быть DatetimeIndex"


def test_task_05_merge_multiple(execute_notebook_task):
    """Проверяет объединение нескольких DataFrame."""
    result = execute_notebook_task(HARD_NOTEBOOK, 5)
    assert 'merged_df' in result, "Переменная 'merged_df' не найдена"
    assert isinstance(result['merged_df'], pd.DataFrame), "'merged_df' должен быть DataFrame"
    
    # Проверяем, что количество столбцов увеличилось после объединения
    if 'df1' in result and 'df2' in result:
        assert result['merged_df'].shape[1] >= result['df1'].shape[1], "Количество столбцов не увеличилось после объединения"


def test_task_06_complex_join(execute_notebook_task):
    """Проверяет сложное объединение данных с использованием различных типов join."""
    result = execute_notebook_task(HARD_NOTEBOOK, 6)
    assert 'complex_join_result' in result, "Переменная 'complex_join_result' не найдена"
    assert isinstance(result['complex_join_result'], pd.DataFrame), "'complex_join_result' должен быть DataFrame"
    # Дополнительные проверки зависят от конкретного типа объединения


def test_task_07_hierarchical_index(execute_notebook_task):
    """Проверяет создание и работу с иерархическими индексами."""
    result = execute_notebook_task(HARD_NOTEBOOK, 7)
    assert 'hierarchical_df' in result, "Переменная 'hierarchical_df' не найдена"
    assert isinstance(result['hierarchical_df'], pd.DataFrame), "'hierarchical_df' должен быть DataFrame"
    assert isinstance(result['hierarchical_df'].index, pd.MultiIndex), "Индекс должен быть MultiIndex"
    assert result['hierarchical_df'].index.nlevels > 2, "Иерархический индекс должен иметь более 2 уровней"


def test_task_08_advanced_groupby(execute_notebook_task):
    """Проверяет продвинутую группировку с множественными агрегациями."""
    result = execute_notebook_task(HARD_NOTEBOOK, 8)
    assert 'grouped_complex' in result, "Переменная 'grouped_complex' не найдена"
    assert isinstance(result['grouped_complex'], pd.DataFrame), "'grouped_complex' должен быть DataFrame"
    if isinstance(result['grouped_complex'].columns, pd.MultiIndex):
        assert result['grouped_complex'].columns.nlevels > 1, "Результат должен иметь многоуровневые колонки"


def test_task_09_pivot_techniques(execute_notebook_task):
    """Проверяет продвинутые техники pivot для анализа многомерных данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 9)
    assert 'advanced_pivot' in result, "Переменная 'advanced_pivot' не найдена"
    assert isinstance(result['advanced_pivot'], pd.DataFrame), "'advanced_pivot' должен быть DataFrame"
    # Сложно предусмотреть все возможные трансформации pivot


def test_task_10_time_series_resampling(execute_notebook_task):
    """Проверяет продвинутый ресемплинг временных рядов."""
    result = execute_notebook_task(HARD_NOTEBOOK, 10)
    assert 'resampled_data' in result, "Переменная 'resampled_data' не найдена"
    assert isinstance(result['resampled_data'], pd.DataFrame) or isinstance(result['resampled_data'], pd.Series), \
        "'resampled_data' должен быть DataFrame или Series"
    if hasattr(result['resampled_data'], 'index'):
        assert isinstance(result['resampled_data'].index, pd.DatetimeIndex), "Индекс должен быть DatetimeIndex"


def test_task_11_cross_sectional_analysis(execute_notebook_task):
    """Проверяет кросс-секционный анализ данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 11)
    assert 'cross_section_result' in result, "Переменная 'cross_section_result' не найдена"
    assert isinstance(result['cross_section_result'], pd.DataFrame), "'cross_section_result' должен быть DataFrame"


def test_task_12_multiindex_operations(execute_notebook_task):
    """Проверяет операции с MultiIndex."""
    result = execute_notebook_task(HARD_NOTEBOOK, 12)
    assert 'processed_multi_index' in result, "Переменная 'processed_multi_index' не найдена"
    assert isinstance(result['processed_multi_index'], pd.DataFrame), "'processed_multi_index' должен быть DataFrame"
    assert isinstance(result['processed_multi_index'].index, pd.MultiIndex), "Индекс должен быть MultiIndex"


def test_task_13_custom_window_functions(execute_notebook_task):
    """Проверяет использование пользовательских оконных функций."""
    result = execute_notebook_task(HARD_NOTEBOOK, 13)
    assert 'window_result' in result, "Переменная 'window_result' не найдена"
    assert isinstance(result['window_result'], pd.DataFrame), "'window_result' должен быть DataFrame"


def test_task_14_advanced_filtering(execute_notebook_task):
    """Проверяет продвинутую фильтрацию с использованием сложных условий."""
    result = execute_notebook_task(HARD_NOTEBOOK, 14)
    assert 'complex_filtered' in result, "Переменная 'complex_filtered' не найдена"
    assert isinstance(result['complex_filtered'], pd.DataFrame), "'complex_filtered' должен быть DataFrame"
    assert result['complex_filtered'].shape[0] < result['df'].shape[0], "Фильтрация не уменьшила количество строк"


def test_task_15_categorical_operations(execute_notebook_task):
    """Проверяет продвинутые операции с категориальными данными."""
    result = execute_notebook_task(HARD_NOTEBOOK, 15)
    assert 'categorical_result' in result, "Переменная 'categorical_result' не найдена"
    assert isinstance(result['categorical_result'], pd.DataFrame), "'categorical_result' должен быть DataFrame"
    assert any(pd.api.types.is_categorical_dtype(dtype) for dtype in result['categorical_result'].dtypes), \
        "В результате должен быть как минимум один категориальный столбец"


def test_task_16_time_series_decomposition(execute_notebook_task):
    """Проверяет декомпозицию временных рядов."""
    result = execute_notebook_task(HARD_NOTEBOOK, 16)
    assert 'decomposition_result' in result, "Переменная 'decomposition_result' не найдена"
    # Тип результата может зависеть от используемого метода декомпозиции


def test_task_17_panel_data_analysis(execute_notebook_task):
    """Проверяет анализ панельных данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 17)
    assert 'panel_result' in result, "Переменная 'panel_result' не найдена"
    assert isinstance(result['panel_result'], pd.DataFrame), "'panel_result' должен быть DataFrame"


def test_task_18_spatial_data_analysis(execute_notebook_task):
    """Проверяет анализ пространственных данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 18)
    assert 'spatial_result' in result, "Переменная 'spatial_result' не найдена"
    # Тип результата может зависеть от используемого метода пространственного анализа


def test_task_19_custom_indexers(execute_notebook_task):
    """Проверяет использование пользовательских индексаторов."""
    result = execute_notebook_task(HARD_NOTEBOOK, 19)
    assert 'indexed_result' in result, "Переменная 'indexed_result' не найдена"
    assert isinstance(result['indexed_result'], pd.DataFrame) or isinstance(result['indexed_result'], pd.Series), \
        "'indexed_result' должен быть DataFrame или Series"


def test_task_20_complex_string_processing(execute_notebook_task):
    """Проверяет сложную обработку строковых данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 20)
    assert 'string_processed' in result, "Переменная 'string_processed' не найдена"
    assert isinstance(result['string_processed'], pd.DataFrame) or isinstance(result['string_processed'], pd.Series), \
        "'string_processed' должен быть DataFrame или Series"


def test_task_21_advanced_visualization(execute_notebook_task):
    """Проверяет продвинутую визуализацию данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 21)
    assert 'viz_data' in result, "Переменная 'viz_data' не найдена"
    # Сложно проверить визуализацию автоматически, поэтому проверяем только наличие подготовленных данных


def test_task_22_custom_accessors(execute_notebook_task):
    """Проверяет создание и использование пользовательских аксессоров."""
    result = execute_notebook_task(HARD_NOTEBOOK, 22)
    assert 'accessor_result' in result, "Переменная 'accessor_result' не найдена"
    # Тип результата зависит от реализации аксессора


def test_task_23_data_pipelines(execute_notebook_task):
    """Проверяет создание и использование конвейеров обработки данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 23)
    assert 'pipeline_result' in result, "Переменная 'pipeline_result' не найдена"
    assert isinstance(result['pipeline_result'], pd.DataFrame), "'pipeline_result' должен быть DataFrame"


def test_task_24_multivariate_analysis(execute_notebook_task):
    """Проверяет многомерный анализ данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 24)
    assert 'multivariate_result' in result, "Переменная 'multivariate_result' не найдена"
    # Тип результата зависит от выбранного метода многомерного анализа


def test_task_25_outlier_detection(execute_notebook_task):
    """Проверяет обнаружение выбросов."""
    result = execute_notebook_task(HARD_NOTEBOOK, 25)
    assert 'outliers' in result, "Переменная 'outliers' не найдена"
    # Тип результата зависит от метода обнаружения выбросов


def test_task_26_sparse_data(execute_notebook_task):
    """Проверяет работу с разреженными данными."""
    result = execute_notebook_task(HARD_NOTEBOOK, 26)
    assert 'sparse_result' in result, "Переменная 'sparse_result' не найдена"
    # В зависимости от реализации может быть pd.DataFrame с разреженными значениями


def test_task_27_data_reshaping(execute_notebook_task):
    """Проверяет продвинутое изменение формы данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 27)
    assert 'reshaped_data' in result, "Переменная 'reshaped_data' не найдена"
    assert isinstance(result['reshaped_data'], pd.DataFrame), "'reshaped_data' должен быть DataFrame"


def test_task_28_memory_optimization(execute_notebook_task):
    """Проверяет оптимизацию памяти при работе с большими данными."""
    result = execute_notebook_task(HARD_NOTEBOOK, 28)
    assert 'optimized_df' in result, "Переменная 'optimized_df' не найдена"
    assert isinstance(result['optimized_df'], pd.DataFrame), "'optimized_df' должен быть DataFrame"
    # Проверка оптимизации может потребовать сравнения с исходным DataFrame


def test_task_29_extension_arrays(execute_notebook_task):
    """Проверяет использование extension arrays."""
    result = execute_notebook_task(HARD_NOTEBOOK, 29)
    assert 'extension_array_df' in result, "Переменная 'extension_array_df' не найдена"
    assert isinstance(result['extension_array_df'], pd.DataFrame), "'extension_array_df' должен быть DataFrame"


def test_task_30_custom_validation(execute_notebook_task):
    """Проверяет пользовательскую валидацию данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 30)
    assert 'validation_result' in result, "Переменная 'validation_result' не найдена"
    # Тип результата зависит от реализации валидации


def test_task_31_non_standard_indexes(execute_notebook_task):
    """Проверяет работу с нестандартными индексами."""
    result = execute_notebook_task(HARD_NOTEBOOK, 31)
    assert 'indexed_df' in result, "Переменная 'indexed_df' не найдена"
    assert isinstance(result['indexed_df'], pd.DataFrame), "'indexed_df' должен быть DataFrame"
    assert not isinstance(result['indexed_df'].index, pd.RangeIndex), "Индекс не должен быть стандартным RangeIndex"


def test_task_32_large_dataset_techniques(execute_notebook_task):
    """Проверяет техники работы с большими наборами данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 32)
    assert 'large_data_result' in result, "Переменная 'large_data_result' не найдена"
    # Тип результата зависит от выбранной техники


def test_task_33_parallel_processing(execute_notebook_task):
    """Проверяет параллельную обработку данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 33)
    assert 'parallel_result' in result, "Переменная 'parallel_result' не найдена"
    assert isinstance(result['parallel_result'], pd.DataFrame), "'parallel_result' должен быть DataFrame"


def test_task_34_custom_reduction(execute_notebook_task):
    """Проверяет пользовательскую функцию редукции данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 34)
    assert 'reduced_data' in result, "Переменная 'reduced_data' не найдена"
    # Тип результата зависит от реализации редукции


def test_task_35_complex_time_series(execute_notebook_task):
    """Проверяет сложные операции с временными рядами."""
    result = execute_notebook_task(HARD_NOTEBOOK, 35)
    assert 'complex_ts_result' in result, "Переменная 'complex_ts_result' не найдена"
    # Тип результата зависит от конкретных операций


def test_task_36_data_imputation(execute_notebook_task):
    """Проверяет продвинутые методы импутации данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 36)
    assert 'imputed_data' in result, "Переменная 'imputed_data' не найдена"
    assert isinstance(result['imputed_data'], pd.DataFrame), "'imputed_data' должен быть DataFrame"
    # Проверка качества импутации требует сравнения с исходными данными


def test_task_37_advanced_joining(execute_notebook_task):
    """Проверяет продвинутые методы объединения данных."""
    result = execute_notebook_task(HARD_NOTEBOOK, 37)
    assert 'advanced_join_result' in result, "Переменная 'advanced_join_result' не найдена"
    assert isinstance(result['advanced_join_result'], pd.DataFrame), "'advanced_join_result' должен быть DataFrame"


def test_task_38_custom_groupby_transform(execute_notebook_task):
    """Проверяет пользовательскую функцию трансформации при группировке."""
    result = execute_notebook_task(HARD_NOTEBOOK, 38)
    assert 'custom_transform_result' in result, "Переменная 'custom_transform_result' не найдена"
    assert isinstance(result['custom_transform_result'], pd.DataFrame), "'custom_transform_result' должен быть DataFrame"


def test_task_39_efficient_dataframe_creation(execute_notebook_task):
    """Проверяет эффективные методы создания DataFrame."""
    result = execute_notebook_task(HARD_NOTEBOOK, 39)
    assert 'efficient_df' in result, "Переменная 'efficient_df' не найдена"
    assert isinstance(result['efficient_df'], pd.DataFrame), "'efficient_df' должен быть DataFrame"


def test_task_40_complex_analysis(execute_notebook_task):
    """Проверяет комплексный анализ данных с использованием нескольких техник."""
    result = execute_notebook_task(HARD_NOTEBOOK, 40)
    assert 'complex_analysis_result' in result, "Переменная 'complex_analysis_result' не найдена"
    assert isinstance(result['complex_analysis_result'], pd.DataFrame), "'complex_analysis_result' должен быть DataFrame"
