import pytest
import pandas as pd
import numpy as np
import os
import json

# Путь к файлу medium.ipynb
MEDIUM_NOTEBOOK = os.path.abspath(os.path.join(os.path.dirname(__file__), '../medium.ipynb'))


def test_task_01_dropna(execute_notebook_task):
    """Проверяет удаление строк с пропущенными значениями."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 1)
    assert 'df_clean' in result, "Переменная 'df_clean' не найдена в результате"
    assert isinstance(result['df_clean'], pd.DataFrame), "'df_clean' должен быть DataFrame"
    assert result['df_clean'].isna().sum().sum() < result['df'].isna().sum().sum(), "Количество NaN не уменьшилось"


def test_task_02_fillna(execute_notebook_task):
    """Проверяет заполнение пропущенных значений."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 2)
    assert 'df_filled' in result, "Переменная 'df_filled' не найдена"
    assert isinstance(result['df_filled'], pd.DataFrame), "'df_filled' должен быть DataFrame"
    assert 'Age' in result['df_filled'].columns, "Колонка 'Age' отсутствует"
    assert result['df_filled']['Age'].isna().sum() == 0, "В колонке 'Age' остались пропущенные значения"


def test_task_03_groupby(execute_notebook_task):
    """Проверяет группировку данных и агрегацию."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 3)
    assert 'grouped_data' in result, "Переменная 'grouped_data' не найдена"
    assert isinstance(result['grouped_data'], pd.DataFrame), "'grouped_data' должен быть DataFrame"
    assert 'Sex' in result['grouped_data'].index.names, "'Sex' должен быть в индексе группировки"


def test_task_04_pivot_table(execute_notebook_task):
    """Проверяет создание сводной таблицы."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 4)
    assert 'pivot_table' in result, "Переменная 'pivot_table' не найдена"
    assert isinstance(result['pivot_table'], pd.DataFrame), "'pivot_table' должен быть DataFrame"
    assert 'Pclass' in result['pivot_table'].index.names or 'Pclass' in result['pivot_table'].columns.names, \
        "'Pclass' должен быть в индексе или столбцах сводной таблицы"


def test_task_05_custom_function(execute_notebook_task):
    """Проверяет применение пользовательской функции к DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 5)
    assert 'processed_df' in result, "Переменная 'processed_df' не найдена"
    assert isinstance(result['processed_df'], pd.DataFrame), "'processed_df' должен быть DataFrame"
    assert 'custom_col' in result['processed_df'].columns, "Колонка 'custom_col' отсутствует в результате"


def test_task_06_merge_dataframes(execute_notebook_task):
    """Проверяет объединение двух DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 6)
    assert 'merged_df' in result, "Переменная 'merged_df' не найдена"
    assert isinstance(result['merged_df'], pd.DataFrame), "'merged_df' должен быть DataFrame"
    # Проверяем, что колонки из обоих DataFrame присутствуют в результате объединения
    if 'df1' in result and 'df2' in result:
        assert result['merged_df'].shape[1] >= result['df1'].shape[1], "В результате должно быть как минимум столько же столбцов, сколько в df1"


def test_task_07_join_dataframes(execute_notebook_task):
    """Проверяет соединение двух DataFrame через метод join."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 7)
    assert 'joined_df' in result, "Переменная 'joined_df' не найдена"
    assert isinstance(result['joined_df'], pd.DataFrame), "'joined_df' должен быть DataFrame"
    assert result['joined_df'].shape[1] > result['df'].shape[1], "Количество столбцов не увеличилось после join"


def test_task_08_concat_dataframes(execute_notebook_task):
    """Проверяет конкатенацию DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 8)
    assert 'concat_result' in result, "Переменная 'concat_result' не найдена"
    assert isinstance(result['concat_result'], pd.DataFrame), "'concat_result' должен быть DataFrame"
    assert result['concat_result'].shape[0] > result['df'].shape[0], "Количество строк не увеличилось после concat"


def test_task_09_cut_function(execute_notebook_task):
    """Проверяет использование функции pd.cut для создания категориальных переменных."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 9)
    assert 'bins_df' in result, "Переменная 'bins_df' не найдена"
    assert isinstance(result['bins_df'], pd.DataFrame), "'bins_df' должен быть DataFrame"
    assert any('category' in str(dtype) for dtype in result['bins_df'].dtypes), "В результате должна быть категориальная переменная"


def test_task_10_qcut_function(execute_notebook_task):
    """Проверяет использование функции pd.qcut для создания равночисленных интервалов."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 10)
    assert 'qcut_df' in result, "Переменная 'qcut_df' не найдена"
    assert isinstance(result['qcut_df'], pd.DataFrame), "'qcut_df' должен быть DataFrame"
    assert any('category' in str(dtype) for dtype in result['qcut_df'].dtypes), "В результате должна быть категориальная переменная"


def test_task_11_melt_function(execute_notebook_task):
    """Проверяет использование функции melt для изменения формы DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 11)
    assert 'melted_df' in result, "Переменная 'melted_df' не найдена"
    assert isinstance(result['melted_df'], pd.DataFrame), "'melted_df' должен быть DataFrame"
    assert 'variable' in result['melted_df'].columns, "В результате должен быть столбец 'variable'"
    assert 'value' in result['melted_df'].columns, "В результате должен быть столбец 'value'"


def test_task_12_pivot_function(execute_notebook_task):
    """Проверяет использование функции pivot для изменения формы DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 12)
    assert 'pivoted_df' in result, "Переменная 'pivoted_df' не найдена"
    assert isinstance(result['pivoted_df'], pd.DataFrame), "'pivoted_df' должен быть DataFrame"
    # Сложно проверить конкретные свойства без знания пивотируемых данных


def test_task_13_get_dummies(execute_notebook_task):
    """Проверяет использование функции get_dummies для one-hot кодирования."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 13)
    assert 'dummies_df' in result, "Переменная 'dummies_df' не найдена"
    assert isinstance(result['dummies_df'], pd.DataFrame), "'dummies_df' должен быть DataFrame"
    assert result['dummies_df'].shape[1] > result['df'].shape[1], "Количество столбцов не увеличилось после применения get_dummies"


def test_task_14_explode_function(execute_notebook_task):
    """Проверяет использование функции explode для превращения списка в строки DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 14)
    assert 'exploded_df' in result, "Переменная 'exploded_df' не найдена"
    assert isinstance(result['exploded_df'], pd.DataFrame), "'exploded_df' должен быть DataFrame"
    assert result['exploded_df'].shape[0] >= result['df_with_lists'].shape[0], "Количество строк не увеличилось после explode"


def test_task_15_corr_method(execute_notebook_task):
    """Проверяет вычисление корреляции между столбцами."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 15)
    assert 'correlation_matrix' in result, "Переменная 'correlation_matrix' не найдена"
    assert isinstance(result['correlation_matrix'], pd.DataFrame), "'correlation_matrix' должен быть DataFrame"
    assert result['correlation_matrix'].shape[0] > 1, "Корреляционная матрица должна содержать несколько строк"


def test_task_16_describe_by_group(execute_notebook_task):
    """Проверяет использование describe с группировкой."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 16)
    assert 'group_description' in result, "Переменная 'group_description' не найдена"
    assert isinstance(result['group_description'], pd.DataFrame), "'group_description' должен быть DataFrame"
    assert result['group_description'].index.nlevels > 1, "Результат должен иметь многоуровневый индекс"


def test_task_17_custom_aggregation(execute_notebook_task):
    """Проверяет применение пользовательской агрегирующей функции."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 17)
    assert 'custom_agg' in result, "Переменная 'custom_agg' не найдена"
    assert isinstance(result['custom_agg'], pd.DataFrame), "'custom_agg' должен быть DataFrame"
    # Сложно проверить конкретные свойства без знания агрегирующей функции


def test_task_18_rolling_function(execute_notebook_task):
    """Проверяет использование функции rolling для скользящего окна."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 18)
    assert 'rolling_result' in result, "Переменная 'rolling_result' не найдена"
    assert isinstance(result['rolling_result'], pd.DataFrame) or isinstance(result['rolling_result'], pd.Series), \
        "'rolling_result' должен быть DataFrame или Series"


def test_task_19_resample_function(execute_notebook_task):
    """Проверяет использование функции resample для временных рядов."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 19)
    assert 'resampled_data' in result, "Переменная 'resampled_data' не найдена"
    assert isinstance(result['resampled_data'], pd.DataFrame) or isinstance(result['resampled_data'], pd.Series), \
        "'resampled_data' должен быть DataFrame или Series"
    # Для проверки потребовалось бы знать, как выглядит исходный временной ряд


def test_task_20_shift_function(execute_notebook_task):
    """Проверяет использование функции shift для смещения данных."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 20)
    assert 'shifted_data' in result, "Переменная 'shifted_data' не найдена"
    assert isinstance(result['shifted_data'], pd.DataFrame) or isinstance(result['shifted_data'], pd.Series), \
        "'shifted_data' должен быть DataFrame или Series"


def test_task_21_diff_function(execute_notebook_task):
    """Проверяет использование функции diff для вычисления разностей."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 21)
    assert 'diff_result' in result, "Переменная 'diff_result' не найдена"
    assert isinstance(result['diff_result'], pd.DataFrame) or isinstance(result['diff_result'], pd.Series), \
        "'diff_result' должен быть DataFrame или Series"


def test_task_22_pct_change_function(execute_notebook_task):
    """Проверяет использование функции pct_change для вычисления процентных изменений."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 22)
    assert 'pct_change_result' in result, "Переменная 'pct_change_result' не найдена"
    assert isinstance(result['pct_change_result'], pd.DataFrame) or isinstance(result['pct_change_result'], pd.Series), \
        "'pct_change_result' должен быть DataFrame или Series"


def test_task_23_cumsum_function(execute_notebook_task):
    """Проверяет использование функции cumsum для вычисления накопительной суммы."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 23)
    assert 'cumsum_result' in result, "Переменная 'cumsum_result' не найдена"
    assert isinstance(result['cumsum_result'], pd.DataFrame) or isinstance(result['cumsum_result'], pd.Series), \
        "'cumsum_result' должен быть DataFrame или Series"


def test_task_24_crosstab_function(execute_notebook_task):
    """Проверяет использование функции pd.crosstab для создания кросс-таблицы."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 24)
    assert 'crosstab_result' in result, "Переменная 'crosstab_result' не найдена"
    assert isinstance(result['crosstab_result'], pd.DataFrame), "'crosstab_result' должен быть DataFrame"


def test_task_25_multiindex_creation(execute_notebook_task):
    """Проверяет создание MultiIndex."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 25)
    assert 'multi_index_df' in result, "Переменная 'multi_index_df' не найдена"
    assert isinstance(result['multi_index_df'], pd.DataFrame), "'multi_index_df' должен быть DataFrame"
    assert isinstance(result['multi_index_df'].index, pd.MultiIndex), "Индекс 'multi_index_df' должен быть MultiIndex"


def test_task_26_unstack_method(execute_notebook_task):
    """Проверяет использование метода unstack для преобразования уровней индекса в столбцы."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 26)
    assert 'unstacked_df' in result, "Переменная 'unstacked_df' не найдена"
    assert isinstance(result['unstacked_df'], pd.DataFrame), "'unstacked_df' должен быть DataFrame"
    # Здесь следовало бы проверить, что количество столбцов увеличилось, но для этого нужно знать исходный DataFrame


def test_task_27_stack_method(execute_notebook_task):
    """Проверяет использование метода stack для преобразования столбцов в уровни индекса."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 27)
    assert 'stacked_df' in result, "Переменная 'stacked_df' не найдена"
    assert isinstance(result['stacked_df'], pd.DataFrame) or isinstance(result['stacked_df'], pd.Series), \
        "'stacked_df' должен быть DataFrame или Series"
    assert result['stacked_df'].index.nlevels > 1, "Результат должен иметь многоуровневый индекс"


def test_task_28_custom_index(execute_notebook_task):
    """Проверяет создание пользовательского индекса."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 28)
    assert 'custom_index_df' in result, "Переменная 'custom_index_df' не найдена"
    assert isinstance(result['custom_index_df'], pd.DataFrame), "'custom_index_df' должен быть DataFrame"
    assert not result['custom_index_df'].index.equals(pd.RangeIndex(start=0, stop=len(result['custom_index_df']))), \
        "Индекс должен быть изменен, а не остаться стандартным RangeIndex"


def test_task_29_interpolate_method(execute_notebook_task):
    """Проверяет использование метода interpolate для заполнения пропущенных значений."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 29)
    assert 'interpolated_df' in result, "Переменная 'interpolated_df' не найдена"
    assert isinstance(result['interpolated_df'], pd.DataFrame), "'interpolated_df' должен быть DataFrame"
    # Здесь следовало бы проверить, что количество NaN уменьшилось, но для этого нужно знать исходный DataFrame


def test_task_30_map_method(execute_notebook_task):
    """Проверяет использование метода map для преобразования значений."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 30)
    assert 'mapped_series' in result, "Переменная 'mapped_series' не найдена"
    assert isinstance(result['mapped_series'], pd.Series), "'mapped_series' должен быть Series"
    # Здесь сложно проверить конкретное преобразование без знания исходных данных


def test_task_31_apply_with_lambda(execute_notebook_task):
    """Проверяет использование метода apply с lambda-функцией."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 31)
    assert 'lambda_result' in result, "Переменная 'lambda_result' не найдена"
    assert isinstance(result['lambda_result'], pd.DataFrame) or isinstance(result['lambda_result'], pd.Series), \
        "'lambda_result' должен быть DataFrame или Series"


def test_task_32_date_range(execute_notebook_task):
    """Проверяет создание диапазона дат с помощью pd.date_range."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 32)
    assert 'date_range' in result, "Переменная 'date_range' не найдена"
    assert isinstance(result['date_range'], pd.DatetimeIndex), "'date_range' должен быть DatetimeIndex"
    assert len(result['date_range']) > 0, "'date_range' не должен быть пустым"


def test_task_33_reindex_method(execute_notebook_task):
    """Проверяет использование метода reindex для изменения индекса."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 33)
    assert 'reindexed_df' in result, "Переменная 'reindexed_df' не найдена"
    assert isinstance(result['reindexed_df'], pd.DataFrame), "'reindexed_df' должен быть DataFrame"
    # Здесь сложно проверить конкретное изменение индекса без знания исходных данных


def test_task_34_applymap_method(execute_notebook_task):
    """Проверяет использование метода applymap для поэлементного преобразования DataFrame."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 34)
    assert 'applymap_result' in result, "Переменная 'applymap_result' не найдена"
    assert isinstance(result['applymap_result'], pd.DataFrame), "'applymap_result' должен быть DataFrame"


def test_task_35_set_index_method(execute_notebook_task):
    """Проверяет использование метода set_index для установки индекса."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 35)
    assert 'indexed_df' in result, "Переменная 'indexed_df' не найдена"
    assert isinstance(result['indexed_df'], pd.DataFrame), "'indexed_df' должен быть DataFrame"
    assert not result['indexed_df'].index.equals(pd.RangeIndex(start=0, stop=len(result['indexed_df']))), \
        "Индекс должен быть изменен, а не остаться стандартным RangeIndex"


def test_task_36_pivot_longer(execute_notebook_task):
    """Проверяет преобразование DataFrame из широкого формата в длинный."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 36)
    assert 'long_format_df' in result, "Переменная 'long_format_df' не найдена"
    assert isinstance(result['long_format_df'], pd.DataFrame), "'long_format_df' должен быть DataFrame"
    assert result['long_format_df'].shape[0] > result['wide_format_df'].shape[0], \
        "Количество строк должно увеличиться при переходе к длинному формату"


def test_task_37_pivot_wider(execute_notebook_task):
    """Проверяет преобразование DataFrame из длинного формата в широкий."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 37)
    assert 'wide_format_df' in result, "Переменная 'wide_format_df' не найдена"
    assert isinstance(result['wide_format_df'], pd.DataFrame), "'wide_format_df' должен быть DataFrame"
    assert result['wide_format_df'].shape[1] > result['long_format_df'].shape[1], \
        "Количество столбцов должно увеличиться при переходе к широкому формату"


def test_task_38_transform_method(execute_notebook_task):
    """Проверяет использование метода transform для групповых операций, сохраняющих форму."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 38)
    assert 'transformed_df' in result, "Переменная 'transformed_df' не найдена"
    assert isinstance(result['transformed_df'], pd.DataFrame), "'transformed_df' должен быть DataFrame"
    assert result['transformed_df'].shape == result['df'].shape, \
        "После transform размер DataFrame должен остаться прежним"


def test_task_39_nlargest_nsmallest(execute_notebook_task):
    """Проверяет использование методов nlargest и nsmallest для выбора n наибольших/наименьших значений."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 39)
    assert 'nlargest_result' in result, "Переменная 'nlargest_result' не найдена"
    assert 'nsmallest_result' in result, "Переменная 'nsmallest_result' не найдена"
    assert isinstance(result['nlargest_result'], pd.DataFrame), "'nlargest_result' должен быть DataFrame"
    assert isinstance(result['nsmallest_result'], pd.DataFrame), "'nsmallest_result' должен быть DataFrame"
    assert 0 < len(result['nlargest_result']) < len(result['df']), "'nlargest_result' должен содержать меньше строк, чем исходный DataFrame"
    assert 0 < len(result['nsmallest_result']) < len(result['df']), "'nsmallest_result' должен содержать меньше строк, чем исходный DataFrame"


def test_task_40_filter_method(execute_notebook_task):
    """Проверяет использование метода filter для выбора строк/столбцов по паттерну."""
    result = execute_notebook_task(MEDIUM_NOTEBOOK, 40)
    assert 'filtered_df' in result, "Переменная 'filtered_df' не найдена"
    assert isinstance(result['filtered_df'], pd.DataFrame), "'filtered_df' должен быть DataFrame"
    assert result['filtered_df'].shape[1] < result['df'].shape[1], "После filter должно быть меньше столбцов"
