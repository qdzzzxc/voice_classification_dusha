## Идентификация личности по голосу

Проект состоит из следующих частей. Каждая часть подробно описана в markdown ячейках в следующих ноутбуках

Загрузка и преобразование датасета
> * data_preparation\get_dataframe.ipynb

Алгоритм получения признаков из аудио
> * mfcc\mfcc_algorithm.ipynb

Применение данного алгоритма к собранным данным
> * data_preparation\mfcc_to_df.ipynb

Построение и сравнение моделей 
> * experiments\
> * cross_validation_scores\best_models_cross_validation.ipynb
> * models_results.ipynb

Использование моделей для проверки на нахождение голоса в базе
> * threshold\find_threshold.ipynb
> * threshold\tpr@fpr.ipynb

Использование модели на реальном примере
> * evaluation_model.ipynb
