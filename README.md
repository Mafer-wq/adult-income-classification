# Adult Income Classification

Este proyecto tiene como objetivo predecir si una persona percibe ingresos anuales superiores a 50K utilizando técnicas de machine learning sobre el dataset **Adult Census Income**.

## Objetivo
Construir y comparar modelos de clasificación para predecir la variable `income` a partir de variables demográficas y laborales.

## Dataset
**Adult Census Income Dataset**  
Fuente: https://archive.ics.uci.edu/dataset/2/adult

Archivo principal utilizado:
- `train_adultos.csv`

## Metodología
- Limpieza y preparación de datos
- Análisis exploratorio de variables numéricas y categóricas
- Preprocesamiento con `Pipeline` y `ColumnTransformer`
- Codificación de variables categóricas con `OneHotEncoder`
- Manejo del desbalance de clases mediante `SMOTE`
- Entrenamiento y comparación de modelos
- Optimización de hiperparámetros con `GridSearchCV`

## Modelos utilizados
- Decision Tree
- Random Forest
- XGBoost

## Resultados
El modelo con mejor desempeño general fue **XGBoost**.

Métricas destacadas del modelo final:
- Accuracy: `0.8414`
- ROC-AUC: `0.9235`
- Precision clase >50K: `0.6313`
- Recall clase >50K: `0.8206`
- F1-score clase >50K: `0.7136`

Además, el análisis mostró que variables como `education-num`, `capital-gain`, `occupation`, `marital-status` y `relationship` tuvieron un aporte importante en la predicción del ingreso.

## Principales hallazgos
- Niveles educativos altos como `Doctorate`, `Prof-school` y `Masters` mostraron mayor proporción de ingresos superiores a 50K.
- Ocupaciones como `Exec-managerial` y `Prof-specialty` se asociaron con mayor presencia en la clase positiva.
- Variables laborales, educativas y familiares aportan información relevante al modelo.

## Limitaciones
- El dataset presenta desbalance de clases.
- Algunas categorías tienen pocos registros y pueden producir estimaciones inestables.
- Variables sensibles como sexo, raza y país de origen deben interpretarse con cautela.
- Los resultados obtenidos son predictivos, no causales.

## Trabajo futuro
- Ajustar el umbral de clasificación según el objetivo del problema
- Evaluar métricas de fairness entre grupos
- Probar modelos adicionales como LightGBM o CatBoost
- Realizar ingeniería de variables
- Validar el modelo con datos externos o validación más robusta

## Estructura del proyecto
- `adult_income_classification.ipynb`: notebook principal del análisis y modelado
- `train_adultos.csv`: dataset utilizado

## Cómo ejecutar
1. Instalar las dependencias del proyecto
2. Abrir `adult_income_classification.ipynb`
3. Ejecutar las celdas en orden
