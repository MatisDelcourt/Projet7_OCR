import evidently
import pandas as pd
import numpy as np
import time
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

# Chargement des données
application_train = pd.read_csv("/content/application_train.csv")
application_test = pd.read_csv("/content/application_test.csv")

# Assurer que les deux DataFrame ont les mêmes colonnes
common_columns = application_train.columns.intersection(application_test.columns)
application_train = application_train[common_columns]
application_test = application_test[common_columns]

# Détermination des colonnes catégorielles
categorical_columns = []
for col in application_train.columns:
    unique_vals = set(application_train[col].unique())
    if unique_vals.issubset({0, 1, np.nan}):
        categorical_columns.append(col)

# Les colonnes numériques sont celles qui ne sont pas catégorielles
numerical_columns = [col for col in application_train.columns if col not in categorical_columns]

# Configuration du mapping des colonnes
column_mapping = ColumnMapping()
column_mapping.numerical_features = numerical_columns
column_mapping.categorical_features = categorical_columns

# Création et exécution du rapport de dérive des données
data_drift_report = Report(metrics=[
    DataDriftPreset(num_stattest='ks', cat_stattest='psi', num_stattest_threshold=0.2, cat_stattest_threshold=0.2),
])

start_time = time.time()
data_drift_report.run(reference_data=application_train, current_data=application_test, column_mapping=column_mapping)
elapsed_time = time.time() - start_time
print(f"Data Drift Report executed in {elapsed_time:.2f} seconds.")

# Sauvegarde du rapport en HTML
data_drift_report.save_html('/content/data_drift_report.html')