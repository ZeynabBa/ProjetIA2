import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st

# Charger le dataset
file_path = 'DatasetmalwareExtrait.csv'  # Assurez-vous que le chemin est correct
data = pd.read_csv(file_path)

# Séparer les features et les labels
X = data.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
Y = data.iloc[:, -1]   # Dernière colonne

# Division des données
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Modèle initial sans optimisation
model_no_optimization = DecisionTreeClassifier()
model_no_optimization.fit(X_train, Y_train)

# Évaluation du modèle initial
Y_pred_no_opt = model_no_optimization.predict(X_test)
report_no_opt = classification_report(Y_test, Y_pred_no_opt, output_dict=True)
conf_matrix_no_opt = confusion_matrix(Y_test, Y_pred_no_opt)

# Hyperparamètres pour RandomizedSearch
param_dist = {
    'max_depth': [None] + list(np.arange(3, 20)),
    'min_samples_split': np.arange(2, 20),
    'criterion': ['gini', 'entropy']
}

# RandomizedSearchCV
model_with_optimization = DecisionTreeClassifier()
random_search = RandomizedSearchCV(
    model_with_optimization,
    param_distributions=param_dist,
    n_iter=50,  # Nombre d'itérations de recherche
    scoring='f1_macro',
    cv=3,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, Y_train)

# Meilleurs paramètres
best_params = random_search.best_params_

# Évaluation du modèle optimisé
best_model = random_search.best_estimator_
Y_pred_opt = best_model.predict(X_test)
report_opt = classification_report(Y_test, Y_pred_opt, output_dict=True)
conf_matrix_opt = confusion_matrix(Y_test, Y_pred_opt)

# Streamlit Dashboard
st.set_page_config(page_title="Dashboard - Zeynabou Ba & Aissatou Fofana", layout="wide")

# Header
st.title("Dashboard d'Évaluation : Modèle Decision Tree")
st.markdown("""
**Binôme** : Zeynabou Ba et Aissatou Fofana  
**Classe** : Master 1 GLSI Jour
""")

# Section des résultats
st.header("Résultats Comparatifs")

# Afficher les résultats sous forme de tableau
results = pd.DataFrame({
    "Modèle": ["Sans optimisation", "Avec optimisation"],
    "Précision": [report_no_opt['weighted avg']['precision'], report_opt['weighted avg']['precision']],
    "Rappel": [report_no_opt['weighted avg']['recall'], report_opt['weighted avg']['recall']],
    "F1-Score": [report_no_opt['weighted avg']['f1-score'], report_opt['weighted avg']['f1-score']]
})
st.table(results)

# Matrices de confusion
st.subheader("Matrices de Confusion")
col1, col2 = st.columns(2)

with col1:
    st.write("### Sans optimisation")
    st.write(conf_matrix_no_opt)

with col2:
    st.write("### Avec optimisation")
    st.write(conf_matrix_opt)

# Meilleurs paramètres trouvés
st.subheader("Meilleurs Paramètres")
st.write(best_params)

# Pied de page
st.markdown("""
<div style="text-align: center; margin-top: 50px;">
    <strong>Master 1 GLSI Jour</strong>
</div>
""", unsafe_allow_html=True)
