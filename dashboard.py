import streamlit as st
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

# Titre du dashboard
st.title('Dashboard de Scoring Crédit')

# Charger le DataFrame traité
df_clients = pd.read_csv('data/X_predictionV1.csv')  

# Description du client
st.header('Informations sur le client')
client_id = st.text_input('ID du client (SK_ID_CURR) :', '')

# Récupérer les informations du client via l'API
def get_client_score(client_id):
    url = f"https://projet7credit-11e509d90e55.herokuapp.com/predict?SK_ID_CURR={client_id}"
    response = requests.get(url)

    # Afficher le statut de la réponse
    st.write(f"Statut de la réponse : {response.status_code}")

    # Vérifier si le statut est 200 (succès)
    if response.status_code == 200:
        try:
            # Essayer de parser la réponse comme JSON
            data = response.json()
            return data['score'], data['feature_importances']
        except ValueError as e:
            # Si la réponse n'est pas en JSON, on retourne simplement le texte brut
            st.write("Erreur lors du décodage de la réponse JSON.")
            st.write(f"Détails de l'erreur : {e}")
            return None, "HTML Response: " + response.text
    else:
        st.write(f"Erreur : {response.status_code}")
        return None, f"Erreur : {response.status_code}"

# Utilisation de la fonction
if client_id:
    score, features = get_client_score(client_id)
    if score is not None:
        st.write(f"Le score de crédit pour le client {client_id} est : {score}")
        st.progress(score)

        if features:
            st.write("Importance des caractéristiques :")
            st.write(features)
        else:
            st.write("Aucune importance des caractéristiques n'a été renvoyée.")
    else:
        st.write("Erreur lors de la récupération des données du client.")

# Affichage des informations et du score
if client_id:
    score, features = get_client_score(client_id)
    if score is not None:
        st.write(f"Le score de crédit pour le client {client_id} est : {score}")
        st.progress(score)
        
        if features:
            st.write("Importance des caractéristiques :")
            st.write(features)
        else:
            st.write("Aucune importance des caractéristiques n'a été renvoyée.")
        
        # Extraire les données du client
        client_data = df_clients[df_clients['SK_ID_CURR'] == int(client_id)]
        
        if not client_data.empty:
            # Comparaison des caractéristiques
            st.header('Comparaison des caractéristiques')
            
            # Par exemple, extraire et afficher l'âge (en jours)
            client_value_age = -client_data['DAYS_BIRTH'].values[0] // 365  # Transformer en années
            st.write(f"Âge du client : {client_value_age} ans")
            
            # Graphique de comparaison pour une feature
            def plot_comparaison(feature):
                plt.figure(figsize=(10, 6))
                sns.histplot(df_clients[feature], kde=True)
                plt.axvline(client_data[feature].values[0], color='red', linestyle='--', label=f'Client {client_id}')
                plt.title(f'Comparaison de la caractéristique {feature}')
                plt.legend()
                st.pyplot(plt)

            # Choisir la feature à comparer
            feature_to_compare = st.selectbox('Choisissez une caractéristique à comparer :', df_clients.columns[1:])  # Exclure SK_ID_CURR
            plot_comparaison(feature_to_compare)