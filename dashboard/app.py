import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


st.set_page_config(
    page_title="Estatia - Prédiction Prix Immobilier",
    page_icon="🏡",
    layout="wide"
)

@st.cache_resource
def load_trained_model():
    model_path = Path(__file__).parent.parent
    model_file = model_path / "models" / "house_price_model.joblib"
    print(model_file)
    print(Path(model_file).exists())
    print(model_file.stat().st_size)
    return joblib.load(model_file)

try:
    model = load_trained_model()
except FileNotFoundError:
    st.error("Le modèle entraîné est introuvable. Veuillez vous assurer que le fichier 'house_price_model.joblib' est présent dans le dossier 'models'.")
    st.stop()

def load_data():
    data_path = Path(__file__).parent.parent
    data_path_file = data_path / "data" / "gold" / "House_Prices_Gold.csv"
    data_frame =  pd.read_csv(data_path_file)
    return data_frame
features_cols = []
try:
    data = load_data()
    for col in data.columns:
        if col not in ["Id", "SalePrice"]:
            features_cols.append(col)
        nums_cols = data[features_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = data[features_cols].select_dtypes(include=["object", "category"]).columns.tolist()

except Exception as e:
    st.warning("Impossible de charger les données Gold pour la visualisation.")
    data = None

st.title("🏡 Estatia - Estimation du Prix d'un Logement")
st.markdown("Saisissez les caractéristiques ci-dessous pour obtenir une prédiction en temps réel via le modèle entraîné.")

st.subheader("📋 Formulaire de Saisie")

users_inputs = {}

with st.form("housing form for price detecting"):
    col1, col2 = st.columns(2)
    if data is not None:
        with col1:
            st.markdown("#### 📏 Caractéristiques Numériques")
            for col in nums_cols:
                min_value = float(data[col].min())
                max_value = float(data[col].max())
                mean_value = float(data[col].mean())

                #int because dataframe columns has int64
                if data[col].dtype == "int64":
                    users_inputs[col] = st.number_input(
                        f"{col}",
                        min_value=int(min_value),
                        max_value=int(max_value),
                        value=int(mean_value)
                    )
                else:
                    users_inputs[col] = st.number_input(
                        f"{col}",
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(mean_value)
                    )
        with col2:
            st.markdown("#### 🏷️ Caractéristiques Catégorielles")
            for col in cat_cols:
                uniques_vals = data[col].dropna().unique().tolist()
                users_inputs[col] = st.selectbox(f"{col}", options=uniques_vals)

    submit_btn = st.form_submit_button("💰 Calculer le Prix Estimé", use_container_width=True)

    if submit_btn:
        inputs_dataframe = pd.DataFrame([users_inputs])

        try:
            predicted_price = model.predict(inputs_dataframe)[0]
            st.success(f"### 💵 Prix Estimé : **${predicted_price:,.2f}**")

            if data is not None:
                st.markdown("---")
                st.subheader("📊 Positionnement par rapport au marché")

                fig, ax = plt.subplots(figsize=(10,4))
                sns.histplot(data["SalePrice"], kde=True, ax=ax, color='skyblue')
                ax.axvline(predicted_price, color='red', linestyle='--', linewidth=2, label=f'Prédiction (${predicted_price:,.0f})')
                ax.set_title("Distribution des Prix Immobiliers (Gold Data)")
                ax.set_xlabel("Prix ($)")
                ax.set_ylabel("Nombre de logements")
                ax.legend()

                st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

st.markdown("""<style>
    .powered-by {
        margin: 0.5rem 0 1.25rem;
        padding: 0.75rem 0.85rem;
        color: #8f9bab;
        background: #232323;
        border: 1px solid #343434;
        border-left: 3px solid #2dd4bf;
        border-radius: 5px;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
    }

    .powered-by strong {
        color: #d9fffb;
        font-weight: 700;
    }

    .admin {
        color: #d9fffb;
        text-decoration: none;
    }
</style>""", unsafe_allow_html=True)

st.sidebar.markdown(
    """<div class="powered-by">Powered by <strong><a class="admin" href="https://github.com/aminelebrini" target="_blank">AMINE LEBRINI</a></strong></div>""",
    unsafe_allow_html=True,
)

st.sidebar.header("ℹ️ À propos")
st.sidebar.info(
    "Cette application utilise un modèle de Machine Learning pré-entraîné (sauvegardé avec `joblib`).\n\n"
    "Le modèle prend en entrée l'ensemble des caractéristiques nettoyées et transformées."
)