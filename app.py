import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
st.set_page_config(page_title="AstroData Pro", page_icon="🔭", layout="wide")

# Liens SVG Bootstrap
LOGOS = {
    "dashboard": "https://icons.getbootstrap.com/assets/icons/globe-americas.svg",
    "univariate": "https://icons.getbootstrap.com/assets/icons/bar-chart-fill.svg",
    "bivariate": "https://icons.getbootstrap.com/assets/icons/diagram-3-fill.svg",
    "pca": "https://icons.getbootstrap.com/assets/icons/layers-fill.svg",
    "ai_sup": "https://icons.getbootstrap.com/assets/icons/robot.svg",
    "ai_unsup": "https://icons.getbootstrap.com/assets/icons/diagram-2-fill.svg",
    "arrow_up": "https://icons.getbootstrap.com/assets/icons/arrow-up-circle-fill.svg"
}

# --- CSS AVANCÉ ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Orbitron:wght@500;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {{ scroll-behavior: smooth !important; }}
    
    @keyframes gradientBackground {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    .stApp {{
        background: linear-gradient(-45deg, #000000, #0a1128, #1c3f60, #000000);
        background-size: 400% 400%;
        animation: gradientBackground 15s ease infinite;
        color: #e0e0e0;
    }}
    
    [data-testid="stSidebar"] {{
        background-image: url('https://images.unsplash.com/photo-1464802686167-b939a6910659?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80');
        background-size: cover; background-position: center;
    }}
    [data-testid="stSidebar"] > div:first-child {{
        backdrop-filter: blur(10px); background-color: rgba(0, 0, 0, 0.6);
    }}

    @keyframes slideInFromRight {{
        0% {{ transform: translateX(100vw); opacity: 0; }}
        100% {{ transform: translateX(0); opacity: 1; }}
    }}
    .animated-title {{
        font-family: 'Orbitron', sans-serif; font-size: 2.8em; font-weight: 700;
        color: #00d4ff; text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        animation: slideInFromRight 1.2s ease-out;
    }}

    /* Ajout du padding horizontal pour aérer le texte dans les onglets */
    .stTabs [data-baseweb="tab"] {{
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        background-color: rgba(22, 27, 34, 0.5); 
        border-radius: 8px 8px 0 0;
        padding: 10px 35px !important; 
        transition: all 0.3s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        transform: scale(1.05); color: #00d4ff !important;
        border: 1px solid rgba(0, 212, 255, 0.5) !important; background-color: rgba(0, 212, 255, 0.1);
    }}

    .stButton > button:hover {{
        transform: scale(1.1) !important; font-weight: bold !important;
        background-color: #00d4ff !important; color: black !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.8) !important;
    }}

    .desc-box {{
        background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid #00d4ff;
        padding: 15px; border-radius: 5px; margin-bottom: 25px; font-family: 'Inter', sans-serif;
    }}

    .end-page-btn {{
        display: block; margin: 40px auto 10px auto; width: 50px; height: 50px;
        background-color: rgba(0, 212, 255, 0.1); border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%; display: flex; justify-content: center; align-items: center;
        transition: all 0.3s ease; text-decoration: none;
    }}
    .end-page-btn:hover {{
        background-color: rgba(0, 212, 255, 0.6); border: 2px solid #00d4ff;
        transform: scale(1.15); box-shadow: 0 0 15px rgba(0, 212, 255, 0.8);
    }}
    .icon-filter {{ filter: invert(65%) sepia(87%) saturate(2331%) hue-rotate(154deg) brightness(101%) contrast(104%); }}
    </style>
    <div id="top"></div>
    """, unsafe_allow_html=True)

# --- DATA ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('exoplanetes_data.csv').dropna()
    except: return None
df = load_data()

def display_scroll_to_top():
    st.markdown(f'<a href="#top" class="end-page-btn" title="Remonter au début"><img src="{LOGOS["arrow_up"]}" class="icon-filter" width="30"></a>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown('<img src="https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png" width="100">', unsafe_allow_html=True)
    st.markdown("### 🌌 Contexte")
    st.write("Analyse des données issues des missions Kepler et TESS pour l'identification de mondes habitables.")
    st.markdown("### 🎯 Objectif")
    st.write("Exploration statistique (INF232) et classification Machine Learning.")
    st.divider()
    st.caption("Développé pour l'UE INF232 - UY1")

st.markdown('<div class="animated-title">🔭 AstroData Explorer Pro</div>', unsafe_allow_html=True)

if df is not None:
    # --- ONGLETS ---
    tabs = st.tabs(["Dashboard", "1&2. Univariée/Bivariée", "3. Réduction (ACP)", "4. Class. Supervisée", "5. Class. Non-Supervisée"])

    # ONGLET 1 : DASHBOARD
    with tabs[0]:
        st.markdown(f'<img src="{LOGOS["dashboard"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
        st.markdown("""<div class="desc-box"><b>Vue d'ensemble :</b> Synthèse de la base de données brute.</div>""", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Exoplanètes", len(df))
        c2.metric("Temp. Moyenne", f"{df['Temp_Etoile_K'].mean():.0f} K")
        c3.metric("Potentiel Habitable", len(df[df['Habitable']==1]))
        
        col_scatter, col_pie = st.columns([2, 1])
        with col_scatter:
            fig_scatter = px.scatter(df, x="Distance_AL", y="Masse_Terre", color="Habitable", size="Rayon_Terre", template="plotly_dark", title="Distribution Spatiale")
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_pie:
            df_pie = df.copy()
            df_pie['Statut'] = df_pie['Habitable'].map({1: 'Habitables', 0: 'Hostiles'})
            fig_pie = px.pie(df_pie, names='Statut', title="Fréquences d'Habitabilité", 
                             template="plotly_dark", hole=0.4, color_discrete_sequence=['#ff4b4b', '#00d4ff'])
            st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("🗃️ Consulter la base de données brute"):
            st.dataframe(df, use_container_width=True, hide_index=True)
            
        # NOUVELLE PRÉSENTATION AJOUTÉE ICI :
        st.markdown("---")
        st.subheader("ℹ️ À propos de l'application et des données")
        st.markdown("""
        **AstroData Explorer Pro** est une application interactive d'analyse de données développée dans le cadre de l'UE **INF232 (Statistique et Analyse des Données)**. Son objectif est d'appliquer concrètement les théories statistiques et les modèles de *Machine Learning* sur des jeux de données complexes et passionnants.

        **📡 Source des données : L'API NASA Exoplanet Archive** Les données physiques analysées dans ce projet proviennent en grande partie de la véritable API publique de la **NASA Exoplanet Archive**. 
        
        Le script de collecte (`data_builder.py`) interroge cette API via une requête SQL-like (Protocole TAP - *Table Access Protocol*) pour filtrer et télécharger les caractéristiques avérées des exoplanètes découvertes par nos télescopes (missions Kepler, TESS, etc.). 
        
        Afin de combler le fort déséquilibre naturel des classes (l'Univers connu contient très peu de planètes habitables par rapport aux planètes hostiles), la base de données réelle a été couplée mathématiquement à des profils de planètes synthétiques générés aléatoirement dans la **Zone Boucles d'or** (Habitable Zone). Cela permet d'optimiser l'apprentissage et la robustesse de notre Intelligence Artificielle.
        """)

        display_scroll_to_top()

    # ONGLET 2 : STATISTIQUES DESCRIPTIVES (AVEC BOÎTE À MOUSTACHES)
    with tabs[1]:
        st.markdown(f'<img src="{LOGOS["univariate"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
        st.markdown("""<div class="desc-box">
            <b>Rappels (Chapitres 1, 2 et 3) :</b> L'analyse descriptive permet d'étudier la distribution 
            des variables (Densité, Boîte à moustaches) et les liaisons entre elles (Matrice de Corrélation).
        </div>""", unsafe_allow_html=True)
        
        var = st.selectbox("Variable à étudier :", ['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K'])
        
        c_l, c_r = st.columns(2)
        with c_l:
            st.plotly_chart(px.histogram(df, x=var, histnorm='density', template="plotly_dark", title=f"Densité de {var}"), use_container_width=True)
        with c_r:
            corr = df[['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Distance_AL']].corr()
            st.plotly_chart(px.imshow(corr, text_auto=".2f", template="plotly_dark", title="Matrice de Corrélations"), use_container_width=True)
            
        st.markdown("---")
        st.subheader("Analyse de la Dispersion (Boîte à moustaches)")
        st.write("Conformément au cours de statistique univariée, ce graphique permet d'observer la médiane, l'écart interquartile et les valeurs extrêmes.")
        
        df_box = df.copy()
        df_box['Statut'] = df_box['Habitable'].map({1: 'Habitable', 0: 'Hostile'})
        
        col_box1, col_box2 = st.columns([3, 1])
        with col_box1:
            fig_box = px.box(df_box, x="Statut", y=var, color="Statut", 
                             color_discrete_sequence=['#ff4b4b', '#00d4ff'],
                             template="plotly_dark", title=f"Dispersion de {var} selon le statut")
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col_box2:
            st.info("""
            **🔍 Guide de lecture :**
            * **Ligne au centre :** La Médiane
            * **La Boîte :** Regroupe 50% des planètes (de $Q_1$ à $Q_3$)
            * **Les Moustaches :** Étendue des valeurs normales
            * **Les Points :** Valeurs extrêmes (aberrantes)
            """)
            
        display_scroll_to_top()

    # ONGLET 3 : ACP
    with tabs[2]:
        st.markdown(f'<img src="{LOGOS["pca"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
        st.markdown("""<div class="desc-box">
            <b>Technique de réduction des dimensionnalités (ACP) :</b> L'objectif de l'Analyse en Composantes Principales 
            est de projeter le nuage des planètes (qui possède plusieurs dimensions) sur un plan 2D, tout en 
            préservant au maximum l'information (la variance). Les variables sont préalablement standardisées.
        </div>""", unsafe_allow_html=True)
        
        features = ['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Periode_Orbitale_Jours', 'Distance_AL']
        X_pca = StandardScaler().fit_transform(df[features])
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_pca)
        
        var_expl_1 = pca.explained_variance_ratio_[0] * 100
        var_expl_2 = pca.explained_variance_ratio_[1] * 100
        
        col1, col2 = st.columns(2)
        col1.metric("Variance préservée par l'Axe 1", f"{var_expl_1:.2f} %")
        col2.metric("Variance préservée par l'Axe 2", f"{var_expl_2:.2f} %")
        st.info(f"**Information totale préservée sur le graphique 2D : {var_expl_1 + var_expl_2:.2f} %**")

        df_pca = pd.DataFrame(components, columns=['Composante Principale 1', 'Composante Principale 2'])
        df_pca['Habitable'] = df['Habitable'].values
        fig = px.scatter(df_pca, x='Composante Principale 1', y='Composante Principale 2', color='Habitable', template="plotly_dark", title="Projection des planètes selon l'ACP")
        st.plotly_chart(fig, use_container_width=True)
        display_scroll_to_top()

    # ONGLET 4 : CLASSIFICATION SUPERVISÉE
    with tabs[3]:
        st.markdown(f'<img src="{LOGOS["ai_sup"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
        st.markdown("""<div class="desc-box">
            <b>Classification Supervisée (Forêt Aléatoire) :</b> L'algorithme apprend à partir de données "étiquetées" (il sait déjà quelles planètes sont habitables dans la base). 
            Nous évaluons sa robustesse en calculant son taux de réussite sur un sous-échantillon qu'il n'a jamais vu (Données de test).
        </div>""", unsafe_allow_html=True)

        with st.expander("📖 GUIDE D'UTILISATION ET SEUILS D'HABITABILITÉ"):
            st.markdown("""
            Ce module utilise un algorithme de **Forêt Aléatoire (Random Forest)** pour classifier les planètes.
            
            **Les seuils d'habitabilité (Critères de la zone Goldilocks) :**
            Pour qu'une planète soit considérée comme candidate à la vie dans notre base de données d'entraînement, elle doit respecter *toutes* ces conditions physiques :
            * **Masse :** Comprise entre **0.5 et 5.0** fois la masse de la Terre.
            * **Rayon :** Compris entre **0.8 et 1.5** fois le rayon de la Terre (planètes rocheuses).
            * **Température de l'étoile hôte :** Comprise entre **4000 K et 6000 K** (Étoiles similaires à notre Soleil).
            
            **Comment tester ?**
            Modifiez les curseurs ci-dessous pour créer une planète de toutes pièces. L'IA comparera ces valeurs aux seuils appris et vous donnera son verdict instantanément.
            """)
        
        X = df[['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K']]
        y = df['Habitable']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_sup = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        accuracy = model_sup.score(X_test, y_test) * 100
        
        st.metric("Taux de réussite du Modèle (Précision sur le jeu de test)", f"{accuracy:.1f} %")
        
        st.subheader("Simulateur de Prédiction")
        c1, c2, c3 = st.columns(3)
        m = c1.slider("Masse", 0.1, 10.0, 1.0)
        r = c2.slider("Rayon", 0.1, 5.0, 1.0)
        t = c3.number_input("Température (K)", 2000, 10000, 5700)

        if st.button("🚀 LANCER L'ANALYSE", use_container_width=True):
            proba = model_sup.predict_proba([[m, r, t]])[0]
            if model_sup.predict([[m, r, t]])[0] == 1:
                st.success(f"🌍 PLANÈTE HABITABLE ! (L'IA est certaine à {proba[1]*100:.1f} %)")
                st.balloons()
            else:
                st.error(f"🌋 PLANÈTE HOSTILE. (L'IA est certaine à {proba[0]*100:.1f} %)")
        display_scroll_to_top()

    # ONGLET 5 : CLASSIFICATION NON-SUPERVISÉE
    with tabs[4]:
        st.markdown(f'<img src="{LOGOS["ai_unsup"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
        st.markdown("""<div class="desc-box">
            <b>Classification Non-Supervisée (Clustering K-Means) :</b> Contrairement au modèle précédent, l'algorithme ne connait pas l'étiquette (Habitable ou non). 
            Son rôle est de regrouper "à l'aveugle" les planètes en Familles (Clusters) basées sur leurs similitudes mathématiques (distances).
        </div>""", unsafe_allow_html=True)
        
        nb_clusters = st.slider("Choisissez le nombre de Familles (K) :", 2, 5, 3)
        
        X_unsup = df[['Masse_Terre', 'Rayon_Terre']]
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_unsup)
        
        repartition = df['Cluster'].value_counts(normalize=True) * 100
        
        st.write("**Répartition de la population par Famille :**")
        cols = st.columns(nb_clusters)
        for i in range(nb_clusters):
            cols[i].metric(f"Famille {i}", f"{repartition[i]:.1f} %")
            
        fig = px.scatter(df, x="Rayon_Terre", y="Masse_Terre", color=df['Cluster'].astype(str), 
                         title=f"Segmentation K-Means en {nb_clusters} familles",
                         color_discrete_sequence=px.colors.qualitative.Set2, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        display_scroll_to_top()

else:
    st.warning("Veuillez générer les données avec data_builder.py d'abord.")