import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
st.set_page_config(page_title="AstroData Pro", page_icon="🔭", layout="wide")

# --- INITIALISATION DE LA MÉMOIRE DE SESSION ---
if 'etape_actuelle' not in st.session_state:
    st.session_state.etape_actuelle = "Accueil" # Peut être "Accueil", "Saisie", ou "Analyse"

if 'choix_source' not in st.session_state:
    st.session_state.choix_source = "NASA"

if 'user_planets' not in st.session_state:
    st.session_state.user_planets = pd.DataFrame(columns=[
        'Nom', 'Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Periode_Orbitale_Jours', 'Distance_AL', 'Habitable'
    ])

# --- FONCTIONS UTILES ---
def changer_etape(nouvelle_etape, source=None):
    st.session_state.etape_actuelle = nouvelle_etape
    if source:
        st.session_state.choix_source = source

def apply_transparent_style(fig):
    """Rend l'arrière-plan des graphiques Plotly transparent"""
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def verifier_coherence_planete(nom, masse, rayon, temp, periode, df_existant):
    # 1. Vérification du Nom (Doublon)
    if nom in df_existant['Nom'].values:
        return False, f"Le nom '{nom}' est déjà utilisé par une autre mission."

    # 2. Vérification des données identiques (Copier-coller)
    # On compare tout sauf le nom
    match_data = df_existant[
        (df_existant['Masse_Terre'] == masse) & 
        (df_existant['Rayon_Terre'] == rayon) & 
        (df_existant['Temp_Etoile_K'] == temp) &
        (df_existant['Periode_Orbitale_Jours'] == periode)
    ]
    if not match_data.empty:
        return False, "Ces coordonnées physiques correspondent exactement à une planète déjà enregistrée."

    # 3. Logique de Densité (Masse vs Rayon)
    # Calcul simplifié : Densité proportionnelle à M/R^3
    densite_relative = masse / (rayon**3)
    
    if densite_relative > 20: # Trop dense (ex: une boule de plomb géante)
        return False, "Incohérence physique : La planète est trop dense. Sa masse est trop élevée pour un si petit rayon."
    if densite_relative < 0.05: # Trop "vaporeuse"
        return False, "Incohérence physique : Planète 'Barbe à papa'. Son rayon est trop grand pour une masse si faible."

    # 4. Logique Température vs Étoile
    if temp > 15000:
        return False, "Attention : Cette température correspond à une étoile massive (Type O/B) non gérée par nos modèles d'habitabilité."
    
    return True, "Paramètres orbitaux validés !"

@st.cache_data(show_spinner=False)
def get_base64_image(image_path, max_width=1280, quality=72):
    """
    Convertit une image locale en base64.
    Mise en cache — encodée une seule fois pour toute la session serveur.
    """
    try:
        from PIL import Image
        import io
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            return ""
    except FileNotFoundError:
        st.warning(f"⚠️ Image introuvable : {image_path}")
        return ""

def get_svg_base64(svg_path):
    """Retourne une data-URL base64 pour un SVG local."""
    try:
        with open(svg_path, "rb") as f:
            return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.warning(f"⚠️ SVG introuvable : {svg_path}")
        return ""

# Liens SVG Bootstrap (Restaurés)
LOGOS = {
    "dashboard": "https://icons.getbootstrap.com/assets/icons/globe-americas.svg",
    "univariate": "https://icons.getbootstrap.com/assets/icons/bar-chart-fill.svg",
    "bivariate": "https://icons.getbootstrap.com/assets/icons/diagram-3-fill.svg",
    "pca": "https://icons.getbootstrap.com/assets/icons/layers-fill.svg",
    "ai_sup": "https://icons.getbootstrap.com/assets/icons/robot.svg",
    "ai_unsup": "https://icons.getbootstrap.com/assets/icons/diagram-2-fill.svg",
    "arrow_up": "https://icons.getbootstrap.com/assets/icons/arrow-up-circle-fill.svg"
}

def display_scroll_to_top():
    st.markdown(f'<a href="#top" class="end-page-btn" title="Remonter au début"><img src="{LOGOS["arrow_up"]}" class="icon-filter" width="30"></a>', unsafe_allow_html=True)

# --- CSS AVANCÉ (Incluant le split-screen à 30° et les filtres de température) ---
# --- CSS AVANCÉ (Optimisé pour la performance et le mobile) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Orbitron:wght@500;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] { scroll-behavior: smooth !important; }

    /* Animation du fond : On la garde pour PC, mais on va la désactiver sur mobile pour économiser la batterie */
    @keyframes gradientBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .stApp {
        background: linear-gradient(-45deg, #000000, #0a1128, #1c3f60, #000000);
        background-size: 400% 400%;
        animation: gradientBackground 15s ease infinite;
        color: #e0e0e0;
    }

    /* --- EFFETS TITRE --- */
    @keyframes slideInFromRight {
        0% { transform: translateX(100vw); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }
    .animated-title {
        font-family: 'Orbitron', sans-serif; font-size: 2.8em; font-weight: 700;
        color: #00d4ff; text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        animation: slideInFromRight 1.2s ease-out; text-align: center; margin-bottom: 20px;
    }

    /* Désactivation de l'animation lourde sur mobile */
    @media (max-width: 768px) {
        .stApp {
            animation: none !important;
            background-position: 50% 50% !important;
        }
        .animated-title {
            font-size: 2em; /* Titre un peu plus petit sur mobile */
        }
    }

    /* --- BOÎTES DE DÉTAIL ONGLETS --- */
    .stTabs [data-baseweb="tab"] {
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        background-color: rgba(22, 27, 34, 0.5); 
        border-radius: 8px 8px 0 0;
        padding: 10px 35px !important; 
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        transform: scale(1.05); color: #00d4ff !important;
        border: 1px solid rgba(0, 212, 255, 0.5) !important; background-color: rgba(0, 212, 255, 0.1);
    }
    .stButton > button:hover {
        transform: scale(1.05) !important; font-weight: bold !important;
        background-color: #00d4ff !important; color: black !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.8) !important;
    }
    .desc-box {
        background-color: rgba(255, 255, 255, 0.05); border-left: 5px solid #00d4ff;
        padding: 15px; border-radius: 5px; margin-bottom: 25px; font-family: 'Inter', sans-serif;
    }
    .end-page-btn {
        display: block; margin: 40px auto 10px auto; width: 50px; height: 50px;
        background-color: rgba(0, 212, 255, 0.1); border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%; display: flex; justify-content: center; align-items: center;
        transition: all 0.3s ease; text-decoration: none;
    }
    .end-page-btn:hover {
        background-color: rgba(0, 212, 255, 0.6); border: 2px solid #00d4ff;
        transform: scale(1.15); box-shadow: 0 0 15px rgba(0, 212, 255, 0.8);
    }
    .icon-filter { filter: invert(65%) sepia(87%) saturate(2331%) hue-rotate(154deg) brightness(101%) contrast(104%); }
    .stPlotlyChart { background-color: transparent !important; }
    
    /* --- STYLE DE LA SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: rgba(0, 8, 20, 0.4) !important; 
        backdrop-filter: blur(10px) !important; 
        -webkit-backdrop-filter: blur(10px) !important; 
        border-right: 1px solid rgba(0, 212, 255, 0.2) !important; 
    }
    [data-testid="stSidebarHeader"] { background-color: transparent !important; }

    /* Réduit le padding vertical Streamlit pour que la page d'accueil tienne sans scroll */
    [data-testid="stAppViewContainer"] > section > div:first-child {
        padding-top: 1rem !important;
    }
    </style>
    <div id="top"></div>
""", unsafe_allow_html=True)

# --- INJECTION DES IMAGES (une seule fois, mise en cache) ---
img_kepler_b64 = get_base64_image("data/Earth.png")
img_terre_b64  = get_base64_image("data/Trappist-1e.png")
svg_saturne_b64 = get_svg_base64("data/genetic-data-svgrepo-com.svg")
svg_rocket_b64 = get_svg_base64("data/rocket-svgrepo-com.svg")

# Classes bg activées par JS selon la page — jamais ré-injectées ensuite.

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    try:
        df_all = pd.read_csv('exoplanetes_data.csv').dropna()
        if 'Source' not in df_all.columns:
            df_all['Source'] = 'NASA'
        return df_all
    except: return None

df_global = load_data()


# Masque la sidebar sur Accueil et Saisie
if st.session_state.etape_actuelle != "Analyse":
    st.markdown("""
        <style>
        [data-testid="stSidebar"]      { display: none !important; }
        [data-testid="collapsedControl"]{ display: none !important; }
        </style>
    """, unsafe_allow_html=True)

# =====================================================================
# ÉTAPE 1 : PAGE D'ACCUEIL
# =====================================================================
if st.session_state.etape_actuelle == "Accueil":
    # Titre principal — margin-top réduit pour rester dans le viewport
    st.markdown('<div class="animated-title" style="margin-top: 2vh; margin-bottom: 8px;">🔭 ASTRODATA EXPLORER</div>', unsafe_allow_html=True)

    st.markdown("""
        <p style='text-align: center; color: #b0b8c8; font-size: 1.1em; font-family: "Inter", sans-serif; margin-bottom: 2vh; font-weight: 400; padding: 0 15px;'>
            Bienvenue dans votre centre de commande planétaire. Plongez dans les véritables archives de l'univers connu, ou entrez dans le laboratoire pour forger vos propres mondes.
        </p>
    """, unsafe_allow_html=True)

    # Boutons Streamlit réels cachés
    col1, col2 = st.columns(2)
    with col1:
        btn_nasa = st.button("nasa_hidden", key="btn_nasa_real")
    with col2:
        btn_simu = st.button("simu_hidden", key="btn_simu_real")
    if btn_nasa:
        changer_etape("Analyse", "NASA")
        st.rerun()
    if btn_simu:
        changer_etape("Saisie", "Simulation")
        st.rerun()

    st.markdown("""
        <style>
        div[data-testid="stHorizontalBlock"] { display: none !important; }
        /* Supprime tout padding Streamlit autour du composant iframe */
        div[data-testid="stCustomComponentV1"] > iframe { display: block; }
        </style>
    """, unsafe_allow_html=True)

    import streamlit.components.v1 as components
    components.html(f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:transparent; font-family:'Inter',sans-serif; overflow:hidden; }}

  .cards-wrapper {{
    display: flex;
    flex-direction: row;
    gap: 24px;
    padding: 4px 2px 8px 2px;
  }}

  .rubrique-card {{
    flex: 1;
    padding: 30px 28px;
    background: rgba(0,0,0,0.4);
    border: 1.5px solid rgba(255,255,255,0.25);
    border-radius: 18px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.3s, box-shadow 0.3s, background 0.3s;
    cursor: default;
  }}
  .rubrique-card:hover {{
    border-color: #00d4ff;
    box-shadow: 0 0 28px rgba(0,212,255,0.45);
    background: rgba(0,0,0,0.6);
  }}
  .rubrique-card.card-simu:hover {{
    border-color: #c1440e;
    box-shadow: 0 0 28px rgba(193,68,14,0.55);
    background: rgba(0,0,0,0.6);
  }}
  .rubrique-card h3 {{
    color: #e0e0e0;
    font-family: 'Orbitron', 'Inter', sans-serif;
    font-size: 1.1em;
    font-weight: 700;
    text-align: center;
  }}
  .rubrique-card p {{
    color: #b0b8c8;
    font-size: 0.92em;
    line-height: 1.5;
    text-align: center;
  }}
  .card-btn {{
    display: block;
    width: 60%;
    margin: 6px auto 0 auto;
    padding: 10px 0;
    background: #00d4ff;
    color: #000;
    font-weight: bold;
    font-family: 'Orbitron', 'Inter', sans-serif;
    font-size: 0.85em;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s, background 0.2s;
  }}
  .card-btn:hover {{
    transform: scale(1.05);
    box-shadow: 0 0 18px rgba(0,212,255,0.7);
    background: #33ddff;
  }}

  /* ── MOBILE : empilement vertical ── */
  @media (max-width: 768px) {{
    html, body {{ overflow-y: auto !important; }}  /* scroll autorisé sur mobile */
    .cards-wrapper {{
      flex-direction: column;
      gap: 16px;
    }}
    .rubrique-card {{
      padding: 20px 16px;
      gap: 8px;
    }}
    .rubrique-card h3 {{ 
        font-size: 1em;
        text-align: center;
        }}
    .rubrique-card p  {{ 
        font-size: 0.85em; 
        text-align: center;
    }}
    .rubrique-card img {{ width: 80px !important; }}
    .card-btn {{
      width: 80%;
      padding: 12px 0;
      font-size: 0.82em;
    }}
  }}
</style>
</head>
<body>

<div class="cards-wrapper" id="cards">
  <div class="rubrique-card"
       onmouseenter="showBg('nasa')"
       onmouseleave="hideBg()">
    <img src="https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png"
         width="110" style="display:block; margin: 0 auto 5px auto;" loading="lazy"/>
    <h3>Base R&eacute;elle NASA</h3>
    <p>Analysez les v&eacute;ritables exoplan&egrave;tes d&eacute;couvertes par nos t&eacute;lescopes (Missions Kepler et TESS).</p>
    <button class="card-btn" onclick="navigate('nasa')">Lancer l&rsquo;exploration</button>
  </div>

  <div class="rubrique-card card-simu"
       onmouseenter="showBg('simu')"
       onmouseleave="hideBg()">
    <img src="{svg_saturne_b64}"
         width="50" style="display:block; margin: 0 auto 5px auto;" loading="lazy"/>
    <h3>Laboratoire (Simulation)</h3>
    <p>Partez d&rsquo;un &eacute;chantillon r&eacute;duit, cr&eacute;ez vos propres plan&egrave;tes et pi&eacute;gez l&rsquo;IA.</p>
    <button class="card-btn" onclick="navigate('simu')">Cr&eacute;er des mondes</button>
  </div>
</div>

<script>
  /* ── Redimensionnement automatique de l'iframe ──
     Double approche : postMessage pour Streamlit + frameElement direct */
  function sendHeight() {{
    var h = document.getElementById('cards').scrollHeight + 8;
    /* Méthode 1 : postMessage Streamlit */
    window.parent.postMessage({{type:'streamlit:setFrameHeight', height: h}}, '*');
    /* Méthode 2 : accès direct à l'élément iframe */
    try {{
      if (window.frameElement) window.frameElement.style.height = h + 'px';
    }} catch(e) {{}}
  }}
  document.addEventListener('DOMContentLoaded', function() {{
    sendHeight();
    setTimeout(sendHeight, 100);
    setTimeout(sendHeight, 400);
  }});
  window.addEventListener('load', sendHeight);
  window.addEventListener('resize', sendHeight);
  if (window.ResizeObserver) {{
    new ResizeObserver(sendHeight).observe(document.getElementById('cards'));
  }}

  var IMG_NASA = 'data:image/png;base64,{img_kepler_b64}';
  var IMG_SIMU = 'data:image/png;base64,{img_terre_b64}';

  function getStApp() {{
    return window.parent.document.querySelector('.stApp');
  }}

  function getOrCreateBgDiv() {{
    var p = window.parent.document;
    var el = p.getElementById('__astro_blur_bg__');
    if (!el) {{
      el = p.createElement('div');
      el.id = '__astro_blur_bg__';
      el.style.cssText = [
        'position:fixed','inset:0','z-index:0',
        'pointer-events:none','opacity:0',
        'background-size:cover','background-position:center',
        'filter:blur(3px)','transform:scale(1.05)',
        'transition:opacity 1.3s ease-in-out'
      ].join(';');
      p.body.insertBefore(el, p.body.firstChild);
    }}
    return el;
  }}

  var _hideTimer = null;

  function showBg(which) {{
    var img = which === 'nasa' ? IMG_NASA : IMG_SIMU;
    var app = getStApp();
    var bgDiv = getOrCreateBgDiv();
    if (_hideTimer) {{ clearTimeout(_hideTimer); _hideTimer = null; }}
    if (app) {{ app.style.transition='none'; app.style.background='transparent'; app.style.animation='none'; }}
    bgDiv.style.transition = 'none';
    bgDiv.style.backgroundImage = 'linear-gradient(rgba(0,0,0,0.45),rgba(0,0,0,0.55)),url('+img+')';
    bgDiv.style.opacity = '0';
    requestAnimationFrame(function() {{
      requestAnimationFrame(function() {{
        bgDiv.style.transition = 'opacity 1.4s ease-in-out';
        bgDiv.style.opacity = '1';
      }});
    }});
  }}

  function hideBg() {{
    var app = getStApp();
    var bgDiv = getOrCreateBgDiv();
    bgDiv.style.transition = 'opacity 1.4s ease-in-out';
    bgDiv.style.opacity = '0';
    _hideTimer = setTimeout(function() {{
      _hideTimer = null;
      if (app) {{
        app.style.transition = 'none';
        app.style.background = 'linear-gradient(-45deg,#000000,#0a1128,#1c3f60,#000000)';
        app.style.backgroundSize = '400% 400%';
        if (window.parent.innerWidth > 768) {{
          app.style.animation = 'gradientBackground 15s ease infinite';
        }}
      }}
    }}, 1400);
  }}

  function navigate(dest) {{
    var buttons = window.parent.document.querySelectorAll('button');
    var keyword = dest === 'nasa' ? 'nasa_hidden' : 'simu_hidden';
    for (var i = 0; i < buttons.length; i++) {{
      if (buttons[i].innerText.trim() === keyword) {{ buttons[i].click(); return; }}
    }}
  }}

  function adjustHeight() {{
    var body = document.body;
    var h = body.getBoundingClientRect().height;
    window.parent.postMessage({{type: 'streamlit:setFrameHeight', height: h}}, '*');
    if (window.frameElement) window.frameElement.style.height = h + 'px';
  }}
  window.addEventListener('load', function() {{ setTimeout(adjustHeight, 80); }});
  window.addEventListener('resize', adjustHeight);
  if (window.ResizeObserver) {{
    new ResizeObserver(adjustHeight).observe(document.body);
  }}
</script>

<p style="text-align:center;color:rgba(255,255,255,0.25);font-size:0.73em;
          font-family:'Inter',sans-serif;margin-top:120px;letter-spacing:0.05em;">
  AstroData Explorer &nbsp;&mdash;&nbsp; Donn&eacute;es NASA · Kepler &amp; TESS
  &nbsp;&mdash;&nbsp; Moteur IA : Random Forest
</p>

<p style="text-align:center;color:rgba(255,255,255,0.25);font-size:0.80em;
          font-family:'Inter',sans-serif;margin-top:15px;letter-spacing:0.07em;">
  Par : Abondo Jean Joël
  <br>Matricule : 23V2214
</p>  

</body>
</html>
    """, height=560, scrolling=False)

# =====================================================================
# ÉTAPE 2 : PAGE DE SAISIE (Uniquement pour Simulation)
# =====================================================================
elif st.session_state.etape_actuelle == "Saisie":
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.75)),
                        url("data:image/jpeg;base64,{img_terre_b64}") center/cover fixed !important;
            animation: none !important;
        }}
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="animated-title">🪐 LABORATOIRE DE CRÉATION</div>', unsafe_allow_html=True)
    st.markdown("Simulez vos propres exoplanètes et testez-les face aux lois de l'astrophysique.")

    

    col_form, col_recap = st.columns([1, 2], gap="large")

    with col_form:
        st.subheader("Forger une nouvelle planète")

        # --- SLIDERS EN DEHORS DU FORMULAIRE pour mise à jour en temps réel ---
        nom_p     = st.text_input("Nom de la planète", value=f"Alpha-{np.random.randint(100, 999)}")
        c1, c2    = st.columns(2)
        with c1:
            masse_p   = st.slider("Masse (Terre)",                0.1,    10.0,   1.0)
            rayon_p   = st.slider("Rayon (Terre)",                0.1,     5.0,   1.0)
            dist_p    = st.slider("Distance (Années-Lumière)",      1,    5000,   100)
        with c2:
            temp_p    = st.slider("Température de l'étoile (K)", 2000,   10000,  5800)
            periode_p = st.slider("Période Orbitale (Jours)",      1.0,  1000.0, 365.0)

        # --- INDICATEUR ML EN TEMPS RÉEL (se met à jour à chaque mouvement de slider) ---
        if df_global is not None:
            df_ml = df_global.copy()
            for col in ['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Habitable']:
                df_ml[col] = pd.to_numeric(df_ml[col], errors='coerce')
            df_ml['Habitable'] = df_ml['Habitable'].fillna(0).astype(int)
            df_ml = df_ml.dropna(subset=['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K'])
            X_ml = df_ml[['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K']]
            y_ml = df_ml['Habitable']
            if len(y_ml.unique()) > 1:
                X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.2, random_state=42)
                model_labo = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
                proba_labo = model_labo.predict_proba([[masse_p, rayon_p, temp_p]])[0]
                pred_labo  = model_labo.predict([[masse_p, rayon_p, temp_p]])[0]
                if pred_labo == 1:
                    st.success(f"🟢 Potentiellement habitable — certitude IA : {proba_labo[1]*100:.1f} %")
                else:
                    st.error(f"🔴 Probablement hostile — certitude IA : {proba_labo[0]*100:.1f} %")

        # --- FORMULAIRE réduit au bouton de validation uniquement ---
        with st.form("form_planete", clear_on_submit=True):
            soumettre = st.form_submit_button("Ajouter à l'univers", use_container_width=True)
            if soumettre:
                est_valide, message = verifier_coherence_planete(
                    nom_p, masse_p, rayon_p, temp_p, periode_p, st.session_state.user_planets
                )
                if est_valide:
                    is_hab = 1 if (0.7 <= masse_p <= 3.5 and 0.9 <= rayon_p <= 1.4 and 4500 <= temp_p <= 5800) else 0
                    nouvelle_ligne = pd.DataFrame([{
                        'Nom': nom_p, 'Masse_Terre': masse_p, 'Rayon_Terre': rayon_p,
                        'Temp_Etoile_K': temp_p, 'Periode_Orbitale_Jours': periode_p,
                        'Distance_AL': dist_p, 'Habitable': is_hab, 'Source': 'Simulation'
                    }])
                    st.session_state.user_planets = pd.concat([st.session_state.user_planets, nouvelle_ligne], ignore_index=True)
                    st.success(f"Succès ! La planète '{nom_p}' a rejoint l'univers.")
                else:
                    st.error(message)

    with col_recap:
        st.subheader("Vos créations actuelles")
        if len(st.session_state.user_planets) == 0:
            st.info("💡 Votre flotte est vide. Utilisez le créateur à gauche pour enregistrer votre première planète.")
        else:
            st.dataframe(st.session_state.user_planets, use_container_width=True, height=400)
            
        st.markdown("---")
        st.markdown("Une fois que vous avez terminé de créer vos planètes, lancez les algorithmes d'analyse et de prédiction.")
        if st.button("Terminer et passer à l'Analyse", type="primary", use_container_width=True):
            changer_etape("Analyse")
            st.rerun()
        if st.button("Retour à l'accueil"):
            changer_etape("Accueil")
            st.rerun()

# =====================================================================
# ÉTAPE 3 : PAGE D'ANALYSE (Onglets Restaurés)
# =====================================================================
elif st.session_state.etape_actuelle == "Analyse":
    fond_b64 = img_kepler_b64 if st.session_state.choix_source == "NASA" else img_terre_b64
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.75)),
                        url("data:image/jpeg;base64,{fond_b64}") center/cover fixed !important;
            animation: none !important;
        }}
        </style>
    """, unsafe_allow_html=True)

    # --- PRÉPARATION DU DATAFRAME FILTRÉ ---
    if df_global is not None:
        if st.session_state.choix_source == "NASA":
            df = df_global[df_global['Source'] == 'NASA'].copy()
        else:
            # On récupère l'échantillon des 20 planètes (Simulation) et on y ajoute les créations de l'utilisateur
            df_sim = df_global[df_global['Source'] == 'Simulation']
            df = pd.concat([df_sim, st.session_state.user_planets], ignore_index=True)

        # 🛠️ Forcer le type numérique pour éviter les erreurs Plotly ET Scikit-Learn
        colonnes_numeriques = ['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Periode_Orbitale_Jours', 'Distance_AL', 'Habitable']
        for col in colonnes_numeriques:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Scikit-Learn exige que la cible (Habitable) soit un entier strictement (int)
        if 'Habitable' in df.columns:
            df['Habitable'] = df['Habitable'].fillna(0).astype(int)

    # --- SIDEBAR (Restaurée pour la vue Analyse) ---
    with st.sidebar:
        st.markdown('<img src="https://www.nasa.gov/wp-content/uploads/2023/03/nasa-logo-web-rgb.png" width="100" loading="lazy">', unsafe_allow_html=True)
        st.markdown("### Contexte")
        st.write("Analyse des données issues des missions Kepler et TESS pour l'identification de mondes habitables.")
        
        st.divider()
        st.markdown(f"**Mission actuelle :** {st.session_state.choix_source}")
        
        if st.button("Retourner à l'accueil", use_container_width=True):
            changer_etape("Accueil")
            st.rerun()

        st.divider()
        st.markdown("""
            <div style="text-align:center; color:rgba(255,255,255,0.55);
                        font-size:0.82em; font-family:'Inter',sans-serif;
                        line-height:1.8; letter-spacing:0.04em;">
                Données NASA · Kepler &amp; TESS<br>
                Moteur IA : Random Forest<br>
                <br>
                <span style="color:rgba(255,255,255,0.35); font-size:0.92em;">
                    Par : Abondo Jean Joël<br>
                    Matricule : 23V2214
                </span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="animated-title">🔭 AstroData Explorer Pro</div>', unsafe_allow_html=True)

    if df_global is not None and len(df) > 0:
        # --- ONGLETS (RESTUCTURÉS POUR LE TP) ---
        tabs = st.tabs(["Dashboard", "1&2. Univariée/Bivariée", "3. Réduction (ACP)", "4. Class. Supervisée", "5. Class. Non-Supervisée"])

        # ONGLET 1 : DASHBOARD
        with tabs[0]:
            st.markdown(f'<img src="{LOGOS["dashboard"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
            st.markdown(f"""<div class="desc-box"><b>Vue d'ensemble ({st.session_state.choix_source}):</b> Synthèse de la base de données.</div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Exoplanètes analysées", len(df))
            c2.metric("Temp. Moyenne", f"{df['Temp_Etoile_K'].mean():.0f} K")
            c3.metric("Potentiel Habitable", len(df[df['Habitable']==1]))
            
            col_scatter, col_pie = st.columns([2, 1])
            with col_scatter:
                fig_scatter = px.scatter(df, x="Distance_AL", y="Masse_Terre", color="Habitable", size="Rayon_Terre", template="plotly_dark", title="Distribution Spatiale")
                st.plotly_chart(apply_transparent_style(fig_scatter), use_container_width=True)
            
            with col_pie:
                # Création d'une copie pour un affichage textuel propre dans le camembert
                df_pie = df.copy()
                df_pie['Statut'] = df_pie['Habitable'].map({1: 'Habitables', 0: 'Hostiles'})
                fig_pie = px.pie(df_pie, names='Statut', title="Fréquences d'Habitabilité", 
                                 template="plotly_dark", hole=0.4, color_discrete_sequence=['#ff4b4b', '#00d4ff'])
                st.plotly_chart(apply_transparent_style(fig_pie), use_container_width=True)

            with st.expander("Consulter la base de données brute"):
                # Si on est en simulation, on affiche une petite pastille pour distinguer
                if st.session_state.choix_source != "NASA" and len(st.session_state.user_planets) > 0:
                    st.info(f"Vos {len(st.session_state.user_planets)} planètes créées sont incluses ci-dessous.")
                st.dataframe(df, use_container_width=True, hide_index=True)
                
            st.markdown("---")
            st.subheader("À propos de l'application et des données")
            st.markdown("""
            **AstroData Explorer Pro** est une application interactive d'analyse de données développée dans le cadre de l'UE **INF232 (Statistique et Analyse des Données)**. Son objectif est d'appliquer concrètement les théories statistiques et les modèles de *Machine Learning* sur des jeux de données complexes et passionnants.

            **Source des données : L'API NASA Exoplanet Archive** Les données physiques analysées dans ce projet proviennent entièrement de la véritable API publique de la **NASA Exoplanet Archive**. 
            """)

            display_scroll_to_top()

        # ONGLET 2 : STATISTIQUES DESCRIPTIVES
        with tabs[1]:
            st.markdown(f'<img src="{LOGOS["univariate"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
            st.markdown("""<div class="desc-box">
                <b>Guide d'analyse :</b> L'analyse descriptive permet d'étudier la distribution 
                des variables (Densité, Boîte à moustaches) et les liaisons entre elles (Matrice de Corrélation).
            </div>""", unsafe_allow_html=True)
            
            var = st.selectbox("Variable à étudier :", ['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K'])
            
            c_l, c_r = st.columns(2)
            with c_l:
                fig_hist = px.histogram(df, x=var, histnorm='density', template="plotly_dark", title=f"Densité de {var}")
                st.plotly_chart(apply_transparent_style(fig_hist), use_container_width=True)
            with c_r:
                corr = df[['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Distance_AL']].corr()
                fig_corr = px.imshow(corr, text_auto=".2f", template="plotly_dark", title="Matrice de Corrélations")
                st.plotly_chart(apply_transparent_style(fig_corr), use_container_width=True)
                
            st.markdown("---")
            st.subheader("Analyse de la Dispersion (Boîte à moustaches)")
            st.write("Ce graphique permet d'observer la médiane, l'écart interquartile et les valeurs extrêmes.")
            
            df_box = df.copy()
            df_box['Statut'] = df_box['Habitable'].map({1: 'Habitable', 0: 'Hostile'})
            
            col_box1, col_box2 = st.columns([3, 1])
            with col_box1:
                # Création de la boîte à moustaches via Plotly
                fig_box = px.box(df_box, x="Statut", y=var, color="Statut", 
                                 color_discrete_sequence=['#ff4b4b', '#00d4ff'],
                                 template="plotly_dark", title=f"Dispersion de {var} selon le statut")
                st.plotly_chart(apply_transparent_style(fig_box), use_container_width=True)
                st.info("""
                    **Interprétation :** Ce graphique montre la répartition de nos découvertes. 
                    Une forte concentration à courte distance (0-500 AL) reflète souvent les limites de nos télescopes 
                    actuels plutôt qu'une réalité galactique. Plus une planète est loin, plus elle est difficile à confirmer.
                """)
                            
            with col_box2:
                st.info("""
                **Guide de lecture :**
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
            
            if len(df) < 5:
                st.warning("Pas assez de données pour faire une ACP. Veuillez ajouter des planètes ou repasser sur la base NASA.")
            else:
                # Opérations mathématiques pour l'ACP
                features = ['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K', 'Periode_Orbitale_Jours', 'Distance_AL']
                X_pca = StandardScaler().fit_transform(df[features])
                pca = PCA(n_components=2)
                components = pca.fit_transform(X_pca)
                
                # Affichage des pourcentages de variance
                var_expl_1 = pca.explained_variance_ratio_[0] * 100
                var_expl_2 = pca.explained_variance_ratio_[1] * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Variance préservée par l'Axe 1", f"{var_expl_1:.2f} %")
                col2.metric("Variance préservée par l'Axe 2", f"{var_expl_2:.2f} %")
                st.info(f"**Information totale préservée sur le graphique 2D : {var_expl_1 + var_expl_2:.2f} %**")

                df_pca = pd.DataFrame(components, columns=['Composante Principale 1', 'Composante Principale 2'])
                df_pca['Habitable'] = df['Habitable'].values
                fig_pca_scatter = px.scatter(df_pca, x='Composante Principale 1', y='Composante Principale 2', color='Habitable', template="plotly_dark", title="Projection des planètes selon l'ACP")
                st.plotly_chart(apply_transparent_style(fig_pca_scatter), use_container_width=True)
                st.success("""
                    **Analyse PCA :** L'IA a réduit toutes les dimensions (masse, rayon, température) en deux axes. 
                    Si vous voyez des points jaunes (Habitables) bien groupés, cela signifie que ces planètes partagent 
                    une signature physique commune, distincte des mondes hostiles.
                """)
            display_scroll_to_top()

        # ONGLET 4 : CLASSIFICATION SUPERVISÉE
        with tabs[3]:
            st.markdown(f'<img src="{LOGOS["ai_sup"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
            st.markdown("""<div class="desc-box">
                <b>Classification Supervisée (Forêt Aléatoire) :</b> L'algorithme s'entraîne en temps réel sur la base de données sélectionnée (NASA ou Simulation) pour apprendre quelles planètes sont habitables.
            </div>""", unsafe_allow_html=True)

            with st.expander("GUIDE D'UTILISATION ET SEUILS D'HABITABILITÉ"):
                st.markdown("""
                Ce module utilise un algorithme de **Forêt Aléatoire (Random Forest)** pour classifier les planètes.
                
                **Les seuils d'habitabilité (Critères de la zone Goldilocks) :**
                Pour qu'une planète soit considérée comme candidate à la vie dans notre base de données d'entraînement, elle doit respecter *toutes* ces conditions physiques :
                * ** Masse **:Coefficient du produit par la masse de la Terre.
                * ** Rayon **: Coefficient du produit par le rayon de la Terre (planètes rocheuses).
                * ** Température de l'étoile hôte **:(Étoiles similaires à notre Soleil).
                
                **Comment tester ?**
                Modifiez les curseurs ci-dessous pour créer une planète de toutes pièces. L'IA comparera ces valeurs aux modèles appris et vous donnera son verdict instantanément.
                """)
            
            if len(df) < 5:
                st.warning("Pas assez de données pour l'apprentissage. Ajoutez des planètes ou repasser sur la base NASA.")
            else:
                # Opérations : Séparation Test/Entraînement et calcul d'Accuracy
                X = df[['Masse_Terre', 'Rayon_Terre', 'Temp_Etoile_K']]
                y = df['Habitable']
                
                # Gestion d'erreur si la classe est unique (ex: toutes les planètes créées sont habitables)
                if len(y.unique()) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model_sup = RandomForestClassifier(random_state=42).fit(X_train, y_train)
                    accuracy = model_sup.score(X_test, y_test) * 100
                    
                    st.metric("Taux de réussite du Modèle (Précision sur le jeu de test)", f"{accuracy:.1f} %")
                    
                    st.subheader("Simulateur de Prédiction")
                    c1, c2, c3 = st.columns(3)
                    m = c1.slider("Masse", 0.1, 10.0, 1.0, key="slider_mass_ml")
                    r = c2.slider("Rayon", 0.1, 5.0, 1.0, key="slider_radius_ml")
                    t = c3.number_input("Température (K)", 2000, 10000, 5700, key="input_temp_ml")

                    st.markdown(
                        f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">'
                        f'<img src="{svg_rocket_b64}" width="30" height="30" style="display:block;" alt="Lancer l\'analyse"/>',
                        unsafe_allow_html=True
                    )
                    if st.button("LANCER L'ANALYSE", use_container_width=True):
                        # Calcul du pourcentage de certitude (Probabilité)
                        proba = model_sup.predict_proba([[m, r, t]])[0]
                        if model_sup.predict([[m, r, t]])[0] == 1:
                            st.success(f"PLANÈTE HABITABLE ! (L'IA est certaine à {proba[1]*100:.1f} %)")
                            st.balloons()
                        else:
                            st.error(f"PLANÈTE HOSTILE. (L'IA est certaine à {proba[0]*100:.1f} %)")
                else:
                    st.warning("L'échantillon actuel ne contient qu'un seul type de planète (soit 100% habitables, soit 100% hostiles). L'IA ne peut pas comparer et apprendre. Ajoutez des planètes du type opposé !")

            display_scroll_to_top()

        # ONGLET 5 : CLASSIFICATION NON-SUPERVISÉE
        with tabs[4]:
            st.markdown(f'<img src="{LOGOS["ai_unsup"]}" class="icon-filter" width="60">', unsafe_allow_html=True)
            st.markdown("""<div class="desc-box">
                <b>Classification Non-Supervisée (Clustering K-Means) :</b> Contrairement au modèle précédent, l'algorithme ne connait pas l'étiquette (Habitable ou non). 
                Son rôle est de regrouper "à l'aveugle" les planètes en Familles (Clusters) basées sur leurs similitudes mathématiques (distances).
            </div>""", unsafe_allow_html=True)
            
            if len(df) < 5:
                st.warning("Pas assez de données pour segmenter les planètes.")
            else:
                nb_clusters = st.slider("Choisissez le nombre de Familles (K) :", 2, 5, 3, key="slider_kmeans")
                
                # Opérations de clustering
                X_unsup = df[['Masse_Terre', 'Rayon_Terre']]
                kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
                df['Cluster'] = kmeans.fit_predict(X_unsup)
                
                # Calcul des pourcentages de répartition par cluster
                repartition = df['Cluster'].value_counts(normalize=True) * 100
                
                st.write("**Répartition de la population par Famille :**")
                cols = st.columns(nb_clusters)
                for i in range(nb_clusters):
                    cols[i].metric(f"Famille {i}", f"{repartition[i]:.1f} %")
                    
                fig_kmeans = px.scatter(df, x="Rayon_Terre", y="Masse_Terre", color=df['Cluster'].astype(str), 
                                 title=f"Segmentation K-Means en {nb_clusters} familles",
                                 color_discrete_sequence=px.colors.qualitative.Set2, template="plotly_dark")
                st.plotly_chart(apply_transparent_style(fig_kmeans), use_container_width=True)
                st.warning("""
                    **Analyse des Familles :** Les "clusters" créés ici par l'IA ne tiennent pas compte du label 'Habitable'. 
                    Ils regroupent les planètes par ressemblance physique pure. Par exemple, une famille peut regrouper 
                    les 'Super-Terres', tandis qu'une autre regroupe les 'Mini-Neptunes' gazeuses.
                """)
                
            display_scroll_to_top()

else:
    st.warning("Veuillez générer les données avec data_builder.py d'abord.")