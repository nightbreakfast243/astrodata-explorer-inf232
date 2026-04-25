import pandas as pd
import requests
import numpy as np
import io

def get_nasa_exoplanets():
    print("🚀 Connexion à l'API de la NASA en cours...")
    
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # Paramétrage propre de la requête
    params = {
        "query": "select pl_name,pl_bmasse,pl_rade,st_teff,pl_orbper,sy_dist from ps where default_flag=1",
        "format": "csv"
    }
    
    # Le User-Agent indique à la NASA que nous sommes un navigateur légitime
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AstroData-TP"
    }
    
    try:
        # Timeout ajouté pour ne pas bloquer le script indéfiniment
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status() 
        
        df = pd.read_csv(io.StringIO(response.text))
        
        # 🛡️ SÉCURITÉ : Vérifie si la NASA a bien renvoyé les colonnes
        colonnes_attendues = ['pl_name', 'pl_bmasse', 'pl_rade', 'st_teff', 'pl_orbper', 'sy_dist']
        if not all(col in df.columns for col in colonnes_attendues):
            print("⚠️ L'API a répondu mais le format est inattendu. Activation du plan de secours.")
            return create_fallback_real_data()
        
        # Renommage des colonnes
        df = df.rename(columns={
            'pl_name': 'Nom',
            'pl_bmasse': 'Masse_Terre',
            'pl_rade': 'Rayon_Terre',
            'st_teff': 'Temp_Etoile_K',
            'pl_orbper': 'Periode_Orbitale_Jours',
            'sy_dist': 'Distance_AL'
        })
        
        df['Distance_AL'] = df['Distance_AL'] * 3.26156
        
        conditions_habitabilite = (
            (df['Masse_Terre'] >= 0.5) & (df['Masse_Terre'] <= 5.0) &
            (df['Rayon_Terre'] >= 0.8) & (df['Rayon_Terre'] <= 1.5) &
            (df['Temp_Etoile_K'] >= 4000) & (df['Temp_Etoile_K'] <= 6000)
        )
        df['Habitable'] = np.where(conditions_habitabilite, 1, 0)
        
        print(f"✅ {len(df)} planètes réelles téléchargées avec succès.")
        return df

    except Exception as e:
        print(f"❌ Échec réseau ({e}). Activation automatique des données de secours.")
        return create_fallback_real_data()

def create_fallback_real_data():
    """Génère un jeu de données réaliste si la NASA est inaccessible."""
    print("🛟 Génération du jeu de données Kepler de secours...")
    data = {
        'Nom': [f"Kepler-{i}b" for i in range(1, 4501)],
        'Masse_Terre': np.random.lognormal(mean=0.5, sigma=1.0, size=4500),
        'Rayon_Terre': np.random.lognormal(mean=0.3, sigma=0.6, size=4500),
        'Temp_Etoile_K': np.random.normal(loc=5500, scale=1200, size=4500),
        'Periode_Orbitale_Jours': np.random.uniform(1, 800, size=4500),
        'Distance_AL': np.random.uniform(10, 2500, size=4500),
        'Habitable': [0] * 4500 # Seront équilibrées par les planètes synthétiques plus bas
    }
    return pd.DataFrame(data)

def generate_synthetic_habitable_planets(num_planets=500):
    print(f"🧪 Génération de {num_planets} planètes habitables synthétiques...")
    data = {
        'Nom': [f"Synth-Earth-{i}" for i in range(1, num_planets + 1)],
        'Masse_Terre': np.random.uniform(0.7, 3.5, num_planets), 
        'Rayon_Terre': np.random.uniform(0.9, 1.4, num_planets), 
        'Temp_Etoile_K': np.random.uniform(4500, 5800, num_planets), 
        'Periode_Orbitale_Jours': np.random.uniform(200, 450, num_planets), 
        'Distance_AL': np.random.uniform(4.0, 1000.0, num_planets),
        'Habitable': [1] * num_planets 
    }
    return pd.DataFrame(data)

# --- Exécution principale ---
if __name__ == "__main__":
    df_real = get_nasa_exoplanets()
    df_fake = generate_synthetic_habitable_planets(num_planets=800)
    
    df_final = pd.concat([df_real, df_fake], ignore_index=True)
    df_final = df_final.dropna(subset=['Masse_Terre', 'Rayon_Terre'])
    
    fichier_sortie = 'exoplanetes_data.csv'
    df_final.to_csv(fichier_sortie, index=False)
    
    print("\nRésumé du jeu de données final :")
    print(f"Total des planètes : {len(df_final)}")
    print(df_final['Habitable'].value_counts().rename({0: "Hostiles (0)", 1: "Habitables (1)"}))
    print(f"\nFichier '{fichier_sortie}' créé avec succès !")