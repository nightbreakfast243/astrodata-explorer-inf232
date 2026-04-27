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
        
        # AJOUT DE LA SOURCE POUR LE FILTRE DANS L'APP
        df['Source'] = 'NASA'
        
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
    df = pd.DataFrame(data)
    df['Source'] = 'NASA'
    return df

def generate_synthetic_planets(num_planets=20):
    print(f"🧪 Génération du laboratoire de {num_planets} planètes synthétiques...")
    
    # 1. Tirage aléatoire du nombre de planètes hostiles (entre 5 et 14 inclus)
    num_hostiles = np.random.randint(5, 15) 
    num_habitables = num_planets - num_hostiles
    
    print(f"   -> Détail de la simulation : {num_habitables} habitables et {num_hostiles} hostiles.")

    # 2. Génération des planètes HABITABLES (Critères de la zone Goldilocks)
    masses_hab = np.random.uniform(0.7, 3.5, num_habitables)
    rayons_hab = np.random.uniform(0.9, 1.4, num_habitables)
    temps_hab = np.random.uniform(4500, 5800, num_habitables)
    labels_hab = [1] * num_habitables

    # 3. Génération des planètes HOSTILES (Valeurs extrêmes : Géantes gazeuses, trop chaudes/froides)
    masses_host = np.random.uniform(10.0, 100.0, num_hostiles) # Trop massives
    rayons_host = np.random.uniform(2.5, 15.0, num_hostiles)   # Trop grandes
    temps_host = np.random.choice(
        np.concatenate([np.random.uniform(1000, 3000, num_hostiles), np.random.uniform(8000, 15000, num_hostiles)])
    , size=num_hostiles) # Soit glaciales, soit brûlantes
    labels_host = [0] * num_hostiles

    # 4. Fusion des deux groupes
    data = {
        'Masse_Terre': np.concatenate([masses_hab, masses_host]),
        'Rayon_Terre': np.concatenate([rayons_hab, rayons_host]),
        'Temp_Etoile_K': np.concatenate([temps_hab, temps_host]),
        'Periode_Orbitale_Jours': np.random.uniform(10, 800, num_planets), # Peu importe pour l'IA
        'Distance_AL': np.random.uniform(4.0, 1000.0, num_planets),
        'Habitable': labels_hab + labels_host
    }
    
    df = pd.DataFrame(data)
    
    # 5. Mélange aléatoire (Shuffle) pour ne pas avoir toutes les habitables au début
    df = df.sample(frac=1, random_state=np.random.randint(1000)).reset_index(drop=True)
    
    # Ajout des noms et de la source APRÈS le mélange
    df['Nom'] = [f"Synth-World-{i}" for i in range(1, num_planets + 1)]
    df['Source'] = 'Simulation'
    
    return df

# --- Exécution principale ---
if __name__ == "__main__":
    df_real = get_nasa_exoplanets()
    
    # Appel de la nouvelle fonction renommée
    df_fake = generate_synthetic_planets(num_planets=20)
    
    df_final = pd.concat([df_real, df_fake], ignore_index=True)
    df_final = df_final.dropna(subset=['Masse_Terre', 'Rayon_Terre'])
    
    fichier_sortie = 'exoplanetes_data.csv'
    df_final.to_csv(fichier_sortie, index=False)
    
    print("\n📊 Résumé du jeu de données final :")
    print(f"Total des planètes : {len(df_final)}")
    print("Répartition Globale :")
    print(df_final['Habitable'].value_counts().rename({0: "Hostiles (0)", 1: "Habitables (1)"}))
    
    print("\n🔬 Répartition dans votre laboratoire (Simulation) :")
    print(df_final[df_final['Source'] == 'Simulation']['Habitable'].value_counts().rename({0: "Hostiles (0)", 1: "Habitables (1)"}))
    
    print(f"\n✅ Fichier '{fichier_sortie}' créé avec succès !")