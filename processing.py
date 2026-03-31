import ee
import json
import streamlit as st


# =========================================================
# 1. Parametres generaux du projet
# =========================================================

# Identifiant du projet Google Cloud / Earth Engine
PROJECT_ID = "habitat-du-macaque-de-barbarie"

# Chemin de l'asset contenant la region de Fes-Meknes
REGION_ASSET = "projects/habitat-du-macaque-de-barbarie/assets/fes_meknes"


# =========================================================
# 2. Initialisation de Google Earth Engine
# =========================================================

def initialize_earth_engine():
    """
    Initialise Earth Engine.

    Deux cas sont possibles :
    1. En ligne (Streamlit Cloud) :
       on lit les credentials depuis st.secrets
    2. En local :
       on utilise simplement ee.Initialize(project=...)
    """
    service_account_json = st.secrets.get("GEE_SERVICE_ACCOUNT_JSON", None)

    if service_account_json:
        try:
            credentials_data = json.loads(service_account_json)

            credentials = ee.ServiceAccountCredentials(
                email=credentials_data["client_email"],
                key_data=service_account_json
            )

            ee.Initialize(credentials=credentials, project=PROJECT_ID)

        except Exception as e:
            raise RuntimeError(
                f"Echec de l'initialisation Earth Engine via st.secrets : {e}"
            )
    else:
        ee.Initialize(project=PROJECT_ID)


# On initialise Earth Engine une seule fois au chargement du fichier
initialize_earth_engine()


# =========================================================
# 3. Chargement de la zone d'etude
# =========================================================

# La region est stockee comme FeatureCollection dans Earth Engine
region_feature_collection = ee.FeatureCollection(REGION_ASSET)

# On extrait la geometrie de cette region pour l'utiliser
# dans les filtres et les calculs
region_geom = region_feature_collection.geometry()


# =========================================================
# 4. Fonction de masquage des nuages
# =========================================================

def mask_sentinel2_clouds(image):
    """
    Supprime les pixels nuageux d'une image Sentinel-2
    en utilisant la bande QA60.

    bit 10 = nuages
    bit 11 = cirrus
    """
    quality_band = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        quality_band.bitwiseAnd(cloud_bit).eq(0)
        .And(quality_band.bitwiseAnd(cirrus_bit).eq(0))
    )

    return image.updateMask(mask).copyProperties(image, image.propertyNames())


# =========================================================
# 5. Recuperation de la collection Sentinel-2
# =========================================================

def get_sentinel2_collection(start_date, end_date, max_cloud_percentage=20):
    """
    Retourne une collection Sentinel-2 filtree :
    - par date
    - par zone d'etude
    - par pourcentage maximal de nuages
    - avec masquage des nuages
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region_geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percentage))
        .map(mask_sentinel2_clouds)
    )

    return collection


# =========================================================
# 6. Construction d'une image composite mediane
# =========================================================

def get_median_composite(start_date, end_date, max_cloud_percentage=20):
    """
    Cree un composite median a partir de la collection Sentinel-2.
    La mediane permet de reduire le bruit et les valeurs extremes.
    """
    collection = get_sentinel2_collection(start_date, end_date, max_cloud_percentage)
    median_image = collection.median()

    return median_image.clip(region_geom)


# =========================================================
# 7. Calcul de l'indice d'humidite NDMI
# =========================================================

def calculate_ndmi(start_date, end_date, max_cloud_percentage=20):
    """
    Calcule l'indice d'humidite NDMI :
    NDMI = (B8 - B11) / (B8 + B11)

    B8  = proche infrarouge (NIR)
    B11 = infrarouge a ondes courtes (SWIR)
    """
    image = get_median_composite(start_date, end_date, max_cloud_percentage)

    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

    return ndmi.clip(region_geom)


# =========================================================
# 8. Calcul de statistiques sur une image
# =========================================================

def calculate_image_statistics(image, band_name, scale=100):
    """
    Calcule des statistiques simples sur une image :
    - moyenne
    - ecart-type
    - minimum
    - maximum

    Ici, scale=100 est utilise pour alleger le calcul
    sur Streamlit Cloud.
    """
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean()
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.minMax(), "", True),
        geometry=region_geom,
        scale=scale,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    return {
        "mean": float(stats.get(f"{band_name}_mean", 0) or 0),
        "stdDev": float(stats.get(f"{band_name}_stdDev", 0) or 0),
        "min": float(stats.get(f"{band_name}_min", 0) or 0),
        "max": float(stats.get(f"{band_name}_max", 0) or 0),
    }


# =========================================================
# 9. Analyse complete du changement d'humidite
# =========================================================

def analyze_moisture_change(
    start_date_1,
    end_date_1,
    start_date_2,
    end_date_2,
    cloud_pct=20,
    threshold=0.10
):
    """
    Fonction principale du traitement.

    Elle :
    1. calcule le NDMI pour la periode 1
    2. calcule le NDMI pour la periode 2
    3. calcule la difference entre les deux
    4. detecte les gains, pertes et changements significatifs
    5. calcule les statistiques associees
    """

    # NDMI pour chaque periode
    moisture_1 = calculate_ndmi(start_date_1, end_date_1, cloud_pct).rename("NDMI_1")
    moisture_2 = calculate_ndmi(start_date_2, end_date_2, cloud_pct).rename("NDMI_2")

    # Difference entre les deux periodes
    moisture_diff = moisture_2.subtract(moisture_1).rename("NDMI_Diff")

    # Valeur absolue de la difference
    absolute_difference = moisture_diff.abs().rename("NDMI_Abs_Diff")

    # Masque de gain d'humidite
    gain_mask = moisture_diff.gt(threshold).rename("Gain")

    # Masque de perte d'humidite
    loss_mask = moisture_diff.lt(-threshold).rename("Loss")

    # Masque de changement significatif (gain ou perte)
    change_mask = absolute_difference.gt(threshold).rename("Significant_Change")

    # Statistiques des trois images principales
    stats_period_1 = calculate_image_statistics(moisture_1, "NDMI_1", scale=100)
    stats_period_2 = calculate_image_statistics(moisture_2, "NDMI_2", scale=100)
    stats_difference = calculate_image_statistics(moisture_diff, "NDMI_Diff", scale=100)

    # Statistiques des masques binaires
    change_stats_raw = change_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=100,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    gain_stats_raw = gain_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=100,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    loss_stats_raw = loss_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=100,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    # Regroupement final des statistiques
    stats_bundle = {
        "period_1": stats_period_1,
        "period_2": stats_period_2,
        "difference": stats_difference,
        "change": {
            "proportion": float(change_stats_raw.get("Significant_Change_mean", 0) or 0),
            "pixels": float(change_stats_raw.get("Significant_Change_sum", 0) or 0),
        },
        "gain": {
            "proportion": float(gain_stats_raw.get("Gain_mean", 0) or 0),
            "pixels": float(gain_stats_raw.get("Gain_sum", 0) or 0),
        },
        "loss": {
            "proportion": float(loss_stats_raw.get("Loss_mean", 0) or 0),
            "pixels": float(loss_stats_raw.get("Loss_sum", 0) or 0),
        },
    }

    # Resultat final retourne a app.py
    return {
        "moisture_1": moisture_1,
        "moisture_2": moisture_2,
        "moisture_diff": moisture_diff,
        "gain_mask": gain_mask,
        "loss_mask": loss_mask,
        "change_mask": change_mask,
        "stats": stats_bundle,
    }


# =========================================================
# 10. Parametres de visualisation des cartes
# =========================================================

MOISTURE_VIS = {
    "min": -0.40,
    "max": 0.60,
    "palette": [
        "#8c510a",  # brun fonce = faible humidite
        "#d8b365",  # brun clair / sable
        "#f6e8c3",  # beige clair
        "#c7eae5",  # bleu tres clair
        "#5ab4ac",  # bleu-vert moyen
        "#01665e"   # bleu-vert fonce = forte humidite
    ]
}

DIFF_VIS = {
    "min": -0.30,
    "max": 0.30,
    "palette": [
        "#b2182b",  # rouge fonce = perte forte
        "#ef8a62",  # rouge clair
        "#fddbc7",  # rose clair
        "#f7f7f7",  # blanc = stable
        "#d1e5f0",  # bleu tres clair
        "#67a9cf",  # bleu moyen
        "#2166ac"   # bleu fonce = gain fort
    ]
}

GAIN_VIS = {
    "min": 0,
    "max": 1,
    "palette": [
        "#deebf7",  # bleu tres clair
        "#9ecae1",  # bleu clair
        "#3182bd",  # bleu moyen
        "#08519c"   # bleu fonce
    ]
}

LOSS_VIS = {
    "min": 0,
    "max": 1,
    "palette": [
        "#fee5d9",  # rouge tres clair
        "#fcae91",  # saumon clair
        "#fb6a4a",  # rouge-orange
        "#cb181d"   # rouge fonce
    ]
}

BINARY_CHANGE_VIS = {
    "min": 0,
    "max": 1,
    "palette": [
        "#ffffff",  # blanc
        "#6a3d9a"   # violet
    ]
}