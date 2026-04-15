import ee
import json
import streamlit as st


# =========================================================
# 1. Parametres generaux
# =========================================================
PROJECT_ID = "habitat-du-macaque-de-barbarie"
DEFAULT_REGION_ASSET = "projects/habitat-du-macaque-de-barbarie/assets/fes_meknes"

DEFAULT_REGION_BBOX = {
    "lat_min": 33.1,
    "lon_min": -6.9,
    "lat_max": 34.9,
    "lon_max": -3.8,
}


# =========================================================
# 2. Initialisation Earth Engine
# =========================================================
def initialize_earth_engine():
    """
    Initialise Earth Engine :
    - en ligne via st.secrets
    - en local via ee.Initialize(project=...)
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


initialize_earth_engine()


# =========================================================
# 3. Construction de la geometrie de travail
# =========================================================
def get_default_region_geometry():
    """Charge la region par defaut depuis l'asset Earth Engine."""
    region_fc = ee.FeatureCollection(DEFAULT_REGION_ASSET)
    return region_fc.geometry()


def get_default_region_bbox():
    """Retourne une bbox approximative de Fes-Meknes pour la preview/export."""
    return DEFAULT_REGION_BBOX


def get_bbox_geometry(lat_min, lon_min, lat_max, lon_max):
    """Cree une bbox a partir de coordonnees."""
    return ee.Geometry.BBox(lon_min, lat_min, lon_max, lat_max)


def get_polygon_geometry_from_geojson(polygon_geojson):
    """
    Transforme un GeoJSON dessine sur la carte en geometrie Earth Engine.
    Accepte :
    - un Feature GeoJSON
    - une Geometry GeoJSON
    """
    if polygon_geojson is None:
        raise ValueError("Aucun polygone n'a ete fourni.")

    if polygon_geojson.get("type") == "Feature":
        geometry = polygon_geojson.get("geometry")
        if geometry is None:
            raise ValueError("Le Feature GeoJSON ne contient pas de geometrie.")
        return ee.Geometry(geometry)

    if polygon_geojson.get("type") in ["Polygon", "MultiPolygon"]:
        return ee.Geometry(polygon_geojson)

    raise ValueError("Format GeoJSON non supporte pour le polygone.")


def get_region_geometry(region_mode, bbox_values=None, polygon_geojson=None):
    """
    Retourne la geometrie selon le mode choisi :
    - region_defaut
    - bbox_personnalisee
    - polygone_dessine
    """
    if region_mode == "region_defaut":
        return get_default_region_geometry()

    if region_mode == "bbox_personnalisee":
        if bbox_values is None:
            raise ValueError("Les coordonnees de la bbox personnalisee sont manquantes.")

        return get_bbox_geometry(
            bbox_values["lat_min"],
            bbox_values["lon_min"],
            bbox_values["lat_max"],
            bbox_values["lon_max"]
        )

    if region_mode == "polygone_dessine":
        return get_polygon_geometry_from_geojson(polygon_geojson)

    raise ValueError("Mode de region inconnu.")


def get_export_bbox_geometry(region_mode, bbox_values=None, polygon_geojson=None):
    """
    Retourne une geometrie simple pour l'export local.
    On exporte sur une bbox simple, tandis que l'image reste clippee a la zone.
    Cela reduit fortement la taille de la requete.
    """
    if region_mode == "region_defaut":
        bbox = get_default_region_bbox()
        return get_bbox_geometry(
            bbox["lat_min"], bbox["lon_min"], bbox["lat_max"], bbox["lon_max"]
        )

    if region_mode == "bbox_personnalisee":
        return get_bbox_geometry(
            bbox_values["lat_min"],
            bbox_values["lon_min"],
            bbox_values["lat_max"],
            bbox_values["lon_max"]
        )

    if region_mode == "polygone_dessine":
        geom = get_polygon_geometry_from_geojson(polygon_geojson)
        return geom.bounds(1)

    raise ValueError("Mode de region inconnu pour l'export.")


def get_simplified_clip_geometry(region_mode, bbox_values=None, polygon_geojson=None):
    """
    Retourne une geometrie de clipping simplifiee pour rendre l'export plus leger.
    """
    geom = get_region_geometry(region_mode, bbox_values, polygon_geojson)

    if region_mode == "region_defaut":
        return geom.simplify(500)

    if region_mode == "polygone_dessine":
        return geom.simplify(200)

    return geom


# =========================================================
# 4. Masquage des nuages Sentinel-2
# =========================================================
def mask_sentinel2_clouds(image):
    """Retire les pixels nuageux avec QA60."""
    quality_band = image.select("QA60")

    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    mask = (
        quality_band.bitwiseAnd(cloud_bit).eq(0)
        .And(quality_band.bitwiseAnd(cirrus_bit).eq(0))
    )

    return image.updateMask(mask).copyProperties(image, image.propertyNames())


# =========================================================
# 5. Collection Sentinel-2
# =========================================================
def get_sentinel2_collection(region_geom, start_date, end_date, max_cloud_percentage=20):
    """Retourne une collection Sentinel-2 filtree."""
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region_geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percentage))
        .map(mask_sentinel2_clouds)
    )


# =========================================================
# 6. Composite median
# =========================================================
def get_median_composite(region_geom, start_date, end_date, max_cloud_percentage=20):
    """Cree un composite median pour la periode choisie."""
    collection = get_sentinel2_collection(region_geom, start_date, end_date, max_cloud_percentage)
    return collection.median().clip(region_geom)


# =========================================================
# 7. Calcul du NDMI
# =========================================================
def calculate_ndmi(region_geom, start_date, end_date, max_cloud_percentage=20):
    """
    NDMI = (B8 - B11) / (B8 + B11)
    """
    image = get_median_composite(region_geom, start_date, end_date, max_cloud_percentage)
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return ndmi.clip(region_geom)


# =========================================================
# 8. Statistiques
# =========================================================
def calculate_image_statistics(image, region_geom, band_name, scale=100):
    """Calcule moyenne, ecart-type, min et max."""
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
# 9. Export GeoTIFF local allege
# =========================================================
def build_lightweight_geotiff_download_url(
    image,
    region_mode,
    bbox_values=None,
    polygon_geojson=None,
    filename="export_geotiff",
    scale=60
):
    """
    Tente un export GeoTIFF direct allege vers le PC.

    Strategie :
    - l'image est clippee a une geometrie simplifiee
    - la region d'export est une bbox simple
    - resolution plus grossiere par defaut
    """
    clip_geom = get_simplified_clip_geometry(
        region_mode=region_mode,
        bbox_values=bbox_values,
        polygon_geojson=polygon_geojson
    )

    export_region = get_export_bbox_geometry(
        region_mode=region_mode,
        bbox_values=bbox_values,
        polygon_geojson=polygon_geojson
    )

    export_image = (
        image
        .clip(clip_geom)
        .toFloat()
        .unmask(-9999)
    )

    url = export_image.getDownloadURL({
        "name": filename,
        "region": export_region,
        "scale": scale,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF"
    })

    return url


# =========================================================
# 10. Analyse complete
# =========================================================
def analyze_moisture_change(
    region_mode,
    start_date_1,
    end_date_1,
    start_date_2,
    end_date_2,
    cloud_pct=20,
    threshold=0.10,
    bbox_values=None,
    polygon_geojson=None,
    stats_scale=100
):
    """
    Fonction principale :
    - choisit la zone
    - calcule NDMI sur 2 periodes
    - calcule la difference
    - detecte gains / pertes / changement
    - calcule les stats
    """
    region_geom = get_region_geometry(
        region_mode=region_mode,
        bbox_values=bbox_values,
        polygon_geojson=polygon_geojson
    )

    moisture_1 = calculate_ndmi(region_geom, start_date_1, end_date_1, cloud_pct).rename("NDMI_1")
    moisture_2 = calculate_ndmi(region_geom, start_date_2, end_date_2, cloud_pct).rename("NDMI_2")

    moisture_diff = moisture_2.subtract(moisture_1).rename("NDMI_Diff")
    absolute_difference = moisture_diff.abs().rename("NDMI_Abs_Diff")

    gain_mask = moisture_diff.gt(threshold).rename("Gain")
    loss_mask = moisture_diff.lt(-threshold).rename("Loss")
    change_mask = absolute_difference.gt(threshold).rename("Significant_Change")

    stats_period_1 = calculate_image_statistics(moisture_1, region_geom, "NDMI_1", scale=stats_scale)
    stats_period_2 = calculate_image_statistics(moisture_2, region_geom, "NDMI_2", scale=stats_scale)
    stats_difference = calculate_image_statistics(moisture_diff, region_geom, "NDMI_Diff", scale=stats_scale)

    change_stats_raw = change_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=stats_scale,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    gain_stats_raw = gain_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=stats_scale,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    loss_stats_raw = loss_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=stats_scale,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

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

    return {
        "region_geom": region_geom,
        "moisture_1": moisture_1,
        "moisture_2": moisture_2,
        "moisture_diff": moisture_diff,
        "gain_mask": gain_mask,
        "loss_mask": loss_mask,
        "change_mask": change_mask,
        "stats": stats_bundle,
    }


# =========================================================
# 11. Palettes de visualisation
# =========================================================
MOISTURE_VIS = {
    "min": -0.40,
    "max": 0.60,
    "palette": [
        "#8c510a",
        "#d8b365",
        "#f6e8c3",
        "#c7eae5",
        "#5ab4ac",
        "#01665e"
    ]
}

DIFF_VIS = {
    "min": -0.30,
    "max": 0.30,
    "palette": [
        "#b2182b",
        "#ef8a62",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#67a9cf",
        "#2166ac"
    ]
}

GAIN_VIS = {
    "min": 0,
    "max": 1,
    "palette": ["#deebf7", "#9ecae1", "#3182bd", "#08519c"]
}

LOSS_VIS = {
    "min": 0,
    "max": 1,
    "palette": ["#fee5d9", "#fcae91", "#fb6a4a", "#cb181d"]
}

BINARY_CHANGE_VIS = {
    "min": 0,
    "max": 1,
    "palette": ["#ffffff", "#6a3d9a"]
}