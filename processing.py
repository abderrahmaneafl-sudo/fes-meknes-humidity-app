import ee


# =========================================================
# 1) Initialisation Google Earth Engine
# =========================================================
PROJECT_ID = "habitat-du-macaque-de-barbarie"
REGION_ASSET = "projects/habitat-du-macaque-de-barbarie/assets/fes_meknes"

ee.Initialize(project=PROJECT_ID)


# =========================================================
# 2) Zone d'etude : Fes-Meknes
# =========================================================
region_fc = ee.FeatureCollection(REGION_ASSET)
region_geom = region_fc.geometry()


# =========================================================
# 3) Masquage simple des nuages Sentinel-2
# =========================================================
def mask_s2_clouds(image):
    qa = image.select("QA60")

    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloud_bit_mask).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    )

    return image.updateMask(mask).copyProperties(image, image.propertyNames())


# =========================================================
# 4) Collection Sentinel-2
# =========================================================
def get_sentinel_collection(start_date, end_date, cloud_pct=20):
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(region_geom)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        .map(mask_s2_clouds)
    )


# =========================================================
# 5) Composite median
# =========================================================
def get_median_image(start_date, end_date, cloud_pct=20):
    collection = get_sentinel_collection(start_date, end_date, cloud_pct)
    image = collection.median()
    return image.clip(region_geom)


# =========================================================
# 6) Indice d'humidite (NDMI)
#    NDMI = (B8 - B11) / (B8 + B11)
# =========================================================
def get_moisture_index_image(start_date, end_date, cloud_pct=20):
    image = get_median_image(start_date, end_date, cloud_pct)
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return ndmi.clip(region_geom)


# =========================================================
# 7) Statistiques d'une image
# =========================================================
def get_image_stats(image, band_name, scale=10):
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
# 8) Analyse complete de changement d'humidite
# =========================================================
def analyze_moisture_change(
    start_date_1,
    end_date_1,
    start_date_2,
    end_date_2,
    cloud_pct=20,
    threshold=0.10
):
    moisture_1 = get_moisture_index_image(start_date_1, end_date_1, cloud_pct).rename("NDMI_1")
    moisture_2 = get_moisture_index_image(start_date_2, end_date_2, cloud_pct).rename("NDMI_2")

    moisture_diff = moisture_2.subtract(moisture_1).rename("NDMI_Diff")
    abs_diff = moisture_diff.abs().rename("NDMI_Abs_Diff")

    gain_mask = moisture_diff.gt(threshold).rename("Gain")
    loss_mask = moisture_diff.lt(-threshold).rename("Loss")
    change_mask = abs_diff.gt(threshold).rename("Significant_Change")

    moisture_1_stats = get_image_stats(moisture_1, "NDMI_1")
    moisture_2_stats = get_image_stats(moisture_2, "NDMI_2")
    diff_stats = get_image_stats(moisture_diff, "NDMI_Diff")

    change_stats_raw = change_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=10,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    gain_stats_raw = gain_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=10,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    loss_stats_raw = loss_mask.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.sum(), "", True),
        geometry=region_geom,
        scale=10,
        maxPixels=1e13,
        bestEffort=True
    ).getInfo()

    stats_bundle = {
        "period_1": moisture_1_stats,
        "period_2": moisture_2_stats,
        "difference": diff_stats,
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
        "moisture_1": moisture_1,
        "moisture_2": moisture_2,
        "moisture_diff": moisture_diff,
        "gain_mask": gain_mask,
        "loss_mask": loss_mask,
        "change_mask": change_mask,
        "stats": stats_bundle,
    }


# =========================================================
# 9) Palettes de couleurs plus lisibles
# =========================================================
MOISTURE_VIS = {
    "min": -0.40,
    "max": 0.60,
    "palette": [
        "#8c510a",  # brun
        "#d8b365",
        "#f6e8c3",
        "#c7eae5",
        "#5ab4ac",
        "#01665e"   # bleu-vert doux
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