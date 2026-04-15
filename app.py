import streamlit as st
import geemap.foliumap as geemap
import folium
from folium.plugins import SideBySideLayers, Geocoder, Draw
from streamlit_folium import st_folium
import pandas as pd
from datetime import date
import plotly.graph_objects as go

from processing import (
    analyze_moisture_change,
    build_lightweight_geotiff_download_url,
    get_default_region_bbox,
    get_default_region_geometry,
    MOISTURE_VIS,
    DIFF_VIS,
    GAIN_VIS,
    LOSS_VIS,
    BINARY_CHANGE_VIS
)

# =========================================================
# 1. Configuration generale
# =========================================================
st.set_page_config(
    page_title="Plateforme de detection de changement d'humidite",
    page_icon="🛰️",
    layout="wide"
)

# =========================================================
# 2. Etat de session
# =========================================================
if "drawn_polygon_geojson" not in st.session_state:
    st.session_state.drawn_polygon_geojson = None

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "analysis_params" not in st.session_state:
    st.session_state.analysis_params = None

if "export_url_p1" not in st.session_state:
    st.session_state.export_url_p1 = None

if "export_url_p2" not in st.session_state:
    st.session_state.export_url_p2 = None

if "export_url_diff" not in st.session_state:
    st.session_state.export_url_diff = None

if "export_error_p1" not in st.session_state:
    st.session_state.export_error_p1 = None

if "export_error_p2" not in st.session_state:
    st.session_state.export_error_p2 = None

if "export_error_diff" not in st.session_state:
    st.session_state.export_error_diff = None


# =========================================================
# 3. Fonctions utilitaires
# =========================================================
MONTHS_FR = {
    1: "janvier", 2: "fevrier", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "aout",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "decembre"
}


def format_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def short_period_label(start_d: date, end_d: date) -> str:
    if start_d.month == end_d.month and start_d.year == end_d.year:
        return f"{MONTHS_FR[start_d.month]} {start_d.year}"
    return f"{format_date(start_d)} -> {format_date(end_d)}"


def build_dynamic_title(start_1: date, end_1: date, start_2: date, end_2: date) -> str:
    return f"Comparaison humidite : {short_period_label(start_1, end_1)} vs {short_period_label(start_2, end_2)}"


def automatic_interpretation(stats: dict) -> str:
    p1 = stats["period_1"]["mean"]
    p2 = stats["period_2"]["mean"]
    diff = stats["difference"]["mean"]
    change_prop = stats["change"]["proportion"]
    gain_prop = stats["gain"]["proportion"]
    loss_prop = stats["loss"]["proportion"]

    if diff > 0:
        tendance = "augmenté"
    elif diff < 0:
        tendance = "diminué"
    else:
        tendance = "peu varié"

    if gain_prop > loss_prop:
        dominance = "les gains d'humidite dominent"
    elif loss_prop > gain_prop:
        dominance = "les pertes d'humidite dominent"
    else:
        dominance = "les gains et les pertes sont proches"

    return (
        f"L'humidite moyenne est passée de {p1:.3f} a {p2:.3f}. "
        f"La variation moyenne a donc {tendance} ({diff:.3f}). "
        f"Environ {change_prop * 100:.2f}% de la zone presente un changement significatif. "
        f"Globalement, {dominance}."
    )


def build_stats_dataframe(stats: dict, threshold: float, cloud_pct: int) -> pd.DataFrame:
    rows = [
        ["Humidite periode 1", stats["period_1"]["mean"], stats["period_1"]["stdDev"], stats["period_1"]["min"], stats["period_1"]["max"]],
        ["Humidite periode 2", stats["period_2"]["mean"], stats["period_2"]["stdDev"], stats["period_2"]["min"], stats["period_2"]["max"]],
        ["Difference humidite", stats["difference"]["mean"], stats["difference"]["stdDev"], stats["difference"]["min"], stats["difference"]["max"]],
        ["Proportion changement", stats["change"]["proportion"], None, None, None],
        ["Proportion gain", stats["gain"]["proportion"], None, None, None],
        ["Proportion perte", stats["loss"]["proportion"], None, None, None],
        ["Pixels changement", stats["change"]["pixels"], None, None, None],
        ["Pixels gain", stats["gain"]["pixels"], None, None, None],
        ["Pixels perte", stats["loss"]["pixels"], None, None, None],
        ["Seuil humidite", threshold, None, None, None],
        ["Nuages max (%)", cloud_pct, None, None, None],
    ]
    return pd.DataFrame(rows, columns=["Indicateur", "Valeur", "StdDev", "Min", "Max"])


def add_compact_legend():
    return """
    <div style="
        position: fixed;
        bottom: 22px;
        left: 22px;
        z-index: 9999;
        background-color: rgba(255,255,255,0.96);
        border: 1px solid #bdbdbd;
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 12px;
        line-height: 1.35;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.15);
        max-width: 220px;
    ">
        <b>Legende</b><br><br>

        <b>Humidite</b><br>
        <div><span style="display:inline-block;width:14px;height:10px;background:#8c510a;margin-right:6px;"></span>Faible</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#f6e8c3;margin-right:6px;"></span>Moyenne</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#01665e;margin-right:6px;"></span>Forte</div>
        <br>

        <b>Difference</b><br>
        <div><span style="display:inline-block;width:14px;height:10px;background:#b2182b;margin-right:6px;"></span>Perte</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#f7f7f7;border:1px solid #888;margin-right:6px;"></span>Stable</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#2166ac;margin-right:6px;"></span>Gain</div>
        <br>

        <b>Masques</b><br>
        <div><span style="display:inline-block;width:14px;height:10px;background:#08519c;margin-right:6px;"></span>Gain detecte</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#cb181d;margin-right:6px;"></span>Perte detectee</div>
        <div><span style="display:inline-block;width:14px;height:10px;background:#6a3d9a;margin-right:6px;"></span>Changement significatif</div>
    </div>
    """


def add_preview_geometry_to_map(m, region_mode, bbox_values):
    if region_mode == "region_defaut":
        bbox = get_default_region_bbox()
        lat_min = bbox["lat_min"]
        lon_min = bbox["lon_min"]
        lat_max = bbox["lat_max"]
        lon_max = bbox["lon_max"]

        m.fit_bounds([
            [lat_min, lon_min],
            [lat_max, lon_max]
        ])

        try:
            default_geom = get_default_region_geometry()
            default_geojson = default_geom.getInfo()

            folium.GeoJson(
                default_geojson,
                name="Region Fes-Meknes",
                style_function=lambda x: {
                    "color": "red",
                    "weight": 2,
                    "fillColor": "red",
                    "fillOpacity": 0.08
                },
                tooltip="Region par defaut : Fes-Meknes"
            ).add_to(m)
        except Exception:
            rectangle = folium.Rectangle(
                bounds=[
                    [lat_min, lon_min],
                    [lat_max, lon_max],
                ],
                color="red",
                weight=2,
                fill=True,
                fill_opacity=0.08,
                tooltip="Region par defaut : Fes-Meknes"
            )
            rectangle.add_to(m)

    elif region_mode == "bbox_personnalisee":
        lat_min = bbox_values["lat_min"]
        lon_min = bbox_values["lon_min"]
        lat_max = bbox_values["lat_max"]
        lon_max = bbox_values["lon_max"]

        m.fit_bounds([
            [lat_min, lon_min],
            [lat_max, lon_max]
        ])

        rectangle = folium.Rectangle(
            bounds=[
                [lat_min, lon_min],
                [lat_max, lon_max],
            ],
            color="red",
            weight=2,
            fill=True,
            fill_opacity=0.08,
            tooltip="BBox personnalisee"
        )
        rectangle.add_to(m)

    elif region_mode == "polygone_dessine":
        if st.session_state.drawn_polygon_geojson is not None:
            folium.GeoJson(
                st.session_state.drawn_polygon_geojson,
                name="Polygone dessine",
                style_function=lambda x: {
                    "color": "red",
                    "weight": 2,
                    "fillColor": "red",
                    "fillOpacity": 0.08
                }
            ).add_to(m)

            try:
                geometry = st.session_state.drawn_polygon_geojson.get("geometry", {})
                coords = []

                if geometry.get("type") == "Polygon":
                    coords = geometry["coordinates"][0]
                elif geometry.get("type") == "MultiPolygon":
                    coords = geometry["coordinates"][0][0]

                if coords:
                    lons = [pt[0] for pt in coords]
                    lats = [pt[1] for pt in coords]

                    m.fit_bounds([
                        [min(lats), min(lons)],
                        [max(lats), max(lons)]
                    ])
            except Exception:
                m.location = [33.8, -5.0]
                m.zoom_start = 7
        else:
            m.location = [33.8, -5.0]
            m.zoom_start = 7


def build_preview_map(region_mode, bbox_values):
    m = folium.Map(location=[33.8, -5.0], zoom_start=7, control_scale=True)

    add_preview_geometry_to_map(m, region_mode, bbox_values)

    Geocoder(collapsed=False, position="topleft").add_to(m)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polygon": True
        },
        edit_options={"edit": True, "remove": True}
    ).add_to(m)

    folium.LayerControl().add_to(m)

    return m


def build_thematic_map(region_geom, layer_mode: str, result: dict, map_title: str):
    m = geemap.Map()
    m.centerObject(region_geom, 8)

    moisture_1 = result["moisture_1"]
    moisture_2 = result["moisture_2"]
    moisture_diff = result["moisture_diff"]
    gain_mask = result["gain_mask"]
    loss_mask = result["loss_mask"]
    change_mask = result["change_mask"]

    if layer_mode == "Voir toutes les couches":
        m.addLayer(moisture_1, MOISTURE_VIS, "Humidite - Periode 1")
        m.addLayer(moisture_2, MOISTURE_VIS, "Humidite - Periode 2")
        m.addLayer(moisture_diff, DIFF_VIS, "Difference humidite")
        m.addLayer(gain_mask.selfMask(), GAIN_VIS, "Gains")
        m.addLayer(loss_mask.selfMask(), LOSS_VIS, "Pertes")
        m.addLayer(change_mask.selfMask(), BINARY_CHANGE_VIS, "Changement detecte")
    elif layer_mode == "Voir seulement periode 1":
        m.addLayer(moisture_1, MOISTURE_VIS, "Humidite - Periode 1")
    elif layer_mode == "Voir seulement periode 2":
        m.addLayer(moisture_2, MOISTURE_VIS, "Humidite - Periode 2")
    elif layer_mode == "Voir seulement difference":
        m.addLayer(moisture_diff, DIFF_VIS, "Difference humidite")
    elif layer_mode == "Voir seulement gains":
        m.addLayer(gain_mask.selfMask(), GAIN_VIS, "Gains")
    elif layer_mode == "Voir seulement pertes":
        m.addLayer(loss_mask.selfMask(), LOSS_VIS, "Pertes")

    m.addLayerControl()

    title_html = f"""
    <div style="
        position: fixed;
        top: 18px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: rgba(255,255,255,0.96);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #cfcfcf;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.12);
    ">
        {map_title}
    </div>
    """

    m.get_root().html.add_child(folium.Element(title_html))
    m.get_root().html.add_child(folium.Element(add_compact_legend()))

    return m


def build_split_map(region_geom, result: dict, left_label: str, right_label: str):
    m = geemap.Map()
    m.centerObject(region_geom, 8)

    left_layer = geemap.ee_tile_layer(result["moisture_1"], MOISTURE_VIS, left_label)
    right_layer = geemap.ee_tile_layer(result["moisture_2"], MOISTURE_VIS, right_label)

    left_layer.add_to(m)
    right_layer.add_to(m)

    SideBySideLayers(left_layer, right_layer).add_to(m)

    split_title = """
    <div style="
        position: fixed;
        top: 18px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        background-color: rgba(255,255,255,0.96);
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid #cfcfcf;
        font-size: 14px;
        font-weight: 700;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.12);
    ">
        Split panel - Comparaison Periode 1 / Periode 2
    </div>
    """

    side_labels = f"""
    <div style="
        position: fixed;
        top: 60px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 9999;
        display: flex;
        gap: 20px;
        font-size: 12px;
        font-weight: 700;
        color: #1f2937;
        background: rgba(255,255,255,0.92);
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 6px 10px;
    ">
        <span>Gauche : {left_label}</span>
        <span>Droite : {right_label}</span>
    </div>
    """

    m.get_root().html.add_child(folium.Element(split_title))
    m.get_root().html.add_child(folium.Element(side_labels))
    m.get_root().html.add_child(folium.Element(add_compact_legend()))

    return m


def build_mean_chart(p1_mean: float, p2_mean: float, diff_mean: float):
    fig = go.Figure()
    fig.add_bar(
        x=["Periode 1", "Periode 2", "Difference"],
        y=[p1_mean, p2_mean, diff_mean],
        text=[f"{p1_mean:.3f}", f"{p2_mean:.3f}", f"{diff_mean:.3f}"],
        textposition="outside"
    )
    fig.update_layout(
        title="Comparaison des humidites moyennes",
        xaxis_title="Indicateurs",
        yaxis_title="Valeur moyenne",
        template="plotly_white",
        height=420
    )
    return fig


def build_proportion_chart(gain_prop: float, loss_prop: float, change_prop: float):
    fig = go.Figure()
    fig.add_bar(
        x=["Gain", "Perte", "Changement total"],
        y=[gain_prop, loss_prop, change_prop],
        text=[f"{gain_prop:.2f}%", f"{loss_prop:.2f}%", f"{change_prop:.2f}%"],
        textposition="outside"
    )
    fig.update_layout(
        title="Proportions de gain, perte et changement",
        xaxis_title="Categorie",
        yaxis_title="Proportion (%)",
        template="plotly_white",
        height=420
    )
    return fig


# =========================================================
# 4. En-tete
# =========================================================
st.title("🛰️ Plateforme de detection de changement d'humidite")
st.caption("Analyse spatio-temporelle de l'humidite par imagerie Sentinel-2 et Google Earth Engine")

# =========================================================
# 5. Barre laterale
# =========================================================
st.sidebar.header("Configuration de l'analyse")

st.sidebar.subheader("Zone d'etude")
region_mode_label = st.sidebar.radio(
    "Choisir la zone",
    [
        "Region par defaut : Fes-Meknes",
        "BBox personnalisee (coordonnees)",
        "Polygone dessine sur la carte"
    ]
)

bbox_values = None
region_mode = None

if region_mode_label == "Region par defaut : Fes-Meknes":
    region_mode = "region_defaut"

elif region_mode_label == "BBox personnalisee (coordonnees)":
    region_mode = "bbox_personnalisee"

    lat_min = st.sidebar.number_input("Latitude minimale", value=33.0, format="%.6f")
    lon_min = st.sidebar.number_input("Longitude minimale", value=-6.0, format="%.6f")
    lat_max = st.sidebar.number_input("Latitude maximale", value=34.5, format="%.6f")
    lon_max = st.sidebar.number_input("Longitude maximale", value=-4.0, format="%.6f")

    bbox_values = {
        "lat_min": lat_min,
        "lon_min": lon_min,
        "lat_max": lat_max,
        "lon_max": lon_max,
    }

else:
    region_mode = "polygone_dessine"

st.sidebar.subheader("Periode 1")
start_date_1 = st.sidebar.date_input("Date debut 1", value=date(2022, 1, 1), key="d1")
end_date_1 = st.sidebar.date_input("Date fin 1", value=date(2022, 1, 31), key="f1")

st.sidebar.subheader("Periode 2")
start_date_2 = st.sidebar.date_input("Date debut 2", value=date(2024, 1, 1), key="d2")
end_date_2 = st.sidebar.date_input("Date fin 2", value=date(2024, 1, 31), key="f2")

st.sidebar.subheader("Parametres")
cloud_pct = st.sidebar.slider("Seuil maximal de nuages (%)", 0, 100, 20)
threshold = st.sidebar.slider("Seuil de changement d'humidite", 0.01, 0.50, 0.10, 0.01)
stats_scale = st.sidebar.slider("Resolution des statistiques (m)", 50, 500, 100, 10)
export_scale = st.sidebar.slider("Resolution export TIF local (m)", 30, 200, 60, 10)

st.sidebar.subheader("Affichage")
layer_mode = st.sidebar.selectbox(
    "Carte thematique",
    [
        "Voir toutes les couches",
        "Voir seulement periode 1",
        "Voir seulement periode 2",
        "Voir seulement difference",
        "Voir seulement gains",
        "Voir seulement pertes",
    ]
)

if st.sidebar.button("Lancer l'analyse", use_container_width=True):
    st.session_state.analysis_done = True
    st.session_state.export_url_p1 = None
    st.session_state.export_url_p2 = None
    st.session_state.export_url_diff = None
    st.session_state.export_error_p1 = None
    st.session_state.export_error_p2 = None
    st.session_state.export_error_diff = None

# =========================================================
# 6. Explication
# =========================================================
with st.expander("A propos de cette application"):
    st.write("""
    Cette application permet de comparer deux periodes temporelles a partir de l'indice NDMI.

    Elle permet aussi de choisir la zone d'etude de trois manieres :
    - region par defaut
    - bbox personnalisee via coordonnees
    - polygone dessine directement sur la carte

    L'export GeoTIFF local est volontairement allege :
    - geometrie simplifiee
    - export sur bbox
    - resolution plus grossiere
    """)

# =========================================================
# 7. Verifications
# =========================================================
if start_date_1 > end_date_1 or start_date_2 > end_date_2:
    st.error("La date de debut doit etre anterieure ou egale a la date de fin.")
    st.stop()

if region_mode == "bbox_personnalisee":
    if lat_min >= lat_max or lon_min >= lon_max:
        st.error("Les bornes de la bbox sont invalides.")
        st.stop()

# =========================================================
# 8. Previsualisation de la zone
# =========================================================
st.subheader("Previsualisation de la zone d'etude")

if region_mode == "region_defaut":
    st.caption("Zone active : region par defaut Fes-Meknes")
elif region_mode == "bbox_personnalisee":
    st.caption(
        f"Zone active : bbox | "
        f"lat_min={bbox_values['lat_min']}, lon_min={bbox_values['lon_min']}, "
        f"lat_max={bbox_values['lat_max']}, lon_max={bbox_values['lon_max']}"
    )
else:
    st.caption("Zone active : polygone dessine sur la carte")

preview_map = build_preview_map(region_mode, bbox_values)

preview_output = st_folium(
    preview_map,
    width=None,
    height=430,
    returned_objects=["last_active_drawing", "all_drawings"]
)

# Boutons juste sous la carte
preview_controls = st.container()
with preview_controls:
    if region_mode == "polygone_dessine":
        col_save, col_clear = st.columns(2)

        with col_save:
            if st.button("Enregistrer le polygone dessine", use_container_width=True):
                drawn = preview_output.get("last_active_drawing", None)

                if drawn is not None:
                    geometry = drawn.get("geometry", {})
                    geometry_type = geometry.get("type", None)

                    if geometry_type in ["Polygon", "MultiPolygon"]:
                        st.session_state.drawn_polygon_geojson = drawn
                        st.success("Polygone enregistre avec succes.")
                    else:
                        st.warning("Merci de dessiner un polygone valide.")
                else:
                    st.warning("Aucun polygone detecte sur la carte.")

        with col_clear:
            if st.button("Effacer le polygone enregistre", use_container_width=True):
                st.session_state.drawn_polygon_geojson = None
                st.info("Le polygone enregistre a ete supprime.")

        if st.session_state.drawn_polygon_geojson is None:
            st.info("Dessine un polygone sur la carte puis clique sur 'Enregistrer le polygone dessine'.")

if not st.session_state.analysis_done:
    st.info("Tu peux d'abord choisir la zone, puis cliquer sur 'Lancer l'analyse'.")
    st.stop()

if region_mode == "polygone_dessine" and st.session_state.drawn_polygon_geojson is None:
    st.error("Aucun polygone n'est enregistre. Dessine puis enregistre un polygone avant de lancer l'analyse.")
    st.stop()

# =========================================================
# 9. Analyse principale
# =========================================================
current_params = {
    "region_mode": region_mode,
    "bbox_values": bbox_values,
    "polygon_geojson": st.session_state.drawn_polygon_geojson,
    "start_date_1": format_date(start_date_1),
    "end_date_1": format_date(end_date_1),
    "start_date_2": format_date(start_date_2),
    "end_date_2": format_date(end_date_2),
    "cloud_pct": cloud_pct,
    "threshold": threshold,
    "stats_scale": stats_scale
}

needs_recompute = st.session_state.analysis_params != current_params

if needs_recompute:
    try:
        with st.spinner("Analyse en cours..."):
            result = analyze_moisture_change(
                region_mode=region_mode,
                start_date_1=format_date(start_date_1),
                end_date_1=format_date(end_date_1),
                start_date_2=format_date(start_date_2),
                end_date_2=format_date(end_date_2),
                cloud_pct=cloud_pct,
                threshold=threshold,
                bbox_values=bbox_values,
                polygon_geojson=st.session_state.drawn_polygon_geojson,
                stats_scale=stats_scale
            )

        st.session_state.analysis_result = result
        st.session_state.analysis_params = current_params

    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
        st.stop()

result = st.session_state.analysis_result
stats = result["stats"]
region_geom = result["region_geom"]

map_title = build_dynamic_title(start_date_1, end_date_1, start_date_2, end_date_2)
interpretation_text = automatic_interpretation(stats)
stats_df = build_stats_dataframe(stats, threshold, cloud_pct)

p1_mean = stats["period_1"]["mean"]
p2_mean = stats["period_2"]["mean"]
diff_mean = stats["difference"]["mean"]

gain_prop = stats["gain"]["proportion"] * 100
loss_prop = stats["loss"]["proportion"] * 100
change_prop = stats["change"]["proportion"] * 100

st.success("Analyse terminee avec succes.")

st.subheader("1. Resume statistique")
a, b = st.columns(2)
a.info(f"**Periode 1**\n\n{format_date(start_date_1)} → {format_date(end_date_1)}")
b.info(f"**Periode 2**\n\n{format_date(start_date_2)} → {format_date(end_date_2)}")

c1, c2, c3 = st.columns(3)
c1.metric("Humidite moyenne P1", f"{p1_mean:.3f}")
c2.metric("Humidite moyenne P2", f"{p2_mean:.3f}")
c3.metric("Difference moyenne", f"{diff_mean:.3f}")

c4, c5, c6 = st.columns(3)
c4.metric("Changement significatif", f"{change_prop:.2f}%")
c5.metric("Gain d'humidite", f"{gain_prop:.2f}%")
c6.metric("Perte d'humidite", f"{loss_prop:.2f}%")

st.subheader("2. Graphiques d'analyse")
g1, g2 = st.columns(2)

with g1:
    fig_means = build_mean_chart(p1_mean, p2_mean, diff_mean)
    st.plotly_chart(fig_means, use_container_width=True)

with g2:
    fig_props = build_proportion_chart(gain_prop, loss_prop, change_prop)
    st.plotly_chart(fig_props, use_container_width=True)

st.subheader("3. Split panel")
split_map = build_split_map(
    region_geom,
    result,
    left_label=f"Periode 1 - {short_period_label(start_date_1, end_date_1)}",
    right_label=f"Periode 2 - {short_period_label(start_date_2, end_date_2)}"
)
split_map.to_streamlit(height=760)

st.subheader("4. Carte thematique")
st.caption(f"Mode d'affichage : {layer_mode}")
thematic_map = build_thematic_map(region_geom, layer_mode, result, map_title)
thematic_map.to_streamlit(height=760)

st.subheader("5. Interpretation automatique")
st.info(interpretation_text)

st.subheader("6. Tableau statistique detaille")
st.dataframe(stats_df, use_container_width=True)

st.subheader("7. Export standard")
csv_bytes = stats_df.to_csv(index=False).encode("utf-8")

summary_text = f"""Detection automatique de changement d'humidite

Periode 1 : {format_date(start_date_1)} -> {format_date(end_date_1)}
Periode 2 : {format_date(start_date_2)} -> {format_date(end_date_2)}

Humidite moyenne periode 1 : {p1_mean:.3f}
Humidite moyenne periode 2 : {p2_mean:.3f}
Difference moyenne : {diff_mean:.3f}

Proportion de changement : {change_prop:.2f}%
Proportion de gain : {gain_prop:.2f}%
Proportion de perte : {loss_prop:.2f}%

Interpretation :
{interpretation_text}
"""

html_map = thematic_map.get_root().render().encode("utf-8")
html_split = split_map.get_root().render().encode("utf-8")

e1, e2, e3, e4 = st.columns(4)

with e1:
    st.download_button(
        "Exporter statistiques CSV",
        data=csv_bytes,
        file_name="statistiques_humidite.csv",
        mime="text/csv",
        use_container_width=True
    )

with e2:
    st.download_button(
        "Exporter resume TXT",
        data=summary_text.encode("utf-8"),
        file_name="resume_humidite.txt",
        mime="text/plain",
        use_container_width=True
    )

with e3:
    st.download_button(
        "Exporter carte HTML",
        data=html_map,
        file_name="carte_humidite.html",
        mime="text/html",
        use_container_width=True
    )

with e4:
    st.download_button(
        "Exporter split HTML",
        data=html_split,
        file_name="split_panel_humidite.html",
        mime="text/html",
        use_container_width=True
    )

st.subheader("8. Export GeoTIFF local allege")
st.caption("En cas d'echec, augmente la resolution d'export locale : 80 m, 100 m, 120 m...")

t1, t2, t3 = st.columns(3)

with t1:
    if st.button("Preparer TIF Periode 1", use_container_width=True):
        try:
            st.session_state.export_url_p1 = build_lightweight_geotiff_download_url(
                image=result["moisture_1"],
                region_mode=region_mode,
                bbox_values=bbox_values,
                polygon_geojson=st.session_state.drawn_polygon_geojson,
                filename="humidite_periode_1",
                scale=export_scale
            )
            st.session_state.export_error_p1 = None
        except Exception as e:
            st.session_state.export_url_p1 = None
            st.session_state.export_error_p1 = str(e)

    if st.session_state.export_url_p1:
        st.markdown(f"[Télécharger le TIF période 1]({st.session_state.export_url_p1})")

    if st.session_state.export_error_p1:
        st.error(st.session_state.export_error_p1)

with t2:
    if st.button("Preparer TIF Periode 2", use_container_width=True):
        try:
            st.session_state.export_url_p2 = build_lightweight_geotiff_download_url(
                image=result["moisture_2"],
                region_mode=region_mode,
                bbox_values=bbox_values,
                polygon_geojson=st.session_state.drawn_polygon_geojson,
                filename="humidite_periode_2",
                scale=export_scale
            )
            st.session_state.export_error_p2 = None
        except Exception as e:
            st.session_state.export_url_p2 = None
            st.session_state.export_error_p2 = str(e)

    if st.session_state.export_url_p2:
        st.markdown(f"[Télécharger le TIF période 2]({st.session_state.export_url_p2})")

    if st.session_state.export_error_p2:
        st.error(st.session_state.export_error_p2)

with t3:
    if st.button("Preparer TIF Difference", use_container_width=True):
        try:
            st.session_state.export_url_diff = build_lightweight_geotiff_download_url(
                image=result["moisture_diff"],
                region_mode=region_mode,
                bbox_values=bbox_values,
                polygon_geojson=st.session_state.drawn_polygon_geojson,
                filename="difference_humidite",
                scale=export_scale
            )
            st.session_state.export_error_diff = None
        except Exception as e:
            st.session_state.export_url_diff = None
            st.session_state.export_error_diff = str(e)

    if st.session_state.export_url_diff:
        st.markdown(f"[Télécharger le TIF différence]({st.session_state.export_url_diff})")

    if st.session_state.export_error_diff:
        st.error(st.session_state.export_error_diff)