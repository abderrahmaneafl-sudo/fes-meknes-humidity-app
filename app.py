import streamlit as st
import geemap.foliumap as geemap
import folium
from folium.plugins import SideBySideLayers
import pandas as pd
from datetime import date
import plotly.graph_objects as go

from processing import (
    analyze_moisture_change,
    region_geom,
    MOISTURE_VIS,
    DIFF_VIS,
    GAIN_VIS,
    LOSS_VIS,
    BINARY_CHANGE_VIS
)

# ---------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------
st.set_page_config(
    page_title="Detection de changement d'humidite",
    page_icon="🛰️",
    layout="wide"
)

# ---------------------------------------------------------
# Styles
# ---------------------------------------------------------
st.markdown("""
<style>
.block-title {
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}
.info-box {
    background-color: #f7f9fc;
    border: 1px solid #dfe6ee;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.section-box {
    background-color: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 18px;
}
.small-muted {
    color: #5b6573;
    font-size: 0.95rem;
}
.stat-card {
    background: #fafbfc;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 14px;
    height: 100%;
}
.stat-title {
    font-size: 0.95rem;
    color: #4b5563;
    margin-bottom: 8px;
    font-weight: 600;
}
.stat-value {
    font-size: 1.7rem;
    font-weight: 700;
    color: #111827;
}
.stat-sub {
    font-size: 0.88rem;
    color: #6b7280;
    margin-top: 6px;
}
.good {
    color: #0f766e;
    font-weight: 700;
}
.bad {
    color: #b91c1c;
    font-weight: 700;
}
.neutral {
    color: #374151;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------
MONTHS_FR = {
    1: "janvier", 2: "fevrier", 3: "mars", 4: "avril",
    5: "mai", 6: "juin", 7: "juillet", 8: "aout",
    9: "septembre", 10: "octobre", 11: "novembre", 12: "decembre"
}


def fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def short_period_label(start_d: date, end_d: date) -> str:
    if start_d.month == end_d.month and start_d.year == end_d.year:
        return f"{MONTHS_FR[start_d.month]} {start_d.year}"
    return f"{fmt_date(start_d)} -> {fmt_date(end_d)}"


def build_dynamic_title(start_1: date, end_1: date, start_2: date, end_2: date) -> str:
    return f"Comparaison humidite : {short_period_label(start_1, end_1)} vs {short_period_label(start_2, end_2)}"


def automatic_interpretation(stats: dict) -> str:
    p1 = stats["period_1"]["mean"]
    p2 = stats["period_2"]["mean"]
    diff = stats["difference"]["mean"]
    change_prop = stats["change"]["proportion"]
    gain_prop = stats["gain"]["proportion"]
    loss_prop = stats["loss"]["proportion"]

    tendance = "augmenté" if diff > 0 else "diminué" if diff < 0 else "peu varié"
    dominance = (
        "les gains d'humidite dominent" if gain_prop > loss_prop
        else "les pertes d'humidite dominent" if loss_prop > gain_prop
        else "les gains et pertes sont proches"
    )

    return (
        f"L'humidite moyenne est passée de {p1:.3f} a {p2:.3f}. "
        f"La variation moyenne a donc {tendance} ({diff:.3f}). "
        f"Environ {change_prop*100:.2f}% de la zone presente un changement significatif. "
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


def build_map(layer_mode: str, result: dict, map_title: str):
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


def build_split_map(result: dict, left_label: str, right_label: str):
    m = geemap.Map()
    m.centerObject(region_geom, 8)

    left_layer = geemap.ee_tile_layer(
        result["moisture_1"],
        MOISTURE_VIS,
        left_label
    )

    right_layer = geemap.ee_tile_layer(
        result["moisture_2"],
        MOISTURE_VIS,
        right_label
    )

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


def get_delta_text(v1: float, v2: float) -> str:
    delta = v2 - v1
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def get_variation_label(value: float) -> str:
    if value > 0:
        return "<span class='good'>Augmentation</span>"
    elif value < 0:
        return "<span class='bad'>Diminution</span>"
    return "<span class='neutral'>Stable</span>"


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


# ---------------------------------------------------------
# Interface
# ---------------------------------------------------------
st.title("🛰️ Detection automatique de changement d'humidite - Region Fes-Meknes")
st.markdown(
    "<div class='small-muted'>Comparaison de deux periodes a l'aide d'un indice d'humidite (NDMI).</div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.header("Parametres d'analyse")

st.sidebar.subheader("Periode 1")
start_date_1 = st.sidebar.date_input("Date debut 1", value=date(2022, 1, 1), key="d1")
end_date_1 = st.sidebar.date_input("Date fin 1", value=date(2022, 1, 31), key="f1")

st.sidebar.subheader("Periode 2")
start_date_2 = st.sidebar.date_input("Date debut 2", value=date(2024, 1, 1), key="d2")
end_date_2 = st.sidebar.date_input("Date fin 2", value=date(2024, 1, 31), key="f2")

cloud_pct = st.sidebar.slider("Seuil maximal de nuages (%)", 0, 100, 20)

threshold = st.sidebar.slider(
    "Seuil de changement d'humidite",
    min_value=0.01,
    max_value=0.50,
    value=0.10,
    step=0.01
)

layer_mode = st.sidebar.selectbox(
    "Affichage de la carte thematique",
    [
        "Voir toutes les couches",
        "Voir seulement periode 1",
        "Voir seulement periode 2",
        "Voir seulement difference",
        "Voir seulement gains",
        "Voir seulement pertes",
    ]
)

run_analysis = st.sidebar.button("Lancer l'analyse", use_container_width=True)

# ---------------------------------------------------------
# Explication
# ---------------------------------------------------------
with st.expander("Comprendre le resultat"):
    st.write("""
    Cette application utilise un indice d'humidite de type **NDMI**.

    Elle permet de suivre :
    - l'humidite du sol
    - l'humidite de la vegetation
    - les petites eaux de surface
    - les variations globales d'humidite

    Le seuil de nuages sert a filtrer les images trop nuageuses.
    Le seuil de changement d'humidite sert a garder seulement les variations importantes.
    """)

if start_date_1 > end_date_1 or start_date_2 > end_date_2:
    st.error("La date de debut doit etre anterieure ou egale a la date de fin.")
    st.stop()

if not run_analysis:
    st.info("Configure les parametres dans la barre laterale, puis clique sur **Lancer l'analyse**.")
    st.stop()

# ---------------------------------------------------------
# Analyse
# ---------------------------------------------------------
try:
    with st.spinner("Calcul en cours..."):
        result = analyze_moisture_change(
            fmt_date(start_date_1),
            fmt_date(end_date_1),
            fmt_date(start_date_2),
            fmt_date(end_date_2),
            cloud_pct,
            threshold
        )

    stats = result["stats"]
    map_title = build_dynamic_title(start_date_1, end_date_1, start_date_2, end_date_2)
    interpretation_text = automatic_interpretation(stats)
    stats_df = build_stats_dataframe(stats, threshold, cloud_pct)

    p1_mean = stats["period_1"]["mean"]
    p2_mean = stats["period_2"]["mean"]
    diff_mean = stats["difference"]["mean"]

    p1_std = stats["period_1"]["stdDev"]
    p2_std = stats["period_2"]["stdDev"]
    diff_std = stats["difference"]["stdDev"]

    p1_min = stats["period_1"]["min"]
    p1_max = stats["period_1"]["max"]
    p2_min = stats["period_2"]["min"]
    p2_max = stats["period_2"]["max"]

    gain_prop = stats["gain"]["proportion"] * 100
    loss_prop = stats["loss"]["proportion"] * 100
    change_prop = stats["change"]["proportion"] * 100

    gain_px = stats["gain"]["pixels"]
    loss_px = stats["loss"]["pixels"]
    change_px = stats["change"]["pixels"]

    st.success("Analyse terminee avec succes.")

    # -----------------------------------------------------
    # Bloc 1 : Resume statistique enrichi
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 1 - Resume statistique detaille</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div class='info-box'>
            <b>Periode 1</b><br>
            {fmt_date(start_date_1)} → {fmt_date(end_date_1)}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class='info-box'>
            <b>Periode 2</b><br>
            {fmt_date(start_date_2)} → {fmt_date(end_date_2)}
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-title'>Humidite moyenne - Periode 1</div>
            <div class='stat-value'>{p1_mean:.3f}</div>
            <div class='stat-sub'>StdDev : {p1_std:.3f} | Min : {p1_min:.3f} | Max : {p1_max:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-title'>Humidite moyenne - Periode 2</div>
            <div class='stat-value'>{p2_mean:.3f}</div>
            <div class='stat-sub'>StdDev : {p2_std:.3f} | Min : {p2_min:.3f} | Max : {p2_max:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-title'>Difference moyenne</div>
            <div class='stat-value'>{diff_mean:.3f}</div>
            <div class='stat-sub'>{get_variation_label(diff_mean)} | StdDev : {diff_std:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-title'>Changement significatif</div>
            <div class='stat-value'>{change_prop:.2f}%</div>
            <div class='stat-sub'>Pixels : {change_px:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-title'>Gain d'humidite</div>
            <div class='stat-value'>{gain_prop:.2f}%</div>
            <div class='stat-sub'>Pixels : {gain_px:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c6:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-title'>Perte d'humidite</div>
            <div class='stat-value'>{loss_prop:.2f}%</div>
            <div class='stat-sub'>Pixels : {loss_px:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Lecture rapide")

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Variation P2 - P1", f"{diff_mean:.3f}", delta=get_delta_text(p1_mean, p2_mean))
    l2.metric("Seuil humidite", f"{threshold:.2f}")
    l3.metric("Nuages max", f"{cloud_pct}%")
    l4.metric("Dominante", "Gain" if gain_prop > loss_prop else "Perte" if loss_prop > gain_prop else "Equilibre")

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 2 : Graphiques
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 2 - Graphiques d'analyse</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)

    with g1:
        fig_means = build_mean_chart(p1_mean, p2_mean, diff_mean)
        st.plotly_chart(fig_means, use_container_width=True)

    with g2:
        fig_props = build_proportion_chart(gain_prop, loss_prop, change_prop)
        st.plotly_chart(fig_props, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 3 : Split panel
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 3 - Split panel Periode 1 / Periode 2</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='small-muted'>Utilise le curseur vertical pour comparer directement les deux periodes.</div>",
        unsafe_allow_html=True
    )

    split_map = build_split_map(
        result,
        left_label=f"Periode 1 - {short_period_label(start_date_1, end_date_1)}",
        right_label=f"Periode 2 - {short_period_label(start_date_2, end_date_2)}"
    )
    split_map.to_streamlit(height=820)

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 4 : Carte thematique
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 4 - Carte thematique</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='small-muted'><b>Mode actuel :</b> {layer_mode}</div>",
        unsafe_allow_html=True
    )

    map_obj = build_map(layer_mode, result, map_title)
    map_obj.to_streamlit(height=820)

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 5 : Interpretation automatique
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 5 - Interpretation automatique</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='info-box'>
        {interpretation_text}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 6 : Tableau statistique detaille
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 6 - Tableau statistique detaille</div>", unsafe_allow_html=True)
    st.dataframe(stats_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 7 : Guide rapide
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 7 - Guide de lecture</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Couches disponibles**
        - Humidite - Periode 1
        - Humidite - Periode 2
        - Difference humidite
        - Gains
        - Pertes
        - Changement detecte
        """)
    with c2:
        st.markdown("""
        **Lecture**
        - Palette humidite : brun -> jaune -> bleu-vert
        - Difference : rouge -> blanc -> bleu
        - Gains : bleu
        - Pertes : rouge
        - Changement detecte : violet
        """)

    st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # Bloc 8 : Export
    # -----------------------------------------------------
    st.markdown("<div class='section-box'>", unsafe_allow_html=True)
    st.markdown("<div class='block-title'>Bloc 8 - Export</div>", unsafe_allow_html=True)

    csv_bytes = stats_df.to_csv(index=False).encode("utf-8")
    summary_text = f"""Detection automatique de changement d'humidite - Region Fes-Meknes

Periode 1 : {fmt_date(start_date_1)} -> {fmt_date(end_date_1)}
Periode 2 : {fmt_date(start_date_2)} -> {fmt_date(end_date_2)}

Humidite moyenne periode 1 : {p1_mean:.3f}
Humidite moyenne periode 2 : {p2_mean:.3f}
Difference moyenne : {diff_mean:.3f}

StdDev P1 : {p1_std:.3f}
StdDev P2 : {p2_std:.3f}
StdDev difference : {diff_std:.3f}

Min/Max P1 : {p1_min:.3f} / {p1_max:.3f}
Min/Max P2 : {p2_min:.3f} / {p2_max:.3f}

Proportion de changement : {change_prop:.2f}%
Proportion de gain : {gain_prop:.2f}%
Proportion de perte : {loss_prop:.2f}%

Pixels changement : {change_px:.0f}
Pixels gain : {gain_px:.0f}
Pixels perte : {loss_px:.0f}

Seuil d'humidite : {threshold:.2f}
Seuil maximal de nuages : {cloud_pct}%

Interpretation automatique :
{interpretation_text}
"""

    html_map = map_obj.get_root().render().encode("utf-8")
    html_split = split_map.get_root().render().encode("utf-8")

    e1, e2, e3, e4 = st.columns(4)

    with e1:
        st.download_button(
            "Exporter les statistiques (CSV)",
            data=csv_bytes,
            file_name="statistiques_humidite.csv",
            mime="text/csv",
            use_container_width=True
        )

    with e2:
        st.download_button(
            "Exporter le resume (TXT)",
            data=summary_text.encode("utf-8"),
            file_name="resume_humidite.txt",
            mime="text/plain",
            use_container_width=True
        )

    with e3:
        st.download_button(
            "Exporter la carte thematique (HTML)",
            data=html_map,
            file_name="carte_humidite.html",
            mime="text/html",
            use_container_width=True
        )

    with e4:
        st.download_button(
            "Exporter le split panel (HTML)",
            data=html_split,
            file_name="split_panel_humidite.html",
            mime="text/html",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Erreur pendant l'analyse : {e}")