# -*- coding: utf-8 -*-
"""
Sentinelle DDoS — détection et prédiction d'attaques DDoS avec Streamlit.

Pipeline (identique à l'atelier « ddos_yassmine ») :
  1. Chargement du CSV           4. Entraînement de 4 classifieurs
  2. Prétraitement               5. Métriques (Accuracy, Précision, Recall, F1)
  3. Découpage 70 / 15 / 15      6. Prédiction affichée sur une jauge de vitesse

Lancer l'application :  streamlit run app.py
"""

import base64
import math
import os
import time
from string import Template

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ----------------------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE
# ----------------------------------------------------------------------------

# Identifiants PROVISOIRES (à remplacer plus tard par une vraie authentification)
IDENTIFIANT = "admin"
MOT_DE_PASSE = "admin"

# Le dataset est cherché à côté de app.py
CHEMIN_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

# Colonnes numériques décrivant le trafic
COLONNES_NUM = [
    "duration_ms", "packet_count", "byte_count", "packets_per_second",
    "bytes_per_second", "avg_packet_size_bytes", "syn_count", "ack_count",
    "connections_10s", "unique_sources_10s",
]
# Colonnes écartées (identifiants/horodatage : risque de « raccourcis » pour le modèle)
COLONNES_EXCLUES = ["timestamp", "src_ip", "dst_ip", "src_port"]
# Colonnes indispensables dans le CSV
COLONNES_REQUISES = COLONNES_NUM + ["protocol", "dst_port", "label"]

# Libellés français pour la saisie manuelle : (colonne, libellé, type)
PARAMETRES_FLUX = [
    ("duration_ms", "Durée du flux (ms)", "float"),
    ("packet_count", "Nombre de paquets", "int"),
    ("byte_count", "Nombre d'octets", "int"),
    ("packets_per_second", "Paquets par seconde", "float"),
    ("bytes_per_second", "Octets par seconde", "float"),
    ("avg_packet_size_bytes", "Taille moyenne d'un paquet (octets)", "float"),
    ("syn_count", "Paquets SYN", "int"),
    ("ack_count", "Paquets ACK", "int"),
    ("connections_10s", "Connexions sur 10 s", "int"),
    ("unique_sources_10s", "Sources uniques sur 10 s", "int"),
]

MODELES_DISPONIBLES = ["Régression logistique", "SVM", "Random Forest", "XGBoost"]
PAGES = ["Jeu de données", "Prétraitement", "Entraînement", "Métriques", "Prédiction"]
NOMS_CLASSES = {0: "Normal", 1: "DDoS"}

# Palette beige
BG = "#F3ECE0"          # fond de page
CARTE = "#FBF8F2"       # cartes / champs
LATERAL = "#E9DFCE"     # barre latérale
BORDURE = "#DDD0BB"
TEXTE = "#3B2F23"
TEXTE_DOUX = "#7A6A58"
ACCENT = "#8A6A43"
ACCENT_FONCE = "#6B4F2E"
ACCENT_CLAIR = "#B89B72"
VERT = "#6F8F6A"
AMBRE = "#D9A441"
ROUGE = "#B5483A"

CSS = Template("""
.stApp { background: $BG; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1150px; }
.stApp p, .stApp li, .stApp label, .stApp .stMarkdown { color: $TEXTE; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    font-family: Georgia, 'Times New Roman', serif; color: $ACCENT_FONCE; letter-spacing: .2px; }
[data-testid="stSidebar"] { background: $LATERAL; border-right: 1px solid $BORDURE; }

/* Boutons */
.stButton button, .stFormSubmitButton button {
    background: $ACCENT; border: none; border-radius: 10px; padding: .55rem 1.5rem; }
.stButton button:hover, .stFormSubmitButton button:hover { background: $ACCENT_FONCE; }
.stButton button *, .stFormSubmitButton button * { color: #FBF8F2 !important; }

/* Champs de saisie */
input, textarea { background: $CARTE !important; color: $TEXTE !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {
    background: $CARTE !important; border-radius: 8px; }
.stNumberInput button { background: #EFE6D6 !important; color: $TEXTE !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: $CARTE; color: $TEXTE; }

/* Indicateurs */
[data-testid="stMetric"] { background: $CARTE; border: 1px solid $BORDURE;
    border-radius: 14px; padding: 14px 18px; }
[data-testid="stMetricValue"] { color: $ACCENT_FONCE; font-family: Georgia, serif; }
[data-testid="stMetricLabel"] p { color: $TEXTE_DOUX; }
[data-testid="stExpander"] { background: $CARTE; border: 1px solid $BORDURE; border-radius: 12px; }

/* Composants maison */
.en-tete h1 { margin-bottom: 0; }
.en-tete p { color: $TEXTE_DOUX; margin-top: .2rem; margin-bottom: 1.2rem; }
.carte { background: $CARTE; border: 1px solid $BORDURE; border-radius: 14px;
    padding: 16px 20px; margin: 8px 0 14px 0; }
.carte p { margin: 0; }
.tableau-wrap { overflow-x: auto; background: $CARTE; border: 1px solid $BORDURE;
    border-radius: 12px; margin-bottom: 12px; }
.tableau-beige { width: 100%; border-collapse: collapse; font-size: .88rem; }
.tableau-beige th { background: $LATERAL; color: $ACCENT_FONCE; padding: 8px 10px;
    text-align: right; font-weight: 600; white-space: nowrap; }
.tableau-beige td { padding: 7px 10px; text-align: right; color: $TEXTE;
    border-top: 1px solid $BORDURE; white-space: nowrap; }
.tableau-beige th:first-child, .tableau-beige td:first-child { text-align: left; }
.barre-ligne { display: flex; align-items: center; gap: 12px; margin: 8px 0; }
.barre-nom { width: 170px; color: $TEXTE; font-size: .92rem; }
.barre-piste { flex: 1; background: $LATERAL; border-radius: 8px; height: 18px; overflow: hidden; }
.barre-remplissage { height: 100%; border-radius: 8px; }
.barre-val { width: 90px; text-align: right; color: $TEXTE; font-size: .9rem; }
.etat-ligne { display: flex; justify-content: space-between; font-size: .88rem;
    padding: 3px 0; color: $TEXTE; }
.matrice { width: 100%; border-collapse: separate; border-spacing: 6px; text-align: center; }
.matrice td, .matrice th { padding: 14px 8px; border-radius: 10px; color: $TEXTE; }
.matrice th { background: transparent; color: $TEXTE_DOUX; font-weight: 600; font-size: .85rem; }
.matrice .val { font-size: 1.6rem; font-family: Georgia, serif; }
.matrice .sous { font-size: .78rem; color: $TEXTE_DOUX; }
""").substitute(
    BG=BG, CARTE=CARTE, LATERAL=LATERAL, BORDURE=BORDURE, TEXTE=TEXTE,
    TEXTE_DOUX=TEXTE_DOUX, ACCENT=ACCENT, ACCENT_FONCE=ACCENT_FONCE,
)


# ----------------------------------------------------------------------------
# 2. LOGO ET JAUGE (dessinés en SVG, sans aucune image externe)
# ----------------------------------------------------------------------------

def point_polaire(cx, cy, r, p):
    """Point d'un demi-cercle : p=0 -> extrémité gauche, p=1 -> extrémité droite."""
    theta = math.pi * (1 - p)
    return cx + r * math.cos(theta), cy - r * math.sin(theta)


def arc_svg(cx, cy, r, p1, p2, couleur, epaisseur):
    """Segment de demi-cercle entre les positions p1 et p2 (entre 0 et 1)."""
    x1, y1 = point_polaire(cx, cy, r, p1)
    x2, y2 = point_polaire(cx, cy, r, p2)
    return (f'<path d="M{x1:.1f} {y1:.1f} A{r} {r} 0 0 1 {x2:.1f} {y2:.1f}" '
            f'fill="none" stroke="{couleur}" stroke-width="{epaisseur}"/>')


def arcs_zones(cx, cy, r, epaisseur):
    """Trois zones : normal (vert), incertitude (ambre), attaque (rouge)."""
    return (arc_svg(cx, cy, r, 0.0, 0.4, VERT, epaisseur)
            + arc_svg(cx, cy, r, 0.4, 0.6, AMBRE, epaisseur)
            + arc_svg(cx, cy, r, 0.6, 1.0, ROUGE, epaisseur))


def logo_svg():
    """Logo : un bouclier (protection) contenant un cadran de vitesse (trafic)."""
    nx, ny = point_polaire(60, 84, 24, 0.78)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 140">'
        f'<path d="M60 6 L108 24 V68 C108 100 86 124 60 134 C34 124 12 100 12 68 V24 Z" '
        f'fill="{CARTE}" stroke="{ACCENT_FONCE}" stroke-width="5" stroke-linejoin="round"/>'
        + arcs_zones(60, 84, 30, 9) +
        f'<line x1="60" y1="84" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{ACCENT_FONCE}" '
        'stroke-width="4" stroke-linecap="round"/>'
        f'<circle cx="60" cy="84" r="5.5" fill="{ACCENT_FONCE}"/>'
        f'<path d="M34 104 H86" stroke="{ACCENT_CLAIR}" stroke-width="3" stroke-linecap="round"/>'
        '</svg>'
    )


def jauge_svg(proba):
    """Jauge façon compteur de vitesse. proba entre 0 et 1 (None = en attente)."""
    cx, cy, r = 150, 150, 105
    graduations = ""
    for p in (0, 0.25, 0.5, 0.75, 1):
        x, y = point_polaire(cx, cy, r + 26, p)
        graduations += (f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
                        f'font-size="12" fill="{TEXTE_DOUX}" font-family="sans-serif">{int(p * 100)}</text>')
    angle = -90 if proba is None else proba * 180 - 90
    texte = "—" if proba is None else f"{proba * 100:.1f} %"
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 205">'
        '<style>@keyframes aig { from { transform: rotate(-90deg); } }</style>'
        + arcs_zones(cx, cy, r, 22) + graduations +
        f'<g style="transform-origin:150px 150px; transform:rotate({angle:.1f}deg); '
        'animation: aig 1.2s ease-out;">'
        f'<polygon points="145,150 155,150 150,52" fill="{TEXTE}"/></g>'
        f'<circle cx="150" cy="150" r="11" fill="{TEXTE}"/>'
        f'<circle cx="150" cy="150" r="4.5" fill="{CARTE}"/>'
        f'<text x="150" y="190" text-anchor="middle" font-size="28" font-weight="bold" '
        f'fill="{ACCENT_FONCE}" font-family="Georgia, serif">{texte}</text>'
        '</svg>'
    )


def afficher_svg(svg, largeur):
    """Affiche un SVG via une balise <img> (encodage base64, fiable partout)."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    st.markdown(
        f'<div style="text-align:center"><img src="data:image/svg+xml;base64,{b64}" '
        f'width="{largeur}"/></div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 3. PETITS COMPOSANTS D'AFFICHAGE (HTML aux couleurs du thème)
# ----------------------------------------------------------------------------

def en_tete(titre, sous_titre):
    st.markdown(f'<div class="en-tete"><h1>{titre}</h1><p>{sous_titre}</p></div>',
                unsafe_allow_html=True)


def carte(texte, couleur=ACCENT):
    """Encadré beige avec un liseré de couleur à gauche."""
    st.markdown(f'<div class="carte" style="border-left:6px solid {couleur}"><p>{texte}</p></div>',
                unsafe_allow_html=True)


def tableau_html(df, decimales=3):
    """Affiche un DataFrame en tableau beige (reste lisible quel que soit le mode du navigateur)."""
    html = df.to_html(classes="tableau-beige", border=0, na_rep="",
                      float_format=lambda v: f"{v:.{decimales}f}")
    html = "".join(ligne.strip() for ligne in html.splitlines())
    st.markdown(f'<div class="tableau-wrap">{html}</div>', unsafe_allow_html=True)


def barres_html(lignes):
    """lignes = liste de (nom, fraction entre 0 et 1, texte affiché, couleur)."""
    html = ""
    for nom, fraction, texte, couleur in lignes:
        largeur = max(0.0, min(1.0, fraction)) * 100
        html += (f'<div class="barre-ligne"><div class="barre-nom">{nom}</div>'
                 f'<div class="barre-piste"><div class="barre-remplissage" '
                 f'style="width:{largeur:.1f}%;background:{couleur}"></div></div>'
                 f'<div class="barre-val">{texte}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def matrice_confusion_html(cm):
    """Matrice de confusion 2x2 : vert = bonnes réponses, rouge = erreurs."""
    noms = [["Vrais négatifs", "Faux positifs"], ["Faux négatifs", "Vrais positifs"]]
    html = ('<table class="matrice"><tr><th></th><th>Prédit : Normal</th>'
            '<th>Prédit : DDoS</th></tr>')
    for i in range(2):
        total_ligne = max(1, int(cm[i].sum()))
        html += f'<tr><th>Réel : {NOMS_CLASSES[i]}</th>'
        for j in range(2):
            couleur = "111,143,106" if i == j else "181,72,58"
            alpha = 0.12 + 0.55 * (cm[i][j] / total_ligne)
            html += (f'<td style="background:rgba({couleur},{alpha:.2f})">'
                     f'<div class="val">{int(cm[i][j])}</div>'
                     f'<div class="sous">{noms[i][j]}</div></td>')
        html += '</tr>'
    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# 4. FONCTIONS DU PIPELINE ML
# ----------------------------------------------------------------------------

@st.cache_data
def lire_csv(chemin):
    """Lit un fichier CSV (mis en cache pour ne pas le relire à chaque clic)."""
    return pd.read_csv(chemin)


def normaliser_label(df):
    """Convertit la colonne label en 0 (normal) / 1 (DDoS), même si elle est écrite en texte."""
    df = df.copy()
    if df["label"].dtype == object:
        correspondance = {"normal": 0, "0": 0, "ddos": 1, "attaque": 1, "1": 1}
        df["label"] = df["label"].astype(str).str.strip().str.lower().map(correspondance)
    return df


def verifier_dataset(df):
    """Retourne un message d'erreur (str) si le dataset est inutilisable, sinon None."""
    manquantes = [c for c in COLONNES_REQUISES if c not in df.columns]
    if manquantes:
        return "Colonnes manquantes : " + ", ".join(manquantes)
    if df["label"].isna().any() or not set(df["label"].unique()) <= {0, 1}:
        return "La colonne label doit contenir uniquement 0 (normal) et 1 (DDoS)."
    if df["label"].nunique() < 2:
        return "La colonne label doit contenir les deux classes."
    return None


def pretraiter(df, mode_nan):
    """
    Prétraitement :
      - suppression des doublons (AVANT le découpage, pour éviter les fuites) ;
      - valeurs manquantes : remplacement par 0 ou suppression des lignes ;
      - choix de X et y, encodage one-hot du protocole et du port de destination ;
      - découpage 70 % / 15 % / 15 % stratifié, puis standardisation (fit sur train seulement).
    """
    n_depart = len(df)
    df = df.drop_duplicates().copy()
    n_doublons = n_depart - len(df)
    n_nan = int(df.isna().sum().sum())
    n_lignes_nan = int(df.isna().any(axis=1).sum())

    if mode_nan == "zero":
        df = df.fillna(0)
    else:
        df = df.dropna()

    y = df["label"].astype(int)
    X = df.drop(columns=["label"] + [c for c in COLONNES_EXCLUES if c in df.columns])
    X = pd.get_dummies(X, columns=["protocol", "dst_port"], dtype=int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    return {
        "mode_nan": mode_nan, "n_depart": n_depart, "n_doublons": n_doublons,
        "n_nan": n_nan, "n_lignes_nan": n_lignes_nan, "n_final": len(df),
        "colonnes": list(X.columns),
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "scaler": scaler,
        "X_train_sc": scaler.fit_transform(X_train),   # fit UNIQUEMENT sur train
        "X_val_sc": scaler.transform(X_val),
        "X_test_sc": scaler.transform(X_test),
        "medianes": X_train[COLONNES_NUM].median(),
        "ports": sorted(int(c.replace("dst_port_", "")) for c in X.columns if c.startswith("dst_port_")),
        "protocoles": [c.replace("protocol_", "") for c in X.columns if c.startswith("protocol_")],
    }


def creer_modele(nom):
    """Retourne (modèle, a_besoin_de_standardisation) — mêmes réglages que l'atelier."""
    if nom == "Régression logistique":
        return LogisticRegression(max_iter=1000, random_state=42), True
    if nom == "SVM":
        # probability=True donne une probabilité pour la jauge (option dépréciée dans scikit-learn 1.9,
        # retirée en 1.11 : requirements.txt limite donc la version)
        return SVC(kernel="rbf", C=1.0, probability=True, random_state=42), True
    if nom == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2), False
    return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                         eval_metric="logloss", random_state=42, n_jobs=2), False


def probabilites(info, X):
    """Probabilité d'attaque DDoS (classe 1) pour une ou plusieurs lignes."""
    X_modele = info["scaler"].transform(X) if info["echelle"] else X
    return info["modele"].predict_proba(X_modele)[:, 1]


def calculer_metriques(y_vrai, y_pred):
    return {
        "Accuracy": accuracy_score(y_vrai, y_pred),
        "Précision": precision_score(y_vrai, y_pred, zero_division=0),
        "Recall": recall_score(y_vrai, y_pred, zero_division=0),
        "F1-score": f1_score(y_vrai, y_pred, zero_division=0),
    }


def entrainer_modeles(prep, noms, barre):
    """Entraîne les modèles sur TRAIN, les compare sur VALIDATION, puis évalue le meilleur
    une seule fois sur TEST."""
    modeles, resultats = {}, []
    for k, nom in enumerate(noms):
        barre.progress(k / len(noms), text=f"Entraînement : {nom}…")
        modele, echelle = creer_modele(nom)
        debut = time.time()
        X_tr = prep["X_train_sc"] if echelle else prep["X_train"]
        modele.fit(X_tr, prep["y_train"])
        duree = time.time() - debut
        modeles[nom] = {"modele": modele, "echelle": echelle,
                        "scaler": prep["scaler"], "temps": duree}
        # Prédictions sur validation (seuil 0,5 sur la probabilité DDoS)
        X_va = prep["X_val_sc"] if echelle else prep["X_val"]
        pred = (modele.predict_proba(X_va)[:, 1] >= 0.5).astype(int)
        ligne = {"Modèle": nom}
        ligne.update(calculer_metriques(prep["y_val"], pred))
        ligne["Temps (s)"] = round(duree, 1)
        resultats.append(ligne)
    barre.progress(1.0, text="Entraînement terminé.")

    # Meilleur modèle = meilleur F1 sur validation
    meilleur = max(resultats, key=lambda r: r["F1-score"])["Modèle"]
    info = modeles[meilleur]
    X_te = prep["X_test_sc"] if info["echelle"] else prep["X_test"]
    pred_test = (info["modele"].predict_proba(X_te)[:, 1] >= 0.5).astype(int)
    test = {
        "modele": meilleur,
        "metriques": calculer_metriques(prep["y_test"], pred_test),
        "cm": confusion_matrix(prep["y_test"], pred_test, labels=[0, 1]),
    }
    return modeles, resultats, meilleur, test


def ligne_vers_X(valeurs, protocole, port, colonnes):
    """Transforme une saisie manuelle en ligne encodée comme les données d'entraînement."""
    ligne = pd.DataFrame(0.0, index=[0], columns=colonnes)
    for col in COLONNES_NUM:
        ligne.loc[0, col] = valeurs[col]
    for nom_col in (f"protocol_{protocole}", f"dst_port_{port}"):
        if nom_col in colonnes:
            ligne.loc[0, nom_col] = 1.0
    return ligne


def colonne_active(ligne, prefixe):
    """Retourne la modalité active d'un encodage one-hot (ex. 'TCP' pour protocol_TCP)."""
    for col in ligne.index:
        if col.startswith(prefixe) and ligne[col] == 1:
            return col[len(prefixe):]
    return "?"


# ----------------------------------------------------------------------------
# 5. PAGES
# ----------------------------------------------------------------------------

def page_connexion():
    """Page de connexion affichée avant tout le reste."""
    st.markdown("<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl']"
                "{display:none}</style>", unsafe_allow_html=True)
    _, centre, _ = st.columns([1, 1.3, 1])
    with centre:
        st.write("")
        afficher_svg(logo_svg(), 110)
        st.markdown(
            f"<h1 style='text-align:center;margin-bottom:0'>Sentinelle DDoS</h1>"
            f"<p style='text-align:center;color:{TEXTE_DOUX}'>Détection et prédiction d'attaques "
            f"par déni de service distribué</p>", unsafe_allow_html=True)
        with st.form("formulaire_connexion"):
            nom = st.text_input("Nom d'utilisateur")
            mdp = st.text_input("Mot de passe", type="password")
            valider = st.form_submit_button("Se connecter")
        if valider:
            if nom == IDENTIFIANT and mdp == MOT_DE_PASSE:
                st.session_state["connecte"] = True
                st.rerun()
            else:
                st.error("Nom d'utilisateur ou mot de passe incorrect.")


def reinitialiser(a_partir_de):
    """Efface les résultats devenus obsolètes quand une étape amont change."""
    cles = {"prep": ["prep", "modeles", "resultats_val", "meilleur", "test"],
            "modeles": ["modeles", "resultats_val", "meilleur", "test"]}[a_partir_de]
    for cle in cles:
        st.session_state.pop(cle, None)


def charger_dataset_par_defaut():
    if "df" not in st.session_state and os.path.exists(CHEMIN_DATASET):
        df = normaliser_label(lire_csv(CHEMIN_DATASET))
        if verifier_dataset(df) is None:
            st.session_state["df"] = df
            st.session_state["source"] = "dataset.csv"


def page_donnees():
    en_tete("Jeu de données", "Chargement et exploration des flux réseau étiquetés.")

    fichier = st.file_uploader("Remplacer par un autre fichier CSV (optionnel)", type="csv")
    if fichier is not None and st.session_state.get("source") != (fichier.name, fichier.size):
        try:
            nouveau = normaliser_label(pd.read_csv(fichier))
            erreur = verifier_dataset(nouveau)
        except Exception as exc:  # fichier illisible
            nouveau, erreur = None, f"Lecture impossible : {exc}"
        if erreur:
            st.error(erreur)
        else:
            st.session_state["df"] = nouveau
            st.session_state["source"] = (fichier.name, fichier.size)
            reinitialiser("prep")

    if "df" not in st.session_state:
        st.warning("Aucun jeu de données chargé. Placez dataset.csv à côté de app.py "
                   "ou importez un fichier CSV.")
        return
    df = st.session_state["df"]

    carte("Selon l'en-tête de votre notebook, ce jeu de données est synthétique : les scores "
          "obtenus illustrent la méthode et ne prouvent pas une efficacité sur un réseau réel.",
          AMBRE)

    n_ddos = int((df["label"] == 1).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{len(df):,}".replace(",", " "))
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Trafic normal", f"{len(df) - n_ddos:,}".replace(",", " "))
    c4.metric("Attaques DDoS", f"{n_ddos:,}".replace(",", " "))

    st.subheader("Répartition des classes")
    barres_html([
        ("Normal (0)", (len(df) - n_ddos) / len(df), f"{(len(df) - n_ddos) / len(df) * 100:.1f} %", VERT),
        ("DDoS (1)", n_ddos / len(df), f"{n_ddos / len(df) * 100:.1f} %", ROUGE),
    ])

    st.subheader("Aperçu des données")
    tableau_html(df.head(10).reset_index(drop=True), decimals=2)

    st.subheader("Valeurs manquantes")
    manquantes = df.isna().sum()
    manquantes = manquantes[manquantes > 0]
    if manquantes.empty:
        st.write("Aucune valeur manquante.")
    else:
        tab = pd.DataFrame({"Valeurs manquantes": manquantes,
                            "Part (%)": (manquantes / len(df) * 100).round(2)})
        tableau_html(tab, decimals=2)

    with st.expander("Statistiques descriptives"):
        tableau_html(df.describe().T, decimals=2)


def page_pretraitement():
    en_tete("Prétraitement", "Nettoyage, encodage, découpage 70 / 15 / 15 et standardisation.")
    if "df" not in st.session_state:
        st.info("Chargez d'abord le jeu de données (page « Jeu de données »).")
        return

    choix = st.radio(
        "Traitement des valeurs manquantes",
        ["Remplacer par 0 (comme dans l'atelier)", "Supprimer les lignes incomplètes"])
    st.caption("Remplacer par 0 conserve toutes les lignes mais peut créer un signal artificiel ; "
               "supprimer réduit la taille des données.")
    if st.button("Lancer le prétraitement"):
        mode = "zero" if choix.startswith("Remplacer") else "supprimer"
        with st.spinner("Prétraitement en cours…"):
            st.session_state["prep"] = pretraiter(st.session_state["df"], mode)
        reinitialiser("modeles")

    prep = st.session_state.get("prep")
    if prep is None:
        return

    st.subheader("Résumé")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes au départ", prep["n_depart"])
    c2.metric("Doublons supprimés", prep["n_doublons"])
    c3.metric("Cellules manquantes", prep["n_nan"])
    c4.metric("Lignes conservées", prep["n_final"])
    st.write(f"Variables explicatives après encodage : **{len(prep['colonnes'])}** colonnes. "
             f"Colonnes écartées : {', '.join(COLONNES_EXCLUES)}.")

    st.subheader("Découpage des données")
    lignes = []
    for nom, y in (("Entraînement (70 %)", prep["y_train"]),
                   ("Validation (15 %)", prep["y_val"]),
                   ("Test (15 %)", prep["y_test"])):
        lignes.append({"Ensemble": nom, "Lignes": len(y), "Part de DDoS (%)": round(y.mean() * 100, 1)})
    tableau_html(pd.DataFrame(lignes).set_index("Ensemble"), decimals=1)


def page_entrainement():
    en_tete("Entraînement", "Apprentissage sur l'ensemble d'entraînement, comparaison sur la validation.")
    prep = st.session_state.get("prep")
    if prep is None:
        st.info("Lancez d'abord le prétraitement.")
        return

    noms = st.multiselect("Algorithmes à entraîner", MODELES_DISPONIBLES, default=MODELES_DISPONIBLES)
    st.caption("La régression linéaire de l'atelier n'est pas proposée : elle ne fournit pas de "
               "probabilité et n'est pas adaptée à la classification.")
    if st.button("Lancer l'entraînement"):
        if not noms:
            st.warning("Sélectionnez au moins un algorithme.")
        else:
            barre = st.progress(0.0, text="Démarrage…")
            modeles, resultats, meilleur, test = entrainer_modeles(prep, noms, barre)
            st.session_state.update(modeles=modeles, resultats_val=resultats,
                                    meilleur=meilleur, test=test)

    if "resultats_val" in st.session_state:
        st.subheader("Modèles entraînés")
        tab = pd.DataFrame(st.session_state["resultats_val"])[["Modèle", "Temps (s)"]]
        tableau_html(tab.set_index("Modèle"), decimals=1)
        carte(f"Meilleur modèle sur la validation (F1-score) : <b>{st.session_state['meilleur']}</b>. "
              "Consultez la page « Métriques » pour le détail.", VERT)


def page_metriques():
    en_tete("Métriques", "Accuracy, Précision, Recall et F1-score calculés sur les données.")
    if "resultats_val" not in st.session_state:
        st.info("Entraînez d'abord les modèles.")
        return

    tableau = (pd.DataFrame(st.session_state["resultats_val"])
               .drop(columns=["Temps (s)"])
               .sort_values("F1-score", ascending=False).reset_index(drop=True))
    meilleur = st.session_state["meilleur"]

    st.subheader("Comparaison sur l'ensemble de validation")
    tableau_html(tableau.set_index("Modèle"))
    barres_html([(r["Modèle"], r["F1-score"], f"{r['F1-score']:.3f}",
                  ACCENT_FONCE if r["Modèle"] == meilleur else ACCENT_CLAIR)
                 for _, r in tableau.iterrows()])
    st.caption("Barres : F1-score de validation.")

    test = st.session_state["test"]
    st.subheader(f"Évaluation finale sur le test — {test['modele']}")
    st.caption("Le jeu de test n'est utilisé qu'une seule fois, pour le meilleur modèle de la validation.")
    colonnes = st.columns(4)
    for col, (nom, val) in zip(colonnes, test["metriques"].items()):
        col.metric(nom, f"{val * 100:.2f} %")

    st.subheader("Matrice de confusion (test)")
    matrice_confusion_html(test["cm"])
    st.caption("Faux négatifs : attaques non détectées. Faux positifs : trafic normal signalé à tort.")

    with st.expander("Définition des métriques"):
        st.write("**Accuracy** : part de prédictions correctes. **Précision** : parmi les alertes, "
                 "part d'attaques réelles. **Recall** : parmi les attaques réelles, part détectée. "
                 "**F1-score** : équilibre entre précision et recall.")


def afficher_resultat(proba, vrai_label=None):
    """Jauge + verdict pour une prédiction."""
    afficher_svg(jauge_svg(proba), 420)
    st.markdown(f"<p style='text-align:center;color:{TEXTE_DOUX};margin-top:-6px'>"
                "Probabilité d'attaque DDoS estimée par le modèle</p>", unsafe_allow_html=True)
    if proba is None:
        return
    if proba >= 0.5:
        verdict, couleur = "Attaque DDoS probable", ROUGE
    else:
        verdict, couleur = "Trafic normal", VERT
    texte = f"<b>{verdict}</b>"
    if 0.4 <= proba <= 0.6:
        texte += " — décision peu tranchée, vérification conseillée."
    if vrai_label is not None:
        correct = (proba >= 0.5) == bool(vrai_label)
        texte += (f"<br>Étiquette réelle du jeu de test : <b>{NOMS_CLASSES[int(vrai_label)]}</b> — "
                  f"prédiction {'correcte' if correct else 'incorrecte'}.")
    carte(texte, couleur)


def tirer_au_hasard(n):
    st.session_state["idx_test"] = int(np.random.randint(0, n))


def page_prediction():
    en_tete("Prédiction", "Estimez si un flux réseau est normal ou s'il s'agit d'une attaque DDoS.")
    if "modeles" not in st.session_state:
        st.info("Entraînez d'abord les modèles.")
        return
    prep, modeles = st.session_state["prep"], st.session_state["modeles"]

    noms = list(modeles.keys())
    defaut = noms.index(st.session_state["meilleur"])
    nom = st.selectbox("Modèle utilisé", noms, index=defaut)
    info = modeles[nom]

    onglet_saisie, onglet_test = st.tabs(["Saisie manuelle", "Exemple du jeu de test"])

    # --- Saisie manuelle -------------------------------------------------------
    with onglet_saisie:
        st.caption("Les valeurs proposées par défaut sont les médianes de l'ensemble d'entraînement.")
        gauche, droite = st.columns([1.2, 1])
        with gauche:
            with st.form("formulaire_prediction"):
                c1, c2 = st.columns(2)
                protocole = c1.selectbox("Protocole", prep["protocoles"])
                port = c2.selectbox("Port de destination", prep["ports"])
                valeurs = {}
                cols = st.columns(2)
                for k, (col, libelle, type_) in enumerate(PARAMETRES_FLUX):
                    med = float(prep["medianes"][col])
                    if type_ == "int":
                        valeurs[col] = cols[k % 2].number_input(
                            libelle, min_value=0, value=int(round(med)), step=1, key=f"in_{col}")
                    else:
                        valeurs[col] = cols[k % 2].number_input(
                            libelle, min_value=0.0, value=round(med, 2), step=1.0,
                            format="%.2f", key=f"in_{col}")
                lancer = st.form_submit_button("Lancer la prédiction")
        with droite:
            if lancer:
                ligne = ligne_vers_X(valeurs, protocole, port, prep["colonnes"])
                afficher_resultat(float(probabilites(info, ligne)[0]))
            else:
                afficher_resultat(None)
                st.caption("Renseignez le flux puis lancez la prédiction.")

    # --- Exemple du jeu de test --------------------------------------------------
    with onglet_test:
        X_test, y_test = prep["X_test"], prep["y_test"]
        n = len(X_test)
        st.session_state.setdefault("idx_test", 0)
        c1, c2 = st.columns([1, 1])
        c1.number_input("Numéro de l'exemple", min_value=0, max_value=n - 1, step=1, key="idx_test")
        c2.write("")
        c2.write("")
        c2.button("Tirer un exemple au hasard", on_click=tirer_au_hasard, args=(n,))
        idx = int(st.session_state["idx_test"])
        ligne = X_test.iloc[[idx]]
        gauche, droite = st.columns([1.2, 1])
        with gauche:
            resume = ligne[COLONNES_NUM].T
            resume.columns = ["Valeur"]
            resume.index = [dict((c, l) for c, l, _ in PARAMETRES_FLUX)[c] for c in resume.index]
            resume.loc["Protocole"] = colonne_active(ligne.iloc[0], "protocol_")
            resume.loc["Port de destination"] = colonne_active(ligne.iloc[0], "dst_port_")
            tableau_html(resume.astype(str))
        with droite:
            afficher_resultat(float(probabilites(info, ligne)[0]), y_test.iloc[idx])


def barre_laterale():
    """Barre latérale : logo, navigation, état d'avancement, déconnexion."""
    with st.sidebar:
        afficher_svg(logo_svg(), 80)
        st.markdown(f"<h3 style='text-align:center;margin:0'>Sentinelle DDoS</h3>"
                    f"<p style='text-align:center;color:{TEXTE_DOUX};font-size:.85rem'>"
                    "Détection d'attaques réseau</p>", unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.markdown("---")
        etapes = [("Données", "df"), ("Prétraitement", "prep"), ("Modèles", "modeles")]
        html = ""
        for libelle, cle in etapes:
            pret = cle in st.session_state
            html += (f'<div class="etat-ligne"><span>{libelle}</span>'
                     f'<span style="color:{VERT if pret else TEXTE_DOUX}">'
                     f'{"Prêt" if pret else "En attente"}</span></div>')
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("---")
        if st.button("Se déconnecter"):
            st.session_state.clear()
            st.rerun()
    return page


def main():
    st.set_page_config(page_title="Sentinelle DDoS", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)

    if not st.session_state.get("connecte", False):
        page_connexion()
        return

    charger_dataset_par_defaut()
    page = barre_laterale()
    {"Jeu de données": page_donnees, "Prétraitement": page_pretraitement,
     "Entraînement": page_entrainement, "Métriques": page_metriques,
     "Prédiction": page_prediction}[page]()


if __name__ == "__main__":
    main()
