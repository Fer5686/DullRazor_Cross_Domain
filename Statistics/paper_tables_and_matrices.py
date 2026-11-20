"""
=========================================================================================
MODULE: Publication Figures, McNemar Agreement & Delta Confusion Matrices
        (statistics/paper_tables_and_matrices.py)
=========================================================================================
DESCRIPTION:
    Generates the statistical tables (PNG & Elsevier/APA Excel workbooks) and confusion 
    matrix visualizations for the manuscript:
    1. Population Analysis: Computes Mean ± Sample SD (ddof=1), Welch's t-test (native 
       unpaired), Paired Student's t-test (cross-domain shift), and pooled Cohen's d.
    2. McNemar's Diagnostic Agreement Test: Selects the median-performing validation 
       model per training condition and evaluates paired diagnostic discordances on the 
       ISIC 2018 test set (N=1,511), automatically applying exact binomial calculation 
       when discordant pairs < 25.
    3. Condensed & 1x3 Delta Confusion Matrices: Visualizes normalized per-class recall 
       alongside absolute prediction shifts induced by domain transitions.
    4. Multi-Language Support (i18n): Exports figures and tables in English ('en'), 
       Spanish ('es'), or Portuguese ('pt').
=========================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats

# ==============================================================================
# 0. PATH CONFIGURATION & CENTRAL DATA ENGINE IMPORT
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_paths
from clean_data_extractor import get_clean_evaluation_data as obtener_datos_limpios

MODELS_DIR = getattr(config_paths, "MODELS_DIR", getattr(config_paths, "REPO_DIR"))
DOCS_DIR = getattr(config_paths, "DOCS_DIR")

# ==============================================================================
# 1. MULTI-LANGUAGE DICTIONARY (Spanish, English, Portuguese)
# ==============================================================================
TEXTOS_I18N = {
    'es': {
        'trained': "Entrenado",
        'tested': "Testeado",
        'with_hair': "Con Pelos",
        'without_hair': "Sin Pelos",
        'ttest_native': "p-valor y d de Cohen\nNativo vs. Nativo",
        'ttest_train_cp': "p-valor y d de Cohen\nEntr. Pelos / Test Limpio",
        'ttest_train_sp': "p-valor y d de Cohen\nEntr. Limpio / Test Pelos",
        'mc_header': "Concordancia Diagnóstica\nMcNemar (N=1511)",
        'mc_only_cp': "Solo 'Con Pelos'\nCorrecto",
        'mc_only_sp': "Solo 'Sin Pelos'\nCorrecto",
        'mc_pval': "p-valor",
        'mc_row_native': "Inferencia Nativa",
        'mc_row_shift_cp': "Cambio Dominio (Oclusiones)",
        'mc_row_shift_sp': "Cambio Dominio (Limpio)",
        'title_multi': "F2-Score Multiclase",
        'title_mel': "F2-Score Melanoma",
        'title_main': "Evaluación de Cambio de Dominio: {arq} ({titulo})",
        'true_pathology': "Patología Real",
        'pred_pathology': "Patología Predicha",
        'delta_title': "Impacto del Cambio de Dominio ({arq})\nPredicciones Absolutas y Variación de Error",
        'native_cp': "Modelo Nativo Con Pelos",
        'native_sp': "Modelo Nativo Sin Pelos",
        'norm_recall': "Recall Normalizado",
        'delta_matrix_title': "Matriz Delta (Desplazamientos Absolutos)\n[Positivo = Modelo B predijo más esta clase]"
    },
    'en': {
        'trained': "Trained",
        'tested': "Tested",
        'with_hair': "With Hair",
        'without_hair': "Without Hair",
        'ttest_native': "p-value & Cohen's d\nNative vs. Native",
        'ttest_train_cp': "p-value & Cohen's d\nTrain Hair / Test Clean",
        'ttest_train_sp': "p-value & Cohen's d\nTrain Clean / Test Hair",
        'mc_header': "McNemar Diagnostic\nAgreement (N=1511)",
        'mc_only_cp': "Only 'With Hair'\nCorrect",
        'mc_only_sp': "Only 'Without Hair'\nCorrect",
        'mc_pval': "p-value",
        'mc_row_native': "Native Inference",
        'mc_row_shift_cp': "Domain Shift (Occlusions)",
        'mc_row_shift_sp': "Domain Shift (Clean)",
        'title_multi': "Multiclass F2-Score",
        'title_mel': "Melanoma F2-Score",
        'title_main': "Cross-Domain Shift Evaluation: {arq} ({titulo})",
        'true_pathology': "True Pathology",
        'pred_pathology': "Predicted Pathology",
        'delta_title': "Domain Shift Impact ({arq})\nAbsolute Predictions & Error Deltas",
        'native_cp': "Native CP Model",
        'native_sp': "Native SP Model",
        'norm_recall': "Normalized Recall",
        'delta_matrix_title': "Delta Matrix (Absolute Shifts)\n[Positive = Model B predicted this more]"
    },
    'pt': {
        'trained': "Treinado",
        'tested': "Testado",
        'with_hair': "Com Pelos",
        'without_hair': "Sem Pelos",
        'ttest_native': "p-valor e d de Cohen\nNativo vs. Nativo",
        'ttest_train_cp': "p-valor e d de Cohen\nTrein. Pelos / Teste Limpo",
        'ttest_train_sp': "p-valor e d de Cohen\nTrein. Limpo / Teste Pelos",
        'mc_header': "Concordância Diagnóstica\nMcNemar (N=1511)",
        'mc_only_cp': "Apenas 'Com Pelos'\nCorreto",
        'mc_only_sp': "Apenas 'Sem Pelos'\nCorreto",
        'mc_pval': "p-valor",
        'mc_row_native': "Inferência Nativa",
        'mc_row_shift_cp': "Mudança Domínio (Oclusões)",
        'mc_row_shift_sp': "Mudança Domínio (Limpo)",
        'title_multi': "F2-Score Multiclasse",
        'title_mel': "F2-Score Melanoma",
        'title_main': "Avaliação de Mudança de Domínio: {arq} ({titulo})",
        'true_pathology': "Patologia Real",
        'pred_pathology': "Patologia Predita",
        'delta_title': "Impacto da Mudança de Domínio ({arq})\nPredições Absolutas e Variação de Erro",
        'native_cp': "Modelo Nativo Com Pelos",
        'native_sp': "Modelo Nativo Sem Pelos",
        'norm_recall': "Recall Normalizado",
        'delta_matrix_title': "Matriz Delta (Deslocamentos Absolutos)\n[Positivo = Modelo B predisse mais esta classe]"
    }
}


def normalize_language(lang_code: str) -> str:
    lang = str(lang_code).lower().strip()
    if lang in ['es', 'castellano', 'español', 'spanish']:
        return 'es'
    if lang in ['pt', 'portugues', 'português', 'portuguese']:
        return 'pt'
    return 'en'


# ==============================================================================
# 2. ACADEMIC VISUAL STYLING & TABLE RENDERING
# ==============================================================================
COLOR_AZUL_TR = "#46CDEB"
COLOR_ROJO_TS = "#EB6346"
COLOR_MORADO = "#EFDCF2"
COLOR_VERDE = "#DFF2DC"
COLOR_AMARILLO = "#FDC10E"
COLOR_CREMA = "#EFE3A4"

COLOR_MCN_HDR = "#497DD1"
COLOR_MCN_FILA = "#9BAED4"
COLOR_MCN_DAT = "#DCE4F2"
COLOR_MCN_PVAL = "#DCE4F2"
COLOR_BORDE = "black"


def _fmt(x, decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{decimals}f}"


def draw_cell(ax, x, y, w, h, color, text="", fontsize=11, fontweight='normal', fontcolor='black'):
    rect = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=COLOR_BORDE, linewidth=1.5)
    ax.add_patch(rect)
    if text:
        ax.text(
            x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fontsize, fontweight=fontweight, color=fontcolor
        )


def draw_diagonal_header_cell(ax, x, y, w, h, t):
    tri_sup = patches.Polygon(
        [(x, y + h), (x + w, y + h), (x + w, y)],
        closed=True, facecolor=COLOR_AZUL_TR, edgecolor=COLOR_BORDE, linewidth=1.5
    )
    tri_inf = patches.Polygon(
        [(x, y + h), (x, y), (x + w, y)],
        closed=True, facecolor=COLOR_ROJO_TS, edgecolor=COLOR_BORDE, linewidth=1.5
    )
    ax.add_patch(tri_inf)
    ax.add_patch(tri_sup)
    ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='none', edgecolor=COLOR_BORDE, linewidth=1.5))
    
    ax.text(x + w * 0.70, y + h * 0.72, t['trained'], ha='center', va='center', fontsize=11, color='black', fontweight='bold')
    ax.text(x + w * 0.30, y + h * 0.28, t['tested'], ha='center', va='center', fontsize=11, color='white', fontweight='bold')


def generate_combined_table_figure(datos_pob: dict, datos_mcnemar: dict, arq: str, nom_metrica: str, nombre_out: str, t: dict):
    lbl_w = 2.4
    col_w = 1.8
    row_h = 0.45
    total_w = lbl_w + 2 * col_w
    total_h = 10 * row_h
    
    fig, ax = plt.subplots(figsize=(9.5, 6))

    y_mcnemar_data3 = 0.0
    y_mcnemar_data2 = y_mcnemar_data3 + row_h
    y_mcnemar_data1 = y_mcnemar_data2 + row_h
    y_mcnemar_header = y_mcnemar_data1 + row_h
    y_gap = y_mcnemar_header + row_h
    y_pob_data_inf = y_gap
    y_pob_header_inf = y_pob_data_inf + row_h
    y_pob_data2 = y_pob_header_inf + row_h
    y_pob_data1 = y_pob_data2 + row_h
    y_pob_header1 = y_pob_data1 + row_h

    # Header Row: Trained (Columns: With Hair | Without Hair)
    draw_diagonal_header_cell(ax, 0, y_pob_header1, lbl_w, row_h, t)
    draw_cell(ax, lbl_w, y_pob_header1, col_w, row_h, COLOR_AZUL_TR, t['with_hair'], fontcolor='black', fontweight='bold', fontsize=14)
    draw_cell(ax, lbl_w + col_w, y_pob_header1, col_w, row_h, COLOR_AZUL_TR, t['without_hair'], fontcolor='black', fontweight='bold', fontsize=14)

    # Row 1: Tested With Hair -> Col 1 (Trained With Hair: cp_cp) | Col 2 (Trained Without Hair: sp_cp)
    draw_cell(ax, 0, y_pob_data1, lbl_w, row_h, COLOR_ROJO_TS, t['with_hair'], fontcolor='white', fontweight='bold', fontsize=14)
    draw_cell(ax, lbl_w, y_pob_data1, col_w, row_h, COLOR_MORADO, f"{_fmt(datos_pob['medias']['cp_cp'])} \u00B1 {_fmt(datos_pob['stds']['cp_cp'])}", fontsize=14)
    draw_cell(ax, lbl_w + col_w, y_pob_data1, col_w, row_h, COLOR_VERDE, f"{_fmt(datos_pob['medias']['sp_cp'])} \u00B1 {_fmt(datos_pob['stds']['sp_cp'])}", fontsize=14)

    # Row 2: Tested Without Hair -> Col 1 (Trained With Hair: cp_sp) | Col 2 (Trained Without Hair: sp_sp)
    draw_cell(ax, 0, y_pob_data2, lbl_w, row_h, COLOR_ROJO_TS, t['without_hair'], fontcolor='white', fontweight='bold', fontsize=14)
    draw_cell(ax, lbl_w, y_pob_data2, col_w, row_h, COLOR_VERDE, f"{_fmt(datos_pob['medias']['cp_sp'])} \u00B1 {_fmt(datos_pob['stds']['cp_sp'])}", fontsize=14)
    draw_cell(ax, lbl_w + col_w, y_pob_data2, col_w, row_h, COLOR_MORADO, f"{_fmt(datos_pob['medias']['sp_sp'])} \u00B1 {_fmt(datos_pob['stds']['sp_sp'])}", fontsize=14)

    header_labels = [t['ttest_native'], t['ttest_train_cp'], t['ttest_train_sp']]
    stat_values = [(datos_pob['p1'], datos_pob['d1']), (datos_pob['p2'], datos_pob['d2']), (datos_pob['p3'], datos_pob['d3'])]
    
    widths_pob = [lbl_w, col_w, col_w]
    x_coords_pob = [0, lbl_w, lbl_w + col_w]

    for i in range(3):
        x = x_coords_pob[i]
        w = widths_pob[i]
        draw_cell(ax, x, y_pob_header_inf, w, row_h, COLOR_AMARILLO, header_labels[i], fontsize=10, fontweight='bold', fontcolor='black')
        p, d = stat_values[i]
        fw = 'bold' if (not pd.isna(p) and p < 0.05) else 'normal'
        data_text = f"p = {_fmt(p, 4)}\nd = {_fmt(d, 2)}"
        draw_cell(ax, x, y_pob_data_inf, w, row_h, COLOR_CREMA, data_text, fontsize=13, fontweight=fw)

    mc_lbl = lbl_w
    mc_cola = 1.3
    mc_colb = 1.3
    mc_pval = 1.0

    draw_cell(ax, 0, y_mcnemar_header, mc_lbl, row_h, COLOR_MCN_HDR, t['mc_header'], fontcolor='white', fontweight='bold', fontsize=11)
    draw_cell(ax, mc_lbl, y_mcnemar_header, mc_cola, row_h, COLOR_MCN_HDR, t['mc_only_cp'], fontcolor='white', fontweight='bold', fontsize=11)
    draw_cell(ax, mc_lbl + mc_cola, y_mcnemar_header, mc_colb, row_h, COLOR_MCN_HDR, t['mc_only_sp'], fontcolor='white', fontweight='bold', fontsize=10)
    draw_cell(ax, mc_lbl + mc_cola + mc_colb, y_mcnemar_header, mc_pval, row_h, COLOR_MCN_HDR, t['mc_pval'], fontcolor='white', fontweight='bold', fontsize=12)

    def draw_mcnemar_row(y, label, key):
        d = datos_mcnemar.get(key, {})
        val_A = str(d.get('A', 'N/A'))
        val_B = str(d.get('B', 'N/A'))
        pval = d.get('p', np.nan)
        
        draw_cell(ax, 0, y, mc_lbl, row_h, COLOR_MCN_HDR, label, fontcolor='white', fontweight='bold', fontsize=12)
        draw_cell(ax, mc_lbl, y, mc_cola, row_h, COLOR_MCN_DAT, val_A, fontsize=14)
        draw_cell(ax, mc_lbl + mc_cola, y, mc_colb, row_h, COLOR_MCN_DAT, val_B, fontsize=14)
        
        fw = 'bold' if (not pd.isna(pval) and pval < 0.05) else 'normal'
        p_text = _fmt(pval, 4) if not pd.isna(pval) else "N/A"
        draw_cell(ax, mc_lbl + mc_cola + mc_colb, y, mc_pval, row_h, COLOR_MCN_PVAL, p_text, fontsize=14, fontweight=fw)

    draw_mcnemar_row(y_mcnemar_data1, t['mc_row_native'], 'nativo')
    draw_mcnemar_row(y_mcnemar_data2, t['mc_row_shift_cp'], 'test_cp')
    draw_mcnemar_row(y_mcnemar_data3, t['mc_row_shift_sp'], 'test_sp')

    sub_title = t['title_multi'] if nom_metrica == "f2_score" else t['title_mel']
    full_title = t['title_main'].format(arq=arq, titulo=sub_title)
    ax.text(
        total_w / 2, total_h - 0.35, full_title,
        ha='center', va='bottom', fontsize=13, fontweight='bold', color=COLOR_BORDE
    )

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h - 0.1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout(pad=0)
    plt.savefig(f"{nombre_out}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


# ==============================================================================
# 3. STATISTICAL FUNCTIONS: WELCH'S T-TEST, PAIRED T-TEST & MCNEMAR
# ==============================================================================
def calc_population_stats(a, b, paired: bool = True, seeds_a=None, seeds_b=None) -> tuple:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    if paired and seeds_a is not None and seeds_b is not None:
        df_a = pd.DataFrame({"Semilla": seeds_a, "val_a": a_arr}).dropna(subset=["Semilla", "val_a"])
        df_b = pd.DataFrame({"Semilla": seeds_b, "val_b": b_arr}).dropna(subset=["Semilla", "val_b"])
        df_a["Semilla"] = df_a["Semilla"].astype(str).str.strip()
        df_b["Semilla"] = df_b["Semilla"].astype(str).str.strip()

        merged = pd.merge(df_a, df_b, on="Semilla", how="inner")
        if len(merged) < 2:
            return np.nan, np.nan

        a_clean = merged["val_a"].to_numpy(dtype=float)
        b_clean = merged["val_b"].to_numpy(dtype=float)
        _, p = stats.ttest_rel(a_clean, b_clean)
        pooled_std = np.sqrt((np.std(a_clean, ddof=1) ** 2 + np.std(b_clean, ddof=1) ** 2) / 2)
        d = abs(np.mean(a_clean) - np.mean(b_clean)) / max(pooled_std, 1e-5)
        return p, d

    a_clean = a_arr[~np.isnan(a_arr)]
    b_clean = b_arr[~np.isnan(b_arr)]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return np.nan, np.nan

    if paired:
        min_len = min(len(a_clean), len(b_clean))
        _, p = stats.ttest_rel(a_clean[:min_len], b_clean[:min_len])
        pooled_std = np.sqrt((np.std(a_clean[:min_len], ddof=1) ** 2 + np.std(b_clean[:min_len], ddof=1) ** 2) / 2)
        d = abs(np.mean(a_clean[:min_len]) - np.mean(b_clean[:min_len])) / max(pooled_std, 1e-5)
    else:
        _, p = stats.ttest_ind(a_clean, b_clean, equal_var=False)
        pooled_std = np.sqrt(
            ((len(a_clean) - 1) * np.std(a_clean, ddof=1) ** 2 + (len(b_clean) - 1) * np.std(b_clean, ddof=1) ** 2) /
            (len(a_clean) + len(b_clean) - 2)
        )
        d = abs(np.mean(a_clean) - np.mean(b_clean)) / max(pooled_std, 1e-5)

    return p, d


def get_predictions(probs_path: str) -> np.ndarray:
    probs = np.load(probs_path)
    if len(probs.shape) > 1 and probs.shape[1] > 1:
        return np.argmax(probs, axis=1)
    return (probs >= 0.5).astype(int)


def compute_mcnemar_discordances(path_A: str, path_B: str, binary_mode: bool = False, target_class: int = 1) -> tuple:
    if not isinstance(path_A, str) or not isinstance(path_B, str):
        return 'N/A', 'N/A', np.nan
        
    labels_path = os.path.join(path_A, "labels.npy")
    probs_A_path = os.path.join(path_A, "probs.npy")
    probs_B_path = os.path.join(path_B, "probs.npy")

    if not all(os.path.exists(p) for p in [labels_path, probs_A_path, probs_B_path]):
        return 'N/A', 'N/A', np.nan

    y_true = np.load(labels_path)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    preds_A = get_predictions(probs_A_path)
    preds_B = get_predictions(probs_B_path)

    if binary_mode:
        y_true = (y_true == target_class).astype(int)
        preds_A = (preds_A == target_class).astype(int)
        preds_B = (preds_B == target_class).astype(int)

    correct_A = (preds_A == y_true)
    correct_B = (preds_B == y_true)
    A_wins = np.sum(correct_A & ~correct_B)
    B_wins = np.sum(~correct_A & correct_B)
    
    contingency_table = [
        [np.sum(correct_A & correct_B), A_wins],
        [B_wins, np.sum(~correct_A & ~correct_B)]
    ]
    result = mcnemar(contingency_table, exact=((A_wins + B_wins) < 25))
    return int(A_wins), int(B_wins), float(result.pvalue)


def select_median_model(df_subset: pd.DataFrame, metric_col: str) -> str:
    df_sorted = df_subset.sort_values(by=metric_col, ascending=True).reset_index(drop=True)
    return df_sorted.iloc[len(df_sorted) // 2]['Modelo']


def plot_condensed_delta_matrix(path_A: str, path_B: str, class_labels: list, plot_title: str, out_name: str, t: dict):
    labels_path = os.path.join(path_A, "labels.npy")
    probs_A_path = os.path.join(path_A, "probs.npy")
    probs_B_path = os.path.join(path_B, "probs.npy")

    if not all(os.path.exists(p) for p in [labels_path, probs_A_path, probs_B_path]):
        return

    y_true = np.load(labels_path)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    preds_A = get_predictions(probs_A_path)
    preds_B = get_predictions(probs_B_path)

    cm_A_abs = confusion_matrix(y_true, preds_A)
    cm_B_abs = confusion_matrix(y_true, preds_B)
    cm_B_norm = confusion_matrix(y_true, preds_B, normalize='true')
    delta_cm = cm_B_abs - cm_A_abs

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('white')

    im = ax.imshow(cm_B_norm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(class_labels, fontsize=11)
    ax.set_title(plot_title, fontweight='bold', pad=15, fontsize=13)
    ax.set_ylabel(t['true_pathology'], fontweight='bold', fontsize=12)
    ax.set_xlabel(t['pred_pathology'], fontweight='bold', fontsize=12)

    threshold = np.max(cm_B_norm) / 2.0
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            val_B = cm_B_abs[i, j]
            delta = delta_cm[i, j]
            txt_delta = f"(+{delta})" if delta > 0 else (f"({delta})" if delta < 0 else "")
            text_color = "white" if cm_B_norm[i, j] > threshold else "black"
            final_text = f"{val_B}\n{txt_delta}" if txt_delta else f"{val_B}"
            ax.text(j, i, final_text, ha="center", va="center", color=text_color, fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{out_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_1x3_mcnemar_matrices(path_A: str, path_B: str, class_labels: list, title_A: str, title_B: str, out_name: str, t: dict):
    labels_path = os.path.join(path_A, "labels.npy")
    probs_A_path = os.path.join(path_A, "probs.npy")
    probs_B_path = os.path.join(path_B, "probs.npy")

    if not all(os.path.exists(p) for p in [labels_path, probs_A_path, probs_B_path]):
        return

    y_true = np.load(labels_path)
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    preds_A = get_predictions(probs_A_path)
    preds_B = get_predictions(probs_B_path)

    cm_A_abs = confusion_matrix(y_true, preds_A)
    cm_B_abs = confusion_matrix(y_true, preds_B)
    cm_A_norm = confusion_matrix(y_true, preds_A, normalize='true')
    cm_B_norm = confusion_matrix(y_true, preds_B, normalize='true')
    delta_cm = cm_B_abs - cm_A_abs

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('white')

    def draw_sub_matrix(ax, matrix, title, cmap, pct_format=False, divergent=False):
        im = ax.imshow(matrix, interpolation='nearest', cmap=cmap)
        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels)
        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_ylabel(t['true_pathology'])
        ax.set_xlabel(t['pred_pathology'])
        
        if divergent:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            vmax = np.max(np.abs(matrix))
            im.set_clim(-vmax, vmax)
        else:
            vmax = np.max(matrix)
            im.set_clim(0, vmax)

        threshold = vmax / 2.0
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                val = matrix[i, j]
                txt = f"{val:.1%}" if pct_format else f"{int(val)}"
                text_color = "white" if (abs(val) > threshold * 1.2 if divergent else val > threshold) else "black"
                ax.text(j, i, txt, ha="center", va="center", color=text_color)

    draw_sub_matrix(axes[0], cm_A_norm, f"{title_A}\n({t['norm_recall']})", cmap=plt.cm.Blues, pct_format=True)
    draw_sub_matrix(axes[1], cm_B_norm, f"{title_B}\n({t['norm_recall']})", cmap=plt.cm.Blues, pct_format=True)
    draw_sub_matrix(axes[2], delta_cm, t['delta_matrix_title'], cmap=plt.cm.RdBu_r, divergent=True)

    plt.tight_layout()
    plt.savefig(f"{out_name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


# ==============================================================================
# 4. EXCEL WORKBOOK EXPORT (APA / Elsevier Standard)
# ==============================================================================
def export_elsevier_excel_tables(datos_pob: dict, datos_mcnemar: dict, arq: str, nom_metrica: str, save_dir: str, t: dict):
    excel_path = os.path.join(save_dir, f"{arq.replace(' ', '_')}_{nom_metrica}_Elsevier_Tables.xlsx")
    
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        fmt_spanner = workbook.add_format({'bold': True, 'bottom': 1, 'top': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_top_left = workbook.add_format({'top': 1})
        fmt_header = workbook.add_format({'bold': True, 'bottom': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_header_izq = workbook.add_format({'bold': True, 'bottom': 1, 'align': 'left', 'valign': 'vcenter'})
        fmt_celda = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
        fmt_celda_izq = workbook.add_format({'align': 'left', 'valign': 'vcenter'})
        fmt_bottom = workbook.add_format({'bottom': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_bottom_izq = workbook.add_format({'bottom': 1, 'align': 'left', 'valign': 'vcenter'})

        ws_pob = workbook.add_worksheet('Population_Stats')
        ws_pob.set_column('A:A', 35)
        ws_pob.set_column('B:C', 25)
        ws_pob.write(0, 0, '', fmt_top_left)
        ws_pob.merge_range(0, 1, 0, 2, t['trained'], fmt_spanner)
        ws_pob.write(1, 0, t['tested'], fmt_header_izq)
        ws_pob.write(1, 1, t['with_hair'], fmt_header)
        ws_pob.write(1, 2, t['without_hair'], fmt_header)

        rows_pob = [
            (f"{t['tested']} {t['with_hair']}", f"{_fmt(datos_pob['medias']['cp_cp'])} ± {_fmt(datos_pob['stds']['cp_cp'])}", f"{_fmt(datos_pob['medias']['sp_cp'])} ± {_fmt(datos_pob['stds']['sp_cp'])}"),
            (f"{t['tested']} {t['without_hair']}", f"{_fmt(datos_pob['medias']['cp_sp'])} ± {_fmt(datos_pob['stds']['cp_sp'])}", f"{_fmt(datos_pob['medias']['sp_sp'])} ± {_fmt(datos_pob['stds']['sp_sp'])}"),
            (f"T-Test ({t['ttest_native'].replace(chr(10), ' ')})", f"p = {_fmt(datos_pob['p1'])}, d = {_fmt(datos_pob['d1'])}", ""),
            (f"T-Test ({t['ttest_train_cp'].replace(chr(10), ' ')})", f"p = {_fmt(datos_pob['p2'])}, d = {_fmt(datos_pob['d2'])}", ""),
            (f"T-Test ({t['ttest_train_sp'].replace(chr(10), ' ')})", f"p = {_fmt(datos_pob['p3'])}, d = {_fmt(datos_pob['d3'])}", "")
        ]

        for i, row in enumerate(rows_pob):
            row_idx = i + 2
            c_izq = fmt_bottom_izq if i == len(rows_pob) - 1 else fmt_celda_izq
            c_cen = fmt_bottom if i == len(rows_pob) - 1 else fmt_celda
            ws_pob.write(row_idx, 0, row[0], c_izq)
            ws_pob.write(row_idx, 1, row[1], c_cen)
            ws_pob.write(row_idx, 2, row[2], c_cen)

        def get_mc(key, field):
            val = datos_mcnemar.get(key, {}).get(field, np.nan)
            return _fmt(val) if field == 'p' else str(val)

        ws_mc = workbook.add_worksheet('McNemar')
        ws_mc.set_column('A:A', 35)
        ws_mc.set_column('B:D', 20)
        ws_mc.write(0, 0, '', fmt_top_left)
        ws_mc.merge_range(0, 1, 0, 3, t['mc_header'].replace('\n', ' '), fmt_spanner)
        ws_mc.write(1, 0, "Scenario", fmt_header_izq)
        ws_mc.write(1, 1, t['mc_only_cp'].replace('\n', ' '), fmt_header)
        ws_mc.write(1, 2, t['mc_only_sp'].replace('\n', ' '), fmt_header)
        ws_mc.write(1, 3, t['mc_pval'], fmt_header)

        rows_mc = [
            (t['mc_row_native'], get_mc('nativo', 'A'), get_mc('nativo', 'B'), get_mc('nativo', 'p')),
            (t['mc_row_shift_cp'], get_mc('test_cp', 'A'), get_mc('test_cp', 'B'), get_mc('test_cp', 'p')),
            (t['mc_row_shift_sp'], get_mc('test_sp', 'A'), get_mc('test_sp', 'B'), get_mc('test_sp', 'p'))
        ]

        for i, row in enumerate(rows_mc):
            row_idx = i + 2
            c_izq = fmt_bottom_izq if i == len(rows_mc) - 1 else fmt_celda_izq
            c_cen = fmt_bottom if i == len(rows_mc) - 1 else fmt_celda
            ws_mc.write(row_idx, 0, row[0], c_izq)
            ws_mc.write(row_idx, 1, row[1], c_cen)
            ws_mc.write(row_idx, 2, row[2], c_cen)
            try:
                if float(row[3]) < 0.05:
                    c_pval = (
                        workbook.add_format({'bottom': 1, 'align': 'center', 'valign': 'vcenter', 'bold': True})
                        if i == len(rows_mc) - 1 else
                        workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True})
                    )
                    ws_mc.write(row_idx, 3, row[3], c_pval)
                else:
                    ws_mc.write(row_idx, 3, row[3], c_cen)
            except Exception:
                ws_mc.write(row_idx, 3, row[3], c_cen)


# ==============================================================================
# 5. MAIN ORCHESTRATOR WITH MULTI-LANGUAGE SUPPORT
# ==============================================================================
def process_and_generate_complete_tables(base_folders: list, save_dir: str, max_n: int = 3, language: str = 'en'):
    lang_code = normalize_language(language)
    t = TEXTOS_I18N[lang_code]
    
    print(f"\n🌍 Selected output language: {lang_code.upper()}")

    configurations = [
        ('f2_score', False),
        ('mel_f2', True)
    ]

    multiclass_labels = ["NV", "MEL", "BKL", "BCC", "AKIEC", "VASC", "DF"]

    for metric_name, binary_mode in configurations:
        print(f"\n{'=' * 70}\n🔬 PROCESSING AND GENERATING FIGURES FOR: {metric_name.upper()}\n{'=' * 70}")
        
        df = obtener_datos_limpios(base_folders, metric_name)
        
        if df.empty:
            print(f"⚠️ Warning: No evaluation records found for '{metric_name}'.")
            continue
            
        for arq in df['Arquitectura'].unique():
            if arq == 'Otro':
                continue
            
            df_arq = df[df['Arquitectura'] == arq]
            
            all_cp_models = sorted(df_arq[df_arq['Entrenado_Con'] == 'Con Pelos']['Modelo'].unique())
            all_sp_models = sorted(df_arq[df_arq['Entrenado_Con'] == 'Sin Pelos']['Modelo'].unique())
            
            allowed_cp_models = all_cp_models[:max_n]
            allowed_sp_models = all_sp_models[:max_n]

            n_cp_actual = len(allowed_cp_models)
            n_sp_actual = len(allowed_sp_models)
            print(f"   ➤ {arq} | Validated N -> Train With Hair: {n_cp_actual}/{max_n} | Train Without Hair: {n_sp_actual}/{max_n}")
            
            if n_cp_actual < max_n or n_sp_actual < max_n:
                print("      ⚠️ WARNING: Incomplete sample group. Check if any run failed survival thresholds.")
            
            df_arq = df_arq[df_arq['Modelo'].isin(list(allowed_cp_models) + list(allowed_sp_models))]
            
            df_test = df_arq[df_arq['Fase'] == 'TEST']
            df_val = df_arq[df_arq['Fase'] == 'VAL']
            
            g1_df = df_test[(df_test['Entrenado_Con'] == 'Con Pelos') & (df_test['Testeado_En'] == 'Con Pelos')].dropna(subset=['Valor_Metrica'])
            g2_df = df_test[(df_test['Entrenado_Con'] == 'Con Pelos') & (df_test['Testeado_En'] == 'Sin Pelos')].dropna(subset=['Valor_Metrica'])
            g3_df = df_test[(df_test['Entrenado_Con'] == 'Sin Pelos') & (df_test['Testeado_En'] == 'Con Pelos')].dropna(subset=['Valor_Metrica'])
            g4_df = df_test[(df_test['Entrenado_Con'] == 'Sin Pelos') & (df_test['Testeado_En'] == 'Sin Pelos')].dropna(subset=['Valor_Metrica'])

            g1 = g1_df['Valor_Metrica']
            g2 = g2_df['Valor_Metrica']
            g3 = g3_df['Valor_Metrica']
            g4 = g4_df['Valor_Metrica']

            if min(len(g1), len(g2), len(g3), len(g4)) == 0:
                continue

            p1, d1 = calc_population_stats(g1.values, g4.values, paired=False)
            p2, d2 = calc_population_stats(
                g1.values, g2.values, paired=True,
                seeds_a=g1_df['Semilla'].values, seeds_b=g2_df['Semilla'].values
            )
            p3, d3 = calc_population_stats(
                g3.values, g4.values, paired=True,
                seeds_a=g3_df['Semilla'].values, seeds_b=g4_df['Semilla'].values
            )

            datos_pob = {
                'medias': {'cp_cp': g1.mean(), 'cp_sp': g2.mean(), 'sp_cp': g3.mean(), 'sp_sp': g4.mean()},
                'stds':   {'cp_cp': g1.std(),  'cp_sp': g2.std(),  'sp_cp': g3.std(),  'sp_sp': g4.std()},
                'p1': p1, 'd1': d1, 'p2': p2, 'd2': d2, 'p3': p3, 'd3': d3,
            }

            val_cp_native = df_val[(df_val['Entrenado_Con'] == 'Con Pelos') & (df_val['Testeado_En'] == 'Con Pelos')]
            val_sp_native = df_val[(df_val['Entrenado_Con'] == 'Sin Pelos') & (df_val['Testeado_En'] == 'Sin Pelos')]
            
            if val_cp_native.empty or val_sp_native.empty:
                continue

            median_cp_model = select_median_model(val_cp_native, 'Valor_Metrica')
            median_sp_model = select_median_model(val_sp_native, 'Valor_Metrica')
            
            df_native_cp = df_test[df_test['Modelo'] == median_cp_model]
            df_native_sp = df_test[df_test['Modelo'] == median_sp_model]

            datos_mcnemar = {}
            for test_cond, key in [('Con Pelos', 'test_cp'), ('Sin Pelos', 'test_sp')]:
                p_A = df_native_cp[df_native_cp['Testeado_En'] == test_cond]
                p_B = df_native_sp[df_native_sp['Testeado_En'] == test_cond]
                if not p_A.empty and not p_B.empty:
                    path_A = p_A.iloc[0]['Ruta_Directorio']
                    path_B = p_B.iloc[0]['Ruta_Directorio']
                    
                    A_val, B_val, p_val = compute_mcnemar_discordances(path_A, path_B, binary_mode=binary_mode)
                    datos_mcnemar[key] = {'A': A_val, 'B': B_val, 'p': p_val}

                    if not binary_mode:
                        condensed_out = os.path.join(save_dir, f"{arq.replace(' ', '_')}_{metric_name}_Condensed_Delta_Matrix_{key}")
                        plot_condensed_delta_matrix(path_A, path_B, multiclass_labels, t['delta_title'].format(arq=arq), condensed_out, t)
                        
                        matrix_1x3_out = os.path.join(save_dir, f"{arq.replace(' ', '_')}_{metric_name}_Matrix_1x3_{key}")
                        plot_1x3_mcnemar_matrices(path_A, path_B, multiclass_labels, t['native_cp'], t['native_sp'], matrix_1x3_out, t)

            p_A_nat = df_native_cp[df_native_cp['Testeado_En'] == 'Con Pelos']
            p_B_nat = df_native_sp[df_native_sp['Testeado_En'] == 'Sin Pelos']
            if not p_A_nat.empty and not p_B_nat.empty:
                A_val, B_val, p_val = compute_mcnemar_discordances(
                    p_A_nat.iloc[0]['Ruta_Directorio'],
                    p_B_nat.iloc[0]['Ruta_Directorio'],
                    binary_mode=binary_mode
                )
                datos_mcnemar['nativo'] = {'A': A_val, 'B': B_val, 'p': p_val}

            combined_out = os.path.join(save_dir, f"{arq.replace(' ', '_')}_{metric_name}_Combined_Table")
            generate_combined_table_figure(datos_pob, datos_mcnemar, arq, metric_name, combined_out, t)
            export_elsevier_excel_tables(datos_pob, datos_mcnemar, arq, metric_name, save_dir, t)
            
            print(f"   ✅ Tables and figures exported in [{lang_code.upper()}] successfully.")


if __name__ == "__main__":
    # Scans the centralized trained_models directory recursively (no per-model paths needed)
    experiment_directories = [
        MODELS_DIR
    ]

    # Saves generated tables and figures inside the repository's Datasets/HAM10000/PAPER_FIGURES_VERIFY folder
    figures_output_dir = os.path.join(DOCS_DIR, "HAM10000", "PAPER_FIGURES_VERIFY")
    os.makedirs(figures_output_dir, exist_ok=True)

    print(f"📂 Scanning models in: {MODELS_DIR}")
    print(f"📊 Exporting figures and tables to: {figures_output_dir}")

    # Supported language options: 'en' (English), 'es' (Spanish), 'pt' (Portuguese)
    process_and_generate_complete_tables(experiment_directories, figures_output_dir, max_n=5, language='en')