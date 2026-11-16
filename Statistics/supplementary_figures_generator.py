"""
=========================================================================================
MODULE: Supplementary PDF Confusion Matrices Compiler (Individual High-Res Layout)
        (statistics/generate_supplementary_pdf.py)
=========================================================================================
DESCRIPTION:
    Compiles all granular multiclass confusion matrices across all independent training 
    runs (N=5) and cross-domain evaluation scenarios into a single publication-ready 
    Supplementary PDF document following Elsevier guidelines. Each of the 200 matrices 
    is rendered individually with its explicit descriptive caption and optimized spacing,
    retaining the normalization and styling standards from the core statistical module.
=========================================================================================
"""

import os
import sys
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ==============================================================================
# 0. ROBUST PATH CONFIGURATION & MODULE RESOLUTION
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MAIN_CODE_DIR = os.path.join(PROJECT_ROOT, "Main_code")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "Tools")

for p in (CURRENT_DIR, PROJECT_ROOT, MAIN_CODE_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import config_paths
from clean_data_extractor import get_clean_evaluation_data as obtener_datos_limpios

MODELS_DIR = getattr(config_paths, "MODELS_DIR", getattr(config_paths, "REPO_DIR"))
DOCS_DIR = getattr(config_paths, "DOCS_DIR")

# Import ReportLab
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


# ==============================================================================
# 1. HELPER FUNCTIONS FOR SEED & MATRIX PATH RESOLUTION
# ==============================================================================

def resolve_model_root_dir(eval_dir_path: str, model_folder_name: str) -> str:
    norm_path = os.path.normpath(eval_dir_path)
    parts = norm_path.split(os.sep)
    if model_folder_name in parts:
        idx = parts.index(model_folder_name)
        return os.sep.join(parts[:idx + 1])
    for token in ("Verificacion_Cruzada", "Cross_Domain_Evaluation", "Resultados_", "TEST_Results", "VAL_Results"):
        if token in norm_path:
            norm_path = norm_path.split(token)[0]
    return norm_path.rstrip(os.sep)


def extract_model_seed(eval_dir_path: str, model_folder_name: str) -> str:
    model_root = resolve_model_root_dir(eval_dir_path, model_folder_name)
    if not os.path.exists(model_root):
        return "N/A"

    try:
        all_filenames = " ".join(os.listdir(model_root))
        match = re.search(r"seed[_:\s]*(\d+)", all_filenames, re.IGNORECASE)
        if match:
            return match.group(1)
    except Exception:
        pass

    info_files = (
        glob.glob(os.path.join(model_root, "info_start*.txt")) +
        glob.glob(os.path.join(model_root, "info_inicio*.txt")) +
        glob.glob(os.path.join(model_root, "info_param*.txt"))
    )
    for info_file in info_files:
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                content = f.read()
                match = re.search(r"seed['\"\s:]*(\d+)", content, re.IGNORECASE)
                if match:
                    return match.group(1)
        except Exception:
            pass

    return "N/A"


def get_predictions(probs_path: str) -> np.ndarray:
    probs = np.load(probs_path)
    if len(probs.shape) > 1 and probs.shape[1] > 1:
        return np.argmax(probs, axis=1)
    return (probs >= 0.5).astype(int)


def generate_high_res_normalized_confusion_matrix(eval_dir_path: str) -> str:
    """
    Genera la matriz de confusión normalizada (Recall) manteniendo exactamente 
    el diseño visual (mapa de calor en escala de azules, fuentes grandes, 
    porcentajes normalizados y conteos absolutos combinados) y eliminando 
    los bordes blancos sobrantes para inserción limpia en ReportLab.
    """
    if not eval_dir_path or not os.path.exists(eval_dir_path):
        return None
        
    labels_path = os.path.join(eval_dir_path, "labels.npy")
    probs_path = os.path.join(eval_dir_path, "probs.npy")
    
    if os.path.exists(labels_path) and os.path.exists(probs_path):
        try:
            y_true = np.load(labels_path)
            if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                y_true = np.argmax(y_true, axis=1)
            preds = get_predictions(probs_path)
            
            # Cálculo exacto de matrices normalizadas (Recall) y absolutas
            cm_norm = confusion_matrix(y_true, preds, normalize='true')
            cm_abs = confusion_matrix(y_true, preds)
            class_labels = ["NV", "MEL", "BKL", "BCC", "AKIEC", "VASC", "DF"]
            
            fig, ax = plt.subplots(figsize=(6, 5))
            fig.patch.set_facecolor('white')

            im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
            
            ax.set_xticks(np.arange(len(class_labels)))
            ax.set_yticks(np.arange(len(class_labels)))
            # Fuentes grandes y legibles para las clases
            ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=12, fontweight='bold')
            ax.set_yticklabels(class_labels, fontsize=12, fontweight='bold')
            
            ax.set_ylabel("True Pathology", fontweight='bold', fontsize=13)
            ax.set_xlabel("Predicted Pathology", fontweight='bold', fontsize=13)

            threshold = np.max(cm_norm) / 2.0
            for i in range(len(class_labels)):
                for j in range(len(class_labels)):
                    val_pct = cm_norm[i, j]
                    val_abs = cm_abs[i, j]
                    txt = f"{val_pct:.1%}\n({int(val_abs)})"
                    text_color = "white" if val_pct > threshold else "black"
                    # Fuentes grandes para los valores internos de la matriz
                    ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

            # Eliminación estricta de márgenes blancos innecesarios
            plt.tight_layout()
            out_png_path = os.path.join(eval_dir_path, "confusion_matrix_test_norm_highres.png")
            plt.savefig(out_png_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close(fig)
            return out_png_path
        except Exception:
            pass

    # Fallback a búsqueda tradicional si no se pudo generar al vuelo
    for candidate in ("confusion_matrix_test.png", "matriz_confusion_test.png"):
        full_path = os.path.join(eval_dir_path, candidate)
        if os.path.exists(full_path):
            return full_path
    return None


# ==============================================================================
# 2. PDF COMPILER USING REPORTLAB (CONTROLLED LAYOUT & KEEP-TOGETHER)
# ==============================================================================

def compile_supplementary_pdf(base_folders: list, output_pdf_path: str, max_n: int = 5):
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, leading=20, alignment=1, spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, alignment=1, spaceAfter=20
    )
    metric_heading = ParagraphStyle(
        'MetricHeading', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=14, leading=18, textColor=colors.HexColor("#002B49"), spaceBefore=15, spaceAfter=10, keepWithNext=True
    )
    arch_heading = ParagraphStyle(
        'ArchHeading', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, leading=16, textColor=colors.HexColor("#1A365D"), spaceBefore=10, spaceAfter=6, keepWithNext=True
    )
    caption_style = ParagraphStyle(
        'FigCaption', parent=styles['Normal'], fontName='Helvetica', fontSize=8.5, leading=11, alignment=1, spaceBefore=4, spaceAfter=8, keepWithNext=True
    )

    story = []

    # Document Header
    story.append(Paragraph("Supplementary Material: Granular Multiclass Confusion Matrices", title_style))
    story.append(Paragraph(
        "<b>Manuscript Title:</b> A statistical validation framework for evaluating artifact preprocessing impact in skin lesion classification.<br/>"
        "<b>Content:</b> Complete set of individual multiclass normalized confusion matrices across all independent training runs (N=5) and cross-domain evaluation scenarios.",
        subtitle_style
    ))
    story.append(PageBreak())

    configurations = [
        ('f2_score', 'Multiclass Performance (F2-Score)'),
        ('mel_f2', 'Oncological Sensitivity - Melanoma Detection (Mel F2-Score)')
    ]

    fig_counter = 1

    for metric_name, metric_label in configurations:
        story.append(Paragraph(f"<b>Evaluation Metric:</b> {metric_label}", metric_heading))
        story.append(Spacer(1, 5))

        df = obtener_datos_limpios(base_folders, metric_name)
        if df.empty:
            continue

        df_test = df[df['Fase'] == 'TEST']

        for arch in sorted(df['Arquitectura'].unique()):
            if arch == 'Otro':
                continue

            df_arch = df[df['Arquitectura'] == arch]
            mod_cp = sorted(df_arch[df_arch['Entrenado_Con'] == 'Con Pelos']['Modelo'].unique())[:max_n]
            mod_sp = sorted(df_arch[df_arch['Entrenado_Con'] == 'Sin Pelos']['Modelo'].unique())[:max_n]

            story.append(Paragraph(f"Architecture: <b>{arch}</b>", arch_heading))

            for row_idx in range(max_n):
                modelo_cp = mod_cp[row_idx] if row_idx < len(mod_cp) else None
                modelo_sp = mod_sp[row_idx] if row_idx < len(mod_sp) else None

                path_native_cp, path_cross_sp = None, None
                path_native_sp, path_cross_cp = None, None
                seed_cp, seed_sp = "N/A", "N/A"

                if modelo_cp:
                    sub_cp = df_test[df_test['Modelo'] == modelo_cp]
                    if not sub_cp.empty:
                        seed_cp = extract_model_seed(sub_cp.iloc[0]['Ruta_Directorio'], modelo_cp)
                    f_nat_cp = sub_cp[sub_cp['Testeado_En'] == 'Con Pelos']
                    if not f_nat_cp.empty:
                        path_native_cp = generate_high_res_normalized_confusion_matrix(f_nat_cp.iloc[0]['Ruta_Directorio'])
                    f_cro_sp = sub_cp[sub_cp['Testeado_En'] == 'Sin Pelos']
                    if not f_cro_sp.empty:
                        path_cross_sp = generate_high_res_normalized_confusion_matrix(f_cro_sp.iloc[0]['Ruta_Directorio'])

                if modelo_sp:
                    sub_sp = df_test[df_test['Modelo'] == modelo_sp]
                    if not sub_sp.empty:
                        seed_sp = extract_model_seed(sub_sp.iloc[0]['Ruta_Directorio'], modelo_sp)
                    f_nat_sp = sub_sp[sub_sp['Testeado_En'] == 'Sin Pelos']
                    if not f_nat_sp.empty:
                        path_native_sp = generate_high_res_normalized_confusion_matrix(f_nat_sp.iloc[0]['Ruta_Directorio'])
                    f_cro_cp = sub_sp[sub_sp['Testeado_En'] == 'Con Pelos']
                    if not f_cro_cp.empty:
                        path_cross_cp = generate_high_res_normalized_confusion_matrix(f_cro_cp.iloc[0]['Ruta_Directorio'])

                scenarios = [
                    (path_native_cp, f"Run #{row_idx + 1} (Seed: {seed_cp}) — Trained: With Hair | Tested: With Hair (Native Raw)"),
                    (path_cross_sp, f"Run #{row_idx + 1} (Seed: {seed_cp}) — Trained: With Hair | Tested: Without Hair (Cross-Domain)"),
                    (path_native_sp, f"Run #{row_idx + 1} (Seed: {seed_sp}) — Trained: Without Hair | Tested: Without Hair (Native Clean)"),
                    (path_cross_cp, f"Run #{row_idx + 1} (Seed: {seed_sp}) — Trained: Without Hair | Tested: With Hair (Cross-Domain)")
                ]

                for img_path, caption_text in scenarios:
                    if img_path and os.path.exists(img_path):
                        # Dimensiones optimizadas para mantener proporción y permitir lectura perfecta
                        img = RLImage(img_path, width=220, height=175)
                        caption = Paragraph(f"<b>Figure S{fig_counter}.</b> {caption_text} [{metric_label}, {arch}].", caption_style)
                        
                        # Agrupación estricta para evitar que la imagen y su epígrafe se separen en páginas distintas
                        story.append(KeepTogether([img, caption, Spacer(1, 6)]))
                        fig_counter += 1

            story.append(PageBreak())

    doc = SimpleDocTemplate(output_pdf_path, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    doc.build(story)
    print(f"✅ Supplementary PDF successfully generated at: {output_pdf_path}")


if __name__ == "__main__":
    experiment_directories = [MODELS_DIR]
    pdf_output_path = os.path.join(DOCS_DIR, "HAM10000", "PAPER_FIGURES_VERIFY", "Supplementary_Material_Confusion_Matrices.pdf")
    os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)

    print(f"📂 Scanning models in: {MODELS_DIR}")
    print(f"📄 Compiling Supplementary PDF to: {pdf_output_path}")

    compile_supplementary_pdf(experiment_directories, pdf_output_path, max_n=5)