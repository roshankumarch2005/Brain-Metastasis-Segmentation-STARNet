import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'

OUT_DIR = "/mnt/DATA/EE23B009/EE23B009_BTP/results/presentation_figures/latex_eqs"
os.makedirs(OUT_DIR, exist_ok=True)

def render_big_equation(latex_str, filename, fontsize=24, figsize=(15, 1.2), color='#0A2540'):
    """
    Renders pure mathematical formula at maximum legibility and scale.
    No vertical crowding; full single-line banner.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    fig.patch.set_alpha(0.0)
    ax.axis('off')
    ax.text(0.5, 0.5, latex_str, fontsize=fontsize, ha='center', va='center', color=color)
    out_path = os.path.join(OUT_DIR, filename)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.03, transparent=True)
    plt.close(fig)
    
    # Auto-tight crop with balanced padding
    im = Image.open(out_path)
    arr = np.array(im)
    alpha = arr[:, :, 3]
    nonzero = np.argwhere(alpha > 0)
    if len(nonzero) > 0:
        ymin, xmin = nonzero.min(axis=0)
        ymax, xmax = nonzero.max(axis=0)
        pad = 6
        ymin = max(0, ymin - pad)
        xmin = max(0, xmin - pad)
        ymax = min(im.height, ymax + pad)
        xmax = min(im.width, xmax + pad)
        cropped = im.crop((xmin, ymin, xmax, ymax))
        cropped.save(out_path)
    print(f"{filename}: size={cropped.size}, aspect={cropped.size[0]/cropped.size[1]:.2f}")
    return out_path

# ==============================================================================
# 1. Slide 3: Dice & HD95 Clinical Metrics Banner (BIG & PROMINENT)
# ==============================================================================
eq_s3_banner = r'$\mathrm{DSC} = \frac{2|\mathbf{P} \cap \mathbf{Y}|}{|\mathbf{P}| + |\mathbf{Y}|} = \frac{2\,\mathrm{TP}}{2\,\mathrm{TP} + \mathrm{FP} + \mathrm{FN}} \qquad\qquad \mathrm{HD}_{95} = \mathrm{P}_{95}\left(\min_{y \in \partial\mathbf{Y}} \|\mathbf{p} - \mathbf{y}\|_2\right) \quad [\mathrm{mm}]$'
render_big_equation(eq_s3_banner, "eq_s3_metrics_banner.png", fontsize=23, figsize=(15, 1.2))

# ==============================================================================
# 2. Slide 5: Preprocessing Mechanics (N4 Bias & NLM Denoising) Banner (BIG)
# ==============================================================================
eq_s5_banner = r'$S(\mathbf{x}) = \mathbf{u}(\mathbf{x}) \cdot B(\mathbf{x}) + \mathbf{n}(\mathbf{x}) \qquad\qquad \hat{\mathbf{u}}(\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \sum_{\mathbf{y} \in \mathcal{N}} w(\mathbf{x}, \mathbf{y}) S(\mathbf{y})$'
render_big_equation(eq_s5_banner, "eq_s6_preprocessing.png", fontsize=23, figsize=(15, 1.2))

# Slide 5 separate equations
render_big_equation(r'$S(\mathbf{x}) = \mathbf{u}(\mathbf{x}) \cdot B(\mathbf{x}) + \mathbf{n}(\mathbf{x})$', "eq_s6_n4.png", fontsize=22, figsize=(6, 1))
render_big_equation(r'$\mathbf{u}(\mathbf{x}) = \frac{1}{Z(\mathbf{x})} \sum_{\mathbf{y} \in \Omega} w(\mathbf{x}, \mathbf{y}) \, \mathbf{v}(\mathbf{y}), \quad w(\mathbf{x}, \mathbf{y}) \propto \exp\left(-\frac{\|\mathbf{v}(N_{\mathbf{x}}) - \mathbf{v}(N_{\mathbf{y}})\|^2}{h^2}\right)$', "eq_s6_nlm.png", fontsize=20, figsize=(9, 1.1))

# ==============================================================================
# 3. Slide 10: Step 1: Baseline Compound Loss (BIG, CLEAN & NO GLYPH BUG)
# ==============================================================================
eq_s11 = r'$\mathcal{L}_{\mathrm{base}}(\mathbf{Y}, \hat{\mathbf{P}}) = \mathcal{L}_{\mathrm{Dice}} + \mathcal{L}_{\mathrm{CE}} = \left( 1 - \frac{2 \sum Y_i \hat{P}_i + \epsilon}{\sum Y_i + \sum \hat{P}_i + \epsilon} \right) - \frac{1}{|\Omega|} \sum_{i \in \Omega} \sum_{k=1}^K Y_{i,k} \log \hat{P}_{i,k}$'
render_big_equation(eq_s11, "eq_s11_baseline.png", fontsize=23, figsize=(15, 1.2))

# ==============================================================================
# 4. Slide 11: SIAM3D Parameter-Free Spatial Gating (BIG)
# ==============================================================================
eq_s12 = r'$\mathbf{F}^\prime = \mathbf{x} + \mathbf{x} \odot \mathbf{A}_{\mathrm{SIAM}}, \quad \mathrm{where} \;\; \mathbf{A}_{\mathrm{SIAM}} = \mathrm{clamp}(\mathbf{m}_d, 0, 1) + \mathrm{clamp}(\mathbf{m}_h, 0, 1) + \mathrm{clamp}(\mathbf{m}_w, 0, 1)$'
render_big_equation(eq_s12, "eq_s12_siam.png", fontsize=22, figsize=(15, 1.2))

# ==============================================================================
# 5. Slide 13: Step 3: Balanced Area-Aware Loss (BIG)
# ==============================================================================
eq_s14 = r'$\mathcal{L}_{\mathrm{BAA}} = \mathcal{L}_{\mathrm{Dice}} + \delta(V) \cdot \mathcal{L}_{\mathrm{CE}}, \quad \mathrm{where} \;\; \delta(V) = 1.0 + \beta \cdot \tanh\left( \left| \log \frac{V_{\mathrm{pred}} + 1.0}{V_{\mathrm{gt}} + 1.0} \right| \right)$'
render_big_equation(eq_s14, "eq_s14_baa.png", fontsize=23, figsize=(15, 1.2))

# ==============================================================================
# 6. Slide 15: Step 4: Inter-Level Quadratic Hinge Loss (BIG)
# ==============================================================================
eq_s16 = r'$\mathcal{L}_{\mathrm{inter}} = \sum_{k=1}^5 w_k \cdot \frac{1}{|\Omega|} \sum_{v \in \Omega} \left[ \max\left(0, \, |p_k^{\mathrm{up}}(v) - p_0^{\mathrm{detach}}(v)| - \mathrm{Tr}\right) \right]^2, \quad \mathrm{Tr} = 0.20$'
render_big_equation(eq_s16, "eq_s16_il_hinge.png", fontsize=22, figsize=(15, 1.2))

# ==============================================================================
# 7. Slide 18: Step 6: Master Flagship Synthesis Objective (BIG)
# ==============================================================================
eq_s19 = r'$\mathcal{L}_{\mathrm{STAR-Net}} = \mathcal{L}_{\mathrm{Dice}}(\mathbf{Y}, \hat{\mathbf{P}}) + \mathcal{L}_{\mathrm{CE}}(\mathbf{Y}, \hat{\mathbf{P}}) + \lambda_{\mathrm{BAA}} \mathcal{L}_{\mathrm{BAA}}(\mathbf{Y}, \hat{\mathbf{P}}) + \lambda_{\mathrm{inter}} \mathcal{L}_{\mathrm{inter}}(p_k^{\mathrm{up}}, p_0)$'
render_big_equation(eq_s19, "eq_s19_flagship.png", fontsize=23, figsize=(15, 1.2))

# ==============================================================================
# 8. Slide 22: Edge Hardware Viability (BIG)
# ==============================================================================
eq_s23 = r'$\mathrm{HR{-}Bypass}: \Omega_{\mathrm{Dec1}} = \frac{1}{8}\Omega_{\mathrm{Dec0}} \Rightarrow -22.1\% \;\mathrm{params} \quad\mid\quad \mathrm{3D\;Factorization}: \rho = \frac{1}{C_{\mathrm{out}}} + \frac{1}{27} \approx 4\% \;\mathrm{to}\; 7\% \Rightarrow -95.2\% \;\mathrm{params}$'
render_big_equation(eq_s23, "eq_s23_params.png", fontsize=20, figsize=(16, 1.2))

print("All large, high-legibility mathematical equation banners rendered successfully!")
