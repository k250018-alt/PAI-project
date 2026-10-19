"""
generate_calculation_visuals.py
Generates visual diagrams and charts illustrating how Random Forest
calculates decisions in Gin Rummy AI:
  1. Shannon Entropy curve & Information Gain reduction
  2. Decision tree split architecture (Parent -> Left & Right child nodes)
  3. Ensemble 10-Tree Bagging & Majority Voting trace
  4. 11-Card Discard Decision composite scoring breakdown
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set global styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def create_entropy_and_tree_split_visual():
    """Figure 1: Mathematical Entropy & Tree Node Splitting Mechanism."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # -------------------------------------------------------------
    # Left: Shannon Entropy Curve
    # -------------------------------------------------------------
    p = np.linspace(0.0001, 0.9999, 500)
    h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    ax1.plot(p, h, color='#1f77b4', lw=2.5, label=r'Entropy $H(p) = -p \log_2(p) - (1-p) \log_2(1-p)$')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Max Impurity (H = 1.0 bit)')
    
    # Highlight Parent node: p=0.30 (3 discard out of 10)
    p_parent = 0.30
    h_parent = - (0.3 * np.log2(0.3) + 0.7 * np.log2(0.7))
    ax1.scatter([p_parent], [h_parent], color='#d62728', s=120, zorder=5)
    ax1.annotate(f'Parent Node\np = 0.30\nH = {h_parent:.3f} bits', 
                 xy=(p_parent, h_parent), xytext=(p_parent + 0.08, h_parent + 0.08),
                 arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
                 fontsize=10, fontweight='bold', color='#d62728')

    # Highlight Left child: p=0.0 (pure Deck, H=0.0)
    ax1.scatter([0.0], [0.0], color='#2ca02c', s=120, zorder=5)
    ax1.annotate('Left Child (Pure Leaf)\np = 0.00\nH = 0.000 bits', 
                 xy=(0.0, 0.0), xytext=(0.05, 0.18),
                 arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.5),
                 fontsize=10, fontweight='bold', color='#2ca02c')

    # Highlight Right child: p=0.75 (3 discard out of 4)
    p_right = 0.75
    h_right = - (0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
    ax1.scatter([p_right], [h_right], color='#ff7f0e', s=120, zorder=5)
    ax1.annotate(f'Right Child\np = 0.75\nH = {h_right:.3f} bits', 
                 xy=(p_right, h_right), xytext=(p_right - 0.25, h_right - 0.15),
                 arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5),
                 fontsize=10, fontweight='bold', color='#ff7f0e')

    # Information gain bracket
    ax1.fill_between([0.2, 0.8], [0.3245, 0.3245], [h_parent, h_parent], color='#9467bd', alpha=0.15)
    ax1.text(0.5, 0.55, f'Information Gain:\nIG = 0.881 - 0.325\n= +0.557 bits', 
             ha='center', fontsize=10, fontweight='bold', color='#6f42c1',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#6f42c1', alpha=0.9))

    ax1.set_title('Shannon Entropy & Information Gain Curve', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel('Probability of Positive Class $p(y = \\text{Take Discard})$', fontsize=10)
    ax1.set_ylabel('Impurity $H(Y)$ [Bits]', fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower center', fontsize=9)

    # -------------------------------------------------------------
    # Right: Tree Node Split Flowchart Diagram
    # -------------------------------------------------------------
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Parent Box
    parent_box = patches.FancyBboxPatch((2.5, 7.2), 5.0, 2.2, boxstyle="round,pad=0.2",
                                        facecolor="#eef2f7", edgecolor="#1f77b4", linewidth=2)
    ax2.add_patch(parent_box)
    ax2.text(5.0, 8.8, "PARENT NODE (10 Samples)", ha="center", va="center", fontsize=11, fontweight="bold", color="#0b3c61")
    ax2.text(5.0, 8.3, "7 Deck (70%) | 3 Take Discard (30%)", ha="center", va="center", fontsize=9.5, color="#333333")
    ax2.text(5.0, 7.8, "Entropy H(Parent) = 0.881 bits", ha="center", va="center", fontsize=10, fontweight="bold", color="#d62728")

    # Decision Split Diamond / Annotation
    ax2.annotate("", xy=(3.0, 5.0), xytext=(4.0, 7.2),
                 arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))
    ax2.annotate("", xy=(7.0, 5.0), xytext=(6.0, 7.2),
                 arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))

    ax2.text(5.0, 6.2, "SPLIT CONDITION\n[ dw_gain > 4 ]", ha="center", va="center",
             fontsize=10, fontweight="bold", color="#111111",
             bbox=dict(boxstyle="square,pad=0.3", facecolor="#fff3cd", edgecolor="#ffeeba", lw=1.5))

    ax2.text(2.6, 5.8, "FALSE (<= 4)\n6 Samples", ha="center", fontsize=9, fontweight="bold", color="#2ca02c")
    ax2.text(7.4, 5.8, "TRUE (> 4)\n4 Samples", ha="center", fontsize=9, fontweight="bold", color="#ff7f0e")

    # Left Child (Pure Leaf)
    left_box = patches.FancyBboxPatch((0.5, 2.4), 4.0, 2.3, boxstyle="round,pad=0.2",
                                      facecolor="#e8f5e9", edgecolor="#2ca02c", linewidth=2)
    ax2.add_patch(left_box)
    ax2.text(2.5, 4.0, "LEFT CHILD (Pure Leaf)", ha="center", va="center", fontsize=10, fontweight="bold", color="#1b5e20")
    ax2.text(2.5, 3.4, "6 Deck (100%) | 0 Discard", ha="center", va="center", fontsize=9, color="#333333")
    ax2.text(2.5, 2.9, "H(Left) = 0.000 bits", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#2ca02c")

    # Right Child (Impure Node)
    right_box = patches.FancyBboxPatch((5.5, 2.4), 4.0, 2.3, boxstyle="round,pad=0.2",
                                       facecolor="#fff8e1", edgecolor="#ff7f0e", linewidth=2)
    ax2.add_patch(right_box)
    ax2.text(7.5, 4.0, "RIGHT CHILD (Branch)", ha="center", va="center", fontsize=10, fontweight="bold", color="#b26a00")
    ax2.text(7.5, 3.4, "1 Deck (25%) | 3 Discard (75%)", ha="center", va="center", fontsize=9, color="#333333")
    ax2.text(7.5, 2.9, "H(Right) = 0.811 bits", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#ff7f0e")

    # Bottom IG Summary Banner
    summary_box = patches.FancyBboxPatch((1.5, 0.4), 7.0, 1.4, boxstyle="round,pad=0.2",
                                         facecolor="#f3e5f5", edgecolor="#6f42c1", linewidth=1.5)
    ax2.add_patch(summary_box)
    ax2.text(5.0, 1.25, "INFORMATION GAIN COMPUTATION:", ha="center", fontsize=9.5, fontweight="bold", color="#4a148c")
    ax2.text(5.0, 0.75, "IG = 0.8813 - [(6/10)*0.000 + (4/10)*0.8113] = +0.5568 bits", 
             ha="center", fontsize=9.5, color="#111111", family="monospace")

    ax2.set_title("ID3 Decision Tree Split Architecture", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    out_file = os.path.join(RESULTS_DIR, "rf_entropy_split_visualization.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


def create_ensemble_voting_visual():
    """Figure 2: 10-Tree Bagging & Majority Voting Inference Visualization."""
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    trees = [f"Tree #{i+1}" for i in range(10)]
    # Actual votes from Decision 1 trace
    votes = [1, 0, 1, 1, 1, 1, 0, 1, 1, 1]  # 8 Discard (1), 2 Deck (0)
    vote_labels = ["Take Discard (1)" if v == 1 else "Deck (0)" for v in votes]
    colors = ['#28a745' if v == 1 else '#17a2b8' for v in votes]

    # Bar chart of individual tree votes
    y_pos = np.arange(len(trees))
    bars = ax.barh(y_pos, [1] * 10, color=colors, height=0.65, edgecolor='#333333', linewidth=1)

    # Feature subsets sampled for each tree
    sample_subsets = [
        "adj2, dw_gain, best_dw, top_num, ...",
        "turn_num, dw_gain, current_dw, best_dw, ...",
        "top_num, adj1, same_num, deck_size, ...",
        "best_dw, adj2, same_num, turn_num, ...",
        "num_unmelded, num_seqs, current_dw, ...",
        "dw_gain, num_sets, highest_unm, ...",
        "turn_num, current_dw, deck_size, ...",
        "same_num, dw_gain, adj1, top_num, ...",
        "num_seqs, best_dw, adj2, highest_unm, ...",
        "dw_gain, current_dw, turn_num, ...",
    ]

    for idx, (bar, label, sub) in enumerate(zip(bars, vote_labels, sample_subsets)):
        ax.text(0.04, bar.get_y() + bar.get_height()/2, f"{label}   [Subsampled: {sub}]",
                va='center', ha='left', fontsize=9.5, fontweight='bold',
                color='white' if label.startswith("Take") else '#111111')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(trees, fontsize=10, fontweight='bold')
    ax.invert_yaxis()  # Top tree first
    ax.set_xlim(0, 1.35)
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Tally result box on the right
    tally_box = patches.FancyBboxPatch((1.05, 1.8), 0.28, 5.5, boxstyle="round,pad=0.04",
                                       facecolor="#f8f9fa", edgecolor="#28a745", linewidth=2.5)
    ax.add_patch(tally_box)
    ax.text(1.19, 6.8, "MAJORITY VOTE", ha="center", fontsize=10.5, fontweight="bold", color="#155724")
    ax.text(1.19, 5.8, "Take Discard (1):\n8 / 10 Trees (80%)", ha="center", fontsize=9.5, fontweight="bold", color="#28a745")
    ax.text(1.19, 4.6, "Draw Deck (0):\n2 / 10 Trees (20%)", ha="center", fontsize=9.5, fontweight="bold", color="#17a2b8")
    ax.text(1.19, 3.4, "FINAL OUTPUT:\nClass 1 (Take)", ha="center", fontsize=10.5, fontweight="bold", color="#155724",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4edda", edgecolor="#c3e6cb"))

    ax.set_title("Random Forest Ensemble Inference: 10-Tree Bagging & Majority Vote Tally", 
                 fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    out_file = os.path.join(RESULTS_DIR, "rf_ensemble_majority_voting.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


def create_discard_scoring_visual():
    """Figure 3: 11-Card Discard Decision Evaluation & Composite Score Breakdown."""
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=300)

    cards = ['S4', 'S7', 'D7', 'S8', 'H8', 'S10', 'H4', 'H5', 'H6', 'S13', 'C7']
    scores = [1604, -3854, -3846, 1752, 1760, 1830, -3779, -3706, -3669, 1830, -3846]
    in_melds = [False, True, True, False, False, False, True, True, True, False, True]

    colors = []
    for c, s, m in zip(cards, scores, in_melds):
        if m:
            colors.append('#dc3545')  # Protected meld card
        elif c in ['S10', 'S13']:
            colors.append('#ffc107' if c == 'S13' else '#28a745')  # Chosen top candidate
        else:
            colors.append('#6c757d')

    x = np.arange(len(cards))
    bars = ax.bar(x, scores, color=colors, width=0.62, edgecolor='#333333', linewidth=1)

    ax.axhline(0, color='black', lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({'MELD' if m else 'DEADWOOD'})" for c, m in zip(cards, in_melds)], 
                       fontsize=9.5, fontweight='bold')

    # Label score values
    for bar, score in zip(bars, scores):
        y_val = bar.get_height()
        va = 'bottom' if y_val >= 0 else 'top'
        y_offset = 60 if y_val >= 0 else -180
        ax.text(bar.get_x() + bar.get_width()/2, y_val + y_offset, f"{score}",
                ha='center', va=va, fontsize=8.5, fontweight='bold',
                color='#155724' if score == 1830 else ('#721c24' if score < 0 else '#333333'))

    # Callout for protected melds
    ax.annotate("MELDDED CARDS PROTECTED (-5000 Penalty)\nNever break H4-H5-H6 or S7-D7-C7",
                xy=(1.5, -3850), xytext=(0.5, -2400),
                arrowprops=dict(arrowstyle="->", color="#dc3545", lw=1.5),
                fontsize=9.5, fontweight="bold", color="#721c24",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8d7da", edgecolor="#f5c6cb"))

    # Callout for chosen discard card S10
    ax.annotate("★ SELECTED DISCARD: S10 (Score: 1830)\n- High isolated deadwood (10 pts)\n- Retains melds & low deadwood (30 pts)",
                xy=(5.0, 1830), xytext=(4.0, 2600),
                arrowprops=dict(arrowstyle="->", color="#28a745", lw=1.8),
                fontsize=10, fontweight="bold", color="#155724",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#d4edda", edgecolor="#c3e6cb"))

    ax.set_ylabel("Composite Discard Score", fontsize=11, fontweight="bold")
    ax.set_title("Decision 2: 11-Card Discard Candidate Scoring Breakdown", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylim(-4800, 3400)
    ax.grid(True, axis='y', linestyle=':', alpha=0.5)

    plt.tight_layout()
    out_file = os.path.join(RESULTS_DIR, "rf_discard_scoring_visualization.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    create_entropy_and_tree_split_visual()
    create_ensemble_voting_visual()
    create_discard_scoring_visual()
    print("All calculation visualizations generated successfully!")
