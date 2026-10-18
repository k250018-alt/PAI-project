"""
evaluate_models.py
Evaluation script for the Gin Rummy AI Random Forest models.

Runs:
  1. Accuracy and baseline comparisons
  2. Confusion matrices for the 3 decision models
  3. Feature importance analysis
  4. Win-rate simulations against Random and Greedy baselines
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "gin rummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from play import Play, Sorter, Hands, DiscardPile, AIRandomForestEngine
from cards import Deck
from RandomForest import Randomforest

BASE_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALL_CARDS = [f"{s}{n}" for s in "CDSH" for n in range(1, 14)]


# =========================================================
# SECTION 1: EVALUATION METRICS
# =========================================================

def precision_recall_f1(y_true, y_pred, pos_label=1):
    """Calculate precision, recall, F1 for a given positive label."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(precision * 100, 1), round(recall * 100, 1), round(f1 * 100, 1)


def accuracy(y_true, y_pred):
    """Simple accuracy percentage."""
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true) * 100, 1) if len(y_true) > 0 else 0.0


def balanced_accuracy(y_true, y_pred):
    """Balanced accuracy = mean of per-class recall."""
    classes = sorted(set(y_true))
    recalls = []
    for cls in classes:
        total = sum(1 for t in y_true if t == cls)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        recalls.append(correct / total if total > 0 else 0.0)
    return round(np.mean(recalls) * 100, 1)


def confusion_matrix(y_true, y_pred, labels):
    """Build a confusion matrix as a 2D numpy array."""
    n = len(labels)
    label_map = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_map and p in label_map:
            cm[label_map[t]][label_map[p]] += 1
    return cm


def plot_confusion_matrix(cm, labels, title, filename):
    """Plot and save a confusion matrix as a PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Greens', interpolation='nearest')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm[i, j] > cm.max() * 0.6 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13, color=color, fontweight='bold')

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def plot_feature_importance(features, importances, title, filename, top_n=15):
    """Plot feature importance bar chart."""
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [features[i] for i in indices]
    top_importances = [importances[i] for i in indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_features)))[::-1]
    bars = ax.barh(range(len(top_features)), top_importances, color=colors, edgecolor='darkgreen', linewidth=0.5)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (Info Gain Contribution)', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Add value labels
    for bar, val in zip(bars, top_importances):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


def estimate_feature_importance(model, X, y, feature_names):
    """
    Permutation-based feature importance:
    For each feature, shuffle it and measure accuracy drop.
    """
    base_acc = accuracy(y, model.predict(X))
    importances = []
    for i in range(X.shape[1]):
        X_perm = X.copy()
        np.random.shuffle(X_perm[:, i])
        perm_acc = accuracy(y, model.predict(X_perm))
        importances.append(max(0, (base_acc - perm_acc) / 100.0))
    # Normalize
    total = sum(importances) if sum(importances) > 0 else 1
    return [imp / total for imp in importances]


# =========================================================
# SECTION 2: EVALUATE ALL MODELS
# =========================================================

def evaluate_draw_model():
    """Evaluate Draw Decision model with baselines."""
    print("\n" + "=" * 60)
    print("DRAW MODEL EVALUATION")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASET_DIR, "draw_decision_dataset.csv")).dropna()
    X = df.drop(columns=["take_discard"]).values
    y = df["take_discard"].values.astype(int)
    feature_names = [c for c in df.columns if c != "take_discard"]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = Randomforest(min_split=4, depth=8, N_trees=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baselines
    baseline_always_deck = [0] * len(y_test)  # Always draw from deck
    baseline_dw_drop = [1 if row[-1] > 0 else 0 for row in X_test]  # Take if deadwood drops

    # Metrics
    rf_acc = accuracy(y_test, y_pred)
    rf_p, rf_r, rf_f1 = precision_recall_f1(y_test, y_pred, pos_label=1)
    bl1_acc = accuracy(y_test, baseline_always_deck)
    bl2_acc = accuracy(y_test, baseline_dw_drop)

    print(f"\n  Random Forest:          Acc={rf_acc}%  P={rf_p}%  R={rf_r}%  F1={rf_f1}%")
    print(f"  Baseline (always deck): Acc={bl1_acc}%")
    print(f"  Baseline (dw drops):    Acc={bl2_acc}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, [0, 1])
    plot_confusion_matrix(cm, ['Draw Deck', 'Take Discard'],
                          'Draw Model — Confusion Matrix', 'draw_confusion_matrix.png')

    # Feature importance
    importances = estimate_feature_importance(model, X_test, y_test, feature_names)
    plot_feature_importance(feature_names, importances,
                            'Draw Model — Feature Importance', 'draw_feature_importance.png')

    return {
        'Model': ['Random Forest', 'Always Deck', 'DW Drop Rule'],
        'Accuracy': [rf_acc, bl1_acc, bl2_acc],
        'Precision': [rf_p, '-', '-'],
        'Recall': [rf_r, '-', '-'],
        'F1': [rf_f1, '-', '-'],
    }


def evaluate_discard_model():
    """Evaluate Discard Decision model with baselines."""
    print("\n" + "=" * 60)
    print("DISCARD MODEL EVALUATION")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASET_DIR, "discard_decision_dataset.csv")).dropna()
    X = df.drop(columns=["best_discard"]).values
    y = df["best_discard"].values.astype(int)
    feature_names = [c for c in df.columns if c != "best_discard"]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = Randomforest(min_split=4, depth=8, N_trees=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baselines
    # Random baseline
    rng = random.Random(42)
    baseline_random = [rng.randint(0, 1) for _ in y_test]
    # Highest card baseline: label 1 if card_number is highest in its group
    baseline_highest = [1 if row[3] >= 10 else 0 for row in X_test]  # card_number column
    # Lowest DW baseline: label 1 if deadwood_reduction is max
    baseline_lowest_dw = [1 if row[9] > 0 else 0 for row in X_test]  # deadwood_reduction column

    rf_acc = accuracy(y_test, y_pred)
    bl_rand_acc = accuracy(y_test, baseline_random)
    bl_high_acc = accuracy(y_test, baseline_highest)
    bl_low_dw_acc = accuracy(y_test, baseline_lowest_dw)

    print(f"\n  Random Forest:              Acc={rf_acc}%")
    print(f"  Baseline (random):          Acc={bl_rand_acc}%")
    print(f"  Baseline (highest card):    Acc={bl_high_acc}%")
    print(f"  Baseline (lowest deadwood): Acc={bl_low_dw_acc}%")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, [0, 1])
    plot_confusion_matrix(cm, ['Not Best', 'Best Discard'],
                          'Discard Model — Confusion Matrix', 'discard_confusion_matrix.png')

    # Feature importance
    importances = estimate_feature_importance(model, X_test, y_test, feature_names)
    plot_feature_importance(feature_names, importances,
                            'Discard Model — Feature Importance', 'discard_feature_importance.png')

    return {
        'Model': ['Random Forest', 'Random', 'Highest Card', 'Lowest Deadwood'],
        'Accuracy': [rf_acc, bl_rand_acc, bl_high_acc, bl_low_dw_acc],
    }


def evaluate_knock_model():
    """Evaluate Knock/Gin Decision model with baselines."""
    print("\n" + "=" * 60)
    print("KNOCK/GIN MODEL EVALUATION")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATASET_DIR, "knock_gin_decision_dataset.csv")).dropna()
    drop_cols = [c for c in ["decision", "label_aux_win_prob_knock", "label_aux_win_prob_continue"] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    y = df["decision"].values.astype(int)
    feature_names = [c for c in df.columns if c not in drop_cols]

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    model = Randomforest(min_split=4, depth=8, N_trees=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Baselines
    # Always knock when allowed (dw <= 10)
    baseline_always_knock = []
    for row in X_test:
        dw = row[2]  # my_deadwood column
        if dw == 0:
            baseline_always_knock.append(2)
        elif dw <= 10:
            baseline_always_knock.append(1)
        else:
            baseline_always_knock.append(0)

    rf_acc = accuracy(y_test, y_pred)
    rf_bal_acc = balanced_accuracy(y_test, y_pred)
    bl_acc = accuracy(y_test, baseline_always_knock)
    bl_bal_acc = balanced_accuracy(y_test, baseline_always_knock)

    # Per-class F1 for macro F1
    classes = sorted(set(y_test))
    f1_scores = []
    for cls in classes:
        _, _, f1 = precision_recall_f1(y_test, y_pred, pos_label=cls)
        f1_scores.append(f1)
    macro_f1 = round(np.mean(f1_scores), 1)

    bl_f1_scores = []
    for cls in classes:
        _, _, f1 = precision_recall_f1(y_test, baseline_always_knock, pos_label=cls)
        bl_f1_scores.append(f1)
    bl_macro_f1 = round(np.mean(bl_f1_scores), 1)

    print(f"\n  Random Forest:            Acc={rf_acc}%  BalAcc={rf_bal_acc}%  MacroF1={macro_f1}%")
    print(f"  Baseline (always knock):  Acc={bl_acc}%  BalAcc={bl_bal_acc}%  MacroF1={bl_macro_f1}%")

    # Confusion matrix
    label_names = ['Continue', 'Knock', 'Gin']
    actual_labels = sorted(set(y_test) | set(y_pred))
    actual_names = [label_names[i] if i < len(label_names) else str(i) for i in actual_labels]
    cm = confusion_matrix(y_test, y_pred, actual_labels)
    plot_confusion_matrix(cm, actual_names,
                          'Knock/Gin Model — Confusion Matrix', 'knock_confusion_matrix.png')

    # Feature importance
    importances = estimate_feature_importance(model, X_test, y_test, feature_names)
    plot_feature_importance(feature_names, importances,
                            'Knock/Gin Model — Feature Importance', 'knock_feature_importance.png')

    return {
        'Model': ['Random Forest', 'Always Knock'],
        'Accuracy': [rf_acc, bl_acc],
        'BalancedAcc': [rf_bal_acc, bl_bal_acc],
        'MacroF1': [macro_f1, bl_macro_f1],
    }


# =========================================================
# SECTION 3: WIN-RATE SIMULATION
# =========================================================

class RandomPlayer:
    """Baseline: random draw, random discard, never knocks."""
    def choose_draw(self, hand, top_discard, deck_size):
        return random.choice([True, False]) if top_discard else False

    def choose_discard(self, hand):
        return random.choice(hand) if hand else None

    def should_knock(self, hand):
        return False  # Never knocks


class GreedyPlayer:
    """Baseline: always minimizes deadwood greedily, always knocks when allowed."""
    def choose_draw(self, hand, top_discard, deck_size):
        if not top_discard:
            return False
        dw_now = Sorter.calculate_cards_deadwood(hand)
        candidate = hand + [top_discard]
        best_dw = min(Sorter.calculate_cards_deadwood([c for c in candidate if c != d]) for d in candidate)
        return best_dw < dw_now

    def choose_discard(self, hand):
        if not hand:
            return None
        return min(hand, key=lambda c: Sorter.calculate_cards_deadwood([x for x in hand if x != c]))

    def should_knock(self, hand):
        dw = Sorter.calculate_cards_deadwood(hand)
        return dw <= 10


class RFPlayer:
    """The AI Random Forest player (same logic as the game)."""
    def __init__(self):
        self.engine = AIRandomForestEngine()
        self.turn = 1

    def choose_draw(self, hand, top_discard, deck_size):
        return self.engine.predict_draw(self.turn, deck_size, hand, top_discard)

    def choose_discard(self, hand):
        return self.engine.predict_discard(hand, self.turn)

    def should_knock(self, hand):
        dw = Sorter.calculate_cards_deadwood(hand)
        return self.engine.predict_knock(dw, hand, self.turn, 20)


def simulate_game(player1, player2, max_turns=50):
    """
    Simulate one complete game between two AI players.
    Returns (p1_score, p2_score, p1_final_dw, p2_final_dw, turns_played)
    """
    deck_cards = ALL_CARDS.copy()
    random.shuffle(deck_cards)

    hand1 = deck_cards[:10]
    hand2 = deck_cards[10:20]
    deck = deck_cards[20:]
    discard_pile = [deck.pop()] if deck else []

    for turn in range(max_turns):
        for player_idx, (player, hand) in enumerate([(player1, hand1), (player2, hand2)]):
            # Draw
            top_disc = discard_pile[-1] if discard_pile else None
            take_disc = player.choose_draw(hand, top_disc, len(deck))

            if take_disc and top_disc:
                hand.append(discard_pile.pop())
            else:
                if deck:
                    hand.append(deck.pop())
                elif discard_pile:
                    # Recycle
                    top = discard_pile.pop()
                    random.shuffle(discard_pile)
                    deck = discard_pile
                    discard_pile = [top]
                    if deck:
                        hand.append(deck.pop())
                    else:
                        continue
                else:
                    continue

            # Discard
            if len(hand) > 10:
                disc = player.choose_discard(hand)
                if disc and disc in hand:
                    hand.remove(disc)
                    discard_pile.append(disc)

            # Check knock/gin
            dw = Sorter.calculate_cards_deadwood(hand)
            if player.should_knock(hand) and dw <= 10:
                opp_hand = hand2 if player_idx == 0 else hand1
                opp_dw = Sorter.calculate_cards_deadwood(opp_hand)

                if dw == 0:
                    score = opp_dw + 25
                    if player_idx == 0:
                        return score, 0, dw, opp_dw, turn + 1
                    else:
                        return 0, score, opp_dw, dw, turn + 1
                elif dw < opp_dw:
                    score = opp_dw - dw
                    if player_idx == 0:
                        return score, 0, dw, opp_dw, turn + 1
                    else:
                        return 0, score, opp_dw, dw, turn + 1
                else:
                    # Undercut
                    score = (dw - opp_dw) + 25
                    if player_idx == 0:
                        return 0, score, dw, opp_dw, turn + 1
                    else:
                        return score, 0, opp_dw, dw, turn + 1

        if hasattr(player1, 'turn'):
            player1.turn += 1

    # Timeout: compare deadwood
    dw1 = Sorter.calculate_cards_deadwood(hand1)
    dw2 = Sorter.calculate_cards_deadwood(hand2)
    if dw1 < dw2:
        return dw2 - dw1, 0, dw1, dw2, max_turns
    elif dw2 < dw1:
        return 0, dw1 - dw2, dw1, dw2, max_turns
    return 0, 0, dw1, dw2, max_turns


def run_win_rate_simulation(n_games=1000):
    """Run AI vs Random and AI vs Greedy simulations."""
    print("\n" + "=" * 60)
    print(f"WIN-RATE SIMULATION ({n_games} games each)")
    print("=" * 60)

    results = {}
    for opponent_name, OpponentClass in [("Random", RandomPlayer), ("Greedy", GreedyPlayer)]:
        print(f"\n  AI vs {opponent_name} Player...")
        ai_wins = 0
        opp_wins = 0
        draws = 0
        ai_dws = []
        opp_dws = []
        total_turns = []

        for i in range(n_games):
            rf = RFPlayer()
            opp = OpponentClass()
            s1, s2, dw1, dw2, turns = simulate_game(rf, opp)

            ai_dws.append(dw1)
            opp_dws.append(dw2)
            total_turns.append(turns)

            if s1 > s2:
                ai_wins += 1
            elif s2 > s1:
                opp_wins += 1
            else:
                draws += 1

            if (i + 1) % 200 == 0:
                print(f"    {i+1}/{n_games} games completed...")

        win_rate = round(ai_wins / n_games * 100, 1)
        avg_ai_dw = round(np.mean(ai_dws), 1)
        avg_opp_dw = round(np.mean(opp_dws), 1)
        avg_turns = round(np.mean(total_turns), 1)

        results[opponent_name] = {
            'AI Wins': ai_wins,
            'Opp Wins': opp_wins,
            'Draws': draws,
            'AI Win Rate': f"{win_rate}%",
            'Avg AI Deadwood': avg_ai_dw,
            'Avg Opp Deadwood': avg_opp_dw,
            'Avg Turns/Game': avg_turns,
        }

        print(f"    Result: AI Win Rate = {win_rate}% ({ai_wins}W / {opp_wins}L / {draws}D)")
        print(f"    Avg AI Deadwood: {avg_ai_dw}  |  Avg {opponent_name} Deadwood: {avg_opp_dw}")
        print(f"    Avg Turns/Game: {avg_turns}")

    # Plot win-rate comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (opp_name, data) in enumerate(results.items()):
        ax = axes[ax_idx]
        categories = ['AI Wins', f'{opp_name} Wins', 'Draws']
        values = [data['AI Wins'], data['Opp Wins'], data['Draws']]
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']
        bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f'AI vs {opp_name} ({n_games} games)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Games', fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                    str(val), ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'win_rate_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: win_rate_comparison.png")

    return results


# =========================================================
# SECTION 4: GENERATE SUMMARY REPORT
# =========================================================

def generate_summary_report(draw_results, discard_results, knock_results, winrate_results):
    """Generate a text summary report."""
    report_path = os.path.join(RESULTS_DIR, "evaluation_summary.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("   GIN RUMMY AI — MODEL EVALUATION SUMMARY REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("1. DRAW MODEL (Binary: Deck vs Take Discard)\n")
        f.write("-" * 50 + "\n")
        for i, model in enumerate(draw_results['Model']):
            f.write(f"  {model:25s}  Acc={draw_results['Accuracy'][i]}%")
            if draw_results['Precision'][i] != '-':
                f.write(f"  P={draw_results['Precision'][i]}%  R={draw_results['Recall'][i]}%  F1={draw_results['F1'][i]}%")
            f.write("\n")

        f.write("\n2. DISCARD MODEL (Binary: Best Card to Discard)\n")
        f.write("-" * 50 + "\n")
        for i, model in enumerate(discard_results['Model']):
            f.write(f"  {model:25s}  Acc={discard_results['Accuracy'][i]}%\n")

        f.write("\n3. KNOCK/GIN MODEL (3-class: Continue/Knock/Gin)\n")
        f.write("-" * 50 + "\n")
        for i, model in enumerate(knock_results['Model']):
            f.write(f"  {model:25s}  Acc={knock_results['Accuracy'][i]}%  BalAcc={knock_results['BalancedAcc'][i]}%  MacroF1={knock_results['MacroF1'][i]}%\n")

        f.write("\n4. WIN-RATE SIMULATION\n")
        f.write("-" * 50 + "\n")
        for opp_name, data in winrate_results.items():
            f.write(f"  AI vs {opp_name}:\n")
            for k, v in data.items():
                f.write(f"    {k:20s} = {v}\n")
            f.write("\n")

        f.write("\n5. GENERATED CHARTS\n")
        f.write("-" * 50 + "\n")
        f.write("  - draw_confusion_matrix.png\n")
        f.write("  - draw_feature_importance.png\n")
        f.write("  - discard_confusion_matrix.png\n")
        f.write("  - discard_feature_importance.png\n")
        f.write("  - knock_confusion_matrix.png\n")
        f.write("  - knock_feature_importance.png\n")
        f.write("  - win_rate_comparison.png\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("End of Report\n")

    print(f"\n  Summary report saved: {report_path}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print("=" * 60)
    print("   GIN RUMMY AI — COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    draw_results = evaluate_draw_model()
    discard_results = evaluate_discard_model()
    knock_results = evaluate_knock_model()
    winrate_results = run_win_rate_simulation(n_games=1000)
    generate_summary_report(draw_results, discard_results, knock_results, winrate_results)

    print("\n" + "=" * 60)
    print("   ALL EVALUATIONS COMPLETE!")
    print(f"   Results saved in: {RESULTS_DIR}")
    print("=" * 60)
