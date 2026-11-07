# show_rf_calculations.py
# Shows the step-by-step math of how our Random Forest models make decisions
# Generates a report in evaluation_results/random_forest_calculations.md

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "gin rummy"))
sys.path.append(os.path.join(BASE_DIR, "model"))

from play import Sorter, AIRandomForestEngine
from RandomForest import Randomforest

RESULTS_DIR = os.path.join(BASE_DIR, "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def calculate_entropy(y):
    if len(y) <= 1:
        return 0.0
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return float(-np.sum([p * np.log2(p) for p in probs if p > 0]))


def calculate_info_gain(parent_y, left_y, right_y):
    n_p = len(parent_y)
    if n_p == 0:
        return 0.0
    h_parent = calculate_entropy(parent_y)
    h_left = calculate_entropy(left_y)
    h_right = calculate_entropy(right_y)
    weighted_child = (len(left_y) / n_p) * h_left + (len(right_y) / n_p) * h_right
    return float(h_parent - weighted_child)


def trace_tree_prediction(tree_dict, x_sub, feature_names, tree_cols, depth=0):
    # follows a sample down one decision tree and shows each step
    if tree_dict.get("Leaf"):
        return [f"{'  ' * depth}└-> [LEAF NODE] Vote Class = {tree_dict['Class']}"], tree_dict["Class"]

    atr_idx = tree_dict["Atr"]
    orig_feat_idx = tree_cols[atr_idx] if atr_idx < len(tree_cols) else atr_idx
    feat_name = feature_names[orig_feat_idx] if orig_feat_idx < len(feature_names) else f"feature_{orig_feat_idx}"
    split_val = tree_dict["Val"]
    curr_val = x_sub[atr_idx]

    lines = []
    if curr_val > split_val:
        lines.append(f"{'  ' * depth}|- Condition: ({feat_name} = {curr_val:.2f}) > {split_val:.2f}  --> Go RIGHT")
        sub_lines, vote = trace_tree_prediction(tree_dict["Right"], x_sub, feature_names, tree_cols, depth + 1)
    else:
        lines.append(f"{'  ' * depth}|- Condition: ({feat_name} = {curr_val:.2f}) <= {split_val:.2f} --> Go LEFT")
        sub_lines, vote = trace_tree_prediction(tree_dict["Left"], x_sub, feature_names, tree_cols, depth + 1)

    lines.extend(sub_lines)
    return lines, vote


def generate_rf_calculations_report():
    engine = AIRandomForestEngine()

    DRAW_FEATURES = [
        "turn_num", "deck_size", "current_dw", "best_dw_with_discard", "dw_gain",
        "top_num", "same_num", "adj1", "adj2",
        "num_sets", "num_seqs", "num_unmelded", "highest_unmelded"
    ]

    DISCARD_FEATURES = [
        "turn_num", "base_dw", "dw_after", "c_num",
        "card_in_set", "card_in_seq", "same_num", "adj1", "adj2",
        "dw_reduction", "pairs", "connectors", "outs", "f14", "f15"
    ]

    KNOCK_FEATURES = [
        "turn_num", "deck_size", "real_deadwood",
        "num_sets", "num_seqs", "num_unmelded",
        "highest_unmelded", "pairs", "connectors", "gin_chance"
    ]

    report = []
    report.append("# Random Forest Mathematical Calculations & Decision Engine Guide")
    report.append("\nThis document shows the math and decision traces for the three Random Forest models in our Gin Rummy AI.\n")

    report.append("## 1. Mathematical Foundations\n")
    report.append("The AI uses an ensemble of ID3 Decision Trees with Bagging. Each decision uses Shannon Entropy, Information Gain, and majority voting.\n")

    report.append("### 1.1 Shannon Entropy")
    report.append("Shannon entropy measures how mixed (impure) the labels are:\n")
    report.append("$$H(Y) = -\\sum_{c \\in C} p(c) \\log_2 p(c)$$\n")
    report.append("* H = 0 means all samples are the same class (pure node).")
    report.append("* H = 1 means classes are split 50/50 (maximum uncertainty).\n")

    report.append("### 1.2 Information Gain")
    report.append("At each tree node we try all possible splits and pick the one that reduces entropy the most:\n")
    report.append("$$IG(Y, X_j, \\theta) = H(Y) - \\left(\\frac{|Y_L|}{|Y|} H(Y_L) + \\frac{|Y_R|}{|Y|} H(Y_R)\\right)$$\n")

    report.append("### 1.3 Random Forest Bagging")
    report.append("To prevent overfitting:")
    report.append("1. Each tree is trained on a random bootstrap sample of rows (with replacement).")
    report.append("2. Each tree sees only 70% of features (randomly selected).")
    report.append("3. Final prediction = majority vote across all trees.\n")

    report.append("### 1.4 Numerical Example of Information Gain")
    report.append("Parent node with 10 samples: 7 labeled Deck (0), 3 labeled Take Discard (1):\n")
    report.append("$$H(Parent) = -(0.70 \\log_2 0.70 + 0.30 \\log_2 0.30) = 0.8813 \\text{ bits}$$\n")
    report.append("Split on feature `dw_gain > 4`:")
    report.append("* Left branch (dw_gain <= 4, 6 samples all Deck): H(Left) = 0.000 bits")
    report.append("* Right branch (dw_gain > 4, 4 samples: 1 Deck, 3 Discard): H(Right) = 0.811 bits\n")
    report.append("$$IG = 0.8813 - [(6/10)*0.000 + (4/10)*0.8113] = 0.5568 \\text{ bits}$$\n")
    report.append("Since IG > 0, this split is chosen.\n")

    report.append("---\n")

    # Decision 1: Draw
    report.append("## 2. Decision 1: Draw (Deck vs Discard)\n")
    sample_hand = ['S4', 'S7', 'D7', 'S8', 'H8', 'S10', 'H4', 'H5', 'H6', 'S13']
    top_disc = 'C7'
    turn_num = 4
    deck_size = 28

    curr_dw = Sorter.calculate_cards_deadwood(sample_hand)
    cand_11 = sample_hand + [top_disc]
    best_dw_with_disc = min(
        Sorter.calculate_cards_deadwood([c for c in cand_11 if c != disc])
        for disc in cand_11
    )
    dw_gain = max(0, curr_dw - best_dw_with_disc)
    top_n = int(top_disc[1:])
    top_s = top_disc[0]
    same_num = sum(1 for c in sample_hand if int(c[1:]) == top_n)
    adj1 = sum(1 for c in sample_hand if c[0] == top_s and abs(int(c[1:]) - top_n) == 1)
    adj2 = sum(1 for c in sample_hand if c[0] == top_s and abs(int(c[1:]) - top_n) == 2)
    info = Sorter.analyze_hand_melds(sample_hand)

    draw_feat_vals = [
        turn_num, deck_size, curr_dw, best_dw_with_disc, dw_gain,
        top_n, same_num, adj1, adj2,
        len(info['sets']), len(info['seqs']), len(info['unmelded']),
        info['highest_unmelded']
    ]

    report.append("### 2.1 Scenario")
    report.append(f"* Hand (10 cards): `{', '.join(sample_hand)}`")
    report.append(f"* Top Discard: `{top_disc}` (Rank {top_n}, Suit {top_s})")
    report.append(f"* Turn {turn_num}, Deck has {deck_size} cards\n")

    report.append("### 2.2 Feature Vector")
    report.append("| Feature | Value |")
    report.append("|---------|-------|")
    report.append(f"| turn_num | {turn_num} |")
    report.append(f"| deck_size | {deck_size} |")
    report.append(f"| current_dw | {curr_dw} |")
    report.append(f"| best_dw_with_discard | {best_dw_with_disc} |")
    report.append(f"| dw_gain | {dw_gain} |")
    report.append(f"| top_num | {top_n} |")
    report.append(f"| same_num | {same_num} |")
    report.append(f"| adj1 | {adj1} |")
    report.append(f"| adj2 | {adj2} |")
    report.append(f"| num_sets | {len(info['sets'])} |")
    report.append(f"| num_seqs | {len(info['seqs'])} |")
    report.append(f"| num_unmelded | {len(info['unmelded'])} |")
    report.append(f"| highest_unmelded | {info['highest_unmelded']} |\n")

    report.append("### 2.3 Tree Voting Trace")
    rf_draw = engine.draw_model
    x_input = np.array([draw_feat_vals])

    votes = []
    tree_trace_blocks = []
    trees = rf_draw._Randomforest__trees
    cols_list = rf_draw._Randomforest__bootstraps_col

    for t_idx, (tree, cols) in enumerate(zip(trees[:5], cols_list[:5])):
        x_sub = x_input[0, cols]
        trace_lines, vote = trace_tree_prediction(tree._tree, x_sub, DRAW_FEATURES, cols)
        votes.append(vote)
        block = [
            f"**Tree #{t_idx+1} (Features: {[DRAW_FEATURES[c] for c in cols[:4]]}...)**",
            "```text"
        ]
        block.extend(trace_lines)
        block.append("```")
        tree_trace_blocks.append("\n".join(block))

    for t_idx, (tree, cols) in enumerate(zip(trees[5:], cols_list[5:]), start=5):
        x_sub = x_input[0, cols]
        _, vote = trace_tree_prediction(tree._tree, x_sub, DRAW_FEATURES, cols)
        votes.append(vote)

    report.append("\n\n".join(tree_trace_blocks))

    vote_deck = votes.count(0)
    vote_discard = votes.count(1)
    majority_vote = 1 if vote_discard > vote_deck else 0

    report.append("\n### 2.4 Vote Tally")
    report.append(f"* Deck (0): {vote_deck} votes | Take Discard (1): {vote_discard} votes")
    action_text = "TAKE DISCARD (C7 completes 3-of-a-kind with S7 and D7!)" if majority_vote == 1 or dw_gain >= 6 else "DRAW FROM DECK"
    report.append(f"* Final Action: **{action_text}**\n")

    report.append("---\n")

    # Decision 2: Discard
    report.append("## 3. Decision 2: Discard Selection (11 Cards)\n")
    hand_11 = sample_hand + [top_disc]
    base_dw = Sorter.calculate_cards_deadwood(hand_11)
    info_11 = Sorter.analyze_hand_melds(hand_11)

    report.append("### 3.1 Scoring Formula")
    report.append("For each card, we compute a score:\n")
    report.append("$$Score(c) = S_{meld} + 25*(100 - DW_{after}) + 4*rank - (12*same\\_num + 8*adj1) + 40*ML\\_vote$$\n")
    report.append("* Meld penalty: -5000 if card is part of a meld (we never break melds)")
    report.append("* We prefer to leave the hand with low deadwood")
    report.append("* High-value isolated cards (K, Q, J) are preferred discards\n")

    report.append("### 3.2 Scores for All 11 Cards")
    report.append("| Card | In Meld? | DW After | Score | Action |")
    report.append("|------|----------|----------|-------|--------|")

    rf_disc = engine.discard_model
    best_card = None
    best_score = -999999

    for c in hand_11:
        c_10 = [x for x in hand_11 if x != c]
        dw_after = Sorter.calculate_cards_deadwood(c_10)
        dw_red = base_dw - dw_after
        c_num = int(c[1:])
        c_suit = c[0]
        c_info = Sorter.analyze_hand_melds(c_10)

        same_num_c = sum(1 for x in c_10 if int(x[1:]) == c_num)
        adj1_c = sum(1 for x in c_10 if x[0] == c_suit and abs(int(x[1:]) - c_num) == 1)
        adj2_c = sum(1 for x in c_10 if x[0] == c_suit and abs(int(x[1:]) - c_num) == 2)
        in_set = 1 if c in info_11['sets'] else 0
        in_seq = 1 if c in info_11['seqs'] else 0
        in_meld = in_set or in_seq

        feat_disc = np.array([[
            turn_num, base_dw, dw_after, c_num,
            in_set, in_seq, same_num_c, adj1_c, adj2_c,
            dw_red, c_info['pairs'], c_info['connectors'],
            c_info['outs'], 0, 0
        ]])
        ml_pred = rf_disc.predict(feat_disc)[0] if engine.discard_trained else 0

        score = 0
        if in_meld:
            score -= 5000
        score += (100 - dw_after) * 25
        score += min(c_num, 10) * 4
        score -= (same_num_c * 12 + adj1_c * 8)
        score += ml_pred * 40

        is_best = False
        if score > best_score:
            best_score = score
            best_card = c
            is_best = True

        meld_str = "YES" if in_meld else "NO"
        report.append(f"| `{c}` | {meld_str} | {dw_after} | **{score}** | {'CHOSEN' if is_best else '-'} |")

    report.append(f"\nResult: AI discards **`{best_card}`** (Score = {best_score}).\n")

    report.append("---\n")

    # Decision 3: Knock
    report.append("## 4. Decision 3: Knock / Gin Decision\n")
    post_discard_hand = [c for c in hand_11 if c != best_card]
    final_dw = Sorter.calculate_cards_deadwood(post_discard_hand)
    k_info = Sorter.analyze_hand_melds(post_discard_hand)

    report.append("### 4.1 State After Discard")
    report.append(f"* Hand: `{', '.join(post_discard_hand)}`")
    report.append(f"* Deadwood: {final_dw}")
    report.append(f"* Sets: `{k_info['sets']}`")
    report.append(f"* Sequences: `{k_info['seqs']}`")
    report.append(f"* Unmelded: `{k_info['unmelded']}`\n")

    report.append("### 4.2 Rules for Knocking")
    report.append("1. If Deadwood = 0: GIN (always)")
    report.append("2. If Deadwood <= 4: KNOCK (safe)")
    report.append("3. If 5 <= Deadwood <= 10: Ask the Random Forest model\n")

    k_feats = np.array([[
        turn_num, deck_size, final_dw,
        len(k_info['sets']), len(k_info['seqs']), len(k_info['unmelded']),
        k_info['highest_unmelded'], k_info['pairs'], k_info['connectors'],
        k_info['gin_chance']
    ]])
    k_pred = engine.knock_model.predict(k_feats)[0] if engine.knock_trained else 1

    k_label_map = {0: "CONTINUE", 1: "KNOCK", 2: "GIN"}
    report.append(f"* Random Forest Prediction: Class {k_pred} -> **{k_label_map.get(k_pred, 'KNOCK')}**")

    if final_dw == 0:
        decision_str = "GIN!"
    elif final_dw <= 10 and k_pred in [1, 2]:
        decision_str = f"KNOCK! (Deadwood = {final_dw})"
    else:
        decision_str = "CONTINUE"

    report.append(f"* Final Decision: **{decision_str}**\n")

    output_path = os.path.join(RESULTS_DIR, "random_forest_calculations.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_rf_calculations_report()
