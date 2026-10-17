"""
Dataset generator for Gin Rummy decision models.
Generates labelled training data for draw, discard, and knock decisions.
"""

import os
import sys
import random
import csv
import numpy as np

# Add paths for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "gin rummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from play import Sorter

SEED = 42
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")

ALL_CARDS = [f"{s}{n}" for s in "CDSH" for n in range(1, 14)]

# Monte Carlo sampling sizes
MC_DRAW_SAMPLES = 25     # Deck cards sampled for draw decision
MC_DISCARD_SAMPLES = 15  # Unseen cards sampled for discard label
MC_KNOCK_OPPONENTS = 20  # Opponent hands sampled for knock decision


def card_value(card: str) -> int:
    """Point value of a card for deadwood calculation."""
    return min(int(card[1:]), 10)


def deal_hand(deck: list, n: int = 10) -> list:
    """Deal n cards from deck (removes them)."""
    hand = deck[:n]
    del deck[:n]
    return hand


def analyze_hand(cards: list) -> dict:
    """Full structural analysis of a hand using the Sorter from play.py."""
    return Sorter.analyze_hand_melds(cards)


def simulate_mid_game_hand(rng: random.Random):
    """
    Simulates a realistic mid-game state:
    1. Shuffle and deal two 10-card hands
    2. Play a random number of turns (2-12) with a greedy bot
    3. Return the resulting hand, discard pile, and remaining deck
    """
    deck = ALL_CARDS.copy()
    rng.shuffle(deck)

    hand = deal_hand(deck, 10)
    opp_hand = deal_hand(deck, 10)
    discard_pile = [deck.pop()]  # Initial upcard

    turns_played = rng.randint(2, 12)
    for turn in range(turns_played):
        # Current player alternates
        current = hand if turn % 2 == 0 else opp_hand

        # Greedy draw: take discard if it reduces deadwood
        dw_before = Sorter.calculate_cards_deadwood(current)
        if discard_pile:
            top = discard_pile[-1]
            test_hand = current + [top]
            best_dw_with = min(
                Sorter.calculate_cards_deadwood([c for c in test_hand if c != d])
                for d in test_hand
            )
            if best_dw_with < dw_before:
                current.append(discard_pile.pop())
            else:
                if deck:
                    current.append(deck.pop())
                else:
                    break
        else:
            if deck:
                current.append(deck.pop())
            else:
                break

        # Greedy discard: remove card that leaves lowest deadwood
        if len(current) > 10:
            best_disc = min(current,
                            key=lambda c: Sorter.calculate_cards_deadwood([x for x in current if x != c]))
            current.remove(best_disc)
            discard_pile.append(best_disc)

    turn_number = turns_played + 1
    return hand, opp_hand, discard_pile, deck, turn_number


def compute_draw_features(hand: list, top_discard: str, turn_number: int, deck_size: int) -> dict:
    """Compute the 13 features for the draw decision model."""
    info = analyze_hand(hand)
    dw_now = info['deadwood']

    top_num = int(top_discard[1:])
    top_suit = top_discard[0]
    same_num = sum(1 for c in hand if int(c[1:]) == top_num)
    adj1 = sum(1 for c in hand if c[0] == top_suit and abs(int(c[1:]) - top_num) == 1)
    adj2 = sum(1 for c in hand if c[0] == top_suit and abs(int(c[1:]) - top_num) == 2)

    # Deadwood gain if taking the discard
    candidate = hand + [top_discard]
    best_dw = min(Sorter.calculate_cards_deadwood([c for c in candidate if c != d]) for d in candidate)
    dw_gain = max(0, dw_now - best_dw)

    return {
        'turn_number': turn_number,
        'deck_size': deck_size,
        'deadwood_now': dw_now,
        'cards_in_sets': len(info['sets']),
        'cards_in_sequences': len(info['seqs']),
        'cards_unmelded': len(info['unmelded']),
        'num_pairs': info['pairs'],
        'num_suit_connectors': info['connectors'],
        'discard_top_number': top_num,
        'top_same_number_in_hand': same_num,
        'top_suit_adjacent1': adj1,
        'top_suit_adjacent2': adj2,
        'deadwood_gain_if_take': dw_gain,
    }


def label_draw_decision(hand: list, top_discard: str, deck: list, rng: random.Random) -> int:
    """
    Monte Carlo label: 1 if taking discard gives deadwood at least as good
    as the average over MC_DRAW_SAMPLES sampled possible deck draws.
    """
    candidate = hand + [top_discard]
    dw_with_discard = min(
        Sorter.calculate_cards_deadwood([c for c in candidate if c != d])
        for d in candidate
    )

    unseen = [c for c in ALL_CARDS if c not in hand and c != top_discard]
    samples = rng.sample(unseen, min(MC_DRAW_SAMPLES, len(unseen))) if unseen else []
    if not samples:
        return 1

    deck_dws = []
    for sample_card in samples:
        test = hand + [sample_card]
        best = min(Sorter.calculate_cards_deadwood([c for c in test if c != d]) for d in test)
        deck_dws.append(best)

    avg_deck_dw = np.mean(deck_dws)
    return 1 if dw_with_discard <= avg_deck_dw else 0


def compute_discard_features(hand_11: list, card: str, turn_number: int) -> dict:
    """Compute the 15 features for a single card's discard candidacy."""
    info_11 = analyze_hand(hand_11)
    base_dw = info_11['deadwood']

    candidate_10 = [c for c in hand_11 if c != card]
    cand_info = analyze_hand(candidate_10)
    dw_after = cand_info['deadwood']
    dw_reduction = base_dw - dw_after

    c_num = int(card[1:])
    c_suit = card[0]
    same_num = sum(1 for c in candidate_10 if int(c[1:]) == c_num)
    adj1 = sum(1 for c in candidate_10 if c[0] == c_suit and abs(int(c[1:]) - c_num) == 1)
    adj2 = sum(1 for c in candidate_10 if c[0] == c_suit and abs(int(c[1:]) - c_num) == 2)
    card_in_set = 1 if card in info_11['sets'] else 0
    card_in_seq = 1 if card in info_11['seqs'] else 0

    return {
        'turn_number': turn_number,
        'deadwood_before_discard': base_dw,
        'deadwood_after_discard': dw_after,
        'card_number': c_num,
        'card_in_set': card_in_set,
        'card_in_sequence': card_in_seq,
        'same_number_in_hand': same_num,
        'suit_adjacent1': adj1,
        'suit_adjacent2': adj2,
        'deadwood_reduction': dw_reduction,
        'pairs_left': cand_info['pairs'],
        'suit_connectors_left': cand_info['connectors'],
        'outs_after_discard': cand_info['outs'],
    }


def label_discard_decision(hand_11: list, rng: random.Random) -> str:
    """
    Monte Carlo label: best card to discard is the one with the lowest expected
    deadwood after the next draw (MC_DISCARD_SAMPLES sampled unseen cards).
    """
    unseen = [c for c in ALL_CARDS if c not in hand_11]
    samples = rng.sample(unseen, min(MC_DISCARD_SAMPLES, len(unseen))) if unseen else []

    best_card = hand_11[0]
    best_expected_dw = 999

    for card in hand_11:
        remaining_10 = [c for c in hand_11 if c != card]
        if not samples:
            exp_dw = Sorter.calculate_cards_deadwood(remaining_10)
        else:
            dws = []
            for drawn in samples:
                test_11 = remaining_10 + [drawn]
                best_disc_dw = min(
                    Sorter.calculate_cards_deadwood([c for c in test_11 if c != d])
                    for d in test_11
                )
                dws.append(best_disc_dw)
            exp_dw = np.mean(dws)

        if exp_dw < best_expected_dw or (exp_dw == best_expected_dw and card_value(card) > card_value(best_card)):
            best_expected_dw = exp_dw
            best_card = card

    return best_card


def compute_knock_features(hand: list, turn_number: int, deck_size: int) -> dict:
    """Compute the 10 features for the knock/gin decision model."""
    info = analyze_hand(hand)
    return {
        'turn_number': turn_number,
        'deck_size': deck_size,
        'my_deadwood': info['deadwood'],
        'cards_in_sets': len(info['sets']),
        'cards_in_sequences': len(info['seqs']),
        'cards_unmelded': len(info['unmelded']),
        'highest_deadwood_card': info['highest_unmelded'],
        'num_pairs': info['pairs'],
        'num_suit_connectors': info['connectors'],
        'gin_chance_next_draw': info['gin_chance'],
    }


def label_knock_decision(hand: list, deck: list, turn_number: int, deck_size: int, rng: random.Random) -> tuple:
    """
    Label: 0=keep_playing, 1=knock, 2=gin
    Also returns auxiliary win probabilities for transparency.
    """
    dw = Sorter.calculate_cards_deadwood(hand)

    if dw == 0:
        return 2, 1.0, 0.0  # Gin

    if dw > 10:
        return 0, 0.0, 0.5  # Cannot knock

    unseen = [c for c in ALL_CARDS if c not in hand]
    knock_wins = 0
    total_sims = min(MC_KNOCK_OPPONENTS, len(unseen) // 10) if len(unseen) >= 10 else 1

    for _ in range(max(1, total_sims)):
        opp_cards = rng.sample(unseen, min(10, len(unseen)))
        opp_dw = Sorter.calculate_cards_deadwood(opp_cards)
        if dw < opp_dw:
            knock_wins += 1

    win_prob_knock = knock_wins / max(1, total_sims)

    continue_dws = []
    sample_draws = rng.sample(unseen, min(10, len(unseen))) if unseen else []
    for drawn in sample_draws:
        test = hand + [drawn]
        best = min(Sorter.calculate_cards_deadwood([c for c in test if c != d]) for d in test)
        continue_dws.append(best)

    avg_continue_dw = np.mean(continue_dws) if continue_dws else dw
    win_prob_continue = max(0.0, min(1.0, (10 - avg_continue_dw) / 10.0))

    if win_prob_knock >= win_prob_continue:
        decision = 1
    else:
        decision = 0

    return decision, round(win_prob_knock, 2), round(win_prob_continue, 2)


def generate_datasets(n_draw=10000, n_discard=3000, n_knock=3000):
    """Generate all three datasets and save as CSV."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # ========== 1. DRAW DATASET ==========
    print(f"Generating draw dataset ({n_draw} samples)...")
    draw_rows = []
    for i in range(n_draw):
        hand, opp, discard_pile, deck, turn_num = simulate_mid_game_hand(rng)
        if not discard_pile:
            continue
        top_disc = discard_pile[-1]
        features = compute_draw_features(hand, top_disc, turn_num, len(deck))
        label = label_draw_decision(hand, top_disc, deck, rng)
        features['take_discard'] = label
        draw_rows.append(features)
        if (i + 1) % 2000 == 0:
            print(f"  Draw: {i+1}/{n_draw}")

    draw_path = os.path.join(DATASET_DIR, "draw_decision_dataset.csv")
    with open(draw_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=draw_rows[0].keys())
        writer.writeheader()
        writer.writerows(draw_rows)
    print(f"  Saved {len(draw_rows)} rows to {draw_path}")

    # ========== 2. DISCARD DATASET ==========
    print(f"Generating discard dataset ({n_discard} hands x 11 cards)...")
    discard_rows = []
    for i in range(n_discard):
        hand, opp, discard_pile, deck, turn_num = simulate_mid_game_hand(rng)
        if deck:
            hand.append(deck.pop())
        elif discard_pile:
            hand.append(discard_pile.pop())
        else:
            continue

        if len(hand) != 11:
            continue

        best_discard = label_discard_decision(hand, rng)

        for card in hand:
            features = compute_discard_features(hand, card, turn_num)
            best_feat = compute_discard_features(hand, best_discard, turn_num)
            features['outs_vs_best_option'] = features['outs_after_discard'] - best_feat['outs_after_discard']
            features['deadwood_vs_best_option'] = features['deadwood_after_discard'] - best_feat['deadwood_after_discard']
            features['best_discard'] = 1 if card == best_discard else 0
            discard_rows.append(features)

        if (i + 1) % 500 == 0:
            print(f"  Discard: {i+1}/{n_discard} hands")

    discard_path = os.path.join(DATASET_DIR, "discard_decision_dataset.csv")
    with open(discard_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=discard_rows[0].keys())
        writer.writeheader()
        writer.writerows(discard_rows)
    print(f"  Saved {len(discard_rows)} rows to {discard_path}")

    # ========== 3. KNOCK/GIN DATASET ==========
    print(f"Generating knock/gin dataset ({n_knock} samples)...")
    knock_rows = []
    for i in range(n_knock):
        hand, opp, discard_pile, deck, turn_num = simulate_mid_game_hand(rng)
        dw = Sorter.calculate_cards_deadwood(hand)

        if dw > 15 and rng.random() > 0.3:
            continue

        features = compute_knock_features(hand, turn_num, len(deck))
        decision, wp_knock, wp_continue = label_knock_decision(hand, deck, turn_num, len(deck), rng)
        features['label_aux_win_prob_knock'] = wp_knock
        features['label_aux_win_prob_continue'] = wp_continue
        features['decision'] = decision
        knock_rows.append(features)

        if (i + 1) % 500 == 0:
            print(f"  Knock: {i+1}/{n_knock}")

    knock_path = os.path.join(DATASET_DIR, "knock_gin_decision_dataset.csv")
    with open(knock_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=knock_rows[0].keys())
        writer.writeheader()
        writer.writerows(knock_rows)
    print(f"  Saved {len(knock_rows)} rows to {knock_path}")

    print(f"\nAll datasets generated successfully!")
    print(f"   Draw:    {len(draw_rows)} rows")
    print(f"   Discard: {len(discard_rows)} rows ({n_discard} hands x ~11 cards)")
    print(f"   Knock:   {len(knock_rows)} rows")


if __name__ == "__main__":
    generate_datasets()
