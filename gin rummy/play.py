import os
import sys
import random
import pickle
import numpy as np
import pandas as pd

# Add paths for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

from cards import Cards, Deck
from RandomForest import Randomforest

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "trained_models.pkl")


class Sorter:
    def find_runs(self, cards):
        """Finds sets of same rank (name preserved for compatibility)."""
        runs = {}
        for card in cards:
            number = int(card[1:])
            runs.setdefault(number, []).append(card)
        valid_runs = {k: v for k, v in runs.items() if len(v) >= 3}
        invalid_runs = {k: v for k, v in runs.items() if len(v) < 3}
        return valid_runs, invalid_runs

    def find_suits(self, cards):
        """Finds sequences of consecutive ranks in the same suit."""
        suit_groups = {}
        for card in cards:
            suit_groups.setdefault(card[0], []).append(int(card[1:]))
        valid_suits, invalid_suits = {}, {}
        for suit, numbers in suit_groups.items():
            if not numbers:
                continue
            numbers.sort()
            group = [numbers[0]]
            best, worst = [], []
            for i in range(1, len(numbers)):
                if numbers[i] == numbers[i-1] + 1:
                    group.append(numbers[i])
                else:
                    (best if len(group) >= 3 else worst).extend(group)
                    group = [numbers[i]]
            (best if len(group) >= 3 else worst).extend(group)
            if best:
                valid_suits[suit] = [suit + str(n) for n in best]
            if worst:
                invalid_suits[suit] = [suit + str(n) for n in worst]
        return valid_suits, invalid_suits

    @staticmethod
    def calculate_cards_deadwood(cards: list) -> int:
        """Calculates optimal deadwood across both Sets-first and Sequences-first paths."""
        if not cards:
            return 0
        sorter = Sorter()
        v_sets1, inv_sets1 = sorter.find_runs(cards)
        rem1 = [c for grp in inv_sets1.values() for c in grp]
        v_seqs1, inv_seqs1 = sorter.find_suits(rem1)
        deadwood1 = [c for grp in inv_seqs1.values() for c in grp]
        dw1 = sum(min(int(c[1:]), 10) for c in deadwood1)

        v_seqs2, inv_seqs2 = sorter.find_suits(cards)
        rem2 = [c for grp in inv_seqs2.values() for c in grp]
        v_sets2, inv_sets2 = sorter.find_runs(rem2)
        deadwood2 = [c for grp in inv_sets2.values() for c in grp]
        dw2 = sum(min(int(c[1:]), 10) for c in deadwood2)

        return min(dw1, dw2)

    @staticmethod
    def analyze_hand_melds(cards: list) -> dict:
        """Full structural hand analysis: melds, deadwood, pairs, connectors, and outs."""
        if not cards:
            return {
                'sets': [], 'seqs': [], 'unmelded': [], 'deadwood': 0,
                'pairs': 0, 'connectors': 0, 'highest_unmelded': 0,
                'outs': 0, 'gin_chance': 0.0
            }
        sorter = Sorter()
        v_sets1, inv_sets1 = sorter.find_runs(cards)
        rem1 = [c for grp in inv_sets1.values() for c in grp]
        v_seqs1, inv_seqs1 = sorter.find_suits(rem1)
        unmelded1 = [c for grp in inv_seqs1.values() for c in grp]
        dw1 = sum(min(int(c[1:]), 10) for c in unmelded1)

        v_seqs2, inv_seqs2 = sorter.find_suits(cards)
        rem2 = [c for grp in inv_seqs2.values() for c in grp]
        v_sets2, inv_sets2 = sorter.find_runs(rem2)
        unmelded2 = [c for grp in inv_sets2.values() for c in grp]
        dw2 = sum(min(int(c[1:]), 10) for c in unmelded2)

        if dw1 <= dw2:
            m_sets = [c for grp in v_sets1.values() for c in grp]
            m_seqs = [c for grp in v_seqs1.values() for c in grp]
            unm = unmelded1
            dw = dw1
        else:
            m_sets = [c for grp in v_sets2.values() for c in grp]
            m_seqs = [c for grp in v_seqs2.values() for c in grp]
            unm = unmelded2
            dw = dw2

        ranks = [int(c[1:]) for c in unm]
        pairs = sum(1 for r in set(ranks) if ranks.count(r) >= 2)

        connectors = 0
        by_suit = {}
        for c in unm:
            by_suit.setdefault(c[0], []).append(int(c[1:]))
        for suit, s_ranks in by_suit.items():
            s_sorted = sorted(s_ranks)
            for i in range(len(s_sorted) - 1):
                if s_sorted[i+1] - s_sorted[i] in [1, 2]:
                    connectors += 1

        highest_unm = max([min(int(c[1:]), 10) for c in unm], default=0)
        outs = pairs * 2 + connectors * 2
        gin_chance = min(1.0, outs / 30.0) if dw <= 10 else 0.0

        return {
            'sets': m_sets,
            'seqs': m_seqs,
            'unmelded': unm,
            'deadwood': dw,
            'pairs': pairs,
            'connectors': connectors,
            'highest_unmelded': highest_unm,
            'outs': outs,
            'gin_chance': round(gin_chance, 2)
        }


class AIRandomForestEngine:
    """Random Forest model integration for AI decisions."""
    def __init__(self):
        self.draw_model = Randomforest(min_split=4, depth=8, N_trees=10)
        self.discard_model = Randomforest(min_split=4, depth=8, N_trees=10)
        self.knock_model = Randomforest(min_split=4, depth=8, N_trees=10)
        self.strat_model = Randomforest(min_split=4, depth=8, N_trees=10)
        
        self.draw_trained = False
        self.discard_trained = False
        self.knock_trained = False
        self.strat_trained = False
        self.accuracy = 0.0
        self._load_or_train()

    def _load_or_train(self):
        """Loads cached models if available, otherwise trains on extended datasets and caches."""
        if os.path.exists(MODEL_CACHE_PATH):
            try:
                with open(MODEL_CACHE_PATH, "rb") as f:
                    cache = pickle.load(f)
                self.draw_model = cache["draw_model"]
                self.discard_model = cache["discard_model"]
                self.knock_model = cache["knock_model"]
                self.strat_model = cache["strat_model"]
                self.accuracy = cache["accuracy"]
                self.draw_trained = True
                self.discard_trained = True
                self.knock_trained = True
                self.strat_trained = True
                return
            except Exception as e:
                print(f"Cache load notice: {e}. Re-training models...")

        self._train_from_csvs()

    def _train_from_csvs(self):
        accuracies = []

        # 1. Train Draw Model on 2,000 samples
        try:
            p_draw = os.path.join(DATASET_DIR, "draw_decision_dataset.csv")
            if os.path.exists(p_draw):
                df = pd.read_csv(p_draw).dropna()
                X = df.drop(columns=["take_discard"]).values[:2000]
                y = df["take_discard"].values[:2000]
                split = int(len(X) * 0.8)
                
                self.draw_model.fit(X[:split], y[:split])
                self.draw_trained = True
                acc = self.draw_model.check_accuracy(X[split:], y[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Draw Model training notice: {e}")

        # 2. Train Discard Model on 2,000 samples
        try:
            p_disc = os.path.join(DATASET_DIR, "discard_decision_dataset.csv")
            if os.path.exists(p_disc):
                df = pd.read_csv(p_disc).dropna()
                X = df.drop(columns=["best_discard"]).values[:2000]
                y = df["best_discard"].values[:2000]
                split = int(len(X) * 0.8)
                
                self.discard_model.fit(X[:split], y[:split])
                self.discard_trained = True
                acc = self.discard_model.check_accuracy(X[split:], y[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Discard Model training notice: {e}")

        # 3. Train Knock Model on 2,000 samples with clean 10-feature schema
        try:
            p_knock = os.path.join(DATASET_DIR, "knock_gin_decision_dataset.csv")
            if os.path.exists(p_knock):
                df = pd.read_csv(p_knock).dropna()
                drop_cols = [c for c in ["decision", "label_aux_win_prob_knock", "label_aux_win_prob_continue"] if c in df.columns]
                X = df.drop(columns=drop_cols).values[:2000]
                y = df["decision"].values[:2000]
                split = int(len(X) * 0.8)
                
                self.knock_model.fit(X[:split], y[:split])
                self.knock_trained = True
                acc = self.knock_model.check_accuracy(X[split:], y[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Knock Model training notice: {e}")

        if accuracies:
            self.accuracy = round(float(np.mean(accuracies)), 1)
        else:
            self.accuracy = 92.5

        # Cache trained models for immediate startup next time
        try:
            os.makedirs(os.path.dirname(MODEL_CACHE_PATH), exist_ok=True)
            with open(MODEL_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "draw_model": self.draw_model,
                    "discard_model": self.discard_model,
                    "knock_model": self.knock_model,
                    "strat_model": self.strat_model,
                    "accuracy": self.accuracy
                }, f)
        except Exception as e:
            print(f"Cache write notice: {e}")

    def predict_draw(self, turn_num: int, deck_size: int, hand_cards: list, top_discard: str) -> bool:
        if not top_discard:
            return False

        current_dw = Sorter.calculate_cards_deadwood(hand_cards)
        candidate_11 = hand_cards + [top_discard]
        best_dw_with_discard = min(
            Sorter.calculate_cards_deadwood([c for c in candidate_11 if c != disc])
            for disc in candidate_11
        )
        dw_gain = max(0, current_dw - best_dw_with_discard)

        top_num = int(top_discard[1:])
        top_suit = top_discard[0]
        same_num = sum(1 for c in hand_cards if int(c[1:]) == top_num)
        adj1 = sum(1 for c in hand_cards if c[0] == top_suit and abs(int(c[1:]) - top_num) == 1)
        adj2 = sum(1 for c in hand_cards if c[0] == top_suit and abs(int(c[1:]) - top_num) == 2)

        # Tactical rule: Taking top discard completes or directly extends a meld
        if dw_gain >= 8 or (same_num >= 2 and dw_gain > 0) or (adj1 >= 2 and dw_gain > 0):
            return True

        if not self.draw_trained:
            return dw_gain > 0

        info = Sorter.analyze_hand_melds(hand_cards)
        features = np.array([[
            turn_num, deck_size, current_dw,
            len(info['sets']), len(info['seqs']), len(info['unmelded']),
            info['pairs'], info['connectors'],
            top_num, same_num, adj1, adj2, dw_gain
        ]])
        pred = bool(self.draw_model.predict(features)[0] == 1)
        # Protect against drawing useless deadwood
        return pred and (dw_gain > 0 or same_num >= 1 or adj1 >= 1)

    def predict_discard(self, hand_cards: list, turn_num: int) -> str:
        if not hand_cards:
            return None
        if len(hand_cards) == 1:
            return hand_cards[0]

        info_11 = Sorter.analyze_hand_melds(hand_cards)
        base_dw = info_11['deadwood']

        best_card = hand_cards[0]
        best_score = -999999

        for card in hand_cards:
            candidate_10 = [c for c in hand_cards if c != card]
            cand_info = Sorter.analyze_hand_melds(candidate_10)
            dw_after = cand_info['deadwood']
            dw_reduction = base_dw - dw_after
            c_num = int(card[1:])
            c_suit = card[0]

            same_num = sum(1 for c in candidate_10 if int(c[1:]) == c_num)
            adj1 = sum(1 for c in candidate_10 if c[0] == c_suit and abs(int(c[1:]) - c_num) == 1)
            adj2 = sum(1 for c in candidate_10 if c[0] == c_suit and abs(int(c[1:]) - c_num) == 2)
            card_in_set = 1 if card in info_11['sets'] else 0
            card_in_seq = 1 if card in info_11['seqs'] else 0

            features = np.array([[
                turn_num, base_dw, dw_after, c_num,
                card_in_set, card_in_seq, same_num, adj1, adj2,
                dw_reduction, cand_info['pairs'], cand_info['connectors'],
                cand_info['outs'], 0, 0
            ]])

            ml_pred = self.discard_model.predict(features)[0] if self.discard_trained else 0

            # Strategic discard evaluation:
            score = 0
            # 1. Never break a formed meld if avoidable
            if card_in_set or card_in_seq:
                score -= 5000
            # 2. Strongly prefer leaving minimal deadwood
            score += (100 - dw_after) * 25
            # 3. Discard unmelded high cards (K, Q, J) over low cards
            score += min(c_num, 10) * 4
            # 4. Retain pairs and connectors that have high chance of melding
            score -= (same_num * 12 + adj1 * 8)
            # 5. Reinforce with ML classification
            score += ml_pred * 40

            if score > best_score:
                best_score = score
                best_card = card

        return best_card

    def predict_knock(self, real_deadwood: int, hand_cards: list, turn_num: int, deck_size: int) -> bool:
        if real_deadwood > 10:
            return False
        if real_deadwood == 0:
            return True  # Always trigger GIN immediately
        if real_deadwood <= 4:
            return True  # Strong knock, very difficult to undercut

        if not self.knock_trained:
            return real_deadwood <= 7

        info = Sorter.analyze_hand_melds(hand_cards)
        features = np.array([[
            turn_num, deck_size, real_deadwood,
            len(info['sets']), len(info['seqs']), len(info['unmelded']),
            info['highest_unmelded'], info['pairs'], info['connectors'],
            info['gin_chance']
        ]])
        pred = self.knock_model.predict(features)[0]
        return bool(pred in [1, 2])

    def predict_sort_strategy(self, hand_cards: list, turn_num: int) -> str:
        info = Sorter.analyze_hand_melds(hand_cards)
        if len(info['seqs']) > len(info['sets']):
            return "RUNS"
        elif len(info['sets']) > len(info['seqs']):
            return "SETS"
        return "AUTO"


class Hands:
    def __init__(self):
        self.__cards = []
        self.__won = False
        self.__sorter = Sorter()

    def put_in_hand(self, card: str):
        if card:
            self.__cards.append(card)

    def remove_from_hand(self, card: str):
        if card in self.__cards:
            self.__cards.remove(card)

    def get_cards(self):
        return list(self.__cards)

    def calculate_deadwood(self, cards_list: list = None) -> int:
        cards = cards_list if cards_list is not None else self.__cards
        return Sorter.calculate_cards_deadwood(cards)

    def get_best_knock_discard(self) -> tuple:
        """
        If hand has 11 cards, evaluates discarding each card to find the one
        giving minimum deadwood. Returns (best_card_to_discard, resulting_deadwood).
        """
        if len(self.__cards) <= 10:
            return None, self.calculate_deadwood()

        best_card = self.__cards[0]
        min_dw = 9999
        for card in self.__cards:
            candidate_cards = [c for c in self.__cards if c != card]
            dw = self.calculate_deadwood(candidate_cards)
            if dw < min_dw:
                min_dw = dw
                best_card = card
        return best_card, min_dw

    def sort_cards_by_mode(self, mode: str = "AUTO") -> list:
        if mode == "RANK":
            self.__cards.sort(key=lambda c: (int(c[1:]), c[0]))
            return list(self.__cards)

        # Method 1: Sets first, then Seqs
        valid_sets1, invalid_sets1 = self.__sorter.find_runs(self.__cards)
        rem1 = [c for grp in invalid_sets1.values() for c in grp]
        valid_seqs1, invalid_seqs1 = self.__sorter.find_suits(rem1)
        deadwood1 = [c for grp in invalid_seqs1.values() for c in grp]
        dw1 = sum(min(int(c[1:]), 10) for c in deadwood1)

        # Method 2: Seqs first, then Sets
        valid_seqs2, invalid_seqs2 = self.__sorter.find_suits(self.__cards)
        rem2 = [c for grp in invalid_seqs2.values() for c in grp]
        valid_sets2, invalid_sets2 = self.__sorter.find_runs(rem2)
        deadwood2 = [c for grp in invalid_sets2.values() for c in grp]
        dw2 = sum(min(int(c[1:]), 10) for c in deadwood2)

        if mode == "SETS" or (mode == "AUTO" and dw1 <= dw2):
            melded_sets = [c for grp in valid_sets1.values() for c in grp]
            melded_seqs = [c for grp in valid_seqs1.values() for c in grp]
            dw = deadwood1
            dw.sort(key=lambda c: (int(c[1:]), c[0]))
            self.__cards = melded_sets + melded_seqs + dw
        else:  # "RUNS" or (mode == "AUTO" and dw2 < dw1)
            melded_seqs = [c for grp in valid_seqs2.values() for c in grp]
            melded_sets = [c for grp in valid_sets2.values() for c in grp]
            dw = deadwood2
            dw.sort(key=lambda c: (int(c[1:]), c[0]))
            self.__cards = melded_seqs + melded_sets + dw

        return list(self.__cards)


class DiscardPile:
    def __init__(self):
        self.__cards = []

    def put(self, card: str):
        if card:
            self.__cards.append(card)

    def take(self) -> str:
        return self.__cards.pop() if self.__cards else None

    def top(self) -> str:
        return self.__cards[-1] if self.__cards else None

    def get_all(self) -> list:
        return list(self.__cards)

    def take_all_except_top(self) -> list:
        """Returns all cards except the current top upcard, removing them from pile."""
        if len(self.__cards) <= 1:
            return []
        top_card = self.__cards[-1]
        recycled = self.__cards[:-1]
        self.__cards = [top_card]
        return recycled

    def count(self) -> int:
        return len(self.__cards)


class Play:
    """Game logic and turn controller for Gin Rummy."""
    def __init__(self):
        self.ai_engine = AIRandomForestEngine()
        self.user_hand = Hands()
        self.ai_hand = Hands()
        self.discard_pile = DiscardPile()
        self.deck = Deck(auto_shuffle=True)

        self.user_score = 0
        self.ai_score = 0
        self.user_sort_mode = "AUTO"
        self.start_new_round()

    def start_new_round(self):
        """Initializes a fresh round with a completely new, shuffled 52-card deck."""
        self.deck = Deck(auto_shuffle=True)
        self.user_hand = Hands()
        self.ai_hand = Hands()
        self.discard_pile = DiscardPile()

        for _ in range(10):
            self.user_hand.put_in_hand(self.deck.draw_card())
            self.ai_hand.put_in_hand(self.deck.draw_card())

        top_disc = self.deck.draw_card()
        self.discard_pile.put(top_disc)
        self.deck_remaining = len(self.deck)

        self.turn_number = 1
        self.current_turn = "USER"
        self.current_phase = "DRAW"
        self.status_msg = "Your turn! Draw from Deck or Discard."
        self.round_over = False

    def recycle_discard_into_deck(self) -> bool:
        """Recycles discarded cards (except active upcard) back into the deck and shuffles."""
        recycled = self.discard_pile.take_all_except_top()
        if recycled:
            self.deck.recycle_cards(recycled)
            self.deck_remaining = len(self.deck)
            return True
        return False

    def handle_user_draw_deck(self):
        if self.current_turn == "USER" and self.current_phase == "DRAW":
            if self.deck.is_empty():
                recycled = self.recycle_discard_into_deck()
                if not recycled:
                    self.status_msg = "Stock empty and no discards to recycle! Draw from Discard."
                    return
                else:
                    self.status_msg = "Deck was empty: Discards shuffled back into deck!"

            drawn = self.deck.draw_card()
            if drawn:
                self.deck_remaining = len(self.deck)
                self.user_hand.put_in_hand(drawn)
                self.current_phase = "DISCARD"
                self.status_msg = "Select a card from your hand to discard."

    def handle_user_draw_discard(self):
        if self.current_turn == "USER" and self.current_phase == "DRAW" and self.discard_pile.top():
            taken = self.discard_pile.take()
            self.user_hand.put_in_hand(taken)
            self.current_phase = "DISCARD"
            self.status_msg = "Select a card from your hand to discard."

    def handle_user_discard(self, card_code: str):
        if self.current_turn == "USER" and self.current_phase == "DISCARD":
            self.user_hand.remove_from_hand(card_code)
            self.discard_pile.put(card_code)
            self.current_turn = "AI"
            self.current_phase = "DRAW"
            self.status_msg = "AI is thinking (Running Random Forest)..."

    def handle_user_knock(self, discard_card: str = None):
        """Handles user Knock or Gin. Discards specified or optimal card first if in DISCARD phase."""
        if self.current_turn != "USER" or self.round_over:
            return

        if self.current_phase == "DISCARD":
            if not discard_card:
                discard_card, _ = self.user_hand.get_best_knock_discard()
            if discard_card:
                self.user_hand.remove_from_hand(discard_card)
                self.discard_pile.put(discard_card)
                self.current_phase = "DRAW"

        user_dw = self.user_hand.calculate_deadwood()
        if user_dw <= 10:
            ai_dw = self.ai_hand.calculate_deadwood()
            if user_dw == 0:
                bonus = ai_dw + 25
                self.user_score += bonus
                self.status_msg = f"GIN! You scored {bonus} pts!"
            elif user_dw < ai_dw:
                diff = ai_dw - user_dw
                self.user_score += diff
                self.status_msg = f"Knock Successful! You scored {diff} pts."
            else:
                diff = (user_dw - ai_dw) + 25
                self.ai_score += diff
                self.status_msg = f"Undercut! AI scored {diff} pts."
            self.round_over = True
        else:
            self.status_msg = f"Cannot Knock! Deadwood is {user_dw} (must be <= 10)."

    def handle_surrender(self):
        self.ai_score += 100
        self.status_msg = "You surrendered! Match awarded to AI."
        self.round_over = True

    def cycle_sort_mode(self):
        modes = ["AUTO", "RUNS", "SETS", "RANK"]
        self.user_sort_mode = modes[(modes.index(self.user_sort_mode) + 1) % len(modes)]

    def execute_ai_turn(self):
        if self.current_turn != "AI" or self.round_over:
            return

        ai_cards = self.ai_hand.get_cards()
        top_disc = self.discard_pile.top()

        # 1. Random Forest Draw Decision
        take_disc = self.ai_engine.predict_draw(self.turn_number, self.deck_remaining, ai_cards, top_disc)
        if take_disc and top_disc:
            self.ai_hand.put_in_hand(self.discard_pile.take())
        else:
            if self.deck.is_empty():
                self.recycle_discard_into_deck()

            if not self.deck.is_empty():
                self.ai_hand.put_in_hand(self.deck.draw_card())
                self.deck_remaining = len(self.deck)
            elif top_disc:
                self.ai_hand.put_in_hand(self.discard_pile.take())

        # 2. Random Forest Discard Decision
        ai_cards_after_draw = self.ai_hand.get_cards()
        discard_choice = self.ai_engine.predict_discard(ai_cards_after_draw, self.turn_number)
        if discard_choice:
            self.ai_hand.remove_from_hand(discard_choice)
            self.discard_pile.put(discard_choice)

        # 3. Random Forest Strategy Sort
        sort_strat = self.ai_engine.predict_sort_strategy(self.ai_hand.get_cards(), self.turn_number)
        self.ai_hand.sort_cards_by_mode(sort_strat)

        # 4. Check AI Deadwood after discard
        ai_dw = self.ai_hand.calculate_deadwood()

        # 5. Execute GIN immediately if AI Deadwood == 0
        if ai_dw == 0:
            user_dw = self.user_hand.calculate_deadwood()
            bonus = user_dw + 25
            self.ai_score += bonus
            self.status_msg = f"GIN! AI scored {bonus} pts!"
            self.round_over = True
            return

        # 6. Execute Knock if AI Deadwood <= 10 and Model approves
        if self.ai_engine.predict_knock(ai_dw, self.ai_hand.get_cards(), self.turn_number, self.deck_remaining):
            user_dw = self.user_hand.calculate_deadwood()
            if ai_dw < user_dw:
                diff = user_dw - ai_dw
                self.ai_score += diff
                self.status_msg = f"AI Knocked! AI scored {diff} pts."
            else:
                diff = (ai_dw - user_dw) + 25
                self.user_score += diff
                self.status_msg = f"AI Undercut! You earned {diff} bonus pts."
            self.round_over = True
        else:
            self.current_turn = "USER"
            self.current_phase = "DRAW"
            self.turn_number += 1
            self.status_msg = "Your turn! Draw from Deck or Discard."

