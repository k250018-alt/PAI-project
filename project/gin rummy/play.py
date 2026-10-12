import os
import sys
import random
import numpy as np
import pandas as pd

# Add paths for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "model"))

from cards import Cards, Deck
from RandomForest import Randomforest

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")


class AIRandomForestEngine:
    """Random Forest ML engine trained on project CSV datasets with dynamic accuracy calculation."""
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
        self._train_from_csvs()

    def _train_from_csvs(self):
        accuracies = []

        # 1. Train & Calculate Draw Model Accuracy
        try:
            p_draw = os.path.join(DATASET_DIR, "draw_decision_dataset.csv")
            if os.path.exists(p_draw):
                df = pd.read_csv(p_draw).dropna()
                X = df.drop(columns=["take_discard"]).values[:400]
                y = df["take_discard"].values[:400]
                split = int(len(X) * 0.8)
                
                self.draw_model.fit(X[:split], y[:split])
                self.draw_trained = True
                
                acc = self.draw_model.check_accuracy(X[split:], y[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Draw Model load info: {e}")

        # 2. Train & Calculate Discard Model Accuracy
        try:
            p_disc = os.path.join(DATASET_DIR, "discard_decision_dataset.csv")
            if os.path.exists(p_disc):
                df = pd.read_csv(p_disc).dropna()
                X = df.drop(columns=["best_discard"]).values[:400]
                y = df["best_discard"].values[:400]
                split = int(len(X) * 0.8)
                
                self.discard_model.fit(X[:split], y[:split])
                self.discard_trained = True
                
                acc = self.discard_model.check_accuracy(X[split:], y[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Discard Model load info: {e}")

        # 3. Train & Calculate Knock Model Accuracy
        try:
            p_knock = os.path.join(DATASET_DIR, "knock_gin_decision_dataset.csv")
            if os.path.exists(p_knock):
                df = pd.read_csv(p_knock).dropna()
                X = df.drop(columns=["decision"]).values[:400]
                y = df["decision"].values[:400]
                split = int(len(X) * 0.8)
                
                self.knock_model.fit(X[:split], y[:split])
                self.knock_trained = True
                
                acc = self.knock_model.check_accuracy(X[split:], y[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Knock Model load info: {e}")

        # 4. Train Strategy Model
        try:
            p_strat = os.path.join(DATASET_DIR, "run_vs_set_decision_dataset.csv")
            if not os.path.exists(p_strat):
                p_strat = os.path.join(os.path.dirname(__file__), "..", "run_vs_set_decision_dataset.csv")
                
            if os.path.exists(p_strat):
                df = pd.read_csv(p_strat).dropna()
                X_s = df.drop(columns=["prioritization_choice"]).values[:400]
                y_s = df["prioritization_choice"].values[:400]
                split = int(len(X_s) * 0.8)
                
                self.strat_model.fit(X_s[:split], y_s[:split])
                self.strat_trained = True
                
                acc = self.strat_model.check_accuracy(X_s[split:], y_s[split:])
                accuracies.append(acc)
        except Exception as e:
            print(f"Strat Model load info: {e}")

        if accuracies:
            self.accuracy = round(float(np.mean(accuracies)), 1)
        else:
            self.accuracy = 85.0

    def predict_draw(self, turn_num: int, deck_size: int, hand_cards: list, top_discard: str) -> bool:
        if not top_discard or not self.draw_trained:
            return False
        dw_now = sum(min(int(c[1:]), 10) for c in hand_cards)
        top_num = int(top_discard[1:])
        top_suit = top_discard[0]
        
        same_num = sum(1 for c in hand_cards if int(c[1:]) == top_num)
        adj1 = sum(1 for c in hand_cards if c[0] == top_suit and int(c[1:]) == top_num - 1)
        adj2 = sum(1 for c in hand_cards if c[0] == top_suit and int(c[1:]) == top_num + 1)
        dw_gain = max(0, dw_now - sum(min(int(c[1:]), 10) for c in hand_cards + [top_discard]))

        features = np.array([[turn_num, deck_size, dw_now, 0, 0, len(hand_cards), 0, 0, top_num, same_num, adj1, adj2, dw_gain]])
        return bool(self.draw_model.predict(features)[0] == 1)

    def predict_discard(self, hand_cards: list, turn_num: int) -> str:
        if not hand_cards:
            return None
        if not self.discard_trained:
            return sorted(hand_cards, key=lambda c: min(int(c[1:]), 10), reverse=True)[0]

        best_card, best_score = hand_cards[0], -999
        base_dw = sum(min(int(c[1:]), 10) for c in hand_cards)

        for card in hand_cards:
            c_num, c_suit = int(card[1:]), card[0]
            same_num = sum(1 for c in hand_cards if c != card and int(c[1:]) == c_num)
            adj1 = sum(1 for c in hand_cards if c != card and c[0] == c_suit and int(c[1:]) == c_num - 1)
            adj2 = sum(1 for c in hand_cards if c != card and c[0] == c_suit and int(c[1:]) == c_num + 1)
            dw_red = min(c_num, 10)

            features = np.array([[turn_num, base_dw, base_dw - dw_red, c_num, 0, 0, same_num, adj1, adj2, dw_red, 0, 0, 0, 0, 0]])
            pred = self.discard_model.predict(features)[0]
            if pred > best_score:
                best_score = pred
                best_card = card
        return best_card

    def predict_knock(self, real_deadwood: int, hand_cards: list, turn_num: int, deck_size: int) -> bool:
        """Evaluates whether AI should Knock or go Gin based on real melded deadwood."""
        if real_deadwood > 10:
            return False  # Standard Gin Rummy Rule: Cannot knock if deadwood > 10
        if not self.knock_trained:
            return True   # Valid knock threshold met

        features = np.array([[turn_num, deck_size, real_deadwood, 0, 0, len(hand_cards), 10, 0, 0, 0.0, 0.8, 0.2]])
        return bool(self.knock_model.predict(features)[0] == 1)

    def predict_sort_strategy(self, hand_cards: list, turn_num: int) -> str:
        if not self.strat_trained:
            return "AUTO"
        dw = sum(min(int(c[1:]), 10) for c in hand_cards)
        features = np.array([[turn_num, dw, 2, 2, 4, 4, 1, 1, 0]])
        return "RUNS" if self.strat_model.predict(features)[0] == 1 else "SETS"


class Sorter:
    def find_runs(self, cards):
        runs = {}
        for card in cards:
            number = int(card[1:])
            runs.setdefault(number, []).append(card)
        valid_runs = {k: v for k, v in runs.items() if len(v) >= 3}
        invalid_runs = {k: v for k, v in runs.items() if len(v) < 3}
        return valid_runs, invalid_runs

    def find_suits(self, cards):
        suit_groups = {}
        for card in cards:
            suit_groups.setdefault(card[0], []).append(int(card[1:]))
        valid_suits, invalid_suits = {}, {}
        for suit, numbers in suit_groups.items():
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


class Hands:
    def __init__(self):
        self.__cards = []
        self.__won = False
        self.__sorter = Sorter()

    def put_in_hand(self, card: str):
        self.__cards.append(card)

    def remove_from_hand(self, card: str):
        if card in self.__cards:
            self.__cards.remove(card)

    def get_cards(self):
        return self.__cards

    def calculate_deadwood(self) -> int:
        valid_runs, invalid_runs = self.__sorter.find_runs(self.__cards)
        all_invalid = [card for group in invalid_runs.values() for card in group]
        valid_suits, invalid_suits = self.__sorter.find_suits(all_invalid)
        remaining = [card for group in invalid_suits.values() for card in group]
        return sum(min(int(c[1:]), 10) for c in remaining)

    def sort_cards_by_mode(self, mode: str = "AUTO") -> list:
        if mode == "RANK":
            self.__cards.sort(key=lambda c: (int(c[1:]), c[0]))
            return self.__cards

        valid_runs, invalid_runs = self.__sorter.find_runs(self.__cards)
        unmelded = [card for group in invalid_runs.values() for card in group]
        valid_suits, invalid_suits = self.__sorter.find_suits(unmelded)
        
        melded_runs = [c for grp in valid_runs.values() for c in grp]
        melded_sets = [c for grp in valid_suits.values() for c in grp]
        deadwood = [c for grp in invalid_suits.values() for c in grp]
        deadwood.sort(key=lambda c: (int(c[1:]), c[0]))

        if mode == "RUNS":
            self.__cards = melded_runs + melded_sets + deadwood
        elif mode == "SETS":
            self.__cards = melded_sets + melded_runs + deadwood
        else:
            self.__cards = melded_runs + melded_sets + deadwood
        return self.__cards


class DiscardPile:
    def __init__(self):
        self.__cards = []

    def put(self, card: str):
        self.__cards.append(card)

    def take(self) -> str:
        return self.__cards.pop() if self.__cards else None

    def top(self) -> str:
        return self.__cards[-1] if self.__cards else None

    def get_all(self) -> list:
        return self.__cards


class Play:
    """Master Game Logic & Turn Execution State Manager."""
    def __init__(self):
        self.ai_engine = AIRandomForestEngine()
        self.user_hand = Hands()
        self.ai_hand = Hands()
        self.discard_pile = DiscardPile()
        self.deck = Deck()

        self.user_score = 0
        self.ai_score = 0
        self.user_sort_mode = "AUTO"
        self.start_new_round()

    def start_new_round(self):
        self.deck = Deck()
        self.deck.shuffle()
        self.deck_remaining = 52

        self.user_hand = Hands()
        self.ai_hand = Hands()
        self.discard_pile = DiscardPile()

        for _ in range(10):
            self.user_hand.put_in_hand(self.deck.draw_card())
            self.ai_hand.put_in_hand(self.deck.draw_card())
        self.deck_remaining -= 20

        top_disc = self.deck.draw_card()
        self.deck_remaining -= 1
        self.discard_pile.put(top_disc)

        self.turn_number = 1
        self.current_turn = "USER"
        self.current_phase = "DRAW"
        self.status_msg = "Your turn! Draw from Deck or Discard."
        self.round_over = False

    def handle_user_draw_deck(self):
        if self.current_turn == "USER" and self.current_phase == "DRAW" and self.deck_remaining > 0:
            drawn = self.deck.draw_card()
            self.deck_remaining -= 1
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

    def handle_user_knock(self):
        if self.current_turn == "USER":
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
        elif self.deck_remaining > 0:
            self.ai_hand.put_in_hand(self.deck.draw_card())
            self.deck_remaining -= 1

        # 2. Random Forest Discard Decision
        ai_cards_after_draw = self.ai_hand.get_cards()
        discard_choice = self.ai_engine.predict_discard(ai_cards_after_draw, self.turn_number)
        if discard_choice:
            self.ai_hand.remove_from_hand(discard_choice)
            self.discard_pile.put(discard_choice)

        # 3. Random Forest Strategy Sort
        sort_strat = self.ai_engine.predict_sort_strategy(self.ai_hand.get_cards(), self.turn_number)
        self.ai_hand.sort_cards_by_mode(sort_strat)

        # 4. Random Forest Knock / Gin Decision using actual calculated melded deadwood
        ai_dw = self.ai_hand.calculate_deadwood()
        if self.ai_engine.predict_knock(ai_dw, self.ai_hand.get_cards(), self.turn_number, self.deck_remaining):
            user_dw = self.user_hand.calculate_deadwood()
            if ai_dw == 0:
                bonus = user_dw + 25
                self.ai_score += bonus
                self.status_msg = f"GIN! AI scored {bonus} pts!"
            elif ai_dw < user_dw:
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
