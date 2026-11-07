import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "gin rummy"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from cards import Cards, Deck
from play import Play, Sorter, Hands, DiscardPile


class TestDeckBasics(unittest.TestCase):
    def test_deck_contains_52_unique_cards(self):
        deck = Deck(auto_shuffle=False)
        cards = []
        while not deck.is_empty():
            cards.append(deck.draw_card())
        self.assertEqual(len(cards), 52)
        self.assertEqual(len(set(cards)), 52)

    def test_deck_auto_shuffled(self):
        deck1 = Deck(auto_shuffle=True)
        deck2 = Deck(auto_shuffle=True)
        cards1 = [deck1.draw_card() for _ in range(20)]
        cards2 = [deck2.draw_card() for _ in range(20)]
        self.assertNotEqual(cards1, cards2)

    def test_deck_reset_and_restock(self):
        deck = Deck()
        for _ in range(15):
            deck.draw_card()
        self.assertEqual(len(deck), 37)
        deck.reset()
        self.assertEqual(len(deck), 52)


class TestDeckRecycling(unittest.TestCase):
    def test_deck_exhaustion_and_recycle_on_draw(self):
        game = Play()
        while game.deck_remaining > 0:
            drawn = game.deck.draw_card()
            game.discard_pile.put(drawn)
            game.deck_remaining = len(game.deck)

        self.assertEqual(game.deck_remaining, 0)
        self.assertTrue(game.deck.is_empty())
        initial_discard_count = game.discard_pile.count()
        self.assertGreater(initial_discard_count, 1)

        top_before = game.discard_pile.top()
        game.handle_user_draw_deck()

        self.assertGreater(game.deck_remaining, 0)
        self.assertEqual(game.discard_pile.count(), 1)
        self.assertEqual(game.discard_pile.top(), top_before)

    def test_ai_turn_does_not_shrink_hand_on_empty_deck(self):
        game = Play()
        while not game.deck.is_empty():
            game.discard_pile.put(game.deck.draw_card())
        game.deck_remaining = 0

        game.current_turn = "AI"
        initial_ai_card_count = len(game.ai_hand.get_cards())
        self.assertEqual(initial_ai_card_count, 10)

        game.execute_ai_turn()

        final_ai_card_count = len(game.ai_hand.get_cards())
        self.assertEqual(final_ai_card_count, 10)


class TestMeldAndDeadwoodCalculation(unittest.TestCase):
    def test_pure_sets_deadwood_zero(self):
        hand = ['S10', 'C10', 'H10', 'D5', 'C5', 'H5']
        dw = Sorter.calculate_cards_deadwood(hand)
        self.assertEqual(dw, 0)

    def test_pure_runs_deadwood_zero(self):
        hand = ['D1', 'D2', 'D3', 'D4']
        dw = Sorter.calculate_cards_deadwood(hand)
        self.assertEqual(dw, 0)

    def test_optimal_meld_selection(self):
        # S7, S8, S9 as a run is better than S7,D7,C7 as a set here
        hand = ['S7', 'D7', 'C7', 'S8', 'S9']
        dw = Sorter.calculate_cards_deadwood(hand)
        self.assertEqual(dw, 14)

    def test_face_card_and_ace_values(self):
        # Ace = 1, Jack/Queen/King = 10 each
        hand = ['S1', 'D11', 'C12', 'H13']
        dw = Sorter.calculate_cards_deadwood(hand)
        self.assertEqual(dw, 1 + 10 + 10 + 10)

    def test_gin_hand_10_cards(self):
        hand = ['S3', 'C3', 'H3', 'D6', 'C6', 'H6', 'S9', 'S10', 'S11', 'S12']
        dw = Sorter.calculate_cards_deadwood(hand)
        self.assertEqual(dw, 0)


class TestKnockingAndScoring(unittest.TestCase):
    def test_gin_scores_bonus_25(self):
        game = Play()
        game.user_hand = Hands()
        for c in ['S3', 'C3', 'H3', 'D6', 'C6', 'H6', 'S9', 'S10', 'S11', 'S12']:
            game.user_hand.put_in_hand(c)

        game.ai_hand = Hands()
        for c in ['D13', 'C13', 'H1', 'S2', 'D4', 'C7', 'H8', 'S10', 'D11', 'C12']:
            game.ai_hand.put_in_hand(c)

        ai_dw = game.ai_hand.calculate_deadwood()
        game.handle_user_knock()

        self.assertTrue(game.round_over)
        expected_score = ai_dw + 25
        self.assertEqual(game.user_score, expected_score)

    def test_undercut_scores_bonus_25_for_defender(self):
        game = Play()
        game.user_hand = Hands()
        for c in ['S3', 'C3', 'H3', 'D6', 'C6', 'H6', 'S6', 'S1', 'S2', 'D5']:
            game.user_hand.put_in_hand(c)
        user_dw = game.user_hand.calculate_deadwood()
        self.assertEqual(user_dw, 8)

        game.ai_hand = Hands()
        for c in ['S4', 'C4', 'H4', 'D7', 'C7', 'H7', 'S8', 'C8', 'H8', 'D2']:
            game.ai_hand.put_in_hand(c)
        ai_dw = game.ai_hand.calculate_deadwood()
        self.assertLess(ai_dw, user_dw)

        game.handle_user_knock()

        self.assertTrue(game.round_over)
        expected_ai_points = (user_dw - ai_dw) + 25
        self.assertEqual(game.ai_score, expected_ai_points)

    def test_illegal_knock_blocked(self):
        game = Play()
        game.user_hand = Hands()
        for c in ['S1', 'D4', 'C6', 'H8', 'S10', 'D11', 'C12', 'H13', 'S13', 'D13']:
            game.user_hand.put_in_hand(c)

        game.handle_user_knock()
        self.assertFalse(game.round_over)
        self.assertIn("Cannot Knock", game.status_msg)


class TestAIEngineBehavior(unittest.TestCase):
    def test_ai_never_discards_melded_cards(self):
        game = Play()
        cards_11 = ['S10', 'C10', 'H10', 'D1', 'D2', 'D3', 'H5', 'H6', 'H7', 'C2', 'S13']
        discard_choice = game.ai_engine.predict_discard(cards_11, turn_num=5)
        self.assertNotIn(discard_choice, ['S10', 'C10', 'H10', 'D1', 'D2', 'D3', 'H5', 'H6', 'H7'])
        self.assertIn(discard_choice, ['S13', 'C2'])

    def test_ai_draw_synergy(self):
        game = Play()
        hand = ['D1', 'D2', 'H5', 'H6', 'H7', 'S10', 'C10', 'H10', 'S12', 'C13']
        top_discard = 'D3'
        should_draw = game.ai_engine.predict_draw(turn_num=3, deck_size=25, hand_cards=hand, top_discard=top_discard)
        self.assertTrue(should_draw)


class TestHandSorting(unittest.TestCase):
    def test_sort_by_rank(self):
        hand = Hands()
        for c in ['S13', 'H4', 'D7', 'S4', 'H6', 'D1']:
            hand.put_in_hand(c)
        sorted_cards = hand.sort_cards_by_mode("RANK")
        ranks = [int(c[1:]) for c in sorted_cards]
        self.assertEqual(ranks, sorted(ranks))

    def test_sort_by_runs(self):
        hand = Hands()
        for c in ['H4', 'S10', 'H6', 'S4', 'D7', 'H5']:
            hand.put_in_hand(c)
        sorted_cards = hand.sort_cards_by_mode("RUNS")
        self.assertEqual(sorted_cards, ['S4', 'S10', 'H4', 'H5', 'H6', 'D7'])

    def test_sort_by_sets(self):
        hand = Hands()
        for c in ['S13', 'H4', 'D7', 'S4', 'S7', 'C7']:
            hand.put_in_hand(c)
        sorted_cards = hand.sort_cards_by_mode("SETS")
        self.assertEqual(sorted_cards[:3], ['S7', 'C7', 'D7'])
        self.assertEqual(sorted_cards[3:5], ['S4', 'H4'])
        self.assertEqual(sorted_cards[5], 'S13')

    def test_cycle_sort_mode(self):
        game = Play()
        self.assertEqual(game.user_sort_mode, "AUTO")
        game.cycle_sort_mode()
        self.assertEqual(game.user_sort_mode, "RUNS")
        game.cycle_sort_mode()
        self.assertEqual(game.user_sort_mode, "SETS")
        game.cycle_sort_mode()
        self.assertEqual(game.user_sort_mode, "RANK")
        game.cycle_sort_mode()
        self.assertEqual(game.user_sort_mode, "AUTO")


if __name__ == "__main__":
    unittest.main()
