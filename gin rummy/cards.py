import random

class Cards:
    def __init__(self, chrtype: str):
        self.__cards = []
        self.__type = chrtype
        t = chrtype.strip().lower()
        if t == "diamond" or t == "diamonds":
            self.sort("D")
        elif t == "club" or t == "clubs":
            self.sort("C")
        elif t == "heart" or t == "hearts":
            self.sort("H")
        elif t == "spade" or t == "spades":
            self.sort("S")

    def sort(self, t: str):
        for i in range(1, 14):
            self.__cards.append(t + str(i))

    def showvalue(self, index: int):
        return self.__cards[index]


class Deck:
    def __init__(self, auto_shuffle: bool = True):
        self._build_deck()
        if auto_shuffle:
            self.shuffle()

    def _build_deck(self):
        self.__diamonds = Cards("Diamond")
        self.__clubs = Cards("Club")
        self.__hearts = Cards("Heart")
        self.__spades = Cards("Spades")
        self.__deck = []
        for i in range(0, 52):
            if 0 <= i <= 12:
                self.__deck.append(self.__diamonds.showvalue(i))
            elif 13 <= i <= 25:
                self.__deck.append(self.__clubs.showvalue(i - 13))
            elif 26 <= i <= 38:
                self.__deck.append(self.__hearts.showvalue(i - 26))
            elif 39 <= i <= 51:
                self.__deck.append(self.__spades.showvalue(i - 39))

    def shuffle(self):
        random.shuffle(self.__deck)

    def draw_card(self):
        return self.__deck.pop() if self.__deck else None

    def __len__(self):
        return len(self.__deck)

    def is_empty(self) -> bool:
        return len(self.__deck) == 0

    def recycle_cards(self, cards: list):
        """Adds discarded/recycled cards back into the deck and shuffles them."""
        if cards:
            self.__deck.extend(cards)
            self.shuffle()

    def restock_deck(self, cards):
        """Restocks the deck excluding cards currently in play, then shuffles."""
        self._build_deck()
        self.__deck = [card for card in self.__deck if card not in cards]
        self.shuffle()

    def reset(self):
        """Re-initializes all 52 cards and shuffles."""
        self._build_deck()
        self.shuffle()