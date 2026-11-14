import random

class Cards:
    def __init__(self, chrtype: str):
        self.__cards = []
        self.__type = chrtype
        if chrtype.lower() == "diamond":
            self.sort("D")
        if chrtype.lower() == "club":
            self.sort("C")
        if chrtype.lower() == "heart":
            self.sort("H")
        if chrtype.lower() == "spades":
            self.sort("S")

    def sort(self, t: str):
        for i in range(1, 14):
            self.__cards.append(t + str(i))

    def showvalue(self, index: int):
        return self.__cards[index]


class Deck:
    def __init__(self):
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

    def restock_deck(self, cards):
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
        self.__deck = [card for card in self.__deck if card not in cards]