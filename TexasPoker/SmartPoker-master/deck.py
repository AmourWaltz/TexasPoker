#---------------------------------------------------------------------------
#	deck.py
#
#	Description:
#
#	Creates a new deck of 52 cards, includes methods for shuffling and
#	drawing cards.
#
#---------------------------------------------------------------------------

from card import Card
import random

random.seed()

#---------------------------------------------------------------------------
#	Deck class
#---------------------------------------------------------------------------
class Deck():
	
	#---------------------------------------------------------------------------
	#	Constructor
	#---------------------------------------------------------------------------
	def __init__(self):
		
		self.cards = [Card("H", "2"), Card("D", "2"), Card("S", "2"), Card("C", "2"),
					  Card("H", "3"), Card("D", "3"), Card("S", "3"), Card("C", "3"),
					  Card("H", "4"), Card("D", "4"), Card("S", "4"), Card("C", "4"),
					  Card("H", "5"), Card("D", "5"), Card("S", "5"), Card("C", "5"),
					  Card("H", "6"), Card("D", "6"), Card("S", "6"), Card("C", "6"),
					  Card("H", "7"), Card("D", "7"), Card("S", "7"), Card("C", "7"),
					  Card("H", "8"), Card("D", "8"), Card("S", "8"), Card("C", "8"),
					  Card("H", "9"), Card("D", "9"), Card("S", "9"), Card("C", "9"),
					  Card("H", "10"), Card("D", "10"), Card("S", "10"), Card("C", "10"),
					  Card("H", "J"), Card("D", "J"), Card("S", "J"), Card("C", "J"),
					  Card("H", "Q"), Card("D", "Q"), Card("S", "Q"), Card("C", "Q"),
					  Card("H", "K"), Card("D", "K"), Card("S", "K"), Card("C", "K"),
					  Card("H", "A"), Card("D", "A"), Card("S", "A"), Card("C", "A"),]
		
		assert len(self.cards) == 52
		for card in self.cards:
			assert card.isValid()
			
	#---------------------------------------------------------------------------
	#	shuffle()
	#	
	#	Shuffles the deck of cards to a randomized order
	#---------------------------------------------------------------------------
	def shuffle(self):
		
		random.shuffle(self.cards)
		
	#---------------------------------------------------------------------------
	#	draw()
	#	
	#	Takes a card from the deck, removing it from the deck and returning the
	#	drawn card. If the deck is empty, None is returned instead.
	#---------------------------------------------------------------------------
	def draw(self):
		
		if len(self.cards) > 0:
			card = self.cards[0]
			self.cards = self.cards[1:]
			return card
		else:
			return None

	#---------------------------------------------------------------------------
	#	pull()
	#	
	#	Takes a specific card from the deck, the card's rank can be given as
	#	a numeric rank or a string number depending on which is easier
	#---------------------------------------------------------------------------
	def pull(self, suit, number="!", rank=-2):
		i = 0
		for c in self.cards:
			if c.suit == suit and (rank == c.getRank() or number == c.number):
				self.cards = self.cards[:i] + self.cards[i+1:]		
				break
			i+=1
	
	
#========================================
#	TESTS
#========================================	

if __name__ == '__main__':
	
	
	print "Testing..."
	deck = Deck()
	card = deck.draw()
	assert card.isMatch(Card("H", "2"))	
	card = deck.draw()
	assert card.isMatch(Card("D", "2"))	
	card = deck.draw()
	assert card.isMatch(Card("S", "2"))
	card = deck.draw()
	assert card.isMatch(Card("C", "2"))
	card = deck.draw()
	assert card.isMatch(Card("H", "3"))
	
	deck = Deck()
	deck.shuffle()
	
	for i in range(52):
		card = deck.draw()
		assert card != None
	
	card = deck.draw()
	assert card == None
	print "Test complete."		