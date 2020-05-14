#---------------------------------------------------------------------------
#	hand.py
#
#	Description:
#
#	Hand class takes a list of cards and an optional ID and evaluates it. 
#	Can be given any number of cards as a list to initialize, in order to
#	make valid poker hands of 5 cards
#---------------------------------------------------------------------------

import card

#---------------------------------------------------------------------------
#	Hand Class
#---------------------------------------------------------------------------
class Hand():
	
	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Given a list of cards and an optional id
	#---------------------------------------------------------------------------
	def __init__(self, cards, ID=-1):
		self.cards = cards
		self.id = ID	
		
	#---------------------------------------------------------------------------
	#	evaluate()
	#
	#	Takes the hand and evaluates it as a tuple of (rank, high1, high2, high3, high4, high5)
	#---------------------------------------------------------------------------
	def evaluate(self):
		
		#---------------------------------------------------------------------------
		#	Algorithm:
		#
		#	Straight-flush? (8, highest card, 0, 0, 0, 0)
		#	4-of-a-kind? (7, card in 4s, kicker, 0, 0, 0)
		#	Full house? (6, card in 3s, card in 2s, 0, 0, 0)
		#	Flush? (5, highest card, 2nd, 3rd, 4th, 5th)
		#	Straight? (4, highest card, 0, 0, 0, 0)
		#	3-of-a-kind? (3, card in 3s, kicker, 2nd, 0, 0)
		#	2 pairs? (2, high pair, low pair, kicker, 0, 0)
		#	Pair? (1, pair, kicker, 2nd, 3rd, 0)
		#	Nothing? (0, 1st, 2nd, 3rd, 4th, 5th)
		#
		#	All cards are stored by rank, which then uses tuple comparison to determine
		#	the greatest value
		#---------------------------------------------------------------------------
		
		cardsInStraightFlush = self.straightFlush()
		if cardsInStraightFlush != None:
			return (8, cardsInStraightFlush[0], 0, 0, 0, 0)
		
		cardsInFours = self.fourOfAKind()
		if cardsInFours != None:
			return (7, cardsInFours[0], cardsInFours[1], 0, 0, 0)
		
		cardsInFullHouse = self.fullHouse()
		if cardsInFullHouse != None:
			return (6, cardsInFullHouse[0], cardsInFullHouse[1], 0, 0, 0)
		
		cardsInFlush = self.flush()
		if cardsInFlush != None:
			return (5, cardsInFlush[0], cardsInFlush[1], cardsInFlush[2], cardsInFlush[3], cardsInFlush[4])
		
		cardsInStraight = self.straight()
		if cardsInStraight != None:
			return (4, cardsInStraight[0], 0, 0, 0, 0)
		
		cardsInThrees = self.threeOfAKind()
		if cardsInThrees != None:
			return (3, cardsInThrees[0], cardsInThrees[1], cardsInThrees[2], 0, 0)
			
		cardsInTwoPairs = self.twoPairs()
		if cardsInTwoPairs != None:
			return (2, cardsInTwoPairs[0], cardsInTwoPairs[1], cardsInTwoPairs[2], 0, 0)
			
		cardsInPair = self.pair()
		if cardsInPair != None:
			return (1, cardsInPair[0], cardsInPair[1], cardsInPair[2], cardsInPair[3], 0)
			
		cardsInHighCard = self.highCard()
		return (0, cardsInHighCard[0], cardsInHighCard[1], cardsInHighCard[2], cardsInHighCard[3], cardsInHighCard[4])
		
	#--------------------------------------------------------------------------
	#	straightFlush()
	#--------------------------------------------------------------------------	
	def straightFlush(self):
		suits = self.groupBySuit()
		hasStraightFlush = False
		bestStraight = 0
		for suit in suits.keys():
			
			highestStraight = 0
			#We have to have at least 5 cards to have a straight
			if len(suits[suit]) >= 5:
				ranks = self.groupByRank(suits[suit])
				numInARow = 0
				for i in range(14, 0, -1):
					if len(ranks[i]) > 0:
						numInARow += 1
					else:
						numInARow = 0
						
					if numInARow >= 5:
						highestStraight = i + 4
						hasStraightFlush = True
						
						#No more need to keep going, we have the highest straight flush for this suit
						break
						
			if highestStraight > bestStraight:
				bestStraight = highestStraight
				
		if hasStraightFlush:
			return [bestStraight]
		else:
			return None
					
		
	#--------------------------------------------------------------------------
	#	fourOfAKind()
	#--------------------------------------------------------------------------	
	def fourOfAKind(self):
		return self.nOfAKind(4)	
				
	#--------------------------------------------------------------------------
	#	fullHouse()
	#--------------------------------------------------------------------------	
	def fullHouse(self):
		ranks = self.groupByRank()
		for r in range(14, 1, -1):
			
			if len(ranks[r]) >= 3:
				#Delete the three of a kind and look for a pair
				ranks[r] = ranks[r][3:]
				
				for p in range(14, 1, -1):
					if len(ranks[p]) >= 2:
						return [r, p]
		
		return None
		
	#--------------------------------------------------------------------------
	#	flush()
	#--------------------------------------------------------------------------	
	def flush(self):
		hasFlush = False
		bestOrder = []
		suits = self.groupBySuit()
		for s in suits.keys():
			if len(suits[s]) >= 5:
				hasFlush = True
				
				#This suit has a flush, now we need to figure out its best flush and compare it
				#to other possible flushes in the other suits
				ranks = self.groupByRank(suits[s])
				cardOrder = []
				for r in range(14, 0, -1):
					for i in range(len(ranks[r])):
						cardOrder.append(r)
						
				if len(bestOrder) == 0:
					bestOrder = cardOrder[:5]
				elif ((cardOrder[0], cardOrder[1], cardOrder[2], cardOrder[3], cardOrder[4]) >
					  (bestOrder[0], bestOrder[1], bestOrder[2], bestOrder[3], bestOrder[4])):
					bestOrder = cardOrder[:5]
					
		if hasFlush:
			return bestOrder
		else:
			return None

	#--------------------------------------------------------------------------
	#	straight()
	#--------------------------------------------------------------------------	
	def straight(self):
		ranks = self.groupByRank()
		numInARow = 0
		for r in range(14, 0, -1):
			if len(ranks[r]) > 0:
				numInARow += 1
			else:
				numInARow = 0
				
			if numInARow >= 5:
				return [r + 4]
				
		return None
		
	#--------------------------------------------------------------------------
	#	threeOfAKind()
	#--------------------------------------------------------------------------	
	def threeOfAKind(self):
		return self.nOfAKind(3)
	
	#--------------------------------------------------------------------------
	#	twoPairs()
	#--------------------------------------------------------------------------	
	def twoPairs(self):
		ranks = self.groupByRank()
		for r in range(14, 1, -1):
			
			if len(ranks[r]) >= 2:
				#Delete the pair and look for another
				ranks[r] = ranks[r][2:]
				
				for p in range(14, 1, -1):
					if len(ranks[p]) >= 2:
						
						#Delete the second pair and look for a kicker
						ranks[p] = ranks[p][2:]
						
						for k in range(14, 1, -1):
							if len(ranks[k]) >= 1:
								return [r, p, k]
		
		return None	
		
	#--------------------------------------------------------------------------
	#	pair()
	#--------------------------------------------------------------------------	
	def pair(self):
		return self.nOfAKind(2)	
		
	#--------------------------------------------------------------------------
	#	highCard()
	#--------------------------------------------------------------------------	
	def highCard(self):
		ranks = self.groupByRank()
		best = []
		for r in range(14, 0, -1):
			for n in range(len(ranks[r])):
				best.append(r)
				
		return best
					
		
	#--------------------------------------------------------------------------
	#	nOfAKind()
	#
	#	Generalizes search for 4-of-a-kind, 3-of-a-kind, 2-of-a-kinds, etc.
	#--------------------------------------------------------------------------
	def nOfAKind(self, n):
		ranks = self.groupByRank()
		for r in range(14, 1, -1):
		
			if len(ranks[r]) >= n:
				#Delete the n of a kind and start looking for the kicker
				ranks[r] = ranks[r][n:]
				returnVal = [r]
				for numKickers in range(5-n):
					for k in range(14, 1, -1):
						if len(ranks[k]) >= 1:
							returnVal.append(k)
							ranks[k] = ranks[k][1:]
							break
				
				return returnVal
				
		return None
	
			
		
	#--------------------------------------------------------------------------
	#	groupBySuit()
	#
	#	Takes all of the cards and splits them into four groups, based on suit
	#	Optionally can be given a different set of cards to group (say a subset)
	#--------------------------------------------------------------------------
	def groupBySuit(self, theCards = None):
		
		if theCards == None:
			theCards = self.cards
			
		groups = {card.Card.DIAMONDS:[],
				  card.Card.HEARTS:[],
				  card.Card.CLUBS:[],
				  card.Card.SPADES:[]}
				
		for c in theCards:
			groups[c.suit].append(c)
			
		return groups
		
	#--------------------------------------------------------------------------
	#	groupByRank()
	#
	#	Takes all of the cards and splits them into fourteen groups, based on rank
	#	Optionally can be given a different set of cards to group (say a subset)
	#	Note: Aces will be double counted as being both 1 and 14, so this operation
	#	DOES NOT preserve number of cards.
	#--------------------------------------------------------------------------
	def groupByRank(self, theCards = None):

		if theCards == None:
			theCards = self.cards

		groups = {1:[], 2:[], 3:[], 4:[], 5:[],
				  6:[], 7:[], 8:[], 9:[], 10:[],
				  11:[], 12:[], 13:[], 14:[]}

		for c in theCards:
			rank = c.getRank()
			groups[rank].append(c)
			
			#Ace is both a high and a low card
			if rank == 14:
				groups[1].append(c)

		return groups	
		
	#--------------------------------------------------------------------------
	#	handName()
	#
	#	Evaluates the hand and returns a string with the name of the hand for
	#	display purposes.
	#
	#--------------------------------------------------------------------------	
	def handName(self):
		
		text = "Nothing."
		ranking = self.evaluate()
		if ranking[0] == 8:
			if ranking[1] == 14:
				text = "royal flush!"
			else:
				text = self.getRankString(ranking[1])+"-high straight flush."
			
		elif ranking[0] == 7:
			text = "four "+self.getRankStringPlural(ranking[1])+"."
		
		elif ranking[0] == 6:
			text = self.getRankStringPlural(ranking[1])+" full of "+self.getRankStringPlural(ranking[2])+"."
		
		elif ranking[0] == 5:
			text = "flush."
		
		elif ranking[0] == 4:
			text = self.getRankString(ranking[1])+"-high straight."
		
		elif ranking[0] == 3:
			text = "three "+self.getRankStringPlural(ranking[1])+"."
		
		elif ranking[0] == 2:
			text = "a pair of "+self.getRankStringPlural(ranking[1])+" and a pair of "+self.getRankStringPlural(ranking[2])+"."
		
		elif ranking[0] == 1:
			text = "a pair of "+self.getRankStringPlural(ranking[1])+"."
		
		elif ranking[0] == 0:
			text = self.getRankString(ranking[1])+" high."
		
		else:
			text = "nothing."
			
		return text
	
	#--------------------------------------------------------------------------
	#	getRankStringPlural()
	#
	#	Pluralizes the result from getRankString()
	#--------------------------------------------------------------------------			
	def getRankStringPlural(self, rank):
		if rank != 6:
			return self.getRankString(rank)+"s"
		else:
			return self.getRankString(rank)+"es"
		
	#--------------------------------------------------------------------------
	#	getRankString()
	#
	#	Given a number between 1 and 14, returns a string describing the rank
	#
	#--------------------------------------------------------------------------			
	def getRankString(self, rank):
		if rank == 1:
			rankStr = "ace"
		elif rank == 2:
			rankStr = "two"
		elif rank == 3:
			rankStr = "three"
		elif rank == 4:
			rankStr = "four"
		elif rank == 5:
			rankStr = "five"
		elif rank == 6:
			rankStr = "six"
		elif rank == 7:
			rankStr = "seven"
		elif rank == 8:
			rankStr = "eight"
		elif rank == 9:
			rankStr = "nine"
		elif rank == 10:
			rankStr = "ten"
		elif rank == 11:
			rankStr = "jack"
		elif rank == 12:
			rankStr = "queen"
		elif rank == 13:
			rankStr = "king"
		elif rank == 14:
			rankStr = "ace"
		else:
			rankStr = "?"
		
		return rankStr
		
							
#--------------------------------------------------------------------------
#	Class Functions 
#--------------------------------------------------------------------------	


#--------------------------------------------------------------------------
#	Winner()
#
#	Evaluates a list of hand objects and returns the id of the winning hand
#--------------------------------------------------------------------------
def winner(hands):
	bestHand = [-1]
	bestScore = (-1,-1,-1,-1,-1,-1)
	for h in hands:
		score = h.evaluate()
		if score > bestScore:
			bestScore = score
			bestHand = [h.id]
		elif score == bestScore:
			bestHand.append(h.id)
	
	return bestHand
	
	
#========================================
#	TESTS
#========================================	
if __name__ == "__main__":
	
	print "Testing constructor."
	
	h = Hand([card.Card("H", "A"), card.Card("H", "3"), card.Card("H", "4"), card.Card("H", "5"),card.Card("H", "6"),
			 card.Card("C", "A"), card.Card("S", "A")])
			
	print "Test complete."
	
	print "Testing groupBy functions."
	
	g = h.groupBySuit()
	assert len(g[card.Card.HEARTS]) == 5
	assert len(g[card.Card.DIAMONDS]) == 0
	
	g = h.groupByRank()
	assert len(g[14]) == 3
	assert len(g[1]) == 3
	assert len(g[3]) == 1
	assert len(g[12]) == 0
	
	print "Test complete."
	
	print "Testing straight flushes"
	
	h = Hand([card.Card("H", "A"), card.Card("H", "2"), card.Card("H", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "A")])
	
	result = h.straightFlush()
	assert result != None and result[0] == 5
	
	h = Hand([card.Card("H", "A"), card.Card("H", "2"), card.Card("H", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("H", "6"), card.Card("H", "7"), card.Card("H", "8"), card.Card("H", "9"),card.Card("H", "J")])
	
	result = h.straightFlush()
	assert result != None and result[0] == 9	
	
	h = Hand([card.Card("H", "A"), card.Card("H", "2"), card.Card("H", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("D", "6"), card.Card("D", "7"), card.Card("D", "8"), card.Card("D", "9"),card.Card("D", "10")])
	
	result = h.straightFlush()
	assert result != None and result[0] == 10	
	
	h = Hand([card.Card("H", "A"), card.Card("H", "2"), card.Card("C", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("D", "6"), card.Card("D", "7"), card.Card("C", "8"), card.Card("D", "9"),card.Card("D", "10")])
	
	result = h.straightFlush()
	assert result == None	
	
	print "Test complete."
	
	print "Testing n's of a kind."
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "A")])	
	
	result = h.fourOfAKind()
	assert result != None 
	assert result[0] == 14 
	assert result[1] == 5	
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "3"),card.Card("H", "3"),
			 card.Card("C", "A"), card.Card("S", "A")], card.Card("H", "3"))	
	
	result = h.fourOfAKind()
	assert result != None and result[0] == 14 and result[1] == 3
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "5"),card.Card("H", "3"),
			 card.Card("C", "A"), card.Card("S", "4")], card.Card("H", "3"))	
	
	result = h.fourOfAKind()
	assert result == None
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "5"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "4")], card.Card("H", "3"))	
	
	result = h.threeOfAKind()
	assert result != None and result[0] == 14 and result[1] == 5 and result[2] == 5	
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "5"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "4")], card.Card("H", "3"))	
	
	result = h.pair()
	assert result != None and result[0] == 14 and result[1] == 14 and result[2] == 5 and result[3] == 5	
	
	print "Test complete."	
	
	print "Testing full house."	
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("S", "3"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "K")])	
	
	result = h.fullHouse()
	assert result != None and result[0] == 14 and result[1] == 3
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("S", "3"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "A"), card.Card("S", "5"), card.Card("S", "6")])	
	
	result = h.fullHouse()
	assert result != None and result[0] == 14 and result[1] == 5			
		
	h = Hand([card.Card("H", "K"), card.Card("D", "A"), card.Card("H", "A"), card.Card("S", "3"),card.Card("H", "10"),
			 card.Card("C", "A")])	
	
	result = h.fullHouse()
	assert result == None
	
	print "Test complete."	
		
	print "Testing flush."
	h = Hand([card.Card("H", "A"), card.Card("H", "2"), card.Card("H", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("C", "K"), card.Card("C", "2"), card.Card("C", "3"), card.Card("C", "4"),card.Card("C", "5"),
			 card.Card("H", "6")])	
			
	result = h.flush()
	assert result != None
	assert result[0] == 14 and result[1] == 6 and result[2] == 5 and result[3] == 4 and result[4] == 3
	
	h = Hand([card.Card("H", "A"), card.Card("H", "2"), card.Card("H", "3"), card.Card("H", "4"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("C", "2"), card.Card("C", "3"), card.Card("C", "4"),card.Card("C", "5"),
			 card.Card("H", "4")])	
			
	result = h.flush()
	assert result != None
	assert result[0] == 14 and result[1] == 5 and result[2] == 4 and result[3] == 4 and result[4] == 3	
	
	h = Hand([card.Card("H", "A"), card.Card("D", "2"), card.Card("H", "3"), card.Card("D", "4"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "2"), card.Card("C", "3"), card.Card("S", "4"),card.Card("C", "5"),
			 card.Card("H", "4")])	
			
	result = h.flush()
	assert result == None
	
	print "Test complete."
		
	print "Testing straights."
	
	h = Hand([card.Card("H", "A"), card.Card("C", "2"), card.Card("H", "3"), card.Card("S", "4"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "A")])
	
	result = h.straight()
	assert result != None and result[0] == 5	
	
	h = Hand([card.Card("H", "A"), card.Card("C", "2"), card.Card("H", "7"), card.Card("S", "4"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "A")])
	
	result = h.straight()
	assert result == None		
	
	h = Hand([card.Card("H", "A"), card.Card("C", "2"), card.Card("H", "3"), card.Card("S", "4"),card.Card("H", "5"),
			 card.Card("C", "K"), card.Card("S", "Q"), card.Card("S", "J"), card.Card("D", "10")])
	
	result = h.straight()
	assert result != None and result[0] == 14
	
	print "Test complete"
	
	print "Testing two pairs."
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "5"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "4"), card.Card("H", "3")])
	
	result = h.twoPairs()
	assert result != None and result[0] == 14 and result[1] == 5 and result[2] == 14
	
	h = Hand([card.Card("H", "A"), card.Card("D", "7"), card.Card("H", "3"), card.Card("H", "5"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "4"), card.Card("H", "3")])
	
	result = h.twoPairs()
	assert result != None and result[0] == 14 and result[1] == 5 and result[2] == 7	
	
	h = Hand([card.Card("H", "K"), card.Card("D", "7"), card.Card("H", "3"), card.Card("H", "6"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "4"), card.Card("H", "3")])
	
	result = h.twoPairs()
	assert result == None
	
	print "Test complete."	
	
	print "Testing high card."
	
	h = Hand([card.Card("H", "A"), card.Card("D", "A"), card.Card("H", "3"), card.Card("H", "5"),card.Card("H", "5"),
			 card.Card("C", "A"), card.Card("S", "4"), card.Card("H", "3")])
	
	result = h.highCard()
	assert result != None and result[0] == 14 and result[1] == 14 and result[2] == 14 and result[3] == 5 and result[4] == 5
	
	h = Hand([card.Card("H", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("H", "2"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("S", "4"), card.Card("H", "8")])
	
	result = h.highCard()
	assert result != None 
	assert result[0] == 13 
	assert result[1] == 12 
	assert result[2] == 8 
	assert result[3] == 6 
	assert result[4] == 5
	
	print "Test complete."	
	
	print "Testing evaluate()"

	print "\tStraight Flush"
	SF = Hand([card.Card("H", "K"), card.Card("H", "6"), card.Card("H", "3"), card.Card("H", "2"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("H", "4"), card.Card("H", "8")])
			
	result = SF.evaluate()
	assert result != None
	assert result[0] == 8
	assert result[1] == 6
	assert result[2] == 0
	assert result[3] == 0
	assert result[4] == 0
	assert result[5] == 0

	print "\tFour-of-a-Kind"
	FK = Hand([card.Card("H", "K"), card.Card("H", "6"), card.Card("H", "3"), card.Card("H", "2"),card.Card("H", "5"),
			 card.Card("C", "K"), card.Card("D", "K"), card.Card("S", "K")])
			
	result = FK.evaluate()
	assert result != None
	assert result[0] == 7
	assert result[1] == 13
	assert result[2] == 6
	assert result[3] == 0
	assert result[4] == 0
	assert result[5] == 0

	print "\tFull House"
	FH = Hand([card.Card("H", "K"), card.Card("H", "6"), card.Card("H", "3"), card.Card("H", "2"),card.Card("H", "5"),
			 card.Card("C", "K"), card.Card("D", "6"), card.Card("S", "6")])
			
	result = FH.evaluate()
	assert result != None
	assert result[0] == 6
	assert result[1] == 6
	assert result[2] == 13
	assert result[3] == 0
	assert result[4] == 0
	assert result[5] == 0

	print "\tFlush"
	F = Hand([card.Card("H", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("H", "2"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("S", "4"), card.Card("H", "8")])
			
	result = F.evaluate()
	assert result != None
	assert result[0] == 5
	assert result[1] == 13
	assert result[2] == 8
	assert result[3] == 5
	assert result[4] == 3
	assert result[5] == 2
	
	print "\tStraight"
	S = Hand([card.Card("D", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("H", "2"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("S", "4"), card.Card("H", "8")])
			
	result = S.evaluate()
	assert result != None
	assert result[0] == 4
	assert result[1] == 6
	assert result[2] == 0
	assert result[3] == 0
	assert result[4] == 0
	assert result[5] == 0	
	
	print "\tThree-of-a-Kind"
	T = Hand([card.Card("D", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("D", "8"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("S", "8"), card.Card("H", "8")])
			
	result = T.evaluate()
	assert result != None
	assert result[0] == 3
	assert result[1] == 8
	assert result[2] == 13
	assert result[3] == 12
	assert result[4] == 0
	assert result[5] == 0	

	print "\tTwo Pairs"
	TP = Hand([card.Card("D", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("D", "8"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("S", "5"), card.Card("H", "8")])
			
	result = TP.evaluate()
	assert result != None
	assert result[0] == 2
	assert result[1] == 8
	assert result[2] == 5
	assert result[3] == 13
	assert result[4] == 0
	assert result[5] == 0	
	
	print "\tPair"
	P = Hand([card.Card("D", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("D", "8"),card.Card("H", "5"),
			 card.Card("C", "Q"), card.Card("S", "5"), card.Card("H", "J")])
			
	result = P.evaluate()
	assert result != None
	assert result[0] == 1
	assert result[1] == 5
	assert result[2] == 13
	assert result[3] == 12
	assert result[4] == 11
	assert result[5] == 0
	
	print "\tHigh Card"
	HC = Hand([card.Card("D", "K"), card.Card("D", "6"), card.Card("H", "3"), card.Card("D", "8"),card.Card("H", "A"),
			 card.Card("C", "Q"), card.Card("S", "5"), card.Card("H", "J")])
			
	result = HC.evaluate()
	assert result != None
	assert result[0] == 0
	assert result[1] == 14
	assert result[2] == 13
	assert result[3] == 12
	assert result[4] == 11
	assert result[5] == 8		
	
	print "Test complete."				
	