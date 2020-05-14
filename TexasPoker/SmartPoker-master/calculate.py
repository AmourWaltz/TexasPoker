#---------------------------------------------------------------------------
#	calculate.py
#
#	Description:
#
#	This is a class for generating tables, probabilities and other data for
#	determining which moves are prudent.
#---------------------------------------------------------------------------

from math import factorial
import card, hand, deck

#---------------------------------------------------------------------------
#	comparePocketsFast - This function will, compare two pockets using an
#	optimized method of calculation to be faster, but still return an
#	identical result to comparePockets()
#
#	p1, and p2 should be in the format [Card1a, Card1b], [Card2a, Card2b]
#---------------------------------------------------------------------------
def comparePocketsFast(p1, p2, c = []):
	
	#---------------------------------------------------------------------------
	#	Algorithm description
	#
	#	For each level of hand rank (straight flush, four-of-a-kind, full house, etc.)
	#	we will run a specialized algorithm that will count the number of combinations
	#	for which p1 has the highest, for which p2 has the highest, and for which there
	#	the community cards have the highest. These counts will be overcounts, so we must
	#	also subtract the overlap between ranks (i.e. where p2 has a Full house, but that
	#	combination also included p1 having 4-of-a-kind)
	#---------------------------------------------------------------------------

	# STRAIGHT FLUSHES
	#
	#	Algorithm description:
	#
	
	
	#Run for each suit
	for suitNum in range(4):
		if suitNum == 0:
			suit = card.Card.HEARTS
		elif suitNum == 1:
			suit = card.Card.DIAMONDS
		elif suitNum == 2:
			suit = card.Card.SPADES
		elif suitNum == 4:
			suit = card.Card.CLUBS
		else:
			assert False
		
		#The low card goes from A to 10
		for lowWindow in range(11):
			# Count the number of cards owned by p1, p2 and the community cards to count the contribution of this window
			# to the w/t/l count
			p1Cards = 0
			p2Cards = 0
			cCards = 0
			overcard = -1 #-1 = deck, 0 = c, 1 = p1, 2 = p2
			
			#the window is the next five cards after the low card. Also go one card further to check who holds the overcard
			for window in range(lowWindow, lowWindow+6):
				if ((p1[0].suit == suit and p1[0].getRank() == window) or
					(p1[1].suit == suit and p1[1].getRank() == window)):
					if window == lowWindow+5:
						overcard = 1
					else:
						p1cards += 1
				elif ((p2[0].suit == suit and p2[0].getRank() == window) or
					 (p2[1].suit == suit and p2[1].getRank() == window)):
					if window == lowWindow+5:
						overcard = 2
					else:
						p1cards += 1	
				for ccard in c:
					if ccard.suit == suit and ccard.getRank() == window:
						if window == lowWindow+5:
							overcard = 0
						else:
							cCards += 1
						break
			
							
			
			# both players held cards in this window then the SF is impossible
			if (p1Cards == 0 or p2cards == 0)
				
			
				
			
			
		
		

	pass
	
	
#---------------------------------------------------------------------------
#	comparePockets - This function will, through brute force compare the 
#	hands generated from all possible permutations of cards and return the
#	number of wins, ties and losses out of a total of 1712304 possibilities
#
#	p1, and p2 should be in the format [Card1a, Card1b], [Card2a, Card2b]
#	optionally can pass a set of community cards to fix.
#
#	If analysis is true, instead of storing wins, ties and losses, comparePockets
#	will give a table of hand evaluation ranks (i.e. # of times A had a SF while 
#	B had a FH) which will be useful for debugging comparePocketsFast()
#---------------------------------------------------------------------------
def comparePockets(p1, p2, c = [], analysis = False):
	
	d = deck.Deck()
	d.pull(p1[0].suit, p1[0].number)
	d.pull(p1[1].suit, p1[1].number)
	d.pull(p2[0].suit, p2[0].number)
	d.pull(p2[1].suit, p2[1].number)
	for ci in c:
		d.pull(ci.suit, ci.number)
		
	assert len(d.cards) == 48 - len(c)
	
	seq = [4, 3, 2, 1, 0]
	seq = seq[len(c):]
	numSeq = 1
	wins = 0
	ties = 0
	losses = 0
	
	if analysis:
		#Table is indexed by A's hand evaluations with an array storing counts of B's hand evalutations
		aTable = {}
		for i in range(9):
			aTable[i] = {}
			for l in range(9):
				aTable[i][l] = {}
				#Store win/tie/loss
				for m in range(3):
					aTable[i][l][m] = 0
		
	handOne = hand.Hand([], 1)
	handTwo = hand.Hand([], 2)
	while True:
		
		communityCards = getComboFromSequence(seq, d.cards) + c
		handOne.cards = p1 + communityCards
		handTwo.cards = p2 + communityCards
		
		if analysis:
			aRank = handOne.evaluate()[0]
			bRank = handTwo.evaluate()[0]
		
		winners = hand.winner([handOne, handTwo])
		if len(winners) == 1:
			if winners[0] == 1:
				wins += 1
				if analysis:
					aTable[aRank][bRank][0] += 1
			else:
				losses += 1
				if analysis:
					aTable[aRank][bRank][2] += 1
		else:
			ties += 1
			if analysis:
				aTable[aRank][bRank][1] += 1
		
		seq = getNextCombinationSequence(seq, d.cards)
		numSeq += 1
		
		if numSeq > 100000:
			numSeq = 0
			print "Another 100000 combinations tested."
		
		if not seq:
			break 
			
	if analysis:
		return aTable
		
	return (wins, ties, losses)
	
#---------------------------------------------------------------------------
#	getNextCombinationSequence(seq, l)
#
#	Pulls the next combination from a list, given a sequence array which
#	stores which elements are being pulled and a list to choose from
#
#	Algorithm:
#
#	Sequence looks like [last, 2nd last, 3rd last, etc.]
#
#	Bump the first sequence index. If you can't, bump the second and
#	put the first sequence index back to one spot ahead of it. If you can't
#	bump the third and put the 2nd and first right after, etc.
#	Only when we can't bump any do we return False
#---------------------------------------------------------------------------	
def getNextCombinationSequence(seq, l):
	
	for bump in range(len(seq)):
		
		#If we can bump this up more
		if seq[bump] < len(l)-(bump+1):
			seq[bump] += 1
			#Reset the previous sequence spots
			for reset in range(bump):
				seq[reset] = seq[bump]+(bump-reset)
				
			return seq
			
	return False
	
#---------------------------------------------------------------------------
#	getComboFromSequence(seq, l)
#	
#	Gets the actual combination given an array of indices
#---------------------------------------------------------------------------	
def getComboFromSequence(seq, l):
	
	combo = []
	for s in seq:
		combo.append(l[s])
		
	return combo
	
#---------------------------------------------------------------------------
#	choose(n, r)
#	
#	Returns nCr = n! / ((n-r)! * r!)
#---------------------------------------------------------------------------
def choose(n, r):
	
	assert n >= r
	
	if n == r:
		return 1
	
	#First, make r equal to whichever is higher r-n or r, since the calculation
	#will be equivalent but we'll have smaller intermediate numbers
	if r < n-r:
		r = n-r
		
	#Now compute n! / r! 
	num = 1
	for i in range(r+1, n+1):
		num *= i
	
	#Now divide by (n-r)!
	den = 1
	for i in range(1, n-r+1):
		den *= i
	
	return num/den
	
def printAnalysis(wins=True, ties=True, losses=True, total=False):

	if wins:
		print "Hands won: "
		print "   0:\t\t1:\t\t2:\t\t3:\t\t4:\t\t5:\t\t6:\t\t7:\t\t8:"
		print ""
		for i in range(9):
			line = str(i)+": "
			for l in range(9):
				line += str(aTable[i][l][0]) + "\t\t"
			
			print line
		print ""
		
	if ties:
		print "Hands tied: "
		print "   0:\t\t1:\t\t2:\t\t3:\t\t4:\t\t5:\t\t6:\t\t7:\t\t8:"
		print ""
		for i in range(9):
			line = str(i)+": "
			for l in range(9):
				line += str(aTable[i][l][1]) + "\t\t"

			print line	
		print ""
			
	if ties:
		print "Hands lost: "
		print "   0:\t\t1:\t\t2:\t\t3:\t\t4:\t\t5:\t\t6:\t\t7:\t\t8:"
		print ""
		for i in range(9):
			line = str(i)+": "
			for l in range(9):
				line += str(aTable[i][l][2]) + "\t\t"

			print line
		print ""
			
	if total:
		print "All hands: "
		print "   0:\t\t1:\t\t2:\t\t3:\t\t4:\t\t5:\t\t6:\t\t7:\t\t8:"
		print ""
		for i in range(9):
			line = str(i)+": "
			for l in range(9):
				line += str(aTable[i][l][0]+aTable[i][l][1]+aTable[i][l][2]) + "\t\t"

			print line
		print ""					
	


if __name__ == "__main__":
	
	p1 = [card.Card("H", "A"), card.Card("D", "2")]
	p2 = [card.Card("C", "K"), card.Card("C", "Q")]
	c = []
	
	numPossibilities = choose(48-len(c), 5-len(c)) 
	
	aTable = comparePockets(p1, p2, c, analysis=True)
	
	printAnalysis()
	
	"""wins, ties, losses = comparePockets(p1, p2, c)
	
	print "Win percentage = "+ str(100.0*wins/numPossibilities)+"%"
	print "Tie percentage = "+ str(100.0*ties/numPossibilities)+"%"
	print "Loss percentage = "+ str(100.0*losses/numPossibilities)+"%" """
		
	
	