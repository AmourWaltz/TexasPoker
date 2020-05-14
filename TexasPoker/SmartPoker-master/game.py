#---------------------------------------------------------------------------
#	game.py
#
#	Description:
#
#	Game manages the rules and procession of the game itself. Initialized with
#	a settings object, the game will then create players and AIs with the 
#	desired settings and start a new hand. The game object will also coordinate 
#	between the player objects sending them states and asking for decisions
#	before proceeding with the next step in the game.
#
#---------------------------------------------------------------------------

import settings, decision, state, card, interface, player, deck, human, hand
import AI_random

#---------------------------------------------------------------------------
#	Game class
#---------------------------------------------------------------------------
class Game():
	
	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Passed a settings object the game will initialize the game settings to
	#	begin playing poker.
	#---------------------------------------------------------------------------	
	def __init__(self, theSettings, theInterfaceConstructor=None, theInterfaces=[]):
		
		self.settings = theSettings
		self.players = []
		
		for i in range(self.settings.numPlayers):
			
			#Make some nicer default names
			if i == 0:
				nameGen = "John"
			elif i == 1:
				nameGen = "Suzy"
			elif i == 2:
				nameGen = "Fred"
			elif i == 3:
				nameGen = "Doug"
			elif i == 4:
				nameGen = "Greg"
			elif i == 5:
				nameGen = "Paul"
			elif i == 6:
				nameGen = "Dino"
			elif i == 7:
				nameGen = "Tony"
			elif i == 8:
				nameGen = "Erza"
			elif i == 9:
				nameGen = "Glen"
			elif i == 10:
				nameGen = "Cali"
			elif i == 11:
				nameGen = "Elsa"
			elif i == 12:
				nameGen = "Zach"
			
			#Change these to human/AI later
			if theInterfaceConstructor != None:
				theInterface = theInterfaceConstructor(i)	
			elif len(theInterfaces) > i:
				theInterface = theInterfaces[i]
			else:
				assert False			
			if i < (self.settings.numPlayers - self.settings.numAIs):
				newPlayer = human.Human(theInterface, i, nameGen, self.settings.numChips)
			else:
				newPlayer = AI_random.AI_Random(i, nameGen, self.settings.numChips)
				#Currently a Human, replace this with the AI class later
				#newPlayer = human.Human(theInterface, i, nameGen, self.settings.numChips)
				
			if i == 0:
				newPlayer.isActive = True
				newPlayer.isDealer = True
			self.players.append(newPlayer)
		
		self.smallBlind = self.settings.smallBlind
		self.bigBlind = 2 * self.smallBlind
		self.currentDealer = 0
		self.currentActive = 0
		self.bettingRound = 0
		self.numVisits = 0
		self.numInGame = self.settings.numPlayers
		self.numInHand = self.settings.numPlayers
		self.communityCards = []
		self.numPlayers = self.settings.numPlayers
		
		#Some variables to handle the revealing sequence in a showdown
		self.lastRaised = 0
		self.canDenyRevealed = False	#First player up in a showdown must reveal (can't fold)
		
		gameOver = False
		while not gameOver:
			gameOver = self.newHand()
			
		winnerText = ""
		for p in self.players:
			if p.bank > 0:
				winnerText = p.name + " has won the game! Thanks for playing."
					
		self.passToPlayers({state.State.CONTINUE_ONLY:True, state.State.CONTINUE_TEXT:winnerText})
		
	#---------------------------------------------------------------------------
	#	newHand()
	#
	#	Sets the game up for a new hand, making a new deck, dealing the pocket
	#	cards to all players still in the game, subtracting the blinds
	#---------------------------------------------------------------------------
	def newHand(self):
		
		#Check that the game hasn't ended
		if self.numInGame <= 1:
			numInGame = 0
			for p in self.players:
				if p.isInGame == True:
					numInGame += 1
			assert numInGame == self.numInGame
			print numInGame
			return True
		
		#Put everyone still playing back in the hand
		for p in self.players:
			if p.isInGame:
				p.isInHand = True
		self.numInHand = self.numInGame
		
		#Make a new deck and shuffle it
		self.deck = deck.Deck()
		self.deck.shuffle()
		
		#Deal the pocket cards
		for p in self.players:
			p.pocket = [self.deck.draw(), self.deck.draw()]
			p.hasRevealed = False
			
			#Do some quick checks to make sure the endHand procedure didn't make any errors
			assert p.pot == 0
			assert p.pocket[0] != None and p.pocket[1] != None
		
		bb = None
		sb = None
		
			
		sb = self.getPlayerInHand(((self.currentDealer + 1) % self.numPlayers))
		if not sb.addToPot(self.smallBlind):
			sb.addToPot(sb.bank)
			
		bb = self.getPlayerInHand(((sb.id + 1) % self.numPlayers))
		if not bb.addToPot(self.bigBlind):
			bb.addToPot(bb.bank)
			
		#in an extreme case, where everyone checks BB and goes to showdown, BB reveals first
		#in our rules of the game, for simplicity
		self.lastRaised = bb.id				
		
		
		#Reset active player statuses and set a new active player
		for p in self.players:
			p.isActive = False	
			
		active = self.getPlayerInHand(((bb.id + 1) % self.numPlayers))
		active.isActive = True
		self.currentActive = active.id		
			
		#Reset the community cards
		self.communityCards = []	
		
		#Betting begins at round 0
		self.bettingRound = 0
		self.numVisits = 1
			
		#Start the game
		handOver = False
		while not handOver:
			handOver = self.nextAction()
			
		return False
		
	#---------------------------------------------------------------------------
	#	nextAction()
	#
	#	Moves the game to the next point where a decision is needed from the
	#	players.
	#---------------------------------------------------------------------------
	def nextAction(self):
		
		#Check to make sure we're not revealing in a showdown
		if self.bettingRound == 4:
			if self.canDenyReveal:
				miscInfo = {state.State.REVEAL_OR_FOLD:True}
			else:
				miscInfo = {state.State.MUST_REVEAL:True}
				
		else:
			miscInfo = {}
		
		theDecision = self.passToPlayers(miscInfo)
		
		activePlayer = self.getActivePlayer()
		
		checkIfHandOver = False
		
		if theDecision.name == decision.Decision.GAMEQUIT:
			raise SystemExit
			
		elif theDecision.name == decision.Decision.FORFEIT:
			activePlayer.bank = 0
			self.removeFromGame(activePlayer)
			checkIfHandOver = True
			
		elif theDecision.name == decision.Decision.FOLD:
			self.removeFromHand(activePlayer)
			checkIfHandOver = True
		
		elif theDecision.name == decision.Decision.CHECK:
			pass
			
		elif theDecision.name == decision.Decision.CALL:
			amountToCall = self.getCallAmount(activePlayer)
			if not activePlayer.addToPot(amountToCall):
				activePlayer.addToPot(activePlayer.bank)
				
		elif theDecision.name == decision.Decision.RAISE:
			amountToCall = self.getCallAmount(activePlayer)
			amountToRaise = theDecision.value
			
			#This person will be first to show cards in a showdown, unless another player re-raises
			self.lastRaised = activePlayer.id
			
			if not activePlayer.addToPot(amountToCall + amountToRaise):
				#Checks should have been in place to ensure we can't raise more than we have
				assert False
				
		elif theDecision.name == decision.Decision.REVEAL:
			activePlayer.revealCards()
			self.canDenyReveal = True
			
		else:
			
			#Currently we shouldn't be able to do any other actions
			assert False
			
		if checkIfHandOver:
			numInHand = 0
			for p in self.players:
				if p.isInHand:
					numInHand += 1
			
			if numInHand <= 1:
				
				self.endHand()
				
				
				return True
		
		
		
		newBettingRound = self.incrementActivePlayer()
		
		handOver = False
		if newBettingRound:
			
			handOver = self.newBettingRound()	
			
		return handOver
			
	#---------------------------------------------------------------------------
	#	incrementActivePlayer()
	#
	#	Moves to the next active player in sequence, increments the number of
	#	visits and compares whether a full loop in the betting has finished.
	#	Returns true if the betting round has finished.
	#---------------------------------------------------------------------------		
	def incrementActivePlayer(self):
		
		p = self.getPlayerByID(self.currentActive)
		p.isActive = False
		
		while True:
			
			self.numVisits += 1
			self.currentActive = (self.currentActive + 1) % self.numPlayers
			
			p = self.getPlayerByID(self.currentActive)
			
			assert p != None
			
			if p.isInHand == True:
				break
				
		p.isActive = True
		
		# Check if there are unresolved raises that other players must respond to
		uncheckedBets = False
		for p2 in self.players:
			callAmount = self.getCallAmount(p2)
			#Either we have an unchecked bet or we're all-in
			if p2.bank > 0 and p2.isInHand and callAmount > 0:
				uncheckedBets = True
				
		# If we've been once around and there are no unchecked raises, the betting round is over		
		if self.numVisits > self.numPlayers and uncheckedBets == False:
			return True
		else:
			return False
		
	#---------------------------------------------------------------------------
	#	newBettingRound()
	#
	#	Resets the numVisits, increments the bettingRound, changes to the correct
	#	active player, deals any community cards and/or ends the hand
	#---------------------------------------------------------------------------		
	def newBettingRound(self):

		self.numVisits = 1
		self.bettingRound += 1
		
		if self.bettingRound <= 4:
		
			p = self.getPlayerByID(self.currentActive)
			p.isActive = False
			
			p = self.getPlayerInHand((self.currentDealer + 1) % self.numPlayers)
			p.isActive = True
			self.currentActive = p.id
		
			#Run specific betting round
		
			if self.bettingRound == 1:
			
				#The Flop
				self.communityCards.append(self.deck.draw())
				self.communityCards.append(self.deck.draw())
				self.communityCards.append(self.deck.draw())
			
			elif self.bettingRound == 2:
			
				#The Turn
				self.communityCards.append(self.deck.draw())
			
			elif self.bettingRound == 3:
			
				#The River
				self.communityCards.append(self.deck.draw())
				
			elif self.bettingRound == 4:
				
				#Set the player to the first raised
				p.isActive = False
				p = self.getPlayerInHand(self.lastRaised)
				p.isActive = True
				self.currentActive = p.id
				self.canDenyReveal = False
		
		# We've reached the end of all betting
		else:
			self.endHand()
			return True
			
		return False
			
	#---------------------------------------------------------------------------
	#	endHand()
	#
	#	Check if there is only one player left inHand, in which case, give him
	#	the pot and call a new hand. If there is more than one, reveal the cards
	#	of those remaining and calculate the winner
	#---------------------------------------------------------------------------
	def endHand(self):
		
		theWinner = None
		
		maxWin = {}
		for p in self.players:
			maxWin[p.id] = p.pot
			
		if self.numInHand == 1:
			for p in self.players:
				if p.isInHand:
					winningIDs = [p.id]
					
		elif self.numInHand > 1:
			
			#NEED TO ADD REVEAL CODE HERE
			
			assert len(self.communityCards) == 5
			
			hands = []
			for p in self.players:
				if p.isInHand:
					hands.append(hand.Hand(p.pocket + self.communityCards, ID = p.id))
					
			winningIDs = hand.winner(hands)
					
					
							
				
			
		else:
			assert False
		
		displayText = ""
		handName = ""	
		for ID in winningIDs:
			p = self.getPlayerByID(ID)
			if len(displayText) == 0:
				displayText += p.name
				#If the winner revealed their hand include some text to show who won
				if p.hasRevealed:
					handName = " with "+hand.Hand(p.pocket + self.communityCards).handName()
				else:
					handName = "."
				token = " has won"
			else:
				displayText += ", "+p.name
				token = " have tied"
		displayText += token + " this hand" + handName
		
		splitPots = []
		while len(winningIDs) > 0:
			
			#First, we need to create split pots for winners who only partially contributed
			lowestPotContribution = -10
			for p in self.players:
				if (p.id in winningIDs) and (p.pot < lowestPotContribution or lowestPotContribution == -10):
					lowestPotContribution = p.pot
			
			#Make a new split pot
			potAmount = 0		
			for p in self.players:
				deduction = min(lowestPotContribution, p.pot)
				p.pot -= deduction
				potAmount += deduction
				
			splitIDs = winningIDs[:]
			splitPots.append((potAmount, splitIDs))
			
			#Remove all players who are no longer inHand from the winningIDs
			for p in self.players:
				if p.id in winningIDs and p.pot <= 0:
					winningIDs.remove(p.id)
					
		#Any remaining money should return back to the original owner
		for p in self.players:
				p.bank += p.pot
				p.pot = 0
					
		#Now we need to distribute the splitPots (or single pot if there was no all-in plays)			
		for splitPot in splitPots:	
			amount, ids = splitPot
			fraction = int(amount / len(ids))
			remainder = amount % len(ids)
			for ID in ids:
				p = self.getPlayerByID(ID)
				p.bank += fraction
				if remainder > 0:
					p.bank += remainder
					remainder -= 1

		
		#Remove bankrupt players from the game
		for p in self.players:
			if p.bank < 0:
				#Something went wrong and the player has negative money
				assert False
			elif p.bank == 0:
				self.removeFromGame(p)
			
		self.passToPlayers({state.State.CONTINUE_ONLY: True, state.State.CONTINUE_TEXT:displayText})
		
		#Increment the dealer
		
		p = self.getPlayerByID(self.currentDealer)
		p.isDealer = False
		while True:
	
			self.currentDealer = (self.currentDealer + 1) % self.numPlayers
		
			p = self.getPlayerByID(self.currentDealer)
		
			assert p != None
		
			if p.isInGame == True:
				break
			
		p.isDealer = True		
			
		
	#---------------------------------------------------------------------------
	#	removeFromHand()
	#
	#	Takes the player out of the hand
	#---------------------------------------------------------------------------	
	def removeFromHand(self, thePlayer):
		if thePlayer.isInHand:
			thePlayer.isInHand = False
			self.numInHand -= 1	
		
	#---------------------------------------------------------------------------
	#	removeFromGame()
	#
	#	Takes the player out of the hand and the game
	#---------------------------------------------------------------------------	
	def removeFromGame(self, thePlayer):
		self.removeFromHand(thePlayer)
		if thePlayer.isInGame:
			thePlayer.isInGame = False
			self.numInGame -= 1			
		
	#---------------------------------------------------------------------------
	#	passToPlayers()
	#
	#	Cleans the game states and passes them to each individual player.
	#---------------------------------------------------------------------------	
	def passToPlayers(self, miscInfo = {}):
			
		for givenPlayer in self.players:		

			#Clean game state
			cleanPlayersInfo = []

			for p in self.players:
				info = p.getInfo()
				if not p.hasRevealed and p.id != givenPlayer.id:
					info.pocket = [card.Card("?", "?"), card.Card("?", "?")]
				cleanPlayersInfo.append(info)
			
			gameState = state.State(cleanPlayersInfo, self.communityCards, miscInfo)
			
			d = givenPlayer.giveDecision(gameState)
			
			if givenPlayer.isActive:
				#Make sure the decision returned by the AI or Human player was valid
				if not gameState.isValidDecision(givenPlayer.id, d):
					print givenPlayer.name + " " + d.name + "  " + int(d.value)
				assert gameState.isValidDecision(givenPlayer.id, d)
				return d
			else:
				assert (d.name == decision.Decision.WAIT or d.name == decision.Decision.FORFEIT or d.name == decision.Decision.GAMEQUIT)
				
		#Something went wrong and now we don't have a decision.		
		assert False
		
	#---------------------------------------------------------------------------
	#	getActivePlayer()
	#
	#	Returns the active player
	#---------------------------------------------------------------------------				
	def getActivePlayer(self):
		numActive = 0
		activePlayer = None
		for p in self.players:
			if p.isActive:
				activePlayer = p
				numActive += 1
		assert numActive == 1
		return activePlayer
		
	#---------------------------------------------------------------------------
	#	getCallAmount()
	#
	#	Returns the amount in the pot that has not been called by the player
	#---------------------------------------------------------------------------		
	def getCallAmount(self, player):
		maxPot = 0
		for p in self.players:
			if p.pot > maxPot:
				maxPot = p.pot
		
		return maxPot - player.pot
		
	#---------------------------------------------------------------------------
	#	getPlayerInHand()
	#
	#	Given an id, returns that player or the next in sequence until the player
	#	is inHand
	#---------------------------------------------------------------------------		
	def getPlayerInHand(self, ID):
		thePlayer = self.getPlayerByID(ID)
		
		while not (thePlayer.isInHand and thePlayer.isInGame):
			ID = (ID + 1) % self.numPlayers
			thePlayer = self.getPlayerByID(ID)
			
		return thePlayer	
		
	#---------------------------------------------------------------------------
	#	getPlayerByID(id)
	#
	#	Returns the player with the specified ID
	#---------------------------------------------------------------------------
	def getPlayerByID(self, ID):
		for p in self.players:
			if p.id == ID:
				return p		
		return None
			
#========================================
#	TESTS
#========================================	

if __name__ == '__main__':
	
	pass	