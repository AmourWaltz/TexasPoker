#---------------------------------------------------------------------------
#	state.py
#
#	Description:
#
#	Current game state available to a player. 
#
#	- List of playerInfo for all players:
#		- ID (also specifies turn order)
#		- Name
#		- Pocket cards (unknown if not revealed yet)
#		- Bank (how much the person has in chips (doesn't include money in pot))
#		- Amount in pot (how much the person has put into the pot)
#		- isInHand
#		- isInGame
#		- isDealer
#		- Active turn (if not active turn, only viable player decision is "Wait")
#	- Community cards
#	- Miscellaneous information
#	
#
#---------------------------------------------------------------------------

from player import PlayerInfo
import card
import decision

#---------------------------------------------------------------------------
#	State class
#---------------------------------------------------------------------------
class State():
	
	CONTINUE_ONLY = "CONTINUE_ONLY"
	CONTINUE_TEXT = "CONTINUE_TEXT"
	REVEAL_OR_FOLD = "REVEAL_OR_FOLD"
	MUST_REVEAL = "MUST_REVEAL"
	
	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Takes a list of playerInfo, a list of community cards and a list of strings
	#	for miscellaneous information which can be resized as needed
	#---------------------------------------------------------------------------
	def __init__(self, playersInfo, communityCards, miscInfo = {}):
		
		self.playersInfo = playersInfo
		self.communityCards = communityCards
		self.miscInfo = miscInfo
		
		#Apply defaults to miscInfo
		if not self.CONTINUE_ONLY in self.miscInfo:
			self.miscInfo[self.CONTINUE_ONLY] = False
			
		if not self.CONTINUE_TEXT in self.miscInfo:
			self.miscInfo[self.CONTINUE_TEXT] = ""
			
		if not self.REVEAL_OR_FOLD in self.miscInfo:
			self.miscInfo[self.REVEAL_OR_FOLD] = False
			
		if not self.MUST_REVEAL in self.miscInfo:
			self.miscInfo[self.MUST_REVEAL] = False							
		
	#---------------------------------------------------------------------------
	#	isValid
	#
	#	Runs through the game state to ensure that the state upholds the rules
	#	of Texas Hold'em. 
	#---------------------------------------------------------------------------
	def isValid(self):
		
		# Confirm all players have valid information
		if len(self.playersInfo) > 2:
			activePlayers = 0
			numDealers = 0
			ids = []
			for playerInfo in self.playersInfo:
				
				if playerInfo.isActive:
					activePlayers += 1
					
				if playerInfo.isDealer:
					numDealers += 1
				
				if not playerInfo.id in ids:
					ids.append(playerInfo.id)
				
				if not playerInfo.isValid():
					return False
					
					
			#Someone must be the active player!		
			if activePlayers != 1:
				return False
				
			#Someone must be the dealer!	
			if numDealers != 1:
				return False
				
			#All ids must be unique	
			if not len(ids) == len(self.playersInfo):
				return False
						
					
		else:
			return False
		
		# Confirm community cards are valid
		if len(self.communityCards) <= 5:
			if len(self.communityCards) > 0:
				
				for card in self.communityCards:
					if not card.isValid() or not card.isKnown():
						return False
						
		else:
			return False
			
		#Passed all checks, is a valid game state	
		return True
		
	#---------------------------------------------------------------------------
	#	decisionOptions
	#
	#	Determines which moves are valid for a player, given the state of the game
	#	Returns a list of decisions without amounts
	#---------------------------------------------------------------------------	
	def decisionOptions(self, playerID):
		
		options = []
		
		playerInfo = None
		mostInPot = 0
		for playerInfo in self.playersInfo:
			if playerInfo.pot > mostInPot:
				mostInPot = playerInfo.pot
			if playerInfo.id == playerID:
				thisPlayer = playerInfo
				
		assert playerInfo != None
				
		options.append(decision.Decision(decision.Decision.GAMEQUIT))
		options.append(decision.Decision(decision.Decision.FORFEIT))
		
		if thisPlayer.isActive:
			
			#We're in a showdown
			if self.miscInfo[self.REVEAL_OR_FOLD]:
				options.append(decision.Decision(decision.Decision.FOLD))
				options.append(decision.Decision(decision.Decision.REVEAL))
			elif self.miscInfo[self.MUST_REVEAL]:	
				options.append(decision.Decision(decision.Decision.REVEAL))
				
			#Normal play
			else:
				#Unmatched raises in pot and we're not all in
				if thisPlayer.pot < mostInPot and thisPlayer.bank > 0 :
					options.append(decision.Decision(decision.Decision.CALL))
					options.append(decision.Decision(decision.Decision.FOLD))
				else:
					options.append(decision.Decision(decision.Decision.CHECK))
				
				#We can also raise the stakes if we have the money (otherwise all we can do is call)	
				if thisPlayer.bank > mostInPot-thisPlayer.pot:
					
					#Quickly check we're not the only player left in betting
					numCanStillCall = 0
					for p in self.playersInfo:
						if p.isInHand and p.bank > mostInPot-p.pot:
							numCanStillCall += 1
					
					if numCanStillCall > 1:
						options.append(decision.Decision(decision.Decision.RAISE))
			
		else:
			options.append(decision.Decision(decision.Decision.WAIT))
			
		return options
		
	#---------------------------------------------------------------------------
	#	isValidDecision
	#
	#	Determines whether a given decision is an available option for the player
	#---------------------------------------------------------------------------	
	def isValidDecision(self, playerID, d):
		
		#If there was no other valid options, this should be valid
		if self.miscInfo[self.CONTINUE_ONLY]:
			return True

		options = self.decisionOptions(playerID)
		
		thisPlayer = None
		for playerInfo in self.playersInfo:
			if playerID == playerInfo.id:
				thisPlayer = playerInfo	
		assert thisPlayer != None
		
		isValid = False
		for option in options:
			if option.name == d.name:
				
				if (option.name == decision.Decision.RAISE and 
					((d.value + self.getCallAmount(thisPlayer)) > thisPlayer.bank or d.value <= 0)):
					isValid = False
				else:
					isValid = True
					
		return isValid
				
				
	#---------------------------------------------------------------------------
	#	getCallAmount()
	#
	#	Returns the amount in the pot that has not been called by the player
	#---------------------------------------------------------------------------		
	def getCallAmount(self, info):
		maxPot = 0
		for p in self.playersInfo:
			if p.pot > maxPot:
				maxPot = p.pot

		return maxPot - info.pot		
			
						



#========================================
#	TESTS
#========================================
if __name__ == '__main__':

	#UNIT TESTS
	print "Testing constructors."
	
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 100, 5, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	
	infos = [player1, player2, player3, player4]
	
	state = State(infos, [])	
	
	assert state.isValid()
	
	#Here the ids aren't unique
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(0, "Susy", True, True, 100, 5, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	
	infos = [player1, player2, player3, player4]
	
	state = State(infos, [])	
	
	assert not state.isValid()	
	
	#Here there is no active player
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 100, 5, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	
	infos = [player1, player2, player3, player4]
	
	state = State(infos, [])	
	
	assert not state.isValid()	
	
	#Here there are too many community cards
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 100, 5, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("H", "2"),card.Card("H", "3"),card.Card("H", "4"),
			card.Card("H", "5"),card.Card("H", "6"),card.Card("H", "7")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()	
	
	#Here one of the community cards is unknown
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 100, 5, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("?", "?")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()	
	
	#Here a busted player has pocket cards
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 0, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], True, True)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("C", "3")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()	
	
	#Here a busted player has cash
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 10, 0, [], False, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("C", "3")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()				
	
	#Here a busted player is the active player
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, False)			
	player2 = PlayerInfo(1, "Susy", True, True, 0, 0, [], False, True)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], True, False)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("C", "3")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()	
	
	#Here we have more than one active player
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], True, True)			
	player2 = PlayerInfo(1, "Susy", True, True, 0, 0, [], False, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("C", "3")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()
	
	#Here we have more than one dealer
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], True, True)			
	player2 = PlayerInfo(1, "Susy", True, True, 0, 0, [], True, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("C", "3")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()		

	#Here we have no dealer
	player1 = PlayerInfo(0, "John", True, True, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False, True)			
	player2 = PlayerInfo(1, "Susy", True, True, 0, 0, [], False, False)
	player3 = PlayerInfo(2, "Hamm", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, True)
	player4 = PlayerInfo(3, "Gunn", True, True, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False, False)
	
	infos = [player1, player2, player3, player4]
	
	cards = [card.Card("H", "A"),card.Card("S", "A"),card.Card("C", "3")]
	
	state = State(infos, cards)	
	
	assert not state.isValid()	
	
	print "Test successful."