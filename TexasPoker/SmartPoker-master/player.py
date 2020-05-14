#---------------------------------------------------------------------------
#	player.py
#
#	Description:
#
#	Player is a template class that has a getDecision function which, in the
#	case of a human player, is passed directly to the interface, and in the
#	case of an AI player, the AI will calculate a decision.
#
#	The player also contains a number of attributes throughout the game and
#	can yield those as a playerInfo object via getInfo()
#
#---------------------------------------------------------------------------

import decision

#---------------------------------------------------------------------------
#	Player class
#---------------------------------------------------------------------------
class Player():

	#---------------------------------------------------------------------------
	#	Constructor
	#---------------------------------------------------------------------------
	def __init__(self, ID, name, bank):
		
		self.id = ID
		self.name = name
		self.bank = bank
		
		self.pot = 0
		self.isInGame = True
		self.isInHand = True
		self.isDealer = False
		self.pocket = []
		self.isActive = False	
		self.miscInfo = []
		self.hasRevealed = False
		
	#---------------------------------------------------------------------------
	#	getInfo()
	#
	#	Returns a set of player info for use in the game state
	#---------------------------------------------------------------------------
	def getInfo(self):
		
		return PlayerInfo(self.id, self.name, self.isInGame, self.isInHand, self.bank, self.pot, self.pocket, 
						  self.isDealer, self.isActive, self.miscInfo)

	#---------------------------------------------------------------------------
	#	giveDecision(state)
	#
	#	Given a game state, returns the decision of the player
	#---------------------------------------------------------------------------	
	def giveDecision(self, state):
		
		default = decision.Decision()
		return default
		
	#---------------------------------------------------------------------------
	#	addToPot(amount)
	#
	#	Transfers the amount from the bank to the pot. If the amount is too large
	#	transfers nothing and returns False. Otherwise returns True.
	#---------------------------------------------------------------------------		
	def addToPot(self, amount):
	
		#Basic sanity test to make sure we're not trying to take money from forfeited/folded players
		assert self.isInGame and self.isInHand
		
		if amount > self.bank:
			return False
		else:
			self.bank -= amount
			self.pot += amount
			return True
	
	#---------------------------------------------------------------------------
	#	revealCards()
	#
	#	Reveals the pocket cards, so passToPlayers passes the real cards instead of (?,?)
	#---------------------------------------------------------------------------
	def revealCards(self):
		self.hasRevealed = True
		
#---------------------------------------------------------------------------
#	PlayerInfo class
#
#	Holds all info per player in the game necessary for the interface & any
#	decisions.
#---------------------------------------------------------------------------		
class PlayerInfo():
	
	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Takes an ID, name, turn order, bank, pot contribution, pocket cards, isActive
	#	and miscInfo string array
	#---------------------------------------------------------------------------
	def __init__(self, ID, name, isInGame, isInHand, bank, pot, pocket, isDealer, isActive, miscInfo = []):
		
		self.id = ID
		self.name = name
		self.isInGame = isInGame
		self.isInHand = isInHand
		self.bank = bank
		self.pot = pot
		self.pocket = pocket
		self.isDealer = isDealer
		self.isActive = isActive
		self.miscInfo = miscInfo
		
	#---------------------------------------------------------------------------
	#	isValid()
	#
	#	Determines whether player info is valid
	#---------------------------------------------------------------------------	
	def isValid(self):
			
		if not self.isInGame:
			if self.bank > 0:
				return False
			if self.pot > 0:
				return False
			if self.isActive:
				return False
			if self.isDealer:
				return False
			if len(self.pocket) > 0:
				return False
		else:
			if self.bank == 0 and self.pot == 0:
				return False
			if len(self.pocket) != 2:
				return False
			elif not self.pocket[0].isValid() or not self.pocket[1].isValid():
				return False
		
		if self.bank < 0:
			return False
			
		if self.pot < 0:
			return False
			
		#All tests are passed
		return True
		
#========================================
#	TESTS
#========================================	
if __name__ == '__main__':	
	pass	