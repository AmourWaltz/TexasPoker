#---------------------------------------------------------------------------
#	interface.py
#
#	Description:
#
#	This is a template for any visualizations for handling the game. All 
#	visualizers must inherit at least these basic methods
#---------------------------------------------------------------------------

import settings, state, decision, card

#---------------------------------------------------------------------------
#	Interface class
#	
#	Contains three methods: 
#		__init__()
#		setupGame()	- Always uses default settings
#		giveDecision() - Always gives default decision
#---------------------------------------------------------------------------
class Interface():

	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Does nothing. 
	#---------------------------------------------------------------------------	
	def __init__(self, playerID = -1):
		self.id = playerID
		pass

	#---------------------------------------------------------------------------
	#	setupGame()
	#
	#	Returns the default game settings
	#---------------------------------------------------------------------------		
	def setupGame(self):
		
		default = settings.Settings()
		return default

	#---------------------------------------------------------------------------
	#	getDecision()
	#
	#	Returns the default decision
	#---------------------------------------------------------------------------		
	def getDecision(self, gameState):
		
		default = decision.Decision()
		return default
		
#========================================
#	TESTS
#========================================	

if __name__ == '__main__':
	
	print "Testing constructor"
	interface = Interface()
	
	settings = interface.setupGame()
	assert settings != None
	
	player1 = state.PlayerInfo(0, "John", 0, 100, 10, [card.Card("H", "3"), card.Card("S", "2")], False)			
	player2 = state.PlayerInfo(1, "Susy", 1, 100, 5, [card.Card("?", "?"), card.Card("?", "?")], False)
	player3 = state.PlayerInfo(2, "Hamm", 2, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], False)
	player4 = state.PlayerInfo(3, "Gunn", 3, 100, 0, [card.Card("?", "?"), card.Card("?", "?")], True)
	
	infos = [player1, player2, player3, player4]
	
	testState = state.State(infos, [])	
	
	decision = interface.getDecision(testState)
	assert decision != None
	
	print "Test complete."