#---------------------------------------------------------------------------
#	settings.py
#
#	Description:
#
#	An object that contains the settings for setting up a new game of Poker
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
#	Settings class
#	
#	Essentially a struct that holds all of the data necessary to start a
#	a game. Also provides default settings if settings passed are corrupt or
#	non-existent.
#---------------------------------------------------------------------------
class Settings():
	
	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Takes an array of strings. Non-string type variables are
	#	first converted to the value then used. Must occur in the order listed
	#	by the default values
	#---------------------------------------------------------------------------	
	def __init__(self, settings = []):
		
		self.numPlayers = 4						#[0]
		self.numChips = 100						#[1]
		self.smallBlind = 5						#[2]
		self.numAIs = 3							#[3]
		
		i = 0
		for setting in settings:
			
			#Number of players
			if i == 0:
				value = int(setting)
				if value < 2:
					value = 2
				elif value > 12:
					value = 12
				self.numPlayers = value
				
			#Number of chips	
			elif i == 1:
				value = int(setting)
				if value < 2:
					value = 2
				self.numChips = value
			
			#Size of small blind
			elif i == 2:
				value = int(setting)
				if value < 0:
					value = 0
				if value > self.numChips/2:
					value = self.numChips/2
				self.smallBlind = value
			
			#Number of AI opponents	
			elif i == 3:
				value = int(setting)
				if value < 0:
					value = 0
				elif value > self.numPlayers:
					value = self.numPlayers
				self.numAIs = value
				
			i += 1
	
	
	
#========================================
#	TESTS
#========================================	

if __name__ == '__main__':

	#UNIT TESTS
	print "Testing constructor."
	
	testsets =[["3", "140", "10", "2"],
			  [],
			  ["3", "140", "140", "2"],
			  ["1", "140", "10", "2"],
			  ["15", "140", "10", "14"],
			  ["3", "140", "10", "4"],
			  ["-1", "-100", "-10", "-4"],
			  ["0", "0", "0", "0"]]
			
	for test in testsets:
		settings = Settings(test)
		
		assert settings.numPlayers >= 2
		assert settings.numChips > 0
		assert settings.smallBlind <= settings.numChips/2
		assert settings.smallBlind >= 0
		assert settings.numAIs <= settings.numPlayers
		assert settings.numPlayers <= 12
		
	print "Test complete."