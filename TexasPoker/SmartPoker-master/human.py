#---------------------------------------------------------------------------
#	human.py
#
#	Description:
#
#	Human extends the Player class by connecting with an interface for which
#	to get decisions. 
#
#---------------------------------------------------------------------------

import player, interface

#---------------------------------------------------------------------------
# Human class
#---------------------------------------------------------------------------
class Human(player.Player):
	
	#---------------------------------------------------------------------------
	# Constructor
	# 
	# Takes an interface as well as an id, name and bank value
	#---------------------------------------------------------------------------
	def __init__(self, interface, ID, name, bank):
		
		player.Player.__init__(self, ID, name, bank)
		self.interface = interface
		
	#---------------------------------------------------------------------------
	#  giveDecision()
	# 
	#  Calls the interface to return a decision
	#---------------------------------------------------------------------------
	def giveDecision(self, state):
		
		return self.interface.getDecision(state)
		
		