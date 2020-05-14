#---------------------------------------------------------------------------
#	testInterface.py
#
#	Description:
#
#	This interface can be fed a sequence of actions for players to take as
#	a list, rather than requiring default actions or a GUI interface for
#	the purposes of testing
#---------------------------------------------------------------------------

import interface, decision

#---------------------------------------------------------------------------
# testInterface
#---------------------------------------------------------------------------
class TestInterface(interface.Interface):
	
	#---------------------------------------------------------------------------
	#	Constructor
	#
	#	Takes a list of decisions which will be applied in sequence by the player
	#---------------------------------------------------------------------------
	def __init__(self, playerID, decisions = []):
		interface.Interface.__init__(self)
		self.id = playerID
		self.decisions = decisions
		
	#---------------------------------------------------------------------------
	#	getDecision()
	#---------------------------------------------------------------------------
	def getDecision(self, theState):
		
		if len(self.decisions) >= 1:
			d = self.decisions[0]
			self.decisions = self.decisions[1:]
		else:
			d = decision.Decision()
		
		return d
		