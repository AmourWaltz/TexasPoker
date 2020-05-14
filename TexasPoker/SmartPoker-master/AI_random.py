#---------------------------------------------------------------------------
#	AI_random.py
#
#	Description:
#
#	AI_random is a player class which does nothing but picks moves totally
#	at random, irrespective of actual cards being held.
#
#---------------------------------------------------------------------------

import random
import player, interface, state, decision

#---------------------------------------------------------------------------
# AI_random class
#---------------------------------------------------------------------------
class AI_Random(player.Player):
	
	#---------------------------------------------------------------------------
	# Constructor
	# 
	# Takes an id, name and bank value
	#---------------------------------------------------------------------------
	def __init__(self, ID, name, bank):
		
		player.Player.__init__(self, ID, name, bank)
		self.interface = interface	
		
	#---------------------------------------------------------------------------
	#  giveDecision()
	# 
	#  Selects a decision at random from the available options
	#---------------------------------------------------------------------------
	def giveDecision(self, theState):	
		
		decisions = theState.decisionOptions(self.id)
		
		choice = random.randint(0,100)
		theDecision = None
		#Generally waiting, folding, checking or revealing should always be an option
		for d in decisions:
			if (d.name == decision.Decision.WAIT or d.name == decision.Decision.FOLD or 
				d.name == decision.Decision.CHECK or d.name == decision.Decision.REVEAL):
				theDecision = d
				
		#Sanity test that we could have checked, folded or revealed		
		assert theDecision != None
		
		#More players, play less aggressively (2 players = 30%, 3 = 20%, 4 = 15%, ... 12 = 5%)
		numActive = 0
		for p in theState.playersInfo:
			if p.isInGame:
				numActive += 1
		raiseChance = 60/numActive
		
		
		for p in theState.playersInfo:
			if p.id == self.id:
				me = p
		
		callAmount = theState.getCallAmount(me)
		if callAmount > me.bank:
			callAmount = me.bank
		possibleRaise = me.bank - callAmount		
				
		#We'll call if the call amount is low, relative to our bank, or low relative to our pot
		callChance = raiseChance + 100 - int(50*(callAmount/(me.pot+callAmount+1.0))) - int(100*(callAmount/(me.bank+callAmount+1.0)))
					
		for d in decisions:
			if d.name == decision.Decision.CALL and choice > 99-callChance:
				theDecision = d
			if d.name == decision.Decision.RAISE and choice > 99-raiseChance and possibleRaise > 0:
				myRaise = possibleRaise
				if possibleRaise > 0:
					if choice > 99-(raiseChance/6): #go all-in
						myRaise = possibleRaise
					elif choice > 99-(raiseChance/3): #big raise
						myRaise = int(me.bank/4) 
						if myRaise > possibleRaise:
							myRaise = possibleRaise
					else:	#small raise
						myRaise = int(me.bank/8)
						if myRaise > possibleRaise:
							myRaise = possibleRaise
							
				theDecision = decision.Decision(decision.Decision.RAISE, myRaise)
				
		return theDecision
				
			
		

#========================================
#	TESTS
#========================================
if __name__ == '__main__':

	ai = AI_Random(3, "John", 100)