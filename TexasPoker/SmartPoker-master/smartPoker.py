#---------------------------------------------------------------------------
#	smartPoker.py
#
#	Description:
#
#	This is the main file. Run to play.	Make sure you run in the terminal to
#	use the default interface.
#
#	Tested on Python 2.7.5,
#	OSX 10.9.3
#
#	Note: Currently the default interface is written to work only with OSX and
#	possibly Linux running in the terminal.
#---------------------------------------------------------------------------

import basicInterface, game

SetupInterface = basicInterface.BasicInterface()
theSettings = SetupInterface.setupGame()

#Change default settings
theSettings.numPlayers = 6
theSettings.numAIs = theSettings.numPlayers-1

newGame = game.Game(theSettings,theInterfaceConstructor = basicInterface.BasicInterface)