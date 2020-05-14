Project start date: June 12th, 2014

Play Texas Hold'em against AI opponents. Make sure you run in Terminal for
Mac so that the basic interface works.

Currently only random_AI available (makes decisions irrespective of cards held
but has some reasoning process so raises/calls are at least somewhat reasonable)

To Do:

- BUG: If you're big blind and are pushed in an insufficient amount, the first round should force you to at least bet
       the big blind, instead of whatever that player pushed in.
- Getting assertion errors for some AI plays -- need to debug!
- Create a getComboSequenceFast() function and benchmark it against the more laborious
  manually calculated version


