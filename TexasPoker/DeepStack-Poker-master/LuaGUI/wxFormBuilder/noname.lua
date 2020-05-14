----------------------------------------------------------------------------
-- Lua code generated with wxFormBuilder (version Apr 19 2018)
-- http://www.wxformbuilder.org/
----------------------------------------------------------------------------

-- Load the wxLua module, does nothing if running from wxLua, wxLuaFreeze, or wxLuaEdit
package.cpath = package.cpath..";./?.dll;./?.so;../lib/?.so;../lib/vc_dll/?.dll;../lib/bcc_dll/?.dll;../lib/mingw_dll/?.dll;"
require("wx")

UI = {}


-- create MainFrame
UI.MainFrame = wx.wxFrame (wx.NULL, wx.wxID_ANY, "DeepStack Experiment Platform", wx.wxDefaultPosition, wx.wxSize( 776,778 ), wx.wxDEFAULT_FRAME_STYLE+wx.wxTAB_TRAVERSAL )
	UI.MainFrame:SetSizeHints( wx.wxDefaultSize, wx.wxDefaultSize )
	
	UI.BottomSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	
	UI.LeftSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.LeftUpperSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.Player1Card = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxPoint( -1,-1 ), wx.wxDefaultSize, 0 )
	UI.Player1Card:SetForegroundColour( wx.wxColour( 41, 169, 39 ) )
	UI.Player1Card:SetBackgroundColour( wx.wxColour( 43, 162, 46 ) )
	
	UI.LeftUpperSizer:Add( UI.Player1Card, 1, wx.wxALL, 5 )
	
	
	UI.LeftSizer:Add( UI.LeftUpperSizer, 1, wx.wxEXPAND, 5 )
	
	UI.LeftMiddleSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	
	UI.SubSizer1 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.PublicCard = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer1:Add( UI.PublicCard, 0, wx.wxALL, 5 )
	
	
	UI.LeftMiddleSizer:Add( UI.SubSizer1, 1, wx.wxEXPAND, 5 )
	
	UI.SubSizer2 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.PotBet = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer2:Add( UI.PotBet, 0, wx.wxALL, 5 )
	
	UI.m_staticText2 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Current Pot Size", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText2:Wrap( -1 )
	UI.SubSizer2:Add( UI.m_staticText2, 0, wx.wxALL, 5 )
	
	UI.CurrentBet = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer2:Add( UI.CurrentBet, 0, wx.wxALL, 5 )
	
	
	UI.LeftMiddleSizer:Add( UI.SubSizer2, 1, wx.wxEXPAND, 5 )
	
	
	UI.LeftSizer:Add( UI.LeftMiddleSizer, 1, wx.wxEXPAND, 5 )
	
	UI.LeftBottomSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.Player2Card = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.LeftBottomSizer:Add( UI.Player2Card, 0, wx.wxALL, 5 )
	
	
	UI.LeftSizer:Add( UI.LeftBottomSizer, 1, wx.wxEXPAND, 5 )
	
	
	UI.BottomSizer:Add( UI.LeftSizer, 1, wx.wxEXPAND, 5 )
	
	UI.RightSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.RightUpperSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.SubSizer3 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	
	UI.m_staticText3 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Stack", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText3:Wrap( -1 )
	UI.SubSizer3:Add( UI.m_staticText3, 0, wx.wxALL, 5 )
	
	UI.Player1Stack = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxPoint( -1,-1 ), wx.wxDefaultSize, 0 )
	UI.SubSizer3:Add( UI.Player1Stack, 0, wx.wxALL, 5 )
	
	
	UI.RightUpperSizer:Add( UI.SubSizer3, 1, wx.wxEXPAND, 5 )
	
	UI.SubSizer4 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	
	UI.m_staticText4 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Action Probabilities", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText4:Wrap( -1 )
	UI.SubSizer4:Add( UI.m_staticText4, 0, wx.wxALL, 5 )
	
	UI.ActionProbabilitiesChoices = {}
	UI.ActionProbabilities = wx.wxListBox( UI.MainFrame, wx.wxID_ANY, wx.wxDefaultPosition, wx.wxDefaultSize, UI.ActionProbabilitiesChoices, 0 )
	UI.SubSizer4:Add( UI.ActionProbabilities, 0, wx.wxALL, 5 )
	
	
	UI.RightUpperSizer:Add( UI.SubSizer4, 1, wx.wxEXPAND, 5 )
	
	UI.SubSizer5 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.Cheat = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Cheat", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer5:Add( UI.Cheat, 0, wx.wxALL, 5 )
	
	
	UI.RightUpperSizer:Add( UI.SubSizer5, 1, wx.wxEXPAND, 5 )
	
	
	UI.RightSizer:Add( UI.RightUpperSizer, 1, wx.wxEXPAND, 5 )
	
	UI.RightMiddleSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.m_staticText1 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "History", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText1:Wrap( -1 )
	UI.RightMiddleSizer:Add( UI.m_staticText1, 0, wx.wxALL, 5 )
	
	UI.HistoryChoices = {}
	UI.History = wx.wxListBox( UI.MainFrame, wx.wxID_ANY, wx.wxDefaultPosition, wx.wxDefaultSize, UI.HistoryChoices, 0 )
	UI.RightMiddleSizer:Add( UI.History, 0, wx.wxALL, 5 )
	
	
	UI.RightSizer:Add( UI.RightMiddleSizer, 1, wx.wxEXPAND, 5 )
	
	UI.RightBottomSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	
	UI.SubSizer6 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.Check = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Check", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer6:Add( UI.Check, 0, wx.wxALL, 5 )
	
	UI.Call = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Call", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer6:Add( UI.Call, 0, wx.wxALL, 5 )
	
	UI.Raise = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Raise", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer6:Add( UI.Raise, 0, wx.wxALL, 5 )
	
	UI.Fold = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Fold", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer6:Add( UI.Fold, 0, wx.wxALL, 5 )
	
	UI.m_staticText7 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Raise Amount", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText7:Wrap( -1 )
	UI.SubSizer6:Add( UI.m_staticText7, 0, wx.wxALL, 5 )
	
	UI.RaiseAmount = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer6:Add( UI.RaiseAmount, 0, wx.wxALL, 5 )
	
	
	UI.RightBottomSizer:Add( UI.SubSizer6, 1, wx.wxEXPAND, 5 )
	
	UI.bSizer19 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.SubSizer7 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	
	UI.m_staticText5 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Stack", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText5:Wrap( -1 )
	UI.SubSizer7:Add( UI.m_staticText5, 0, wx.wxALL, 5 )
	
	UI.Player2Stack = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer7:Add( UI.Player2Stack, 0, wx.wxALL, 5 )
	
	
	UI.bSizer19:Add( UI.SubSizer7, 1, wx.wxEXPAND, 5 )
	
	UI.SubSizer8 = wx.wxBoxSizer( wx.wxVERTICAL )
	
	UI.NextRound = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "New Hand", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer8:Add( UI.NextRound, 0, wx.wxALL, 5 )
	
	
	UI.bSizer19:Add( UI.SubSizer8, 1, wx.wxEXPAND, 5 )
	
	
	UI.RightBottomSizer:Add( UI.bSizer19, 1, wx.wxEXPAND, 5 )
	
	
	UI.RightSizer:Add( UI.RightBottomSizer, 1, wx.wxEXPAND, 5 )
	
	
	UI.BottomSizer:Add( UI.RightSizer, 1, wx.wxEXPAND, 5 )
	
	
	UI.MainFrame:SetSizer( UI.BottomSizer )
	UI.MainFrame:Layout()
	
	UI.MainFrame:Centre( wx.wxBOTH )
	
	-- Connect Events
	
	UI.Cheat:Connect( wx.wxEVT_COMMAND_BUTTON_CLICKED, function(event)
	--implements test
	
	end )
	


--wx.wxGetApp():MainLoop()
