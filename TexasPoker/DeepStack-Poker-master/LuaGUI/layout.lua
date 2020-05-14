----------------------------------------------------------------------------
-- Lua code generated with wxFormBlder (version Apr 19 2018)
-- http://www.wxformbuilder.org/
----------------------------------------------------------------------------

-- Load the wxLua module, does nothing if running from wxLua, wxLuaFreeze, or wxLuaEdit
package.cpath = package.cpath..";./?.dll;./?.so;../lib/?.so;../lib/vc_dll/?.dll;../lib/bcc_dll/?.dll;../lib/mingw_dll/?.dll;"
package.path = package.path..";../DeepStack-Leduc/Source/Settings/?.lua;"
require("wx")

local settings = require("game_settings")

UI = {}

-- create MainFrame
function UI:init_layout()
--	Main Frame definitions
	UI.MainFrame = wx.wxFrame (wx.NULL, wx.wxID_ANY, "DeepStack Experiment Platform", wx.wxDefaultPosition, wx.wxSize( 1000,778 ), wx.wxDEFAULT_FRAME_STYLE+wx.wxTAB_TRAVERSAL )
	UI.MainFrame :SetSizeHints( wx.wxDefaultSize, wx.wxDefaultSize )
	UI.MainFrame :SetFont( wx.wxFont( 14, wx.wxFONTFAMILY_SWISS, wx.wxFONTSTYLE_NORMAL, wx.wxFONTWEIGHT_BOLD, False, "FreeSans" ) )
	UI.MainFrame :SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_INACTIVECAPTIONTEXT ) )
	UI.MainFrame :SetBackgroundColour( wx.wxColour( 33, 151, 35 ) )
--  Left upper sizer and player 1 card display area
	UI.BottomSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.LeftSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.LeftUpperSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.Player1Cards = {}

	for i = 1, settings.hole_card_count do
		UI.Player1Cards[i] = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxPoint( -1,-1 ), wx.wxDefaultSize, 0 )
		if(i == 1) then
			UI.LeftUpperSizer:Add( UI.Player1Cards[i], 0, (wx.wxALIGN_CENTER or wx.wxALIGN_RIGHT), 20 )
		else
			UI.LeftUpperSizer:Add( UI.Player1Cards[i], 0, wx.wxALIGN_CENTER, 20 )
		end
	end

	UI.LeftSizer:Add( UI.LeftUpperSizer, 1, wx.wxEXPAND, 5 )
--  Left middle left sizer and public card display area
	UI.LeftMiddleSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.SubSizer1 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.PublicCard = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer1:Add( UI.PublicCard, 1, wx.wxALIGN_CENTER, 5 )
	UI.LeftMiddleSizer:Add( UI.SubSizer1, 1, wx.wxALIGN_CENTER, 5 )
--  Left middle right sizer and pot display area
	UI.SubSizer2 = wx.wxBoxSizer( wx.wxVERTICAL )
	image = wx.wxImage( "./images/png/poker_pot.png", wx.wxBITMAP_TYPE_PNG )
	w = image:GetWidth()
	h = image:GetHeight()
	image:Rescale(w/7, h/7, wx.wxIMAGE_QUALITY_HIGH )
	image = wx.wxBitmap(image, -1)
	UI.PotBet = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, image, wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer2:Add( 0, 0, 1, wx.wxALIGN_CENTER, 3 )
	UI.SubSizer2:Add( UI.PotBet, 0, wx.wxALIGN_CENTER, 5 )
	UI.SubSizer2:Add( 0, 0, 1, wx.wxEXPAND, 3 )
	UI.m_staticText2 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Current Pot Size", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText2:Wrap( -1 )
	UI.SubSizer2:Add( UI.m_staticText2, 0, wx.wxALIGN_CENTER, 5 )
	UI.CurrentBet = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer2:Add( 0, 0, 1, wx.wxALIGN_CENTER, 3 )
	UI.SubSizer2:Add( UI.CurrentBet, 0, wx.wxALIGN_CENTER, 5 )
	UI.SubSizer2:Add( 0, 0, 1, wx.wxALIGN_CENTER, 3 )
	UI.LeftMiddleSizer:Add( UI.SubSizer2, 1, wx.wxEXPAND, 5 )
	UI.LeftSizer:Add( UI.LeftMiddleSizer, 1, wx.wxEXPAND, 5 )
--	Left bottom sizer and player 2 card display area
	UI.LeftBottomSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.Player2Card = wx.wxStaticBitmap( UI.MainFrame, wx.wxID_ANY, wx.wxNullBitmap, wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.LeftBottomSizer:Add( UI.Player2Card, 0, wx.wxALIGN_CENTER, 5 )
	UI.LeftSizer:Add( UI.LeftBottomSizer, 1, wx.wxALIGN_CENTER, 5 )
	UI.BottomSizer:Add( UI.LeftSizer, 1, wx.wxEXPAND, 5 )
--  Right upper sizer and player 1 stack display area
	UI.RightSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.RightUpperSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.SubSizer3 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.m_staticText3 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Stack", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText3:Wrap( -1 )
	UI.SubSizer3:Add( UI.m_staticText3, 0, wx.wxALL, 5 )
	UI.Player1Stack = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxPoint( -1,-1 ), wx.wxDefaultSize, 0 )
	UI.SubSizer3:Add( UI.Player1Stack, 0, wx.wxALL, 5 )
	UI.RightUpperSizer:Add( UI.SubSizer3, 1, wx.wxEXPAND, 5 )
--  DeepStack action probabilities display area
	UI.SubSizer4 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.m_staticText4 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Action Probabilities", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText4:Wrap( -1 )
	UI.SubSizer4:Add( UI.m_staticText4, 0, wx.wxALL, 5 )
	UI.ActionProbabilitiesChoices = {}
	UI.ActionProbabilities = wx.wxListBox( UI.MainFrame, wx.wxID_ANY, wx.wxDefaultPosition, wx.wxDefaultSize, UI.ActionProbabilitiesChoices, 0 )
	UI.SubSizer4:Add( UI.ActionProbabilities, 0, wx.wxALL, 5 )
	UI.RightUpperSizer:Add( UI.SubSizer4, 1, wx.wxEXPAND, 5 )
--  Cheat button to see which card player 1 holds
	UI.SubSizer5 = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.Cheat = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Cheat", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.Cheat:SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_HIGHLIGHT ) )
	UI.SubSizer5:Add( UI.Cheat, 0, wx.wxALL, 5 )
	UI.RightUpperSizer:Add( UI.SubSizer5, 1, wx.wxEXPAND, 5 )
	UI.RightSizer:Add( UI.RightUpperSizer, 1, wx.wxEXPAND, 5 )
--  Right middle area to show history including actions and past hands results
	UI.RightMiddleSizer = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.m_staticText1 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "History", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText1:Wrap( -1 )
	UI.RightMiddleSizer:Add( UI.m_staticText1, 0, wx.wxALL, 5 )
	UI.HistoryChoices = {}
	UI.History = wx.wxListBox( UI.MainFrame, wx.wxID_ANY, wx.wxDefaultPosition, wx.wxDefaultSize, UI.HistoryChoices, 0 )
	UI.RightMiddleSizer:Add( UI.History, 0, wx.wxALL, 5 )
	UI.RightSizer:Add( UI.RightMiddleSizer, 1, wx.wxEXPAND, 5 )
--  Right bottom left area to place normal action buttons
	UI.RightBottomSizer = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.SubSizer6 = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.Check = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Check", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.Check:SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_HIGHLIGHT ) )
	UI.SubSizer6:Add( UI.Check, 0, wx.wxALL, 5 )
	UI.Call = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Call", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.Call:SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_HIGHLIGHT ) )
	UI.SubSizer6:Add( UI.Call, 0, wx.wxALL, 5 )
	UI.Raise = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Raise", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.Raise:SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_HIGHLIGHT ) )
	UI.SubSizer6:Add( UI.Raise, 0, wx.wxALL, 5 )
	UI.Fold = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "Fold", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.Fold:SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_HIGHLIGHT ) )
	UI.SubSizer6:Add( UI.Fold, 0, wx.wxALL, 5 )
--  Right bottom left bottom area to give the area for raise amount
	UI.m_staticText7 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Raise Amount", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText7:Wrap( -1 )
	UI.SubSizer6:Add( UI.m_staticText7, 0, wx.wxALL, 5 )
	UI.RaiseAmount = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer6:Add( UI.RaiseAmount, 0, wx.wxALL, 5 )
	UI.RightBottomSizer:Add( UI.SubSizer6, 1, wx.wxEXPAND, 5 )
--  Right bottom right area to show player 2 stack
	UI.bSizer19 = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.SubSizer7 = wx.wxBoxSizer( wx.wxHORIZONTAL )
	UI.m_staticText5 = wx.wxStaticText( UI.MainFrame, wx.wxID_ANY, "Stack", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.m_staticText5:Wrap( -1 )
	UI.SubSizer7:Add( UI.m_staticText5, 0, wx.wxALL, 5 )
	UI.Player2Stack = wx.wxTextCtrl( UI.MainFrame, wx.wxID_ANY, "", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.SubSizer7:Add( UI.Player2Stack, 0, wx.wxALL, 5 )
	UI.bSizer19:Add( UI.SubSizer7, 1, wx.wxEXPAND, 5 )
--  Right bottom right area to place button for next round
	UI.SubSizer8 = wx.wxBoxSizer( wx.wxVERTICAL )
	UI.NextRound = wx.wxButton( UI.MainFrame, wx.wxID_ANY, "New Hand", wx.wxDefaultPosition, wx.wxDefaultSize, 0 )
	UI.NextRound:SetForegroundColour( wx.wxSystemSettings.GetColour( wx.wxSYS_COLOUR_HIGHLIGHT ) )
	UI.SubSizer8:Add( UI.NextRound, 0, wx.wxALL, 5 )
	UI.bSizer19:Add( UI.SubSizer8, 1, wx.wxEXPAND, 5 )
	UI.RightBottomSizer:Add( UI.bSizer19, 1, wx.wxEXPAND, 5 )
	UI.RightSizer:Add( UI.RightBottomSizer, 1, wx.wxEXPAND, 5 )
	UI.BottomSizer:Add( UI.RightSizer, 1, wx.wxEXPAND, 5 )
	UI.MainFrame:SetSizer( UI.BottomSizer )

	UI.MainFrame:Layout()
	UI.MainFrame:Centre( wx.wxBOTH )
	UI.MainFrame:Show(true)
end

function UI:display_cards(cards_ids, container)
	assert(table.getn(UI.Player1Cards) == table.getn(cards_ids))
	for i = 1, table.getn(UI.Player1Cards) do
		image = wx.wxImage( "./images/png/card_" .. cards_ids[i] .. ".png", wx.wxBITMAP_TYPE_PNG )
		w = image:GetWidth()
		h = image:GetHeight()
		image:Rescale(w/3, h/3, wx.wxIMAGE_QUALITY_HIGH )
		image = wx.wxBitmap(image, -1)
--		if(i == 1) then
--			image = image:GetSubBitmap(wx.wxRect(0,0,w/10,h/3))
--		end
		UI.Player1Cards[i]:SetBitmap(image)
	end
--	image = wx.wxImage( "./images/png/card_C2.png", wx.wxBITMAP_TYPE_PNG )
--	w = image:GetWidth()
--	h = image:GetHeight()
--	image:Rescale(w/3, h/3, wx.wxIMAGE_QUALITY_HIGH )
--	image = wx.wxBitmap(image, -1)
--	partial_image = image:GetSubBitmap(wx.wxRect(0,0,w/10,h/3))
--	UI.Player1Card:SetBitmap(partial_image)
--
--	image = wx.wxImage( "./images/png/card_S9.png", wx.wxBITMAP_TYPE_PNG )
--	w = image:GetWidth()
--	h = image:GetHeight()
--	image:Rescale(w/3, h/3, wx.wxIMAGE_QUALITY_HIGH )
--	image = wx.wxBitmap(image, -1)
--	UI.Player2Card:SetBitmap(image)
--
--	image = wx.wxImage( "./images/png/card_DQ.png", wx.wxBITMAP_TYPE_PNG )
--	w = image:GetWidth()
--	h = image:GetHeight()
--	image:Rescale(w/3, h/3, wx.wxIMAGE_QUALITY_HIGH )
--	image = wx.wxBitmap(image, -1)
--	UI.PublicCard:SetBitmap(image)
end
	
return UI
--wx.wxGetApp():MainLoop()
