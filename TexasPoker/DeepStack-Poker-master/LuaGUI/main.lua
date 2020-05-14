require('wx')

local M = require('layout') 

function main()
	M:init_layout()
	M:display_cards({"C2", "C3"})
end

main()
wx.wxGetApp():MainLoop()
	
