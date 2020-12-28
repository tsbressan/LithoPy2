import wx
  
class Exec1(wx.Frame): 
   def __init__(self, parent, title): 
      super(Exec1, self).__init__(parent, title = title,size = (550, 450))
		
      panel = wx.Panel(self) 
      vbox = wx.BoxSizer(wx.VERTICAL) 

      
      text = wx.StaticText(panel, -1, "Create New Dataset - start 2")
      text.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD))
      text.SetSize(text.GetBestSize())      
      vbox.Add(text, 0, wx.ALL, 10)
      
      
      hbox2 = wx.BoxSizer(wx.HORIZONTAL)
     
      l2 = wx.StaticText(panel, -1, "Select Site:")
      hbox2.Add(l2, 1, wx.ALIGN_LEFT|wx.ALL,5) 
      
      self.load_file_button = wx.Button(panel, -1, "U1480")
      self.load_file_button.Bind(wx.EVT_BUTTON, self.SelectSiteu1480) 
      self.load_file_button1 = wx.Button(panel, -1, "U1481")
      self.load_file_button1.Bind(wx.EVT_BUTTON, self.SelectSiteu1481) 
      
      
      hbox2.Add(self.load_file_button,1,wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
      hbox2.Add(self.load_file_button1,1,wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
      vbox.Add(hbox2)       
      

      panel.SetSizer(vbox) 
        
      self.Centre() 
      self.Show() 
      self.Fit()  
		

      
   def SelectSiteu1480(self,event):
      import SelectSiteu1480_2
      ex = SelectSiteu1480_2.Exec(None)
      ex.Show()
      
   def SelectSiteu1481(self,event):
      import SelectSiteu1481_2
      ex = SelectSiteu1481_2.Exec(None)
      ex.Show()    


def main():

    app = wx.App() 
    ex.Exec1(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()