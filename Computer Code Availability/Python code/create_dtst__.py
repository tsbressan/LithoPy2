import wx
  
class Exec1(wx.Frame): 
   def __init__(self, parent, title): 
      super(Exec1, self).__init__(parent, title = title,size = (550, 450))
		
      panel = wx.Panel(self) 
      vbox = wx.BoxSizer(wx.VERTICAL) 
         
      
      text = wx.StaticText(panel, -1, "Create New Dataset - start 3")
      text.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD))
      text.SetSize(text.GetBestSize())      
      vbox.Add(text, 0, wx.ALL, 10)
      
      
      hbox2 = wx.BoxSizer(wx.HORIZONTAL)
     
      l2 = wx.StaticText(panel, -1, "Ajuster and finish:")
      hbox2.Add(l2, 1, wx.ALIGN_LEFT|wx.ALL,5) 
      
      self.load_file_button = wx.Button(panel, -1, "Start")
      self.load_file_button.Bind(wx.EVT_BUTTON, self.Selectcreate) 


      hbox2.Add(self.load_file_button,1,wx.EXPAND|wx.ALIGN_LEFT|wx.ALL,5) 
      vbox.Add(hbox2)       
      

      panel.SetSizer(vbox) 
        
      self.Centre() 
      self.Show() 
      self.Fit()  
		

      
   def Selectcreate(self,event):
      import Selectcreate_fin
      ex = Selectcreate_fin.Exec(None)
      ex.Show()
      


def main():

    app = wx.App() 
    ex.Exec1(None)
    ex.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()