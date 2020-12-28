import wx

import create_dtst
import create_dtst_
import create_dtst__
import create_interpo
import create_RF

class Exec(wx.Frame):

    def __init__(self, *args, **kwargs):
        super(Exec, self).__init__(*args, **kwargs)

        self.InitUI()

    def InitUI(self):

        menubar = wx.MenuBar()

        fileMenu = wx.Menu()
        #----------------------------------------------------------
        
        imp = wx.Menu()
        imp.Append(wx.ID_OPEN, 'Start 1')
        imp.Append(wx.ID_NEW, 'Start 2')
        imp.Append(wx.ID_FILE, 'Start 3')
        self.Bind(wx.EVT_MENU, self.OnOpenStart1, id=wx.ID_OPEN)
        self.Bind(wx.EVT_MENU, self.OnOpenStart2, id=wx.ID_NEW)
        self.Bind(wx.EVT_MENU, self.OnOpenStart4, id=wx.ID_FILE)
        
        fileMenu.AppendMenu(wx.ID_ANY, 'Create dataset', imp)
        
        #----------------------------------------------------------
        fileMenu.AppendSeparator()
        #----------------------------------------------------------
        
        imp1 = wx.Menu()
        imp1.Append(wx.ID_EDIT, 'Start')
        self.Bind(wx.EVT_MENU, self.OnOpenStart3, id=wx.ID_EDIT)
        
        fileMenu.AppendMenu(wx.ID_ANY, 'Interpolation', imp1)
        
        #----------------------------------------------------------
        fileMenu.AppendSeparator()
        #----------------------------------------------------------
        
        imp2 = wx.Menu()
        imp2.Append(wx.ID_FILE1, 'Start')
        self.Bind(wx.EVT_MENU, self.OnOpenStart5, id=wx.ID_FILE1)
        
        fileMenu.AppendMenu(wx.ID_ANY, 'Predict RF', imp2)      
        
        #----------------------------------------------------------
        fileMenu.AppendSeparator()
        #----------------------------------------------------------
        
        qmi = wx.MenuItem(fileMenu, wx.ID_EXIT, '&Quit\tCtrl+W')
        fileMenu.AppendItem(qmi)

        self.Bind(wx.EVT_MENU, self.OnQuit, qmi)

        menubar.Append(fileMenu, '&System')
        self.SetMenuBar(menubar)
        #----------------------------------------------------------
        
        self.SetSize((550, 450))
        self.SetTitle('LITHOPY - Classification of lithologies')
        self.Centre()

    def OnQuit(self, e):
        self.Close()
        
    def OnOpenStart1(self, e):
        ex = create_dtst.Exec1(None, 'Create dataset')
        ex.Show()
        
    def OnOpenStart2(self, e):
        ex = create_dtst_.Exec1(None, 'Create dataset')
        ex.Show()

    def OnOpenStart4(self, e):
        ex = create_dtst__.Exec1(None, 'Create dataset')
        ex.Show()
        
    def OnOpenStart3(self, e):
        ex = create_interpo.Exec1(None, 'Interpolation')
        ex.Show()
    def OnOpenStart5(self, e):
        ex = create_RF.Exec1(None, 'Predict RF')
        ex.Show()

def main():

    app = wx.App()
    ex = Exec(None)
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()