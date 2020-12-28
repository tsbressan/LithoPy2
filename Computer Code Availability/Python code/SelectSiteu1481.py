import wx
import subprocess
import os

class Exec(wx.Frame):

    def __init__(self, parent, id=-1, title='Create dataset U1481 - wait to finish',
                 pos=wx.DefaultPosition, size=(550, 450)):
        wx.Frame.__init__(self, parent, id, title, pos, size)
        self.text1 = wx.TextCtrl(self, -1, '', wx.DefaultPosition, wx.Size(550, 450),
                            wx.NO_BORDER | wx.TE_MULTILINE)
        self.Show()
       
        p = subprocess.Popen(["python", "-u", "cut_image_poly_create_csv_U1481.py"], stdout=subprocess.PIPE, bufsize=-1)
        self.pid = p.pid
        #Poll process for output
        while p.poll() is None:
            x = p.stdout.readline().decode() 
            self.text1.write(x)
            wx.GetApp().Yield() 
        self.text1.write("\nProcess finalized. Close the screen")


if __name__ == '__main__':
    app = wx.App()
    frame = Exec(None)
    app.MainLoop()