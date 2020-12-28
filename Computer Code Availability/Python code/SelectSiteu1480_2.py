import wx
import subprocess
import os

class Exec(wx.Frame):

    def __init__(self, parent, id=-1, title='Create dataset U1480 - wait to finish',
                 pos=wx.DefaultPosition, size=(550, 450)):
        wx.Frame.__init__(self, parent, id, title, pos, size)
        self.text1 = wx.TextCtrl(self, -1, '', wx.DefaultPosition, wx.Size(550, 450),
                            wx.NO_BORDER | wx.TE_MULTILINE)
        self.Show()
        
        p = subprocess.Popen(["python", "-u", "join_interpolation_dataset0.py"], stdout=subprocess.PIPE, bufsize=-1)
        self.pid = p.pid
        while p.poll() is None:
            x = p.stdout.readline().decode() 
            self.text1.write(x)
            wx.GetApp().Yield() 
            

        p = subprocess.Popen(["python", "-u", "join_interpolation_dataset1_U1480.py"], stdout=subprocess.PIPE, bufsize=-1)
        self.pid = p.pid
        while p.poll() is None:
            x = p.stdout.readline().decode() 
            self.text1.write(x)
            wx.GetApp().Yield() 
            
        
        p = subprocess.Popen(["python", "-u", "create_dataset2_U1480.py"], stdout=subprocess.PIPE, bufsize=-1)
        self.pid = p.pid
        while p.poll() is None:
            x = p.stdout.readline().decode() 
            self.text1.write(x)
            wx.GetApp().Yield() 
            
        self.text1.write("\nProcess finalized. Close the screen")

        
if __name__ == '__main__':
    app = wx.App()
    frame = Exec(None)
    app.MainLoop()