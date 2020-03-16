import sys
import os
from PyQt5.QtWidgets import QApplication

import Interface


def main():
    app = QApplication(sys.argv)
    ex = Interface.App_Window()
    os.system("mode con: cols=50 lines=50") 
    sys.exit(app.exec_())

    
    


if __name__== '__main__':
    main()