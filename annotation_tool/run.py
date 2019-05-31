import numpy as np

from annotate_gui import AnnotationGUI

if __name__ == '__main__':
    data_dir = '/home/kalpitthakkar/g17_mount/Kalpit/test_set/Easy'
    output_dir = '/home/kalpitthakkar/g17_mount/Kalpit/test_set/Easy'
    gui = AnnotationGUI(data_dir=data_dir, output_dir=output_dir)

    root = gui.build_gui()
    root.mainloop()
