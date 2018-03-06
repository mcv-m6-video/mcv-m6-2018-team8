import cv2
import sys
import os
import glob
import numpy as np

class Database:
    def __init__(self, db_dir, start_frame = 0, end_frame=-1):
        self.db_dir = db_dir
        self.start_frame = start_frame if start_frame==0 else (start_frame-1)
        self.end_frame = end_frame

        assert(self.start_frame >=0)

    def loadDB(self, im_color=cv2.IMREAD_COLOR, im_show=False, normImage=False):

        file_images = glob.glob(os.path.join(self.db_dir, "*.png")) + glob.glob(os.path.join(self.db_dir, "*.jpg"))

        if self.end_frame == -1:
            files = file_images[self.start_frame:]
        else:
            files = file_images[self.start_frame:self.end_frame]

        print("Number of files: {} (from {} to {})".format(len(files), self.start_frame+1, self.end_frame))

        db = []
        for id, file in enumerate(files):

            im = cv2.imread(file, im_color)

            if normImage:
                im = self.normImage(im)

            if im_show:
                cv2.imshow("frame", im)
                cv2.waitKey(1)

            db.append(im)
            sys.stdout.write("\r>  Loading %s folder %s/%s ... " % (self.db_dir, id+1,len(files)))
            sys.stdout.flush()

        sys.stdout.write("Loaded!\n")

        return np.array(db)

    def normImage(sef, im):
        max = 0.001 if np.max(im) == 0 else np.max(im)
        im = cv2.convertScaleAbs(im, alpha=np.iinfo(im.dtype).max / max)

        return im