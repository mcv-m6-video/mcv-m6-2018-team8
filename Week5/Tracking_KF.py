from skimage.measure import *
import cv2
import numpy as np
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class KF_Class:

    def __init__(self, id):
        self.id = id
        self.objKF = cv2.KalmanFilter(4, 2, 1)
        self.objKF.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.objKF.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.objKF.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

    def Upgrade(self, measure):
        self.objKF.correct(measure)

    def Predict(self):
        return self.objKF.predict()

def SpeedDetector(d):
    kilometer_per_pixel = 80
    return np.round(np.abs(np.log(d)+kilometer_per_pixel), 1)

def HexToBGR(hex):
    if "#" in hex:
        hex = hex.lstrip('#')

    return tuple(int(hex[i:i + 2], 16) for i in (4, 2, 0))


def Tracking_KF(gt):

    # init kalman filter object


    # kalman = cv2.KalmanFilter(2, 1, 0)
    # kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]])
    # kalman.measurementMatrix = 1. * np.ones((1, 2))
    # kalman.processNoiseCov = 1e-5 * np.eye(2)
    # kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1))
    # kalman.errorCovPost = 1. * np.ones((2, 2))
    # kalman.statePost = 0.1 * np.random.randn(2, 1)

    palette = sns.color_palette(None, 100).as_hex() #TODO assuming a max of 100 cars

    kalmanFilters = {}
    valid_regions = {}
    pts_predicted = {}

    offset_id = 0
    for num_frame, im in enumerate(gt):

        sys.stdout.write("\r  {}/{}".format(num_frame, len(gt)))
        sys.stdout.flush()

        if len(im.shape) > 2:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            thresh, im = cv2.threshold(im,170,255,0)

        connected_region = label(im, background=0)

        # image_color = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
        image_color = cv2.cvtColor(cv2.convertScaleAbs(connected_region, alpha=255.0 / np.max(connected_region)), cv2.COLOR_GRAY2BGR)

        regions_per_frame = regionprops(connected_region)
        error_id = 0
        for region_id, region in enumerate(reversed(regions_per_frame), start=1):

            car_still_alive = False
            enough_objects = True

            speed = -1
            # id = new_id + region_id
            id = region_id + offset_id - error_id
            if region.area >= 200:
                # bbox (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox

                im_test = image_color.copy()

                if id in valid_regions:
                    cv2.circle(im_test,
                               (np.int32(valid_regions[id].centroid[1]), np.int32(valid_regions[id].centroid[0])),
                               3, (0, 255, 0), -1)

                    # for i in range(max(1, id-1), id+2):
                    #     if i in valid_regions:
                    distance = np.sqrt(np.power(valid_regions[id].centroid[0] - region.centroid[0], 2)
                                       + np.power(valid_regions[id].centroid[1] - region.centroid[1], 4))

                    speed = SpeedDetector(distance)

                    if distance < 50:
                        car_still_alive = True
                else:
                    # only once for each id
                    enough_objects = False
                    kalmanFilters[id] = KF_Class(id)

                # calculate distance between old region=>valid_regions[id] and current region=>region
                # one car lefts or new one car appear
                if not car_still_alive and enough_objects:
                    im_2 = im.copy()
                    cv2.circle(im_2,
                               (np.int32(region.centroid[1]), np.int32(region.centroid[0])),
                               2, (0, 0, 0), -1)
                    cv2.imshow("Next Frame", im_2)
                    cv2.waitKey(1)

                    if id == (offset_id + 1):
                        offset_id += 1
                        print(" - ID {} has left".format(id))
                        id = region_id + offset_id - error_id
                    else:
                        pass # some errors

                current_measurement = np.array([[np.float32(region.centroid[0])], [np.float32(region.centroid[1])]])
                valid_regions[id] = region

                if id in kalmanFilters:
                    kalmanFilters[id].Upgrade(current_measurement)
                else:
                    kalmanFilters[id] = KF_Class(id)

                if id in pts_predicted:
                    pts_predicted[id] = np.vstack((pts_predicted[id], kalmanFilters[id].Predict().reshape(4)))
                else:
                    pts_predicted[id] = np.array([kalmanFilters[id].Predict().reshape(4)])

                # draw current rectangle for each id
                cv2.rectangle(image_color, (minc, minr), (maxc, maxr), HexToBGR(palette[id]), thickness=2)

                # draw last point of the tracked line [-1]
                cv2.circle(image_color,
                           (np.int32(pts_predicted[id][-1][1]), np.int32(pts_predicted[id][-1][0])),
                           2, HexToBGR(palette[id]), -1)

                cv2.imshow("Current Point", im_test)
                cv2.waitKey(1)

                speed = speed if speed != -1 else np.nan

                cv2.putText(image_color, "Car {} {:.2f}".format(id, speed), (minc, minr), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=HexToBGR(palette[id]))

                cv2.circle(image_color, (np.int32(region.centroid[1]), np.int32(region.centroid[0])), 3, (0, 0, 255), -1)

                draw_pts = np.flip(pts_predicted[id][4:,0:2], 1)
                cv2.polylines(image_color, np.array([draw_pts], dtype=np.int32), False, HexToBGR(palette[id]), thickness=1)

                # cv2.line(image_color, (np.int32(pts_predicted[id][1][0]), np.int32(pts_predicted[id][0][0])), (np.int32(region.centroid[1]), np.int32(region.centroid[0])), (0,180,200))

                cv2.circle(im_test, (np.int32(region.centroid[1]), np.int32(region.centroid[0])), 2, (0, 0, 255), -1)

                cv2.imshow("Current Point", im_test)
                cv2.waitKey(1)

            else:
                error_id+=1

        cv2.imshow("Kalman Filter", image_color)
        cv2.waitKey(1)

    return True