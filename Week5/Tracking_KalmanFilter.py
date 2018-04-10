from skimage.measure import *
import cv2
import numpy as np
import seaborn as sns
import sys, warnings

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

def checkImages(im):

    if isinstance(im, (list,)):
        im = np.array(im)

    if not im.dtype == 'uint8':
        im = np.uint8(im)
        warnings.warn("The '{}' image it should be uint8 and it may not be displayed correctly.".format(im.dtype))

    return im

def SpeedDetector(d, speed_estimator):
    return np.round(np.abs(np.log(d)+speed_estimator), 1)

def HexToBGR(hex):
    if "#" in hex:
        hex = hex.lstrip('#')

    return tuple(int(hex[i:i + 2], 16) for i in (4, 2, 0))


def Tracking_KalmanFilter(input, gt, threshold_min_area=500, speed_estimator=0, debug=False):

    assert (len(input) == len(gt))

    input = checkImages(input)
    gt = checkImages(gt)

    output_images = []
    palette = sns.color_palette(None, 100).as_hex()  # TODO assuming a max of 100 cars
    offset_id = 0
    kalmanFilters = {}
    valid_regions = {}
    pts_predicted = {}
    for num_frame, (frame, mask) in enumerate(zip(input, gt)):

        sys.stdout.write("\r  {}/{}".format(num_frame, len(gt)))
        sys.stdout.flush()

        # if mask is not thresholded (binary)
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            thresh, mask = cv2.threshold(mask,170,255,0)

        connected_region = label(mask, background=0)

        # choose if you want to return the prediction over the mask or over the input color images
        # image_color = cv2.cvtColor(cv2.convertScaleAbs(connected_region, alpha=255.0 / np.max(connected_region)), cv2.COLOR_GRAY2BGR)
        image_color = frame

        regions_per_frame = regionprops(connected_region)
        error_id = 0
        for region_id, region in enumerate(reversed(regions_per_frame), start=1):

            car_still_alive = False
            enough_objects = True

            speed = -1 # by default
            id = region_id + offset_id - error_id

            if region.area >= threshold_min_area:
                # bbox (min_row, min_col, max_row, max_col)
                minr, minc, maxr, maxc = region.bbox

                if id in valid_regions:

                    if debug:
                        im_test = image_color.copy()
                        old_measurement = (
                        np.int32(valid_regions[id].centroid[1]), np.int32(valid_regions[id].centroid[0]))
                        cv2.circle(im_test, old_measurement, 3, (0, 255, 0), -1)
                        cv2.circle(im_test, (np.int32(region.centroid[1]), np.int32(region.centroid[0])), 2,
                                   (0, 0, 255), -1)
                        cv2.imshow("Current Point", im_test)
                        cv2.waitKey(50)

                    # for i in range(max(1, id-1), id+2):
                    #     if i in valid_regions:
                    distance = np.sqrt(np.power(valid_regions[id].centroid[0] - region.centroid[0], 2)
                                       + np.power(valid_regions[id].centroid[1] - region.centroid[1], 4))

                    speed = SpeedDetector(distance, speed_estimator)

                    if distance < 500:
                        car_still_alive = True
                else:
                    # here enters when a new id is found
                    enough_objects = False
                    kalmanFilters[id] = KF_Class(id)

                # calculate distance between old region=>valid_regions[id] and current region=>region
                # one car lefts or new one car appear
                if not car_still_alive and enough_objects:
                    if debug:
                        im_debug = mask.copy()
                        cv2.circle(im_debug,
                                   (np.int32(region.centroid[1]), np.int32(region.centroid[0])),
                                   2, (0, 0, 0), -1)
                        cv2.imshow("Next Frame", im_debug)
                        cv2.waitKey(1)

                    if id == (offset_id + 1):
                        offset_id += 1
                        print(" - ID {} has left".format(id))
                        id = region_id + offset_id - error_id
                    else:
                        pass # some errors

                # current measurement needs an "special" format for OpenCV's KalmanFilter
                current_measurement = np.array([[np.float32(region.centroid[0])], [np.float32(region.centroid[1])]])

                valid_regions[id] = region

                if id in kalmanFilters:
                    kalmanFilters[id].Upgrade(current_measurement)
                else:
                    # when new car appear for any reason
                    kalmanFilters[id] = KF_Class(id)

                if id in pts_predicted:
                    pts_predicted[id] = np.vstack((pts_predicted[id], kalmanFilters[id].Predict().reshape(4)))
                else:
                    pts_predicted[id] = np.array([kalmanFilters[id].Predict().reshape(4)])

                speed = speed if speed != -1 else np.nan

                # draw the ID car and its speed
                cv2.putText(image_color, "Car {} {:.2f}".format(id, speed), (minc, minr), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=HexToBGR(palette[id]))

                # draw current rectangle for each id (from regions props)
                thickness = 4 if (speed > speed_estimator) else 2
                cv2.rectangle(image_color, (minc, minr), (maxc, maxr), HexToBGR(palette[id]), thickness=thickness)

                # draw last point of the tracked line (from points predicted) [-1]
                cv2.circle(image_color,
                           (np.int32(pts_predicted[id][-1][1]), np.int32(pts_predicted[id][-1][0])),
                           3, HexToBGR(palette[id]), -1)

                # draw current center point dected  by region props (point measured)
                cv2.circle(image_color, (np.int32(region.centroid[1]), np.int32(region.centroid[0])), 1, (0, 0, 255), -1)

                # draw trajectory
                draw_pts = np.flip(pts_predicted[id][4:,0:2], 1)
                cv2.polylines(image_color, np.array([draw_pts], dtype=np.int32), False, HexToBGR(palette[id]), thickness=1)

            else:
                error_id+=1

        cv2.imshow("Kalman Filter", image_color)
        cv2.imshow("Mask", mask)
        cv2.waitKey(1)

        output_images.append(image_color)

    cv2.destroyAllWindows()

    return np.array(output_images)