from skimage.measure import *
import numpy as np
import cv2
import seaborn as sns
import sys, warnings
import matplotlib.pyplot as plt

class CS_Class:

    def __init__(self, id, frame, mask, start_window):
        self.id = id
        self.bbox = start_window

        (c, r, w, h) = self.bbox
        self.roi = frame[r:r + h, c:c + w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.mask = mask
        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.mask[r:r + h, c:c + w], [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        # Setup the termination criteria, either 10 iteration or move by atleast 1 pt
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def GetRoiHist(self):
        return self.roi_hist

    def GetPoints(self):
        return np.int32(cv2.boxPoints(self.rectangle))

    def GetWindow(self):
        return self.bbox

    def GetMidPoint(self):
        c1, r1, c2, r2 = (self.bbox[0], self.bbox[1], self.bbox[0] + self.bbox[2], self.bbox[1] + self.bbox[3])
        return (c1+(c2-c1)//2, r1+(r2-r1)//2)

    def Predict(self, dst):
        # apply meanshift to get the new location
        self.rectangle, self.bbox = cv2.CamShift(dst, self.bbox, self.term_crit)


def checkImages(im):

    if isinstance(im, (list,)):
        im = np.array(im)

    if not im.dtype == 'uint8':
        im = np.uint8(im)
        warnings.warn("The '{}' image it should be uint8 and it may not be displayed correctly.".format(im.dtype))

    return im

def HexToBGR(hex):
    if "#" in hex:
        hex = hex.lstrip('#')

    return tuple(int(hex[i:i + 2], 16) for i in (4, 2, 0))

def SpeedDetector(d, speed_estimator, last_value=0):
    if last_value:
        return np.round( (np.abs(np.log(d)) * speed_estimator + last_value) / 2, 1)

    return np.round(np.abs(np.log(d))*speed_estimator, 1)

def Tracking_CamShift(input, gt, threshold_min_area=500, speed_estimator=0, debug=False):

    assert (len(input) == len(gt))

    input = checkImages(input)
    gt = checkImages(gt)

    output_images = []
    palette = sns.color_palette(None, 100).as_hex()  # TODO assuming a max of 100 cars
    offset_id = 0
    CamShiftMethods = {}
    valid_regions = {}
    pts_predicted = {}
    speed_predicted = {}
    for num_frame, (frame, mask) in enumerate(zip(input, gt)):

        sys.stdout.write("\r  {}/{}".format(num_frame, len(gt)))
        sys.stdout.flush()

        # calculate the HSV for each frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if debug:
            # cv2.imwrite("output_images/hsv_frame_{}.png".format(num_frame), hsv_frame)
            cv2.imshow("HSV", hsv_frame)
            cv2.waitKey(1)

        # if mask is not thresholded (binary)
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            thresh, mask = cv2.threshold(mask,170,255,0)

        connected_region = label(mask, background=0)

        # image_color = cv2.cvtColor(cv2.convertScaleAbs(connected_region, alpha=255.0 / np.max(connected_region)), cv2.COLOR_GRAY2BGR)
        image_color = frame

        regions_per_frame = regionprops(connected_region)
        error_id = 0
        for region_id, region in enumerate(reversed(regions_per_frame), start=1):

            car_still_alive = False
            enough_objects = True

            speed = -1
            # id = new_id + region_id
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
                        cv2.circle(im_test, CamShiftMethods[id].GetMidPoint(), 2, HexToBGR(palette[id]), -1)
                        cv2.circle(im_test, (np.int32(region.centroid[1]), np.int32(region.centroid[0])), 1,
                                   (0, 0, 255), -1)
                        if id in pts_predicted:
                            cv2.polylines(im_test, np.array([pts_predicted[id][-4:]], dtype=np.int32), True,
                                          HexToBGR(palette[id]), thickness=1)
                        cv2.imshow("Current Point", im_test)
                        cv2.waitKey(50)

                    # for i in range(max(1, id-1), id+2):
                    #     if i in valid_regions:
                    distance = np.sqrt(np.power(valid_regions[id].centroid[0] - region.centroid[0], 4)
                                       + np.power(valid_regions[id].centroid[1] - region.centroid[1], 2))

                    if id in speed_predicted:
                        speed_predicted[id] = SpeedDetector(distance, speed_estimator, last_value=speed_predicted[id])
                    else:
                        speed_predicted[id] = SpeedDetector(distance, speed_estimator)

                    speed = speed_predicted[id]

                    if distance < 500:
                        car_still_alive = True
                else:
                    # here enters when a new id is found
                    enough_objects = False
                    CamShiftMethods[id] = CS_Class(id, frame, mask, (minc, minr, maxc-minc, maxr-minr))

                if not car_still_alive:
                    a = 1
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

                    # if current id is the current first one in the queue
                    if id == (offset_id + 1):
                        offset_id += 1
                        print(" - ID {} has left".format(id))
                        id = region_id + offset_id - error_id
                    else:
                        pass  # some errors

                current_measurement = np.array([[np.float32(region.centroid[0])], [np.float32(region.centroid[1])]])
                valid_regions[id] = region

                dst = None
                if id in CamShiftMethods:
                    dst = cv2.calcBackProject([hsv_frame], [0], CamShiftMethods[id].GetRoiHist(), [0, 180], 1)
                else:
                    # when new car appear for any reason
                    CamShiftMethods[id] = CS_Class(id, frame, mask, (minc, minr, maxc-minc, maxr-minr))
                    continue

                CamShiftMethods[id].Predict(dst)

                if id in pts_predicted:
                    pts_predicted[id] = np.vstack((pts_predicted[id], CamShiftMethods[id].GetPoints()))
                else:
                    pts_predicted[id] = CamShiftMethods[id].GetPoints()

                # draw current rectangle for each id
                cv2.rectangle(image_color, (minc, minr), (maxc, maxr), HexToBGR(palette[id]), thickness=2)

                # draw last point of the tracked line [-1]
                (center_c, center_r) = (pts_predicted[id][-4:][0][0]+(pts_predicted[id][-4:][1][0]-pts_predicted[id][-4:][0][0])//2,
                                        pts_predicted[id][-4:][1][1]+(pts_predicted[id][-4:][2][1]-pts_predicted[id][-4:][1][1])//2)

                cv2.circle(image_color, (center_c, center_r), 2, HexToBGR(palette[id]), -1)

                speed = speed if speed != -1 else np.nan

                cv2.putText(image_color, "Car {} {:.2f}".format(id, speed), (minc, minr), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, color=HexToBGR(palette[id]))

                cv2.polylines(image_color, np.array([pts_predicted[id][-4:]], dtype=np.int32), True, HexToBGR(palette[id]),
                              thickness=1)

            else:
                error_id += 1

        cv2.imshow("CamShift", image_color)
        cv2.imshow("Mask", mask)
        cv2.waitKey(40)

        output_images.append(image_color)

    cv2.destroyAllWindows()

    return np.array(output_images), speed_predicted