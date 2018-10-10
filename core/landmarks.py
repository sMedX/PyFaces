__author__ = 'Ruslan N. Kosarev'

import os
import numpy as np


class Pair:
    def __init__(self, name, point):
        self.name = name
        self.point = np.array(point)


model2017_1_bfm_nomouth = list()
model2017_1_bfm_nomouth.append(Pair('center.chin.tip', [-0.0955040976, -73.366539, 102.963692]))
model2017_1_bfm_nomouth.append(Pair('center.front.trichion', [-0.377398103, 84.9062271, 104.559608]))
model2017_1_bfm_nomouth.append(Pair('center.chin.tip', [-0.0955040976, -73.366539, 102.963692]))
model2017_1_bfm_nomouth.append(Pair('center.front.trichion', [-0.377398103, 84.9062271, 104.559608]))
model2017_1_bfm_nomouth.append(Pair('center.lips.lower.inner', [-0.00852414407, -32.5953026, 109.807968]))
model2017_1_bfm_nomouth.append(Pair('center.lips.lower.outer', [0.00494586537, -41.5086899, 111.808228]))
model2017_1_bfm_nomouth.append(Pair('center.lips.upper.inner', [0.00292040454, -32.4464722, 109.903816]))
model2017_1_bfm_nomouth.append(Pair('center.lips.upper.outer', [0.0115437824, -26.1546135, 115.271103]))
model2017_1_bfm_nomouth.append(Pair('center.nose.attachement_to_philtrum', [0.0253486931, -10.950614, 116.14006]))
model2017_1_bfm_nomouth.append(Pair('center.nose.tip', [0.203733876, 0.97169292, 131.823807]))
model2017_1_bfm_nomouth.append(Pair('left.ear.antihelix.center', [82.5063705, 6.57122803, -4.90304041]))
model2017_1_bfm_nomouth.append(Pair('left.ear.antihelix.tip', [75.1497192, -2.98772359, 10.0038261]))
model2017_1_bfm_nomouth.append(Pair('left.ear.helix.attachement', [79.9922104, 37.9479218, 4.55147552]))
model2017_1_bfm_nomouth.append(Pair('left.ear.helix.center', [82.73983, 0.960383058, -7.79163694]))
model2017_1_bfm_nomouth.append(Pair('left.ear.helix.outer', [87.5009003, 17.4589539, -15.073473]))
model2017_1_bfm_nomouth.append(Pair('left.ear.helix.top', [85.9402008, 38.4805679, -0.328399032]))
model2017_1_bfm_nomouth.append(Pair('left.ear.lobule.attachement', [66.251709, -17.3369293, 16.2358322]))
model2017_1_bfm_nomouth.append(Pair('left.ear.lobule.center', [72.7483749, -13.2787151, 12.7273741]))
model2017_1_bfm_nomouth.append(Pair('left.ear.lobule.tip', [68.5077209, -18.5463467, 12.5550623]))
model2017_1_bfm_nomouth.append(Pair('left.ear.tragus.tip', [72.4910507, 4.89446688, 14.015893]))
model2017_1_bfm_nomouth.append(Pair('left.eye.bottom', [30.8319283, 28.8719139, 93.7743988]))
model2017_1_bfm_nomouth.append(Pair('left.eyebrow.bend.lower', [46.931263, 49.027935, 94.9866104]))
model2017_1_bfm_nomouth.append(Pair('left.eyebrow.bend.upper', [44.4016685, 55.4631691, 96.1824265]))
model2017_1_bfm_nomouth.append(Pair('left.eyebrow.inner_lower', [19.0661564, 48.1254311, 105.932648]))
model2017_1_bfm_nomouth.append(Pair('left.eyebrow.inner_upper', [20.4881687, 53.7197075, 108.116074]))
model2017_1_bfm_nomouth.append(Pair('left.eye.corner_inner', [16.1573315, 31.8608284, 92.9888535]))
model2017_1_bfm_nomouth.append(Pair('left.eye.corner_outer', [43.0308571, 32.5797234, 86.7202759]))
model2017_1_bfm_nomouth.append(Pair('left.eye.pupil.center', [30.8336086, 33.2174835, 95.220459]))
model2017_1_bfm_nomouth.append(Pair('left.eye.top', [30.8075256, 37.1855469, 95.3033066]))
model2017_1_bfm_nomouth.append(Pair('left.lips.corner', [23.5172367, -33.2380562, 97.9552231]))
model2017_1_bfm_nomouth.append(Pair('left.lips.philtrum_ridge', [5.4616003, -25.0098629, 114.817848]))
model2017_1_bfm_nomouth.append(Pair('left.nasolabial_fold.bottom', [36.2107506, -32.7677345, 93.8350906]))
model2017_1_bfm_nomouth.append(Pair('left.nasolabial_fold.center', [29.9009743, -18.6213512, 99.6171494]))
model2017_1_bfm_nomouth.append(Pair('left.nose.hole.center', [7.49403143, -8.7173214, 111.962296]))
model2017_1_bfm_nomouth.append(Pair('left.nose.wing.outer', [17.5326176, -2.76806164, 103.668877]))
model2017_1_bfm_nomouth.append(Pair('left.nose.wing.tip', [13.7165737, -10.4465151, 105.995186]))
model2017_1_bfm_nomouth.append(Pair('right.ear.antihelix.center', [-83.7815704, 7.65924788, -5.45491505]))
model2017_1_bfm_nomouth.append(Pair('right.ear.antihelix.tip', [-76.5435944, -2.8854816, 9.55981541]))
model2017_1_bfm_nomouth.append(Pair('right.ear.helix.attachement', [-79.9036865, 37.3048668, 5.14881897]))
model2017_1_bfm_nomouth.append(Pair('right.ear.helix.center', [-84.0007324, 2.37693, -8.8059988]))
model2017_1_bfm_nomouth.append(Pair('right.ear.helix.outer', [-87.9340668, 19.6208534, -15.712657]))
model2017_1_bfm_nomouth.append(Pair('right.ear.helix.top', [-85.7635574, 38.3022842, 0.423802406]))
model2017_1_bfm_nomouth.append(Pair('right.ear.lobule.attachement', [-66.5214462, -17.565424, 15.6900816]))
model2017_1_bfm_nomouth.append(Pair('right.ear.lobule.center', [-73.6610794, -13.4019375, 12.1780186]))
model2017_1_bfm_nomouth.append(Pair('right.ear.lobule.tip', [-68.7841873, -18.7374401, 11.5634184]))
model2017_1_bfm_nomouth.append(Pair('right.ear.tragus.tip', [-73.8605194, 4.79287958, 13.7834711]))
model2017_1_bfm_nomouth.append(Pair('right.eye.bottom', [-31.677433, 28.7479153, 93.8427048]))
model2017_1_bfm_nomouth.append(Pair('right.eyebrow.bend.lower', [-47.6376495, 48.420105, 95.4094543]))
model2017_1_bfm_nomouth.append(Pair('right.eyebrow.bend.upper', [-45.0199318, 55.1247902, 96.4838409]))
model2017_1_bfm_nomouth.append(Pair('right.eyebrow.inner_lower', [-19.6622295, 47.7437057, 106.096184]))
model2017_1_bfm_nomouth.append(Pair('right.eyebrow.inner_upper', [-21.0835247, 53.1678391, 108.294319]))
model2017_1_bfm_nomouth.append(Pair('right.eye.corner_inner', [-17.0949841, 31.7714787, 93.1282425]))
model2017_1_bfm_nomouth.append(Pair('right.eye.corner_outer', [-44.0275459, 32.4399872, 86.6149063]))
model2017_1_bfm_nomouth.append(Pair('right.eye.pupil.center', [-31.5763073, 33.1489372, 95.2801437]))
model2017_1_bfm_nomouth.append(Pair('right.eye.top', [-31.5933933, 37.0980988, 95.341156]))
model2017_1_bfm_nomouth.append(Pair('right.lips.corner', [-23.7050438, -33.4582787, 98.3342056]))
model2017_1_bfm_nomouth.append(Pair('right.lips.philtrum_ridge', [-5.45049906, -24.9918728, 114.845917]))
model2017_1_bfm_nomouth.append(Pair('right.nasolabial_fold.bottom', [-36.3713493, -33.0296097, 94.2529373]))
model2017_1_bfm_nomouth.append(Pair('right.nasolabial_fold.center', [-30.0923862, -18.8879261, 100.020355]))
model2017_1_bfm_nomouth.append(Pair('right.nose.hole.center', [-7.57467985, -8.7219038, 112.035164]))
model2017_1_bfm_nomouth.append(Pair('right.nose.wing.outer', [-17.6506786, -2.86107826, 103.952957]))
model2017_1_bfm_nomouth.append(Pair('right.nose.wing.tip', [-13.8790989, -10.5827579, 106.113472]))
model2017_1_bfm_nomouth.append(Pair('right.eyebrow.bend.center', [-44.643, 50.9685, 97.8512]))
model2017_1_bfm_nomouth.append(Pair('right.eyebrow.inner.center', [-20.398, 49.4756, 107.178]))
model2017_1_bfm_nomouth.append(Pair('left.eyebrow.inner.center', [19.8027, 49.9112, 107.024]))
model2017_1_bfm_nomouth.append(Pair('left.eyebrow.bend.center', [44.0197, 51.5964, 97.4669]))

model2017_1_face12_nomouth = list()


def get_list(filename):
    filename = os.path.basename(filename)

    if filename == 'model2017-1_bfm_nomouth.h5':
        return model2017_1_bfm_nomouth
    elif filename == 'model2017-1_face12_nomouth.h5':
        return model2017_1_face12_nomouth
    else:
        return None


def to_array(landmarks):
    return np.array([pair.point for pair in landmarks])
