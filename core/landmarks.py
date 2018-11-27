__author__ = 'Ruslan N, Kosarev'

import numpy as np


class Landmark:
    def __init__(self, point, index=None, name=None, weight=1):
        self.point = np.array(point)
        self.name = name
        self.weight = weight
        self.index = index


class LandmarkListBase:
    def __init__(self):
        self._landmarks = None

    @property
    def landmarks(self):
        return self._landmarks

    def to_array(self):
        return np.array([landmark.point for landmark in self.landmarks])

    def get_weights(self):
        return np.array([landmark.weight for landmark in self.landmarks])

    def get_binary_weights(self, threshold=0):
        return np.array([landmark.weight > threshold for landmark in self.landmarks])


class BaselFaceModeNoMouth2017Dlib(LandmarkListBase):
    def __init__(self):
        super().__init__()
        self._landmarks = (
            Landmark([-72.2,  25.0,  43.6], 0, weight=0),
            Landmark([-69.8,   6.5,  43.0], 1, weight=0),
            Landmark([-68.4, -12.4,  44.4], 2, weight=0),
            Landmark([-63.2, -29.0,  57.5], 3, weight=0),
            Landmark([-55.5, -45.0,  65.4], 4, weight=0),
            Landmark([-44.5, -58.2,  73.3], 5, weight=0),
            Landmark([-29.6, -68.0,  88.0], 6, weight=0),
            Landmark([-15.6, -76.0,  96.3], 7, weight=0),
            Landmark([  0.0, -77.6,  99.0], 8, weight=0),
            Landmark([ 15.6, -75.6,  95.5], 9, weight=0),
            Landmark([ 29.2, -67.6,  89.3], 10, weight=0),
            Landmark([ 44.5, -58.2,  73.4], 11, weight=0),
            Landmark([ 55.5, -45.0,  65.0], 12, weight=0),
            Landmark([ 63.2, -29.0,  56.4], 13, weight=0),
            Landmark([ 68.4, -12.4,  44.4], 14, weight=0),
            Landmark([ 69.8,   6.5,  43.0], 15, weight=0),
            Landmark([ 72.2,  25.0,  43.6], 16, weight=0),
            Landmark([-56.0,  47.5,  86.8], 17, weight=1),
            Landmark([-45.3,  53.5,  97.7], 18, weight=1),
            Landmark([-34.4,  52.8, 103.1], 19, weight=1),
            Landmark([-22.4,  51.2, 106.6], 20, weight=1),
            Landmark([-10.8,  46.8, 107.2], 21, weight=1),
            Landmark([  9.6,  47.6, 107.6], 22, weight=1),
            Landmark([ 21.2,  52.0, 107.0], 23, weight=1),
            Landmark([ 33.6,  54.0, 102.8], 24, weight=1),
            Landmark([ 45.3,  53.5,  96.7], 25, weight=1),
            Landmark([ 56.0,  47.5,  84.9], 26, weight=1),
            Landmark([ -0.8,  34.8, 111.0], 27, weight=1),
            Landmark([ -0.4,  23.2, 119.6], 28, weight=1),
            Landmark([  0.0,  11.6, 128.2], 29, weight=1),
            Landmark([  0.0,   0.0, 131.6], 30, weight=1),
            Landmark([-14.0,  -9.6, 107.4], 31, weight=1),
            Landmark([ -7.6, -11.2, 112.7], 32, weight=1),
            Landmark([ -0.4, -12.8, 114.7], 33, weight=1),
            Landmark([  6.8, -11.2, 113.2], 34, weight=1),
            Landmark([ 13.2,  -9.6, 108.3], 35, weight=1),
            Landmark([-45.0,  33.0,  86.7], 36, weight=1),
            Landmark([-36.9,  38.8,  96.2], 37, weight=1),
            Landmark([-27.1,  39.0,  96.2], 38, weight=1),
            Landmark([-19.0,  34.0,  92.3], 39, weight=1),
            Landmark([-27.1,  31.3,  93.7], 40, weight=1),
            Landmark([-36.9,  30.5,  92.6], 41, weight=1),
            Landmark([ 17.0,  34.0,  92.2], 42, weight=1),
            Landmark([ 26.8,  39.0,  95.4], 43, weight=1),
            Landmark([ 35.8,  38.5,  95.2], 44, weight=1),
            Landmark([ 44.0,  34.0,  87.9], 45, weight=1),
            Landmark([ 35.8,  30.5,  92.8], 46, weight=1),
            Landmark([ 26.8,  31.1,  93.7], 47, weight=1),
            Landmark([-24.8, -33.9,  99.2], 48, weight=1),
            Landmark([-14.8, -27.6, 110.6], 49, weight=1),
            Landmark([ -6.8, -26.6, 114.1], 50, weight=1),
            Landmark([  0.0, -26.2, 115.0], 51, weight=1),
            Landmark([  6.8, -26.6, 114.1], 52, weight=1),
            Landmark([ 14.8, -27.6, 110.5], 53, weight=1),
            Landmark([ 24.8, -33.9,  98.4], 54, weight=1),
            Landmark([ 15.6, -39.0, 106.8], 55, weight=1),
            Landmark([  7.4, -41.0, 111.2], 56, weight=1),
            Landmark([  0.0, -41.3, 112.1], 57, weight=1),
            Landmark([ -7.4, -41.0, 111.2], 58, weight=1),
            Landmark([-15.2, -39.0, 107.9], 59, weight=1),
            Landmark([-20.8, -34.0, 101.1], 60, weight=1),
            Landmark([ -6.8, -32.0, 110.0], 61, weight=1),
            Landmark([  0.0, -32.8, 110.6], 62, weight=1),
            Landmark([  6.8, -31.6, 110.0], 63, weight=1),
            Landmark([ 20.8, -34.0, 100.9], 64, weight=1),
            Landmark([  6.8, -31.2, 111.2], 65, weight=1),
            Landmark([  0.0, -32.0, 110.9], 66, weight=1),
            Landmark([ -6.8, -31.6, 110.0], 67, weight=1))


# class BaselFaceModeNoMouth2017(LandmarkBase):
#     def __init__(self):
#         super().__init__()
#         self._landmarks = (
#             Landmark([-0.0955040976, -73.366539, 102.963692], 'center.chin.tip'),
#             Landmark([-0.377398103, 84.9062271, 104.559608], 'center.front.trichion'),
#             Landmark([-0.00852414407, -32.5953026, 109.807968], 'center.lips.lower.inner'),
#             Landmark([0.00494586537, -41.5086899, 111.808228], 'center.lips.lower.outer'),
#             Landmark([0.00292040454, -32.4464722, 109.903816], 'center.lips.upper.inner'),
#             Landmark([0.0115437824, -26.1546135, 115.271103], 'center.lips.upper.outer'),
#             Landmark([0.0253486931, -10.950614, 116.14006], 'center.nose.attachement_to_philtrum'),
#             Landmark([0.203733876, 0.97169292, 131.823807], 'center.nose.tip'),
#             Landmark([82.5063705, 6.57122803, -4.90304041], 'left.ear.antihelix.center'),
#             Landmark([75.1497192, -2.98772359, 10.0038261], 'left.ear.antihelix.tip'),
#             Landmark([79.9922104, 37.9479218, 4.55147552], 'left.ear.helix.attachement'),
#             Landmark([82.73983, 0.960383058, -7.79163694], 'left.ear.helix.center'),
#             Landmark([87.5009003, 17.4589539, -15.073473], 'left.ear.helix.outer'),
#             Landmark([85.9402008, 38.4805679, -0.328399032], 'left.ear.helix.top'),
#             Landmark([66.251709, -17.3369293, 16.2358322], 'left.ear.lobule.attachement'),
#             Landmark([72.7483749, -13.2787151, 12.7273741], 'left.ear.lobule.center'),
#             Landmark([68.5077209, -18.5463467, 12.5550623], 'left.ear.lobule.tip'),
#             Landmark([72.4910507, 4.89446688, 14.015893], 'left.ear.tragus.tip'),
#             Landmark([30.8319283, 28.8719139, 93.7743988], 'left.eye.bottom'),
#             Landmark([46.931263, 49.027935, 94.9866104], 'left.eyebrow.bend.lower'),
#             Landmark([44.4016685, 55.4631691, 96.1824265], 'left.eyebrow.bend.upper'),
#             Landmark([19.0661564, 48.1254311, 105.932648], 'left.eyebrow.inner_lower'),
#             Landmark([20.4881687, 53.7197075, 108.116074], 'left.eyebrow.inner_upper'),
#             Landmark([16.1573315, 31.8608284, 92.9888535], 'left.eye.corner_inner'),
#             Landmark([43.0308571, 32.5797234, 86.7202759], 'left.eye.corner_outer'),
#             Landmark([30.8336086, 33.2174835, 95.220459], 'left.eye.pupil.center'),
#             Landmark([30.8075256, 37.1855469, 95.3033066], 'left.eye.top'),
#             Landmark([23.5172367, -33.2380562, 97.9552231], 'left.lips.corner'),
#             Landmark([5.4616003, -25.0098629, 114.817848], 'left.lips.philtrum_ridge'),
#             Landmark([36.2107506, -32.7677345, 93.8350906], 'left.nasolabial_fold.bottom'),
#             Landmark([29.9009743, -18.6213512, 99.6171494], 'left.nasolabial_fold.center'),
#             Landmark([7.49403143, -8.7173214, 111.962296], 'left.nose.hole.center'),
#             Landmark([17.5326176, -2.76806164, 103.668877], 'left.nose.wing.outer'),
#             Landmark([13.7165737, -10.4465151, 105.995186], 'left.nose.wing.tip'),
#             Landmark([-83.7815704, 7.65924788, -5.45491505], 'right.ear.antihelix.center'),
#             Landmark([-76.5435944, -2.8854816, 9.55981541], 'right.ear.antihelix.tip'),
#             Landmark([-79.9036865, 37.3048668, 5.14881897], 'right.ear.helix.attachement'),
#             Landmark([-84.0007324, 2.37693, -8.8059988], 'right.ear.helix.center'),
#             Landmark([-87.9340668, 19.6208534, -15.712657], 'right.ear.helix.outer'),
#             Landmark([-85.7635574, 38.3022842, 0.423802406], 'right.ear.helix.top'),
#             Landmark([-66.5214462, -17.565424, 15.6900816], 'right.ear.lobule.attachement'),
#             Landmark([-73.6610794, -13.4019375, 12.1780186], 'right.ear.lobule.center'),
#             Landmark([-68.7841873, -18.7374401, 11.5634184], 'right.ear.lobule.tip'),
#             Landmark([-73.8605194, 4.79287958, 13.7834711], 'right.ear.tragus.tip'),
#             Landmark([-31.677433, 28.7479153, 93.8427048], 'right.eye.bottom'),
#             Landmark([-47.6376495, 48.420105, 95.4094543], 'right.eyebrow.bend.lower'),
#             Landmark([-45.0199318, 55.1247902, 96.4838409], 'right.eyebrow.bend.upper'),
#             Landmark([-19.6622295, 47.7437057, 106.096184], 'right.eyebrow.inner_lower'),
#             Landmark([-21.0835247, 53.1678391, 108.294319], 'right.eyebrow.inner_upper'),
#             Landmark([-17.0949841, 31.7714787, 93.1282425], 'right.eye.corner_inner'),
#             Landmark([-44.0275459, 32.4399872, 86.6149063], 'right.eye.corner_outer'),
#             Landmark([-31.5763073, 33.1489372, 95.2801437], 'right.eye.pupil.center'),
#             Landmark([-31.5933933, 37.0980988, 95.341156], 'right.eye.top'),
#             Landmark([-23.7050438, -33.4582787, 98.3342056], 'right.lips.corner'),
#             Landmark([-5.45049906, -24.9918728, 114.845917], 'right.lips.philtrum_ridge'),
#             Landmark([-36.3713493, -33.0296097, 94.2529373], 'right.nasolabial_fold.bottom'),
#             Landmark([-30.0923862, -18.8879261, 100.020355], 'right.nasolabial_fold.center'),
#             Landmark([-7.57467985, -8.7219038, 112.035164], 'right.nose.hole.center'),
#             Landmark([-17.6506786, -2.86107826, 103.952957], 'right.nose.wing.outer'),
#             Landmark([-13.8790989, -10.5827579, 106.113472], 'right.nose.wing.tip'),
#             Landmark([-44.643, 50.9685, 97.8512], 'right.eyebrow.bend.center'),
#             Landmark([-20.398, 49.4756, 107.178], 'right.eyebrow.inner.center'),
#             Landmark([19.8027, 49.9112, 107.024], 'left.eyebrow.inner.center'),
#             Landmark([44.0197, 51.5964, 97.4669], 'left.eyebrow.bend.center')
#         )
