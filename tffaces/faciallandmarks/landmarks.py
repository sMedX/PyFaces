__author__ = 'Ruslan N, Kosarev'

import numpy as np


class Landmark:
    def __init__(self, point, name=None):
        self.point = np.array(point)
        self.name = name


class LandmarksBase:
    def __init__(self):
        self._landmarks = None

    @property
    def number_of_landmarks(self):
        return len(self._landmarks)

    @property
    def landmarks(self):
        return self._landmarks

    @property
    def points(self):
        return np.array([landmark.point for landmark in self.landmarks])

    @property
    def names(self):
        return tuple([landmark.name for landmark in self.landmarks])

    def __repr__(self):
        info = '{} (number of landmarks {})\n'.format(self.__class__.__name__, self.number_of_landmarks)
        for landmark in self.landmarks:
            info = info + '{}, {}\n'.format(landmark.point.tolist(), landmark.name)
        return info


class BaselFaceModeLandmarks(LandmarksBase):
    def __init__(self):
        super().__init__()
        self._landmarks = (
            Landmark([-0.0955040976, -73.366539, 102.963692], 'center.chin.tip'),
            Landmark([-0.3773981030, 84.9062271, 104.559608], 'center.front.trichion'),
            Landmark([-0.00852414407, -32.5953026, 109.807968], 'center.lips.lower.inner'),
            Landmark([0.00494586537, -41.5086899, 111.808228], 'center.lips.lower.outer'),
            Landmark([0.00292040454, -32.4464722, 109.903816], 'center.lips.upper.inner'),
            Landmark([0.0115437824, -26.1546135, 115.271103], 'center.lips.upper.outer'),
            Landmark([0.0253486931, -10.950614, 116.14006], 'center.nose.attachement_to_philtrum'),
            Landmark([0.203733876, 0.97169292, 131.823807], 'center.nose.tip'),
            Landmark([82.5063705, 6.57122803, -4.90304041], 'left.ear.antihelix.center'),
            Landmark([75.1497192, -2.98772359, 10.0038261], 'left.ear.antihelix.tip'),
            Landmark([79.9922104, 37.9479218, 4.55147552], 'left.ear.helix.attachement'),
            Landmark([82.73983, 0.960383058, -7.79163694], 'left.ear.helix.center'),
            Landmark([87.5009003, 17.4589539, -15.073473], 'left.ear.helix.outer'),
            Landmark([85.9402008, 38.4805679, -0.328399032], 'left.ear.helix.top'),
            Landmark([66.251709, -17.3369293, 16.2358322], 'left.ear.lobule.attachement'),
            Landmark([72.7483749, -13.2787151, 12.7273741], 'left.ear.lobule.center'),
            Landmark([68.5077209, -18.5463467, 12.5550623], 'left.ear.lobule.tip'),
            Landmark([72.4910507, 4.89446688, 14.015893], 'left.ear.tragus.tip'),
            Landmark([30.8319283, 28.8719139, 93.7743988], 'left.eye.bottom'),
            Landmark([46.931263, 49.027935, 94.9866104], 'left.eyebrow.bend.lower'),
            Landmark([44.4016685, 55.4631691, 96.1824265], 'left.eyebrow.bend.upper'),
            Landmark([19.0661564, 48.1254311, 105.932648], 'left.eyebrow.inner_lower'),
            Landmark([20.4881687, 53.7197075, 108.116074], 'left.eyebrow.inner_upper'),
            Landmark([16.1573315, 31.8608284, 92.9888535], 'left.eye.corner_inner'),
            Landmark([43.0308571, 32.5797234, 86.7202759], 'left.eye.corner_outer'),
            Landmark([30.8336086, 33.2174835, 95.220459], 'left.eye.pupil.center'),
            Landmark([30.8075256, 37.1855469, 95.3033066], 'left.eye.top'),
            Landmark([23.5172367, -33.2380562, 97.9552231], 'left.lips.corner'),
            Landmark([5.4616003, -25.0098629, 114.817848], 'left.lips.philtrum_ridge'),
            Landmark([36.2107506, -32.7677345, 93.8350906], 'left.nasolabial_fold.bottom'),
            Landmark([29.9009743, -18.6213512, 99.6171494], 'left.nasolabial_fold.center'),
            Landmark([7.49403143, -8.7173214, 111.962296], 'left.nose.hole.center'),
            Landmark([17.5326176, -2.76806164, 103.668877], 'left.nose.wing.outer'),
            Landmark([13.7165737, -10.4465151, 105.995186], 'left.nose.wing.tip'),
            Landmark([-83.7815704, 7.65924788, -5.45491505], 'right.ear.antihelix.center'),
            Landmark([-76.5435944, -2.8854816, 9.55981541], 'right.ear.antihelix.tip'),
            Landmark([-79.9036865, 37.3048668, 5.14881897], 'right.ear.helix.attachement'),
            Landmark([-84.0007324, 2.37693, -8.8059988], 'right.ear.helix.center'),
            Landmark([-87.9340668, 19.6208534, -15.712657], 'right.ear.helix.outer'),
            Landmark([-85.7635574, 38.3022842, 0.423802406], 'right.ear.helix.top'),
            Landmark([-66.5214462, -17.565424, 15.6900816], 'right.ear.lobule.attachement'),
            Landmark([-73.6610794, -13.4019375, 12.1780186], 'right.ear.lobule.center'),
            Landmark([-68.7841873, -18.7374401, 11.5634184], 'right.ear.lobule.tip'),
            Landmark([-73.8605194, 4.79287958, 13.7834711], 'right.ear.tragus.tip'),
            Landmark([-31.677433, 28.7479153, 93.8427048], 'right.eye.bottom'),
            Landmark([-47.6376495, 48.420105, 95.4094543], 'right.eyebrow.bend.lower'),
            Landmark([-45.0199318, 55.1247902, 96.4838409], 'right.eyebrow.bend.upper'),
            Landmark([-19.6622295, 47.7437057, 106.096184], 'right.eyebrow.inner_lower'),
            Landmark([-21.0835247, 53.1678391, 108.294319], 'right.eyebrow.inner_upper'),
            Landmark([-17.0949841, 31.7714787, 93.1282425], 'right.eye.corner_inner'),
            Landmark([-44.0275459, 32.4399872, 86.6149063], 'right.eye.corner_outer'),
            Landmark([-31.5763073, 33.1489372, 95.2801437], 'right.eye.pupil.center'),
            Landmark([-31.5933933, 37.0980988, 95.341156], 'right.eye.top'),
            Landmark([-23.7050438, -33.4582787, 98.3342056], 'right.lips.corner'),
            Landmark([-5.45049906, -24.9918728, 114.845917], 'right.lips.philtrum_ridge'),
            Landmark([-36.3713493, -33.0296097, 94.2529373], 'right.nasolabial_fold.bottom'),
            Landmark([-30.0923862, -18.8879261, 100.020355], 'right.nasolabial_fold.center'),
            Landmark([-7.57467985, -8.7219038, 112.035164], 'right.nose.hole.center'),
            Landmark([-17.6506786, -2.86107826, 103.952957], 'right.nose.wing.outer'),
            Landmark([-13.8790989, -10.5827579, 106.113472], 'right.nose.wing.tip'),
            Landmark([-44.643, 50.9685, 97.8512], 'right.eyebrow.bend.center'),
            Landmark([-20.398, 49.4756, 107.178], 'right.eyebrow.inner.center'),
            Landmark([19.8027, 49.9112, 107.024], 'left.eyebrow.inner.center'),
            Landmark([44.0197, 51.5964, 97.4669], 'left.eyebrow.bend.center')
        )
