import math
from typing import *


class Schedule():
    def __init__(self, period, offset):
        self.period = period
        self.offset = offset

    def __call__(self, step: int, epoch: int, x: object) -> object:
        if step < self.offset:
            return 0
        else:
            return float(step - self.offset) / self.period


class LinearSchedule(Schedule):
    def __init__(self, v_init, v_final, period, offset: int = 0):
        super().__init__(period, offset)
        self.v_init = v_init
        self.v_final = v_final

    def __call__(self, *args, **kwargs) -> object:
        u = super().__call__(*args, **kwargs)
        x = min(1, u)
        return self.v_init + x * (self.v_final - self.v_init)

    def __repr__(self):
        return f"LinearSchedule: init = {self.v_init}, final = {self.v_final}, period = {self.period}, offset = {self.offset}"


class PieceWiseSchedule():
    def __init__(self, schedules: List[Schedule]):
        """
        piece-wise schedule
        :param points: Schedules)]
        """

        _schedules = []
        t = 0
        for sch in schedules:
            a_x = t
            b_x = sch.offset + sch.period
            _schedules += [(a_x, b_x, sch)]

            t = b_x

        self.schedules = _schedules

    def __call__(self, step: int, epoch: int, value: float):

        s = [s for a, b, s in self.schedules if step >= a and step < b]

        if len(s):
            return s[0](step, epoch, value)
        else:
            return self.schedules[-1][2].v_final


class PieceWiseLinearSchedule():
    def __init__(self, points: List[Tuple[int, float]]):
        """
        piece-wise linear schedule
        :param points: coordinates list[(step, values)]
        """
        assert points[0][0] == 0, "value for step 0 must be provided"
        t = 0

        schedules = []
        for a, b in zip(points[:-1], points[1:]):
            a_x, a_y = a
            b_x, b_y = b

            assert b_x > a_x, "next step corrdinate must be greater than the previous one"

            t = b_x
            s = LinearSchedule(a_y, b_y, b_x - a_x, offset=a_x)

            schedules += [(a_x, b_x, s)]

        self.schedules = schedules

    def __call__(self, step: int, epoch: int, value: float):

        s = [s for a, b, s in self.schedules if step >= a and step < b]

        if len(s):
            return s[0](step, epoch, value)
        else:
            return self.schedules[-1][2].v_final


class ExponentialSchedule(Schedule):
    def __init__(self, v_init, v_final, period, offset: int = 0):
        super().__init__(period, offset)
        self.v_init = v_init
        self.v_final = v_final

    def __call__(self, step: int, epoch: int, x: object) -> object:
        if step < self.offset:
            return self.v_init
        else:
            u = super().__call__(step, epoch, x)
            x = 1 - math.exp(- u)
            return self.v_init + x * (self.v_final - self.v_init)


class DecaySchedule(Schedule):
    def __init__(self, decay, min_value, offset: int = 0):
        self.decay = decay
        self.min_value = min_value
        self.offset = offset
        self.period = math.inf

    def __call__(self, step: int, epoch: int, x: object) -> object:
        if step < self.offset:
            return self.v_init
        else:
            return x * self.decay if x > self.min_value else x
