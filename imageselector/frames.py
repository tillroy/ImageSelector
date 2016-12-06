# coding: utf-8
import logging
import math

from numpy import apply_along_axis, abs as absolute, array, where
from pandas import DataFrame

from color import Color
from exceptions import IndexMoreThanHeight, IndexMoreThanWidth, IndexLessThanWidth,\
    IndexLessThanHeight, FrameLoop, UnreflectedGleam, LeakInMistake, LeakedThrough


class Cell(object):
    """save data about cell features, local x, y , etc."""
    CELL_LOCATION = {
        # frame INDEXES
        1: array([0, 0]),
        8: array([0, 1]),
        7: array([0, 2]),
        2: array([1, 0]),
        6: array([1, 2]),
        3: array([2, 0]),
        4: array([2, 1]),
        5: array([2, 2])
    }

    CELL_NAME = {
        # frame INDEXES
        (0, 0): 1,
        (0, 1): 8,
        (0, 2): 7,
        (1, 0): 2,
        (1, 2): 6,
        (2, 0): 3,
        (2, 1): 4,
        (2, 2): 5,
    }

    def __init__(self):
        self.code = None
        self.reverse = False
        self.hub = False
        self.value = None

    def set_value(self, value):
        self.value = value
        return self

    @staticmethod
    def validate(code):
        if 1 <= code <= 8:
            return code
        elif code > 8:
            full_part = code // 8
            res = code - (full_part * 8)
            return res
        elif code < 1:
            r = (abs(code) // 8) + 1
            res = 8 * r - abs(code)
            return res

    def set(self, code):
        self.code = self.validate(code)
        return self

    def set_reverse(self):
        self.reverse = True

    def unset_reverse(self):
        self.reverse = False

    def get_xy(self):
        if self.hub:
            return array([1, 1])
        else:
            return self.CELL_LOCATION[self.code]

    def as_hub(self):
        self.hub = True
        return self

    def set_by_xy(self, coord):
        code = self.CELL_NAME[coord]
        self.set(code)

        return self

    def get_bias(self):
        """координати для мап зміщення за годинниковою стрілкою та проти неї"""
        nfc = self.get_xy()
        bias = nfc - array([1, 1])
        return bias

    def get_direction(self):
        """повертає координати наступної позиці в зананому напрямку"""
        direction_offset = self.get_xy() - array([1, 1])
        direction = self.get_xy() + direction_offset
        return direction

    def get_global_xy(self, gx, gy):
        global_origin_xy = array([gx, gy])
        global_direction_xy = global_origin_xy + self.get_bias()

        return global_direction_xy

    def is_main(self):
        if self.code in (2, 4, 6, 8):
            return True
        else:
            return False

    def get_next(self, steps=1):
        if self.reverse:
            next_code = self.validate(self.code - steps)
            return next_code
        else:
            next_code = self.validate(self.code + steps)
            return next_code

    def get_prev(self, steps=1):
        if self.reverse:
            next_code = self.validate(self.code + steps)
            return next_code
        else:
            next_code = self.validate(self.code - steps)
            return next_code

    def get_opposite(self):
        return self.get_next(4)


class Gleam_old(object):
    """
    class for gleam line object that could be an input for FrameGleam
    and keep info about reflection directions local and global location
    and also is it reflected(obstacle has been found) gleam or not
    """
    def __init__(self, x=0, y=0):
        # mean that stops near obstacle
        self.reflected = None

        self.half_step_back = False
        # list of reflection direction
        self.reflection = None
        # coordinates of the gleam returned by FrameGleam
        self.direction = array([x, y])
        self.value = None
        self.previous_value = None
        # by default it is a centroid of a frame 3x3
        self.obstacle_point = array([1, 1])

    def set_value(self, value):
        self.value = value

    def set_previous_value(self, prev_value):
        self.previous_value = prev_value

    def set_reflected(self, mode=False):
        self.reflected = mode
        return self

    def set_half_step_back(self, mode=False):
        self.half_step_back = mode
        return self

    def is_half_step_back(self):
        return self.half_step_back

    def is_reflected(self):
        if self.reflected is None:
            raise ValueError
        else:
            return self.reflected

    def get_bias(self):
        bias = self.direction - array([1, 1])
        return bias

    def set_reflection(self, refl_set):
        self.reflection = refl_set
        return self

    def next_global_xy(self, gx, gy):
        if self.reflected:
            return array([gx, gy])
        else:
            global_origin_xy = array([gx, gy])
            global_direction_xy = global_origin_xy + self.get_bias()

            return global_direction_xy

    def change_obstacle_point(self, obstacle_point_code=None):
        """
        change obstacle point where exactly start smth else on frame 3x3
        by default it is the centroid of the frame
        """
        if obstacle_point_code is not None:
            cell = Cell().set(obstacle_point_code)
            cell_xy = cell.get_xy()
            self.obstacle_point = array([cell_xy[0], cell_xy[1]])

        return self


class Leak(object):
    def __init__(self, x=0, y=0):
        self.leaked_in = None
        self.leaked_out = None

        self.reflection = None
        # coordinates of the gleam returned by FrameGleam
        self.direction = array([x, y])
        self.value = None
        self.previous_value = None

    def set_leaked_in(self, value=False):
        self.leaked_in = value
        return self

    def is_leaked_in(self):
        return self.leaked_in

    def get_bias(self):
        bias = self.direction - array([1, 1])
        return bias

    def next_global_xy(self, gx, gy):
        global_origin_xy = array([gx, gy])
        global_direction_xy = global_origin_xy + self.get_bias()

        return global_direction_xy


class Route(object):
    """
    методи для руху по фрейму вперед та назад, а аткож підразунок маршшруту та стартової точки
    для фрейму із номерацією
    1   8   7
    2   x   6
    3   4   5
    """
    def __init__(self):
        self.start_point = None
        self.track = None
        self.count_steps = 0
        self.count_circle = 0

    def set_start_point(self, start_point, reverse=False):
        """встановлює нову початкову точку від якою беде розраховуватись трек в одну чи іншу сторону"""
        self.start_point = Cell().set(start_point)
        self.track = self.__calc_track(reverse)
        return self

    def current(self, ind=None):
        """перетворення індексу на значення позиції на калькульованому відрізку"""
        if ind is None:
            index = self.__get_index()
            point = self.track[index]
            return point
        else:
            index = self.__get_index(ind)
            point = self.track[index]
            return point

    def __get_index(self, steps=None):
        """return correct index from 0 to 7"""
        if steps is None:
            if self.count_circle == 0:
                full_steps = 0
            else:
                full_steps = 8 * self.count_circle

            ind = self.count_steps - full_steps
            # print("st",self.count_steps, "|fs",full_steps,"|ind", ind)
            return ind
        else:
            count_circle = steps // 8

            if count_circle == 0:
                full_steps = 0
            else:
                full_steps = 8 * count_circle

            ind = steps - full_steps
            # print("st",self.count_steps, "|fs",full_steps,"|ind", ind)
            return ind

    def __calc_circle(self):
        """пирведення індексів до відрузку від 0 до 7"""
        self.count_circle = int(self.count_steps / 8)

    def __calc_track(self, reverse=False):
        point = self.start_point
        if reverse:
            point.set_reverse()

        steps = list()
        for el in range(8):
            steps.append(point)
            # point = self.__point_to_circle(point.code + 1)
            if reverse:
                point = Cell().set(point.code - 1)
                point.set_reverse()
            else:
                point = Cell().set(point.code + 1)

        return steps

    def on_main(self):
        if self.current().code in (2, 4, 6, 8):
            return True
        else:
            return False

    def next(self, steps_count=1):
        if self.start_point is not None and self.track is not None:
            cur_ind = self.count_steps
            nex_ind = cur_ind + steps_count

            self.count_steps = nex_ind
            self.__calc_circle()

            return self.current()

    def get_next(self, steps_count=1):
        cur_count = self.count_steps
        nex_count_steps = cur_count + steps_count

        next_cell = self.current(nex_count_steps)

        return next_cell.code

    def get_prev(self, steps_count=1):
        cur_count = self.count_steps
        prev_count_steps = cur_count - steps_count

        prev_cell = self.current(prev_count_steps)

        return prev_cell.code

    def prev(self, steps_count=1):
        if self.start_point is not None and self.track is not None:
            cur_ind = self.count_steps
            prev_ind = cur_ind - steps_count

            self.count_steps = prev_ind
            self.__calc_circle()

            return self.current()


class Frame(Route):
    def __init__(self):
        super(Frame, self).__init__()
        self.frame_base = None
        self.frame_contrast = None
        self.frame_gray = None
        self.frame_binary = None
        self.reverse = False
        self.sensitivity = 0.30

    def set(self, frame, reverse=False):
        self.frame_base = frame
        self.reverse = reverse
        return self

    def set_frame_contrast(self, contrast):
        self.frame_contrast = contrast

    def set_frame_gray(self, gray_scale):
        self.frame_gray = gray_scale

    def set_frame_base(self, frame):
        self.frame_base = frame
        return self

    def value(self, cell, source="contrast"):
        point_coord = cell.get_xy()
        if source == "base":
            value = self.frame_base[point_coord[0], point_coord[1]]
            return value
        elif source == "contrast":
            value = self.frame_contrast[point_coord[0], point_coord[1]]
            return value
        elif source == "gray":
            value = self.frame_gray[point_coord[0], point_coord[1]]
            return value
        elif source == "binary":
            value = self.frame_binary[point_coord[0], point_coord[1]]
            return value
        else:
            value = self.frame_gray[point_coord[0], point_coord[1]]
            return value

    def set_value(self, cell, value, source="contrast"):
        """set value on cell position"""
        point_coord = cell.get_xy()
        if source == "base":
            self.frame_base[point_coord[0], point_coord[1]] = value
            return self
        elif source == "contrast":
            self.frame_contrast[point_coord[0], point_coord[1]] = value
            return self
        elif source == "gray":
            self.frame_gray[point_coord[0], point_coord[1]] = value
            return self
        elif source == "binary":
            self.frame_binary[point_coord[0], point_coord[1]] = value
            return self
        else:
            self.frame_gray[point_coord[0], point_coord[1]] = value
            return self

    def coerce_grey(self):
        if self.frame_base.ndim == 3:
            grey_frame = apply_along_axis(
                lambda x: Color().set(*x).coerce_grey().value(),
                2,
                self.frame_base
            )
            self.set_frame_gray(grey_frame)

            return self
        elif self.frame_base.ndim == 2:
            self.set_frame_gray(self.frame_base)
            return self
        else:
            raise ValueError("Wrong array frame size")

    def binarize(self):
        if self.frame_gray is None:
            self.coerce_grey()

        df_grey = self.frame_gray

        df_grey *= df_grey

        _min = df_grey.min()
        _max = df_grey.max()
        _df_min = df_grey - _min
        _df_max = _max - df_grey
        mask = _df_max > _df_min

        binary = where(mask, 1, 0)
        self.frame_binary = binary

        # set hub as background 0 mean white
        hub = Cell().as_hub()
        self.set_value(hub, 0, source="binary")
        # print(self.frame_binary)

    def centroid(self, source="gray"):
        """тільки для фрейму 3 на 3"""
        if source == "base":
            return self.frame_base[1, 1]
        elif source == "contrast":
            return self.frame_contrast[1, 1]
        elif source == "gray":
            return self.frame_gray[1, 1]
        else:
            return self.frame_gray[1, 1]

    def calc_centroid_contrast(self, absolute_value=True):
        if self.frame_gray is None:
            self.coerce_grey()
        contrast = self.frame_gray - self.centroid(source="grey")
        if absolute_value:
            contrast = absolute(contrast)
        self.set_frame_contrast(contrast)

    def show(self, source="gray"):
        if source == "base":
            print(self.frame_base)
        elif source == "contrast":
            print(self.frame_contrast)
        elif source == "gray":
            print(self.frame_gray)
        elif source == "binary":
            print(self.frame_binary)
        else:
            print(self.frame_gray)


class FrameFollow(Frame):
    """позиціонується центром завджи в місці пошуку"""
    def __init__(self):
        super(FrameFollow, self).__init__()
        self.checked_point = list()
        # default value for flow, because of lack of info about previous flow
        self.previous_flow_cell = None

    def set_previous_flow(self, prev_flow_code=None):
        """Chose current flow based on previous one's value.

        Keep info about previous flow value.

        Parameters
        ----------
        prev_flow_code : {1, 2, 3, 4, 5, 6, 7, 8}
            Cell code.

        """
        if prev_flow_code is not None:
            cell = Cell().set(prev_flow_code)
            self.previous_flow_cell = cell
        else:
            # becouse net step will get an opposite value
            cell = Cell().set(8)
            self.previous_flow_cell = cell

    def get_start_point(self):
        """Set opposite to previous flow value as track start point.

        Get opposite meaning to previous flow value as track start point.

        Returns
        -------
        {1, 2, 3, 4, 5, 6, 7, 8}
            Start point code.
        """

        start_point_code = self.previous_flow_cell.get_opposite()

        return start_point_code

    def make_tack(self):
        """Make track sequence from start point."""

        self.set_start_point(self.get_start_point(), self.reverse)

    def get_next_frame_centroid(self):
        """повертає імя наступного пікселя в який можливо попасти із якоря та значення якого не є нижчім за поріг"""
        self.make_tack()
        self.binarize()

        # track = [c.code for c in self.track]
        # print("track", track)
        logging.debug("Frame:\n{0}".format(self.frame_binary))
        values = [self.value(c, source="binary") for c in self.track]

        # TODO make validation for not main directions

        border_end_ind = None
        for ind, border in enumerate(values):
            if border_end_ind is None and border:
                border_end_ind = ind
            elif border:
                border_end_ind = ind
            elif border_end_ind is not None and not border:
                break

        direction_cell_ind = border_end_ind + 1
        logging.debug("direction_cell_ind: {0}".format(direction_cell_ind))
        direction_cell = self.current(direction_cell_ind)

        logging.debug("Flow direction: {0}".format(direction_cell.code))

        return direction_cell


class FrameGleam_old(Frame):
    def __init__(self):
        super(FrameGleam, self).__init__()
        # direction of flowing 5 means 45 degrees right down flow
        self.flow = None
        self.previous_value = None
        # None means - centroid coordinates
        self.obstacle_point = None
        self.half_step_back = False

    def set_previous_value(self, value):
        self.previous_value = value

    def set_flow(self, flow):
        """
        flow direction that takes from the default frame map
        1   8   7
        2   x   6
        3   4   5
        """
        self.flow = Cell().set(flow)
        return self

    def get_next_flow(self):
        """get local coordinates of the next centroid frame"""
        if self.reverse:
            self.flow.set_reverse()

        next_flow = self.flow.get_direction()
        return next_flow

    def set_image_border(self, error):
        print(type(error))

        if isinstance(error, IndexLessThanHeight):
            cell1 = Cell().set(1)
            cell8 = Cell().set(8)
            cell7 = Cell().set(7)
            self.set_value(cell1, 3)
            self.set_value(cell8, 3)
            self.set_value(cell7, 3)

        if isinstance(error, IndexMoreThanHeight):
            cell3 = Cell().set(3)
            cell4 = Cell().set(4)
            cell5 = Cell().set(5)
            self.set_value(cell3, 3)
            self.set_value(cell4, 3)
            self.set_value(cell5, 3)

        if isinstance(error, IndexMoreThanWidth):
            cell7 = Cell().set(7)
            cell6 = Cell().set(6)
            cell5 = Cell().set(5)
            self.set_value(cell7, 3)
            self.set_value(cell6, 3)
            self.set_value(cell5, 3)

        if isinstance(error, IndexLessThanWidth):
            cell1 = Cell().set(1)
            cell2 = Cell().set(2)
            cell3 = Cell().set(3)
            self.set_value(cell1, 3)
            self.set_value(cell2, 3)
            self.set_value(cell3, 3)

    def reflected_directions(self):
        """return set of directions where reflection could occur"""
        # set default configuration and make track from 1
        self.set_start_point(1)
        directions = {c.code for c in self.track if self.value(c) < self.sensitivity}
        directions1 = {(c.code, self.value(c)) for c in self.track}
        print(directions1)

        odd = set()

        for cell_code in directions:
            # print(directions)
            curr = Cell().set(cell_code)
            # print(curr.code)
            if self.reverse:
                curr.set_reverse()

            prev = curr.get_prev()
            nex = curr.get_next()

            if not curr.is_main():
                if prev not in directions and nex not in directions:
                    odd.add(curr.code)

        diff = directions.difference(odd)

        # remove duplicated reflaction diirection
        diff = diff.difference({self.flow.get_opposite()})

        if not diff:
            # raise UnreflectedGleam
            return diff
        else:
            return diff

    def is_obstacle(self):
        if self.frame_contrast is None:
            self.calc_centroid_contrast()

        flow_value = self.value(self.flow, source="contrast")

        if self.previous_value is not None:
            # gleam_diff = self.previous_value - self.value(self.flow, source="grey")
            gleam_diff = self.previous_value - self.centroid(source="grey")
            if gleam_diff >= self.sensitivity:
                self.obstacle_point = self.flow.get_opposite()
                self.half_step_back = True
                # print("the same: {0}|{1}".format(self.previous_value, self.centroid(source="grey")))
                # print("border")
                return False
        if flow_value >= self.sensitivity:
            return True
        elif flow_value < self.sensitivity:
            # if on border conner with 1px width
            if not self.flow.is_main():
                nex = Cell().set(self.flow.get_next())
                prev = Cell().set(self.flow.get_prev())
                if self.value(nex) >= self.sensitivity and self.value(prev) >= self.sensitivity:
                    return True
            else:
                return False
        else:
            return False

    def get_flow(self):
        if self.is_obstacle():
            gl = Gleam().set_reflected(True)
            gl.set_value(self.centroid(source="gray"))
            gl.set_reflection(self.reflected_directions())
            return gl
        else:
            if self.half_step_back:
                # print("repeat half of step")
                prev_flow_code = self.flow.get_opposite()
                cell_step_back = Cell().set(prev_flow_code)
                # print(cell_step_back.code)
                gl = Gleam(*cell_step_back.get_xy()).set_reflected(False)
                gl.set_half_step_back(True)
                # gl.set_value(self.previous_value)
                # gl.change_obstacle_point(self.obstacle_point)
                return gl
            else:
                flow = self.get_next_flow()
                gl = Gleam(*flow).set_reflected(False)
                gl.set_value(self.centroid(source="gray"))
                return gl


class FrameSection(Frame):
    """
    Attributes
    ----------
    flow: Cell
        Direction of moving.
    anchor: Cell
        Base position in the frame
    """
    def __init__(self):
        super(FrameSection, self).__init__()
        # direction of flowing 5 means 45 degrees right down flow
        self.flow = None
        self.anchor = None

    def set_flow(self, flow):
        """
        flow direction that takes from the default frame map
        1   8   7
        2   x   6
        3   4   5
        """
        self.flow = Cell().set(flow)
        self.anchor = Cell().set(self.flow.get_opposite())
        return self

    def get_nfc(self):
        """get local coordinates of the next centroid frame"""
        # next_flow = self.flow.get_direction()
        # return next_flow
        return self.flow

    def get_section(self):
        self.coerce_grey()
        section = list()
        anchor = self.anchor

        if not anchor.is_main():
            pre = Cell().set(self.flow.get_prev())
            # pre = Cell().set(anchor.get_prev())
            pre_value = self.value(pre, source="gray")

            # nex = Cell().set(anchor.get_next())
            nex = Cell().set(self.flow.get_prev())
            nex_value = self.value(nex, source="gray")

            pseudo_hub_val = (pre_value + nex_value) / 2
            # pseudo_hub_val = math.sqrt((pre_value**2 + nex_value**2) / 2)

            hub = Cell().as_hub()
            hub_value = self.value(hub, source="gray")

            updated_hub_value = hub_value if hub_value > pseudo_hub_val else pseudo_hub_val

            # hub = hub.set_value(updated_hub_value)
            hub = hub.set_value(hub_value)
            section.append(hub)
        else:
            hub = Cell().as_hub()
            hub = hub.set_value(self.value(hub, source="gray"))
            section.append(hub)

        return section


class FrameLeak(Frame):
    """Leaking frame. Going through obstacles and return coordinates behind obstacle"""
    def __init__(self):
        super(FrameLeak, self).__init__()
        self.flow = None
        self.leaked_in = False
        self.leaked_out = False

    def set_flow(self, flow):
        """
        flow direction that takes from the default frame map
        1   8   7
        2   x   6
        3   4   5
        """
        self.flow = Cell().set(flow)
        return self

    def set_leaked_in(self):
        self.leaked_in = True
        return self

    def is_leaked_in(self):
        return self.leaked_in

    def leak_in(self):
        """leak into obstacle"""
        hub = Cell().as_hub()
        diff = self.value(self.flow, source="grey") - self.value(hub, source="grey")
        print(diff)
        if abs(diff) >= self.sensitivity:
            self.set_leaked_in()
        else:
            print(diff)
            self.show()

            print("center",self.value(hub, source="grey"))
            print("flow", self.value(self.flow, source="grey"))
            raise LeakInMistake

    def get_next_leak_old(self):
        self.coerce_grey()
        self.show()
        if not self.is_leaked_in():
            self.leak_in()
            flow = self.flow.get_xy()
            leak = Leak(*flow).set_leaked_in(self.leaked_in)
            return leak
        else:
            hub = Cell().as_hub()
            diff = abs(self.value(hub, source="grey") - self.value(self.flow, source="grey"))
            if diff < self.sensitivity:

                flow = self.flow.get_xy()
                leak = Leak(*flow).set_leaked_in(self.leaked_in)

                return leak
            elif diff >= self.sensitivity:
                raise LeakedThrough

    def get_next_leak(self, mode="wb"):
        # TODO make ability working in two modes WB and BW
        """wb - means moving from white to black colour"""
        self.coerce_grey()
        # self.show()
        if not self.is_leaked_in():
            self.leak_in()
            flow = self.flow.get_xy()
            # print(flow)
            leak = Leak(*flow).set_leaked_in(self.leaked_in)
            return leak
        else:
            if mode == "wb":
                # print("ss")
                # self.show(source="contrast")
                hub = Cell().as_hub()
                # diff = abs(self.value(hub, source="grey") - self.value(self.flow, source="grey"))
                diff = self.value(self.flow, source="grey") - self.value(hub, source="grey")
                print(diff)
                if diff < self.sensitivity:
                    flow = self.flow.get_xy()
                    leak = Leak(*flow).set_leaked_in(self.leaked_in)

                    # print("do smth else and return Leak object")
                    return leak

                # if diff < self.sensitivity:
                #
                #     flow = self.flow.get_xy()
                #     leak = Leak(*flow).set_leaked_in(self.leaked_in)
                #
                #     print("do smth else and return Leak object")
                #     return leak
                elif diff >= self.sensitivity:
                    raise LeakedThrough


if __name__ == "__main__":
    from PIL import Image

    # img = Image.open("img/problem1.jpg")

    img = Image.open("img/problem_27_79.png")
    res = array(img)

    data = array([
        [.05882353,    .05882353,     .05882353],
        [.05882353,    0.,             0.19803922],
        [.05882353,    0.2372549,      .05882353]
    ])

    # f = FrameGleam().set(res).set_flow(7)
    # # f.calc_centroid_contrast(absolute_value=False)
    # f.calc_centroid_contrast(absolute_value=True)
    # f.show(source="contrast")
    # f.show(source="grey")
    # # f.calc_centroid_contrast(absolute_value=False)

    ff = FrameSection().set(res)
    # ff.calc_centroid_contrast(absolute_value=False)
    # ff.show(source="contrast")

    # print("")
    # ff.show(source="grey")
    # r = ff.get_next_frame_centroid_old_3()

    ff.set_flow(1)
    ff.get_section()
    ff.show(source="grey")

    # ff.show_track_chart()

    # print(r.code)
    # print(r.get_bias())

    # l = FrameLeak().set(res).set_flow(6)
    # #
    # l.set_leaked_in()
    # print(l.is_leaked_in())
    # l.calc_centroid_contrast(absolute_value=False)
    # res = l.get_next_leak()
    # l.show(source="grey")
    # l.show(source="contrast")

    # print(res.is_leaked_in())
    # print(res.direction)
    # print(res.next_global_xy(1, 1))

    # r = Route().set_start_point(2, reverse=True)
    # tract = [c.code for c in r.track]
    # print(tract)
    # print(r.next().code)

    # c = Cell().set(2, reverse=True)
    # print(c.code)
    # print(c.get_xy())