# coding: utf-8
import logging
from operator import itemgetter

from exceptions import IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight,\
    IndexMoreThanWidth, LeakedThrough, NoFollowDirections
from frames import FrameFollow, FrameSection, FrameLeak, Leak, Cell
from numpy import array
import pandas as pd
from objects import Contour, GleamStream, LeakStream, Section


class Matrix(object):
    def __init__(self):
        self.matrix = None

    def set_matrix(self, matrix):
        self.matrix = matrix
        return self

    def get_frame(self, x, y, size=3):
        """get frame from hole matrix in general coordinate system

        Get neighborhood frame
        """
        half = size // 2
        not_included = 1
        xmin = x - half
        ymin = y - half
        xmax = x + half + not_included
        ymax = y + half + not_included

        # print("""[{0}:{1},{2}:{3}]""".format(xmin, xmax, ymin, ymax))
        height = self.matrix.shape[0] - 1
        width = self.matrix.shape[1] - 1
        if x >= height:
            raise IndexMoreThanHeight
        if x <= 0:
            raise IndexLessThanHeight

        if y >= width:
            raise IndexMoreThanWidth
        if y <= 0:
            raise IndexLessThanWidth

        # frame = self.matrix[int(xmin):int(xmax), int(ymin):int(ymax)]
        frame = self.matrix[xmin:xmax, ymin:ymax]
        return frame

    def get_explored_frame(self, x, y, flow):
        """get frame from hole matrix in general coordinate system

        Get directed neighborhood based on frame flow
        Parameters
        ----------
        flow: {1, 2, 3, 4, 5, 6, 7, 8}

        Returns
        -------
        numpy.array
            Matrix 3x3.
        """
        not_included = 1
        delta = {
            1: {"row": (-2, 0),
                "col": (0, 2)},
            2: {"row": (-1, 1),
                "col": (-2, 0)},
            3: {"row": (0, 2),
                "col": (-2, )},
            4: {"row": (0, 2),
                "col": (-1, 1)},
            5: {"row": (0, 2),
                "col": (0, 2)},
            6: {"row": (-1, 1),
                "col": (0, 2)},
            7: {"row": (-2, 0),
                "col": (-2, 0)},
            8: {"row": (-2, 2),
                "col": (-1, 1)}
        }

        xmin = x + delta[flow]["row"][0]
        ymin = y + delta[flow]["col"][0]
        xmax = x + delta[flow]["row"][1] + not_included
        ymax = y + delta[flow]["col"][1] + not_included

        # print("""[{0}:{1},{2}:{3}]""".format(xmin, xmax, ymin, ymax))
        height = self.matrix.shape[0] - 1
        width = self.matrix.shape[1] - 1
        if xmax >= height:
            raise IndexMoreThanHeight
        if xmin <= 0:
            raise IndexLessThanHeight

        if ymax >= width:
            raise IndexMoreThanWidth
        if ymin <= 0:
            raise IndexLessThanWidth

        # frame = self.matrix[int(xmin):int(xmax), int(ymin):int(ymax)]
        frame = self.matrix[xmin:xmax, ymin:ymax]
        return frame


class MatrixFollow(Matrix):
    def __init__(self, x, y):
        super(MatrixFollow, self).__init__()
        self.enter_point = (x, y)

    def follow_old_4(self, reverse=False, contour_object=None):
        """слідуж лінії, та поертає один обєкт"""
        start_point = None
        if start_point is None:
            start_point = self.enter_point
            # reverse = False

        contour = Contour().set_reverse(reverse)
        if contour_object is not None:
            contour = contour_object

        while True:
            try:
                print("[{0}, {1}],".format(start_point[0], start_point[1]))

                frame = self.get_frame(start_point[0], start_point[1])
                fframe = FrameFollow().set(frame, reverse=reverse)
                nfc = fframe.get_next_frame_centroid()

                if nfc is None:
                    raise IndexLessThanHeight

                current_centroid = array([start_point[0], start_point[1]])
                next_centroid = current_centroid + nfc.get_bias()

                # print("current_centroid", current_centroid)

                if self.enter_point[0] == next_centroid[0] and self.enter_point[1] == next_centroid[1] and len(contour.body) != 0:
                    contour.closed = True
                    # print("""closed:\n\tlen:{0}""".format(len(contour.body)))
                    break

                if (self.enter_point[0] == next_centroid[0] + 1 or self.enter_point[0] == next_centroid[0] - 1)\
                        and (self.enter_point[1] == next_centroid[1] + 1 or self.enter_point[1] == next_centroid[1] - 1)\
                        and len(contour.body) != 0:
                    contour.closed = True
                    # print("""closed:\n\tlen:{0}""".format(len(contour.body)))
                    break

                # TODO зробити поправку на перевірку геометрії а також її дублікатів
                contour.add_point(current_centroid)
                contour.add_point(next_centroid)

                start_point = next_centroid
            except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight, IndexMoreThanWidth):
                break

        return contour

    def follow_old_5(self, reverse=False, contour_object=None):
        """слідуж лінії, та поертає один обєкт"""
        prev_follow_direction = None

        start_point = None
        if start_point is None:
            start_point = self.enter_point
            # reverse = False

        contour = Contour().set_reverse(reverse)
        if contour_object is not None:
            contour = contour_object

        #
        logging.debug("\n\nCONTOUR BODY(reverse: {0})".format(reverse))
        while True:
            try:
                # print("[{0}, {1}],".format(start_point[0], start_point[1]))
                logging.debug("\n\tStart point: [{0}, {1}]".format(start_point[0], start_point[1]))
                frame = self.get_frame(start_point[0], start_point[1])
                fframe = FrameFollow().set(frame, reverse=reverse)
                possible_directions = fframe.get_next_frame_centroid()
                logging.debug("possible directions: {0}".format([c.code for c in possible_directions]))

                nfc = None
                if len(possible_directions) == 1:
                    for el in possible_directions:
                        nfc = el
                elif len(possible_directions) > 1:
                    directions = {c.code for c in possible_directions}
                    if prev_follow_direction is not None and prev_follow_direction in directions:
                        for el in possible_directions:
                            if el.code == prev_follow_direction:
                                nfc = el
                                break
                    else:
                        for el in possible_directions:
                            if el.code == min(directions):
                                nfc = el
                                break

                prev_follow_direction = nfc.code

                current_centroid = array([start_point[0], start_point[1]])
                next_centroid = current_centroid + nfc.get_bias()
                # current_centroid position

                logging.debug("current: [{0}, {1}], next: [{2}, {3}], flow: {4}".format(
                    current_centroid[0], current_centroid[1],
                    next_centroid[0], next_centroid[1],
                    nfc.code
                ))
                # print("current_centroid", current_centroid)

                if self.enter_point[0] == next_centroid[0] and self.enter_point[1] == next_centroid[1] and len(contour.body) != 0:
                    contour.closed = True
                    # print("""closed:\n\tlen:{0}""".format(len(contour.body)))
                    break

                if (self.enter_point[0] == next_centroid[0] + 1 or self.enter_point[0] == next_centroid[0] - 1)\
                        and (self.enter_point[1] == next_centroid[1] + 1 or self.enter_point[1] == next_centroid[1] - 1)\
                        and len(contour.body) != 0:
                    contour.closed = True
                    # print("""closed:\n\tlen:{0}""".format(len(contour.body)))
                    break

                # TODO зробити поправку на перевірку геометрії а також її дублікатів
                contour.add_point(current_centroid)
                contour.add_point(next_centroid)

                start_point = next_centroid
            except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight, IndexMoreThanWidth, NoFollowDirections):
                break

        return contour

    def follow(self, reverse=False, contour_object=None):
        """слідуж лінії, та поертає один обєкт"""
        previous_flow_code = None
        start_point = None
        if start_point is None:
            start_point = self.enter_point
            # reverse = False

        contour = Contour().set_reverse(reverse)
        if contour_object is not None:
            contour = contour_object

        #
        logging.debug("\n\nCONTOUR BODY(reverse: {0})".format(reverse))
        while True:
            try:
                # print("[{0}, {1}],".format(start_point[0], start_point[1]))
                logging.debug("\n\tStart point: [{0}, {1}]".format(start_point[0], start_point[1]))
                frame = self.get_frame(start_point[0], start_point[1])
                fframe = FrameFollow().set(frame, reverse=reverse)

                logging.debug("Previous flow direction: {0}".format(previous_flow_code))
                fframe.set_previous_flow(previous_flow_code)

                nfc = fframe.get_next_frame_centroid()
                logging.debug("Next flow direction: {0}".format(nfc.code))
                previous_flow_code = nfc.code

                current_centroid = array([start_point[0], start_point[1]])
                next_centroid = current_centroid + nfc.get_bias()
                # current_centroid position

                logging.debug("current: [{0}, {1}], next: [{2}, {3}], flow: {4}".format(
                    current_centroid[0], current_centroid[1],
                    next_centroid[0], next_centroid[1],
                    nfc.code
                ))
                # print("current_centroid", current_centroid)

                if self.enter_point[0] == next_centroid[0] and self.enter_point[1] == next_centroid[1] and len(contour.body) != 0:
                    contour.closed = True
                    # print("""closed:\n\tlen:{0}""".format(len(contour.body)))
                    break

                if (self.enter_point[0] == next_centroid[0] + 1 or self.enter_point[0] == next_centroid[0] - 1)\
                        and (self.enter_point[1] == next_centroid[1] + 1 or self.enter_point[1] == next_centroid[1] - 1)\
                        and len(contour.body) != 0:
                    contour.closed = True
                    # print("""closed:\n\tlen:{0}""".format(len(contour.body)))
                    break

                # TODO зробити поправку на перевірку геометрії а також її дублікатів
                contour.add_point(current_centroid)
                contour.add_point(next_centroid)

                start_point = next_centroid
            except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight, IndexMoreThanWidth, NoFollowDirections):
                break

        return contour

    def get_contour(self):
        contour = self.follow()
        print(contour.closed)
        if not contour.closed:
            contour = self.follow(reverse=contour.switch_mode(), contour_object=contour)

        return contour


class MatrixGleam_old(Matrix):
    def __init__(self, x, y):
        super(MatrixGleam, self).__init__()
        self.origin = (x, y)
        self.direction = 5

    def change_origin(self, x, y):
        self.origin = (x, y)

    def set_direction(self, direction):
        self.direction = direction

    def get_reflected_old_working(self, direction=None):
        """
        default direction is diagonal flowing(5)
        return GleamStream object
        """
        if direction is None:
            direction = self.direction

        prev_point = None

        start_point = None

        if start_point is None:
            start_point = self.origin

        gleam_stream = GleamStream()

        prev_gleam_value = None

        while True:
            try:
                frame = self.get_frame(start_point[0], start_point[1])

                gframe = FrameGleam().set(frame).set_flow(direction)

                gframe.set_previous_value(prev_gleam_value)

                gleam = gframe.get_flow()

                gleam_stream.add_point(start_point[0], start_point[1])

                prev_gleam_value = gleam.value

                prev_point = start_point
                # next start point
                start_point = gleam.next_global_xy(prev_point[0], prev_point[1])

                if gleam.is_reflected():
                    gleam_stream.set_break_point(start_point[0], start_point[1], gleam.reflection)
                    break

            except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight, IndexMoreThanWidth) as error:
                frame = self.get_frame(prev_point[0], prev_point[1])
                gframe = FrameGleam().set(frame).set_flow(direction)
                gframe.calc_centroid_contrast()
                gframe.set_image_border(error)
                gleam = gframe.get_flow()

                gleam_stream.set_break_point(prev_point[0], prev_point[1], gleam.reflection)
                break

        return gleam_stream

    def get_reflected(self, direction=None):
        """
        default direction is diagonal flowing(5)
        return GleamStream object
        """
        if direction is None:
            direction = self.direction

        prev_point = None

        start_point = None

        if start_point is None:
            start_point = self.origin

        gleam_stream = GleamStream()

        prev_gleam_value = None

        while True:
            try:
                frame = self.get_frame(start_point[0], start_point[1])

                gframe = FrameGleam().set(frame).set_flow(direction)

                gframe.set_previous_value(prev_gleam_value)

                gleam = gframe.get_flow()

                # print("GLEAM:\n\tback: {0},\n\trefl: {1},\n\tcoor: {2} | {3}".format(
                #     gleam.half_step_back,
                #     gleam.reflected,
                #     start_point[0],
                #     start_point[1])
                # )

                if not gleam.is_half_step_back():
                    gleam_stream.add_point(start_point[0], start_point[1])

                prev_gleam_value = gleam.value

                prev_point = start_point
                # next start point

                start_point = gleam.next_global_xy(prev_point[0], prev_point[1])

                if gleam.is_reflected():
                    gleam_stream.set_break_point(start_point[0], start_point[1], gleam.reflection)
                    break

            except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight, IndexMoreThanWidth) as error:
                frame = self.get_frame(prev_point[0], prev_point[1])
                gframe = FrameGleam().set(frame).set_flow(direction)
                gframe.calc_centroid_contrast()
                gframe.set_image_border(error)
                gleam = gframe.get_flow()

                gleam_stream.set_break_point(prev_point[0], prev_point[1], gleam.reflection)
                break

        return gleam_stream

    def reflect(self, level=1):
        # global list of break points
        global_break_points = list()

        reflect_count = 0

        all_reflections = list()

        while level - reflect_count != 0:
            # level_reflections = list()
            if reflect_count == 0:
                default_gleam_stream = GleamStream().set_break_point(1, 1, {self.direction})
                level_reflections = [default_gleam_stream]

            tmp_list = list()
            for gleam_stream in level_reflections:
                if reflect_count != 0:
                    self.change_origin(gleam_stream.break_point[0], (gleam_stream.break_point[1]))
                reflection = gleam_stream.reflection

                # print(gleam_stream.reflection, reflect_count)

                for refl_direction in reflection:
                    reflected = self.get_reflected(refl_direction)
                    # print("reflected", reflected.break_point)
                    if reflected.break_point.tolist() not in global_break_points:
                        global_break_points.append(reflected.break_point.tolist())
                        tmp_list.append(reflected)

            level_reflections = tmp_list
            all_reflections.append(level_reflections)
            reflect_count += 1

        return all_reflections


class MatrixSection(Matrix):
    def __init__(self, x, y):
        super(MatrixSection, self).__init__()
        self.local_origin = (x, y)

    def change_origin(self, x, y):
        self.local_origin = (x, y)

    @staticmethod
    def split_into_parts(gleam_stream):
        # FIXME add doc string
        data = [{"value": el[1], "row": el[0][0], "col": el[0][1]} for el in gleam_stream]

        df = pd.DataFrame(data, columns=("value", "row", "col"))
        df["diff"] = round(df["value"].diff() ** 3, 2)

        df = df.dropna()
        # if not equival to 0 set as 1
        df.ix[df["diff"] != 0, "diff"] = 1

        df["group"] = (df["diff"].diff() != 0).cumsum()
        df = df.set_index(["group", "diff"])
        # get groups for further processing
        # res = sorted([el for el in df.groupby(["group", "diff"]).groups], key=itemgetter(0))
        res = sorted([el for el in df.groupby(by=[df.index.get_level_values(0), df.index.get_level_values(1)]).groups], key=itemgetter(0))
        if res[0][1] != 1.0:
            del res[0]


        # take transitions as a start of object and fill as a body of an object

        split_gleam_stream = list()

        for start_group_ind, end_group_ind in zip(res[::2], res[1::2]):
            start_group = df.ix[start_group_ind[0], :]
            end_group = df.ix[end_group_ind[0], :]

            start_point = start_group.tail(1)
            end_point = end_group.tail(1)

            sample_body = start_point.append(end_group)
            # print(sample)
            sample = {
                "start_point": start_point[["row", "col"]].values[0],
                "end_point": end_point[["row", "col"]].values[0],
                "body": [el for el in sample_body[["row", "col"]].values],
                "mean_value": sample_body["value"].mean(),
            }
            split_gleam_stream.append(sample)

        return df, split_gleam_stream

    def get_section(self, direction, depth=None):
        """
        default direction is diagonal flowing(5)
        return GleamStream object
        """
        start_point = None

        if start_point is None:
            start_point = self.local_origin

        row_section = list()

        # prev_gleam_value = None
        counter = 0
        while True:
            try:
                frame = self.get_frame(start_point[0], start_point[1])
                sframe = FrameSection().set(frame).set_flow(direction)

                section_part = sframe.get_section()
                for cell in section_part:
                    global_loc = cell.get_global_xy(start_point[0], start_point[1])
                    row_section.append((global_loc, cell.value))

                nfc = sframe.get_nfc()
                start_point = nfc.get_global_xy(start_point[0], start_point[1])

                if depth is not None:
                    counter += 1
                if counter > depth:
                    break

            except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanHeight, IndexMoreThanWidth):
                break

        # return self.split_into_parts(row_section)
        return Section(row_section)


class MatrixLeak(Matrix):
    def __init__(self, x, y):
        super(MatrixLeak, self).__init__()
        self.origin = (x, y)
        self.direction = None

    def change_origin(self, x, y):
        self.origin = (x, y)

    def set_direction(self, direction):
        self.direction = direction

    def leak_old(self):
        start_point = None
        if start_point is None:
            start_point = self.origin

        _leaked_in_value = False

        res = list()
        while True:
            try:
                print(start_point)
                frame = self.get_frame(start_point[0], start_point[1])
                lframe = FrameLeak().set(frame).set_flow(self.direction)
                if _leaked_in_value:
                    lframe.set_leaked_in()
                # print(lframe.direction)

                leak = lframe.get_next_leak()
                if leak.is_leaked_in():
                    _leaked_in_value = True
                print(leak.is_leaked_in())

                start_point = leak.next_global_xy(start_point[0], start_point[1])
                res.append(start_point)
                print(start_point)
            except LeakedThrough:
                break

        return res

    def leak(self):
        start_point = None
        if start_point is None:
            start_point = self.origin

        _leaked_in_value = False

        leak_stream = LeakStream()
        while True:
            try:
                frame = self.get_frame(start_point[0], start_point[1])
                lframe = FrameLeak().set(frame).set_flow(self.direction)
                if _leaked_in_value:
                    lframe.set_leaked_in()
                # print(lframe.direction)

                leak = lframe.get_next_leak()
                if leak.is_leaked_in():
                    _leaked_in_value = True
                # print(leak.is_leaked_in())

                start_point = leak.next_global_xy(start_point[0], start_point[1])

                leak_stream.add_point(start_point[0], start_point[1])

            except LeakedThrough:
                _fllow_cell = Cell().set(self.direction).get_xy()
                _leak = Leak(*_fllow_cell)
                leaked_point = _leak.next_global_xy(start_point[0], start_point[1])
                leak_stream.set_leaked_point(leaked_point[0], leaked_point[1])
                break

        return leak_stream


