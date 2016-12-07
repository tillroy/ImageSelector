# coding: utf-8
import logging
from operator import itemgetter
import math

import numpy as np
import pandas as pd

from exceptions import ContourLoop, EmptySection


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Threshold(object):
    def __init__(self):
        # FIXME don't use pixel based metrics
        self.PIXEL_OFFSET = 5
        self.COLOR_DIFFERENCE = 0.30


class EnterPointPool(Threshold, metaclass=Singleton):
    def __init__(self):
        super(EnterPointPool, self).__init__()
        self.pool = None
        self.last_elected = None
        # name of column
        self.by = "enter_point_row"

    def set_base_pool_old(self, first_enter_points):
        """asdasdasdasd

        dasdasdasd

        Parameters
        ----------
        first_enter_points : list
            adasdasdasd
        """
        if self.pool is None:
            _pool = pd.DataFrame(first_enter_points, columns=("row", "col"))
            print(_pool)
            self.pool = _pool
            logging.debug("""\nBase pool:\n{0}""".format(self.pool))

    def set_base_pool(self, first_enter_points):
        """asdasdasdasd

        dasdasdasd

        Parameters
        ----------
        first_enter_points : list
            adasdasdasd
        """
        if self.pool is None:
            _pool = pd.DataFrame(first_enter_points,
                                 columns=("enter_point_row", "enter_point_col", "next_cell_contour"))
            self.pool = _pool
            logging.debug("""\nBase pool:\n{0}""".format(self.pool))

    def extract_leak_points(self):
        # _max = self.pool.max()
        _min = self.pool.min()

        min_item = self.pool.ix[self.pool[self.by] == _min[self.by]]

        logging.debug("""\nMIN {0} value:\n{1}""".format(self.by.upper(), min_item))

        next_enter_points = self.pool.ix[abs(self.pool.enter_point_row - _min.enter_point_row) < self.PIXEL_OFFSET, :]
        logging.debug("""Next row enter points:\n{0}""".format(next_enter_points))
        # drop next point from pool
        self.pool = self.pool.drop(next_enter_points.index.values)
        # print(self.pool)
        logging.debug("""Dropped pool:\n{0}""".format(self.pool))

        return next_enter_points.values

    def extend_pool(self, new_enter_points):
        _tmp_df = pd.DataFrame(new_enter_points, columns=("enter_point_row", "enter_point_col", "next_cell_contour"))
        # don't use duplicated indices
        self.pool = self.pool.append(_tmp_df, ignore_index=True).sort_values(by=["enter_point_col", "enter_point_row"])
        # print(self.pool)
        self.__validate_enter_points()

    def __validate_enter_points(self):
        """Validate enter points.

        Check whether previous point has better enter point. Try get more left enter point if it
        on the same level(height) as current enter point is.

        """
        # sort in descending direction, because we need the most LEFT enter point in group
        _pool = self.pool.sort_values(by=["enter_point_col", "enter_point_row"], ascending=False)

        values = _pool.values
        res = list()
        for ind, el in enumerate(values):
            try:
                cur = values[ind]
                nex = values[ind + 1]
                diff = cur[0] - nex[0]
                if abs(diff) > self.PIXEL_OFFSET:
                    res.append(cur)

            except IndexError:
                cur = values[ind]
                res.append(cur)

        # sort in default direction
        validated = pd.DataFrame(
            res,
            columns=("enter_point_row", "enter_point_col", "next_cell_contour")
        ).sort_values(by=["enter_point_col", "enter_point_row"])
        print("RES", self.pool)
        self.pool = validated
        print("RES", self.pool)

    def show(self):
        print(self.pool)


class RowCellSet(object):
    # TODO make docstring
    def __init__(self):
        self.body = list()
        self.prev_bbox = None

    def add_contour(self, contour):
        bbox = contour.calc_bbox()
        if self.prev_bbox is not None:
            prev_row_min = self.prev_bbox.ix["min", "row"]
            prev_row_max = self.prev_bbox.ix["max", "row"]
            curr_row_min = bbox.ix["min", "row"]
            curr_row_max = bbox.ix["max", "row"]
            print("""
            prev height: {0}
            curr height: {1}
            BBOX difference: {2}
            """.format(
                prev_row_max - prev_row_min,
                curr_row_max - curr_row_min,
                (prev_row_max - prev_row_min) - (curr_row_max - curr_row_min)
            ))
        self.prev_bbox = bbox
        self.body.append(contour)

    def get_cells_size(self):
        """Get row set cells size.

        Calculated from bbox of each cell.

        Returns
        -------
        dict of list of int
            Dictionary with values of height and width of cells in row set.

        Examples
        --------
        Result returns as:
            {'width': [26, 36, 36, 119], 'height': [31, 31, 31, 14]}
        """
        widths = list()
        heights = list()
        for cell in self.body:
            cell.calc_bbox()
            bbox = cell.bbox
            cell_width = bbox.ix["max", "col"] - bbox.ix["min", "col"]
            cell_height = bbox.ix["max", "row"] - bbox.ix["min", "row"]
            widths.append(cell_width)
            heights.append(cell_height)

        res = {"height": heights, "width": widths}
        return res

    def __get_row_segments_indices(self, feature="height"):
        """Get enter points.

        Get enter points from difference between cells based on they height (by default) or width.

        Returns
        -------
        list of list
            Pairs start index, end index.

        Examples
        --------
        Result returns as:
            [[0, 1, 2], [3], [4, 5], ...]
        Where elements of each group inside global list represent indices of <Contour> objects
        which are in self.body.
        """
        # FIXME add also width measure
        size = self.get_cells_size()
        # Could be for width as well
        measure = size[feature]

        # t = [30, 50, 30, 30, 30, 30, 50]
        # t = [31, 31, 31, 30, 100, 100, 30, 30, 50]
        # measure = t
        # print(measure)

        segments = list()
        _sub_element = list()
        for ind, el in enumerate(measure):
            try:
                cur = measure[ind]
                nex = measure[ind + 1]
                diff = nex - cur
                # FIXME don't use pixel based metrics
                if 0 <= abs(diff) < 5:
                    _sub_element.append(ind)
                else:
                    _sub_element.append(ind)
                    segments.append(_sub_element)
                    _sub_element = list()
            except IndexError:
                cur = measure[ind]
                prev = measure[ind - 1]
                diff = prev - cur
                # FIXME don't use pixel based metrics
                if 0 <= abs(diff) < 5:
                    _sub_element.append(ind)
                    segments.append(_sub_element)
                else:
                    _sub_element = list()
                    _sub_element.append(ind)
                    segments.append(_sub_element)
                    _sub_element = list()
        logging.debug("""
        Sequence {0}:  {1}
        Sequence segments:  {2}""".format(feature, measure, segments))

        return segments

    def get_all_enter_points_old(self):
        """Calculate all possible enter points(bottom left connor) for next row.

        Calculate enter points in each segment group.

        Returns
        -------
        list of numpy.array
        """
        segments = self.__get_row_segments_indices()
        # print("segments", segments)

        all_enter_points = list()
        all_enter_points_new = list()
        for segment in segments:
            # get index of next segment
            next_segment_ind = segments.index(segment) + 1
            # check validity, if it is end left None
            next_cell_contour = None
            if next_segment_ind < len(segments):
                next_segment = segments[next_segment_ind]
                next_cell_contour = self.body[next_segment[0]]

            # print(segment, next_cell_contour)

            segment_start = self.body[segment[0]]
            start_loc = segment_start.get_conner("bl")

            cell_enter_options = {
                "enter_point": start_loc,
                "next_cell_contour": next_cell_contour
            }

            # print(start_loc)
            all_enter_points.append(start_loc)
            all_enter_points_new.append(cell_enter_options)

        print(all_enter_points)
        print(all_enter_points_new)

        logging.debug("""All enter points: {0}""".format(all_enter_points))

        return all_enter_points

    def get_all_enter_points(self):
        """Calculate all possible enter points(bottom left connor) for next row.

        Calculate enter points in each segment group.

        Returns
        -------
        list of numpy.array
        """
        segments = self.__get_row_segments_indices()
        print("segments", segments)

        all_enter_points = list()
        for segment in segments:
            # get index of next segment
            curr_segment_ind = segments.index(segment)
            next_segment_ind = curr_segment_ind + 1
            # check validity, if it is end left None
            next_cell_contour = None
            if next_segment_ind < len(segments):
                curr_segment = segments[curr_segment_ind]
                next_segment = segments[next_segment_ind]
                next_cell_contour = self.body[next_segment[0]]

            # print(segment, next_cell_contour)

            segment_start = self.body[segment[0]]
            start_loc = segment_start.get_conner("bl")

            cell_enter_options = {
                "enter_point_row": start_loc[0],
                "enter_point_col": start_loc[1],
                "next_cell_contour": next_cell_contour
            }

            # print(start_loc)
            all_enter_points.append(cell_enter_options)

        # print(all_enter_points)

        logging.debug("""All enter points: {0}""".format(all_enter_points))

        return all_enter_points

    # @staticmethod
    # def __get_min_old(enter_points, value="row"):
    #     """
    #
    #     Parameters
    #     ----------
    #     enter_points : list of numpy.array
    #         List of coordinates
    #     value : {"row", "col"}
    #         Define which position should be calculated as a minimum.
    #     Returns
    #     -------
    #     numpy.array
    #         The most extreme enter point value
    #     """
    #     if value not in ("row", "col"):
    #         raise ValueError
    #
    #     coord_ind = {"row": 0, "col": 1}
    #
    #     min_value = None
    #     for el in enter_points:
    #         if min_value is None:
    #             min_value = el
    #         else:
    #             # select which index in array(<row_coord>, <coll_coord>) should take
    #             coord = coord_ind[value]
    #             if el[coord] < min_value[coord]:
    #                 min_value = el
    #
    #     logging.debug("MIN value in {0}: {1}".format(value.upper(), min_value))
    #
    #     return min_value
    #
    # @staticmethod
    # def __get_min(enter_points, value="row"):
    #     """
    #
    #     Parameters
    #     ----------
    #     enter_points : list of numpy.array
    #         List of coordinates
    #     value : {"row", "col"}
    #         Define which position should be calculated as a minimum.
    #     Returns
    #     -------
    #     numpy.array
    #         The most extreme enter point value
    #     """
    #     if value not in ("row", "col"):
    #         raise ValueError
    #
    #     coord_ind = {"row": 0, "col": 1}
    #
    #     min_value = None
    #     for el in enter_points:
    #         if min_value is None:
    #             min_value = el
    #         else:
    #             # select which index in array(<row_coord>, <coll_coord>) should take
    #             coord = coord_ind[value]
    #             print(min_value)
    #             if el[0][coord] < min_value[0][coord]:
    #                 min_value = el
    #
    #     logging.debug("MIN value in {0}: {1}".format(value.upper(), min_value))
    #
    #     return min_value
    #
    # def get_next_leak_points(self):
    #     all_enter_points = self.get_all_enter_points()
    #     # FIXME
    #     min_row_val = self.__get_min(all_enter_points)
    #     # print(min_row_val)
    #     next_enter_points = list()
    #     for ind, el in enumerate(all_enter_points):
    #         print("sss")
    #         cur = all_enter_points[ind]
    #         diff = min_row_val[0] - cur[0]
    #         # FIXME don't use pixel based metrics
    #         if 0 <= abs(diff) < 5:
    #             next_enter_points.append(cur)
    #
    #     logging.debug("""Next row enter points: {0}""".format(next_enter_points))
    #     return next_enter_points

    def is_my_point(self, row, col):
        """Check whether point belong row set or not.

        Check point in round (offset) equal to 1.

        Parameters
        ----------
        row : int
            Row index, count from 0.
        col : int
            Column index, count from 0.
        Returns
        -------
        bool
            True if point belong row set and false if not.
        """
        is_my = False
        for cell in self.body:
            if (cell.point_counter.get((row, col)) is not None or
                        cell.point_counter.get((row + 1, col + 1)) is not None or
                        cell.point_counter.get((row - 1, col - 1)) is not None or
                        cell.point_counter.get((row - 1, col + 1)) is not None or
                        cell.point_counter.get((row + 1, col)) is not None or
                        cell.point_counter.get((row - 1, col)) is not None or
                        cell.point_counter.get((row, col + 1)) is not None or
                        cell.point_counter.get((row, col - 1)) is not None or
                        cell.point_counter.get((row + 1, col - 1)) is not None):
                is_my = True
                break

        return is_my


class Contour(object):
    """Shape contour object.

    Represent vector model geometry.

    Attributes
    ----------
    reverse: bool
        Shows in which direction contour has been drawn. It is important when contour while drawing has been stopped
        because of image end. Tats why it is possible to draw contour in two different directions from origin based on
        this attribute.
    closed: bool
        Shows whether contour closed, like polygon, or not, like segment.
    body: list
        Coordinates of all points of the contour.
    point_counter: dict
        Additional attribute for counting duplicated points.
    table: pandas.DataFrame
        Holder for converted body into table base data frame
    """

    def __init__(self):
        self.reverse = False
        self.closed = False
        self.body = list()
        self.point_counter = dict()
        self.table = None
        self.bbox = None

    def add_point(self, point):
        """Add point as numpy.array to body attribute.

        Add points to contour body and, at the same time validate income data.
        Check how many point have been duplicated. Keep base point count statistic.

        Parameters
        ----------
        point: numpy.array
            point representation. Row index in array always comes in 0
            position and column index in 1 position.
        Raises
        ------
        ContourLoop
            More than 2 point duplicates in contour body
        Returns
        -------
        None

        """
        tuple_point = tuple(point)
        if self.point_counter.get(tuple_point) is None:
            self.point_counter[tuple_point] = 1

            self.body.append(point)

        elif self.point_counter.get(tuple_point) is not None:
            self.point_counter[tuple_point] = self.point_counter.get(tuple_point) + 1

        if self.point_counter.get(tuple_point) > 10:
            raise ContourLoop

    def set_reverse(self, reverse=False):
        """Set `reverse` value

        Parameters
        ----------
        reverse: bool
            Value of `reverse` attribute.

        Returns
        -------
        Contour
            Self object.
        """
        self.reverse = reverse
        return self

    def switch_mode(self):
        """Switch `reverse` attribute from one boolean value to another.

        Returns
        -------
        bool
            Boolean value.
        """
        if self.reverse:
            return False
        else:
            return True

    def calc_structure(self):
        """Transform contour geometry list into Pandas DataFrame.

        Returns
        -------
        None
        """
        self.table = pd.DataFrame(self.body, columns=("row", "col"))

        logging.debug("""Structure had been calculated.""")

    def calc_sides(self):
        """Calculate side learion of the coordinates.

        collape both axes coordinates and get extime points in each grioup

        """

        # get max and min value in groups based on row and col
        side_right = self.table.groupby(by=["row"])['col'].idxmax()
        side_left = self.table.groupby(by=["row"])['col'].idxmin()
        side_top = self.table.groupby(by=["col"])['row'].idxmin()
        side_bottom = self.table.groupby(by=["col"])['row'].idxmax()

        self.table.ix[side_right, "side"] = "right"
        self.table.ix[side_left, "side"] = "left"
        self.table.ix[side_top, "side"] = "top"
        self.table.ix[side_bottom, "side"] = "bottom"

    def calc_bbox(self):
        """Calculate contour bounding box.

        Returns
        -------
        dict of int
            Dictionary of max min values for row and columns.

        Examples
        --------
        Returned bbox result:
            {"colmin": 120, "colmax": 239, "rowmin": 12, "rowmax": 26}
        """
        if self.table is not None:
            bbox = pd.DataFrame({"min": self.table.min(), "max": self.table.max()},
                                index=["row", "col"],
                                columns=["min", "max"])
            res = bbox.T
        else:
            self.calc_structure()
            bbox = pd.DataFrame({"min": self.table.min(), "max": self.table.max()},
                                index=["row", "col"],
                                columns=["min", "max"])
            res = bbox.T

        logging.debug("""BBOX: {0}""".format(res))
        self.bbox = res

    @staticmethod
    def remove_outliers(series):
        """Remove outlines by 1.5 of interquartile range

        Parameters
        ----------
        series : pandas.Series

        Returns
        -------
        tuple of int
        """
        q25 = series.describe()["25%"]
        q75 = series.describe()["75%"]
        # interquartile range
        iqr = q75 - q25
        # thresholds
        thr75 = q75 + iqr * 1.5
        thr25 = q75 - iqr * 1.5

        return thr25, thr75

    def get_conner(self, name):
        allowed_names = ("rt", "bl", "tl")

        if name not in allowed_names:
            raise ValueError

        self.calc_sides()

        if name == "rt":
            side_right = self.table[self.table["side"] == "right"]
            r_thr = self.remove_outliers(side_right["col"])
            side_right = side_right[(side_right["col"] >= r_thr[0]) & (side_right["col"] <= r_thr[1])]

            res = side_right.ix[side_right["row"].idxmin()]
            rt = np.array([int(res.row), int(res.col)])
            return rt

        elif name == "bl":
            side_bottom = self.table[self.table["side"] == "bottom"]
            b_thr = self.remove_outliers(side_bottom["row"])
            side_bottom = side_bottom[(side_bottom["row"] >= b_thr[0]) & (side_bottom["row"] <= b_thr[1])]

            res = side_bottom.ix[side_bottom["col"].idxmin()]
            bl = np.array([int(res.row), int(res.col)])
            return bl

        elif name == "tl":
            side_top = self.table[self.table["side"] == "top"]
            t_thr = self.remove_outliers(side_top["row"])
            side_top = side_top[(side_top["row"] >= t_thr[0]) & (side_top["row"] <= t_thr[1])]

            res = side_top.ix[side_top["col"].idxmin()]
            tl = np.array([int(res.row), int(res.col)])
            return tl

            # side_left = self.table[self.table["side"] == "left"]
            # l_thr = self.remove_outliers(side_left["col"])
            # side_left = side_left[(side_left["col"] >= l_thr[0]) & (side_left["col"] <= l_thr[1])]

    def set_table(self, dataframe):
        self.table = dataframe.copy()
        return self

    def get_conner_old(self, name):
        """Get one of the most remote conner of contour.

        Parameters
        ----------
        name: {"lt", "rt", "lb", "rb"}
            Shortening of names for different corners.

            Explanation:

            - `lt` - the most left top coordinate;
            - `rt` - the most right top coordinate;
            - `lb` - the most left bottom coordinate;
            - `rb` - the most right bottom coordinate;

        Returns
        -------
        numpy.array
            Row-column coordinate format NumPy array.

        Raises
        ------
        ValueError
            Name is not allowed.
        """
        # TODO extend to 8 possible combination not only 4
        allowed_names = ("lt", "rt", "lb", "rb",)

        if self.bbox is None:
            self.calc_bbox()

        bbox = self.bbox

        if name not in allowed_names:
            raise ValueError

        res = None
        df = self.table.copy()
        if name == "lt":
            df["rb_dist"] = math.sqrt(
                (bbox.ix["max", "col"] - df["col"]) ** 2 + (bbox.ix["max", "row"] - df["row"]) ** 2)
            max_dist_ind = df["rb_dist"].idxmax()

            # self.table.row.astype(int)
            # self.table.col.astype(int)

            res = df.iloc[max_dist_ind]
            lt = np.array([int(res.row), int(res.col)])
            res = lt

        elif name == "rt":
            df["lb_dist"] = math.sqrt(
                (bbox.ix["min", "col"] - df["col"]) ** 2 + (bbox.ix["max", "row"] - df["row"]) ** 2)
            max_dist_ind = df["lb_dist"].idxmax()

            # self.table.row.astype(int)
            # self.table.col.astype(int)

            res = df.iloc[max_dist_ind]
            rt = np.array([int(res.row), int(res.col)])
            res = rt

        elif name == "lb":
            df["rt_dist"] = math.sqrt(
                (bbox.ix["max", "col"] - df["col"]) ** 2 + (bbox.ix["min", "row"] - df["row"]) ** 2)
            max_dist_ind = df["rt_dist"].idxmax()

            # self.table.row.astype(int)
            # self.table.col.astype(int)

            res = df.iloc[max_dist_ind]
            lb = np.array([int(res.row), int(res.col)])
            res = lb

        elif name == "rb":
            df["lt_dist"] = math.sqrt(
                (bbox.ix["min", "col"] - df["col"]) ** 2 + (bbox.ix["max", "row"] - df["row"]) ** 2)
            max_dist_ind = df["rt_dist"].idxmax()

            res = df.iloc[max_dist_ind]
            rb = np.array([int(res.row), int(res.col)])
            res = rb

        logging.debug("""Value of {0} corner: {1}""".format(name.upper(), res))

        return res

    def clip_by_bbox(self, bbox, inverse=False):
        """Clip data by BBOX values.

        Parameters
        ----------
        bbox : pandas.DataFrame
           pandas.DataFrame
        inverse :

        Returns
        -------
        pandas.DataFrame
        """
        df = self.table.copy()
        if not inverse:
            clipped = df[(df["row"].between(bbox.ix["min", "row"], bbox.ix["max", "row"], inclusive=True)) &
                         (df["col"].between(bbox.ix["min", "col"], bbox.ix["max", "col"], inclusive=True))]
            return clipped
        else:
            clipped = df[(df["row"] <= bbox.ix["min", "row"]) &
                         (df["row"] >= bbox.ix["max", "row"]) &
                         (df["col"] <= bbox.ix["min", "col"]) &
                         (df["col"] >= bbox.ix["max", "col"])]
            return clipped

    def intersection(self, contour, bbox=True):
        # TODO make bbox checker
        if bbox:
            res = self.clip_by_bbox(contour.bbox)
            return res
        else:
            pass


class Keeper(object):
    """Gathering point data into sequence.

    Attributes
    ----------
    body: list
        List of numpy arrays.
    """

    def __init__(self):
        self.body = list()

    def add_point(self, row, col):
        """Add point as numpy array to class body.

        Add new geometry point to it self body attribute.

        Parameters
        ----------
        row: int
            row index(count from 0)

        col: int
            column index(count from 0)
        Returns
        -------
        None
        """
        self.body.append(np.array([row, col]))


class Gleam(object):
    """
    class for gleam line object that could be an input for FrameGleam
    and keep info about reflection directions local and global location
    and also is it reflected(obstacle has been found) gleam or not

    Attributes
    ----------
    direction: {1, 2, 3, 4, 5, 6, 7, 8}
        Flow direction and anchor of the frame if get opposite value.

    value: float or int
        Value of cell in position 5 (centroid).

    """

    def __init__(self, x=0, y=0):
        self.direction = None
        self.value = None

        # mean that stops near obstacle
        self.reflected = None

        self.half_step_back = False
        # list of reflection direction
        self.reflection = None
        # coordinates of the gleam returned by FrameGleam
        self.direction = np.array([x, y])

        self.previous_value = None
        # by default it is a centroid of a frame 3x3
        self.obstacle_point = np.array([1, 1])

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
        bias = self.direction - np.array([1, 1])
        return bias

    def next_global_xy(self, gx, gy):
        if self.reflected:
            return np.array([gx, gy])
        else:
            global_origin_xy = np.array([gx, gy])
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
            self.obstacle_point = np.array([cell_xy[0], cell_xy[1]])

        return self


class GleamStream(Keeper):
    """Represent trajectory of straight stream.

    Attributes
    ----------
    break_point: numpy.array
        End of stream.
    reflection: set
        Represent set of possible direction to reflect on 3x3 matrix.
    """

    def __init__(self):
        super(GleamStream, self).__init__()
        self.break_point = None
        self.reflection = None

    def set_break_point(self, row, col, reflection):
        """Set value of the end of body line.

        Parameters
        ----------
        row: int
            row index(count from 0)
        col: int
            column index(count from 0)
        reflection: set of int
            cell value on 3x3 matrix
        Returns
        -------
        GleamStream
            self object
        """
        self.break_point = np.array([row, col])
        self.reflection = reflection
        return self


class LeakStream(Keeper):
    # TODO make docstring
    def __init__(self):
        super(LeakStream, self).__init__()
        self.leaked_point = None

    def set_leaked_point(self, x, y):
        """bp is Gleam object"""
        self.leaked_point = np.array([x, y])
        return self


class Section(object):
    def __init__(self, row_section):
        self.table = self.outlines_segmentation(row_section)

    @staticmethod
    def outlines_segmentation(gleam_stream):
        # FIXME add doc string
        data = [{"value": el[1], "row": el[0][0], "col": el[0][1]} for el in gleam_stream]

        df = pd.DataFrame(data, columns=("value", "row", "col"))
        df["diff"] = round(df["value"].diff() ** 3, 2)

        df = df.dropna()
        # if not equival to 0 set as 1
        df.ix[df["diff"] != 0, "diff"] = 1

        df["group"] = (df["diff"].diff() != 0).cumsum()
        df = df.set_index(["group", "diff"])

        return df

    def get_all_section_parts(self):
        try:
            # get groups for further processing
            # res = sorted([el for el in df.groupby(["group", "diff"]).groups], key=itemgetter(0))
            res = sorted([el for el in self.table.groupby(by=[self.table.index.get_level_values(0),
                                                              self.table.index.get_level_values(1)]).groups],
                         key=itemgetter(0))

            # if start from transition part drop it
            if res[0][1] != 1.0:
                del res[0]

            # take transitions as a start of object and fill as a body of an object
            section_parts = list()

            for start_group_ind, end_group_ind in zip(res[::2], res[1::2]):
                start_group = self.table.ix[start_group_ind[0], :]
                end_group = self.table.ix[end_group_ind[0], :]

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
                section_parts.append(sample)

            return section_parts
        except IndexError:
            raise EmptySection

    def get_first_section(self, threshold):
        try:
            my_start_point = [el for el in self.get_all_section_parts() if el["mean_value"] >= threshold]
            return my_start_point[0]
        except IndexError:
            raise EmptySection


if __name__ == "__main__":
    ep1 = [
        [20, 0],
        [40, 10],
        [20, 20],
        [10, 30],
    ]
    ep2 = [
        [20, 30],
    ]
    # ep = [31, 31, 31, 30, 100, 100, 30, 30, 50]

    t = EnterPointPool()
    t.set_base_pool(ep1)

    t.show()
    print(t.extract_leak_points())
    t.extend_pool(ep2)
    t.show()
    t.validate_enter_points()
    t.show()
