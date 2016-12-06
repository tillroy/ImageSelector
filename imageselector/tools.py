# coding: utf-8

"""
:Author: Roman Peresoliak

Module with high level tools for managing image structure
"""

import logging

from matrices import MatrixFollow, MatrixLeak, MatrixSection
from exceptions import (
    IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanWidth, IndexMoreThanHeight, OutOfContourBorder, LeakInMistake,
    EmptySection
)
from objects import RowCellSet, EnterPointPool


class Table(object):
    def __init__(self, matrix, border_contour):
        self.matrix = matrix
        self.table_border = border_contour
        border_contour.calc_bbox()
        self.table_bbox = border_contour.bbox

    def __get_table_border_conner(self):
        """get left top corner of general contour of table"""
        lt = self.table_border.get_conner("tl")
        logging.debug("""Left-top general table border point: {0}""".format(lt))
        return lt

    def __get_cell_contour_from(self, row, col):
        # FIXME docstring not in Numpy style.
        """Draw cell inner border.

        Draw and follow border that has been obtained after leaking through obstacle.

        :param row: int row index (count from 0)
        :param col: int column index (count from 0)

        :return Contour object
        :rtype matrices.Contour

        """
        logging.debug("""Start following border from: [{0} {1}]""".format(row, col))

        m = MatrixFollow(row, col).set_matrix(self.matrix)
        cell_contour = m.get_contour()

        return cell_contour

    def __leak_and_get_start_point_from_old(self, row, col, direction):
        """Leak through obstacle and return firs start point.

        Leak through obstacle or border and return firs start point for following border via
        MatrixFollow class.

        Parameters
        ----------
        row: int
            Row index(count from 0).
        col: int
            Column index(count from 0).
        direction: int
            Cell value on 3x3 matrix.

        Returns
        -------
        list of numpy.array
            Coordinates of following start point in row-col format.

        Raises
        ------
        OutOfContourBorder
            Value out of general border of table.

        """
        try:
            m = MatrixLeak(row, col).set_matrix(self.matrix)
            m.set_direction(direction)
            leak = m.leak()
        except LeakInMistake:
            mg = MatrixGleam(row, col).set_matrix(self.matrix)
            mg.set_direction(direction)
            res = mg.reflect(1)
            # only firs one is interesting
            res = res[0][0]
            print("!!!!!!!!!!", res)

            _row = res.break_point[0]
            _col = res.break_point[1]

            m = MatrixLeak(_row, _col).set_matrix(self.matrix)
            m.set_direction(direction)
            leak = m.leak()

        # m = MatrixLeak(row, col).set_matrix(self.matrix)
        # m.set_direction(direction)
        # leak = m.leak()
        if (self.table_bbox["rowmin"] < leak.leaked_point[0] < self.table_bbox["rowmax"] and
                self.table_bbox["colmin"] < leak.leaked_point[1] < self.table_bbox["colmax"]):

            logging.debug("""Leaking in direction({0})...\n\tLeaked to the point: {1}""".format(
                direction, leak.leaked_point
            ))

            return leak.leaked_point
        else:
            logging.warning(
                """Leaking in direction({0})...
                Leaked point is out of general table border: {0}, direction: {1}""".format(
                    direction, leak.leaked_point
                ))

            raise OutOfContourBorder

    def __leak_and_get_start_point_from(self, row, col, direction):
        """Leak through obstacle and return firs start point.

        Leak through obstacle or border and return firs start point for following border via
        MatrixFollow class.

        Parameters
        ----------
        row: int
            Row index(count from 0).
        col: int
            Column index(count from 0).
        direction: int
            Cell value on 3x3 matrix.

        Returns
        -------
        list of numpy.array
            Coordinates of following start point in row-col format.

        Raises
        ------
        OutOfContourBorder
            Value out of general border of table.

        """
        ms = MatrixSection(row, col).set_matrix(self.matrix)
        section = ms.get_section(direction, 10)
        res = section.get_first_section(.6)

        if (self.table_bbox.ix["min","row"] < res["start_point"][0] < self.table_bbox.ix["max", "row"] and
                self.table_bbox.ix["min", "col"] < res["start_point"][1] < self.table_bbox.ix["max", "col"]):

            logging.debug("""Leaking in direction({0})...\n\tLeaked to the point: {1}""".format(
                direction, res["start_point"]
            ))
            return res["start_point"]
        else:
            logging.warning(
                """Leaking in direction({0})...
                Leaked point is out of general table border: {0}, direction: {1}""".format(
                    direction, res["start_point"]
                ))

            raise OutOfContourBorder

    def __get_first_start_point(self):
        """get start point from general table border"""
        table_border_conner = self.__get_table_border_conner()
        first_start_point = self.__leak_and_get_start_point_from(table_border_conner[0], table_border_conner[1], 5)
        # stat, end coordinates of the row
        first_start_point = [first_start_point]

        logging.debug("""TABLE START POINT: {0}""".format(first_start_point))

        return first_start_point

    def get_row_cell_set(self, start_points=None):
        """get leaked start point from contours for the next drawing"""
        logging.debug("\n\nSTART NEW ROW CELL SET")
        # start_point = None
        if start_points is None:
            start_points = self.__get_first_start_point()
            logging.debug("Start line from: {0}".format(start_points))

        # TODO add somehow offset with neighbor bigger cell

        row_cell_set = RowCellSet()
        for sp in start_points:
            cell_start_point = sp
            while True:
                logging.debug("Follow from: {0}".format(sp))

                contour = self.__get_cell_contour_from(cell_start_point[0], cell_start_point[1])
                row_cell_set.add_contour(contour)
                leak_start_point = contour.get_conner("rt")
                try:
                    next_draw_start_point = self.__leak_and_get_start_point_from(
                        leak_start_point[0], leak_start_point[1], direction=5
                    )
                    cell_start_point = next_draw_start_point
                except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanWidth, IndexMoreThanHeight,
                        OutOfContourBorder, EmptySection):
                    break

        return row_cell_set

    def get_table_row_set_old2(self):
        start_points = None
        if start_points is None:
            start_points = self.__get_first_start_point()

        table_row_set = list()

        for el in range(1):
            row_cell_set = self.get_row_cell_set(start_points)
            next_leak_points = row_cell_set.get_next_leak_points()

            # START test
            ep_pool = EnterPointPool()
            if ep_pool.pool is None:
                ep_pool.set_base_pool(row_cell_set.get_all_enter_points())

            # ep_pool.remove_prev_enter_points(next_leak_points)
            ep_pool.extract_leak_points()
            # END test

            table_row_set.append(row_cell_set)
            _start_points = list()
            for leak_point in next_leak_points:
                try:
                    next_draw_start_point = self.__leak_and_get_start_point_from(
                        leak_point[0], leak_point[1], direction=5
                    )
                    _start_points.append(next_draw_start_point)
                except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanWidth, IndexMoreThanHeight):
                    break
            start_points = _start_points
        #
        return table_row_set

    def get_table_row_set(self):
        start_points = None
        if start_points is None:
            start_points = self.__get_first_start_point()

        table_row_set = list()

        for el in range(2):
            row_cell_set = self.get_row_cell_set(start_points)
            # next_leak_points = row_cell_set.get_next_leak_points()

            # START test
            ep_pool = EnterPointPool()
            if ep_pool.pool is None:
                ep_pool.set_base_pool(row_cell_set.get_all_enter_points())
            else:
                ep_pool.extend_pool(row_cell_set.get_all_enter_points())

            next_leak_points = ep_pool.extract_leak_points()
            # END test

            table_row_set.append(row_cell_set)
            _start_points = list()
            for leak_point in next_leak_points:
                try:
                    next_draw_start_point = self.__leak_and_get_start_point_from(
                        leak_point[0], leak_point[1], direction=5
                    )
                    _start_points.append(next_draw_start_point)
                except (IndexLessThanHeight, IndexLessThanWidth, IndexMoreThanWidth, IndexMoreThanHeight):
                    break
            start_points = _start_points
        #
        return table_row_set
