import logging
from operator import itemgetter
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from numpy import array

from matrices import MatrixFollow, MatrixSection, MatrixLeak, IndexMoreThanHeight, IndexMoreThanWidth, IndexLessThanWidth, IndexLessThanHeight
from tools import Table


class Img(object):
    def __init__(self, path):
        self.img = Image.open(path)
        self.draw = ImageDraw.Draw(self.img)

    def make_point(self, row, col, r=0, color=(255, 0, 0)):
        """

        :param row: Y
        :param col: X
        :param r:
        :param color:
        :return:
        """
        self.draw.ellipse((col - r, row - r, col + r, row + r), fill=color)
        # self.draw.ellipse((x , y , x , y ), fill=color)

    def get_matrix(self):
        matrix = np.array(self.img)
        return matrix

    def simplify_matrix(self):

        matrix = np.array(self.img)
        height = matrix.shape[0]
        width = matrix.shape[1]

        image = self.img.resize((width // 3,  height // 3), Image.ANTIALIAS)
        # image = self.img.resize((width // 3,  height // 3))
        print(self.img.size)
        print(image.size)

        image.save("img/ans2.jpg", "png")

    def show(self):
        self.img.show()

    def save(self):
        self.img.save("img/ans.png")

    def contour_from(self, row, col):
        m = MatrixFollow(row, col).set_matrix(self.get_matrix())
        contour = m.get_contour()
        contour.calc_structure()

        for el in contour.body:
            img.make_point(el[0], el[1])

        # print(contour.get_conner("lt"))
        # img.make_point(*contour.get_conner("lt"), color=(255, 0, 255))
        # img.make_point(*contour.get_conner("rt"), color=(0, 255, 255))
        # img.make_point(*contour.get_conner("rb"), color=(255, 255, 0))
        # img.make_point(*contour.get_conner("lb"), color=(100, 100, 255))

        return contour

    def rain(self, step, repeat, direction=4, mode="horizontal"):
        origin = array([1, 1])
        if mode == "horizontal":
            bias = array([0, step])
        elif mode == "vertical":
            bias = array([step, 0])
        else:
            bias = array([0, step])
        for n in range(repeat):
            mg = MatrixGleam(*origin).set_matrix(self.get_matrix())

            origin = origin + bias

            mg.set_direction(direction)
            res = mg.reflect(1)
            colors = ((23, 95, 243), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 255, 255), (255, 0, 255),(0, 255, 255), (255, 0, 255),(0, 255, 255), (255, 0, 255),(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (0, 255, 255), (255, 0, 255),(0, 255, 255), (255, 0, 255),(0, 255, 255), (255, 0, 255),)
            for num, levels in enumerate(res):
                color = colors[num]
                for el in levels:
                    print(el.reflection, el.break_point, el.body)
                    for p in el.body:
                        img.make_point(p[0], p[1], color=color)

                    img.make_point(el.break_point[0], el.break_point[1], color=(255, 0, 0), r=0)

    def gleam_from(self, row, col, direction):
        mg = MatrixGleam(row, col).set_matrix(self.get_matrix())
        mg.set_direction(direction)
        res = mg.reflect(1)

        for num, levels in enumerate(res):
            color = (0, 255, 255)
            for el in levels:
                print(el.reflection, el.break_point, el.body)
                for p in el.body:
                    img.make_point(p[0], p[1], color=color)

                img.make_point(el.break_point[0], el.break_point[1], color=(255, 0, 0), r=0)
        print("res", res)

    def leak_from(self, x, y, direction):
        m = MatrixLeak(x, y).set_matrix(self.get_matrix())
        m.set_direction(direction)
        leak_stream = m.leak()
        print(leak_stream.body)
        print(leak_stream.leaked_point)
        for el in leak_stream.body:
            img.make_point(el[0], el[1])

        img.make_point(leak_stream.leaked_point[0], leak_stream.leaked_point[1], color=(0, 0, 255))

    def test(self):
        table_contour = self.contour_from(9, 41)
        table = Table(self.get_matrix(),  table_contour)
        # res = table.get_row_cell_set(start_point=(65, 18))
        res = table.get_row_cell_set()

        # print("sda", res.is_my_point(100, 100))
        print(len(res.body))
        for con in res.body:
            for el in con.body:
                img.make_point(el[0], el[1], color=(0, 0, 255))

    def test2(self):
        # table_contour = self.contour_from(9, 41)
        table_contour = self.contour_from(8, 8)
        table = Table(self.get_matrix(), table_contour)
        res = table.get_table_row_set()
        # print(res)
        for ind, row_set in enumerate(res):
            colors = ((23, 95, 243), (0, 255, 0), (0, 0, 255), (0, 255, 255),(23, 95, 243), (0, 255, 0), (0, 0, 255),
                      (0, 255, 255), (23, 95, 243), (0, 255, 0), (0, 0, 255), (0, 255, 255), (23, 95, 243), (0, 255, 0),
                      (0, 255, 255), (23, 95, 243), (0, 255, 0), (0, 0, 255), (0, 255, 255), (23, 95, 243), (0, 255, 0),
                      (0, 255, 255), (23, 95, 243), (0, 255, 0), (0, 0, 255), (0, 255, 255), (23, 95, 243), (0, 255, 0),
                      (0, 255, 255), (23, 95, 243), (0, 255, 0), (0, 0, 255), (0, 255, 255), (23, 95, 243), (0, 255, 0),
                      (0, 0, 255), (0, 255, 255),)
            for cell in row_set.body:
                for el in cell.body:
                    img.make_point(el[0], el[1], color=colors[ind])

        # print(res)
        # for row in res:
        #     for cell in row:
        #         for el in cell.body:
        #             img.make_point(el[0], el[1], color=(0, 0, 255))

    def get_section_old(self, row, col, direction):
        import matplotlib.pyplot as plt
        ms = MatrixSection(row, col).set_matrix(self.get_matrix())
        ms.set_direction(direction)
        section = ms.get_section(direction)

        data = [{"value": el[1], "row": el[0][0], "col": el[0][1]} for el in section]

        df = pd.DataFrame(data, columns=("value", "row", "col"))
        df["diff"] = round(df["value"].diff() ** 3, 2)

        df = df.dropna()
        # if not equival to 0 set as 1
        df.ix[df["diff"] != 0, "diff"] = 1
        print("sdsdad", df)

        df["group"] = (df["diff"].diff() != 0).cumsum()

        # get groups for further processing
        res = sorted([el for el in df.groupby(["group", "diff"]).groups], key=itemgetter(0))
        if res[0][1] != 1.0:
            del res[0]

        # print(res)

        df = df.set_index(["group", "diff"])
        # take transitions as a start of object and fill as a body of an object
        ind_counter = 0

        for start_group_ind, end_group_ind in zip(res[::2], res[1::2]):
            color = (0, 0, 200) if ind_counter % 2 == 0 else (0, 200, 0)
            # print(color, ind_counter)

            start_group = df.ix[start_group_ind[0], :]
            end_group = df.ix[end_group_ind[0], :]

            start_point = start_group.tail(1)
            end_point = end_group.tail(1)

            sample = start_point.append(end_group)
            print(sample)
            for el in sample.values:
                img.make_point(el[1], el[2], color=color)

            ind_counter += 1
            # img.make_point(start_point["row"], start_point["col"], color=rand)
            # img.make_point(end_point["row"], end_point["col"], color=rand)

        # print(df)

        # for el in df.values:
        #     # print(el)
        #     cell_class = el[1]
        #     if cell_class == 1:
        #         img.make_point(el[2], el[3], color=(0, 0, 255))
        #     else:
        #         img.make_point(el[2], el[3], color=(255, 0, 0))

        # for el in df[df["group"] == 5].values:
        #     img.make_point(el[2], el[3], color=(0, 255, 0, 50))
        #
        # plt.plot(range(len(df[df["diff"] != 0]["group"])), df[df["diff"] != 0]["group"], ".")
        # plt.plot(range(len(df[df["diff"] == 0]["group"])), df[df["diff"] == 0]["group"], ".")
        # plt.show()

    def get_section(self, row, col, direction, depth):
        ms = MatrixSection(row, col).set_matrix(self.get_matrix())
        section = ms.get_section(direction, depth)
        print(section.get_first_section(.8))
        # my_start_point = [el for el in section if el["mean_value"] >= .8]
        # my_start_point = my_start_point[0]
        # print("start_point", my_start_point)
        # print(section_df)
        # print(section)
        #
        # for el in section[0]["body"]:
        #     img.make_point(el[0], el[1], color=(0, 0, 200))

    def make_frame_sample(self, row, col, size=3):
        """get frame from hole matrix in general coordinate system"""
        half = size // 2
        not_included = 1
        xmin = row - half
        ymin = col - half
        xmax = row + half + not_included
        ymax = col + half + not_included

        # print("""[{0}:{1},{2}:{3}]""".format(xmin, xmax, ymin, ymax))
        _matrix = self.get_matrix()
        height = _matrix.shape[0] - 1
        width = _matrix.shape[1] - 1
        if row >= height:
            raise IndexMoreThanHeight
        if row <= 0:
            raise IndexLessThanHeight

        if col >= width:
            raise IndexMoreThanWidth
        if col <= 0:
            raise IndexLessThanWidth

        frame = _matrix[xmin:xmax, ymin:ymax]
        pillow_image = Image.fromarray(frame, 'RGB')
        pillow_image.save("img/problem_{0}_{1}.png".format(row, col),)
        # return frame


if __name__ == "__main__":
    logging.basicConfig(
        filename='imageselector.log',
        level=logging.DEBUG,
        format='%(asctime)s | %(filename)s:%(lineno)s | %(message)s'
    )

    # path = "img/sample.jpg"
    path = "img/sample1_1.jpg"
    # path = "img/problem5.jpg"
    img = Img(path)
    # img.simplify_matrix()

    # img.get_section(62, 181, direction=5, depth=10)


    # img.make_frame_sample(27, 79)
    #
    # 14, 43
    # img.contour_from(9, 41)
    # img.get_section(10, 13, direction=5, depth=10)
    # img.contour_from(12, 15)

    # img.rain(100, 1, 5, "horizontal")
    # img.gleam_from(21, 115, 5)
    # img.gleam_from(41, 105, 5)
    # img.leak_from(21, 115, direction=5)

    img.test2()

    # problem
    # img.make_point(102, 488, r=1)
    # problem_cell = img.contour_from(102, 488)
    # problem_cell.table.to_csv("res.csv", index=False)
    #
    # print(problem_cell.bbox)

    # img.make_point(21, 115)
    # img.make_point(38, 332)
    # img.make_point(32, 621, color=(0,200,0))
    # img.make_point(31, 620)
    # img.make_point(32, 621, color=(0,100,100))

    # img.make_point(8, 115)
    # img.make_point(7, 331)
    # img.make_point(7, 364)
    # img.make_point(7, 487)

    # img.make_point(31, 620)
    # img.make_point(32, 621, color=(0,0,255))
    # img.make_point(12, 43)



    # img.make_point(43, 79, color=(0, 0, 255))

    # for el in res:
    #     img.make_point(el[0], el[1])

    # img.contour_from(65, 18)
    # img.make_point(12, 117)
    # img.make_point(12, 79, color=(0,0,255))
    img.save()





