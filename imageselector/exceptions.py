# coding: utf-8


class FrameLoop(Exception):
    """too many loops in frame analysis"""
    pass


class ContourLoop(Exception):
    """mistake in countering"""
    pass


class OutOfContourBorder(Exception):
    """leaked through table border"""
    pass


class NoFollowDirections(Exception):
    """directions set is empty"""
    pass


class IndexMoreThanWidth(Exception):
    """Indices bigger tha matrix size"""
    pass


class IndexLessThanWidth(Exception):
    """Indices less tha matrix size"""
    pass


class IndexMoreThanHeight(Exception):
    """Indices bigger tha matrix size"""
    pass


class IndexLessThanHeight(Exception):
    """Indices less tha matrix size"""
    pass


class UnreflectedGleam(Exception):
    """there is now any reflection in obstacle"""
    pass


class LeakInMistake(Exception):
    """difference between centroid and follow values is less then sensitivity"""
    pass


class LeakedThrough(Exception):
    """leaked trougth object"""
    pass

class EmptySection(Exception):
    pass



