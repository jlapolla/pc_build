class Price:

    def __init__(self, amount, **kwargs):
        self._amount = amount
        self._date = kwargs.get('date') # Default to None
        self._url = kwargs.get('url') # Default to None

    def get_amount(self):
        return self._amount

    def get_date(self):
        return self._date

    def get_url(self):
        """URL string where price is quoted.
        """
        return self._url


class CpuScore:

    def __init__(self, **kwargs):
        self._mt_score = kwargs['mt_score']
        self._st_score = kwargs['st_score']
        self._date = kwargs.get('date')

    def get_mt_score(self):
        return self._mt_score

    def get_st_score(self):
        return self._st_score

    def get_date(self):
        return self._date


class CPU:

    def __init__(self, **kwargs):
        self._name = kwargs['name']
        self._score = kwargs['score']
        self._price = kwargs.get('price')

    def get_name(self):
        return self._name

    def get_price(self):
        return self._price

    def get_score(self):
        return self._score


class GpuScore:

    def __init__(self, **kwargs):
        self._g3d_score = kwargs['g3d_score']
        self._date = kwargs.get('date')

    def get_g3d_score(self):
        return self._g3d_score

    def get_date(self):
        return self._date


class GPU:

    def __init__(self, **kwargs):
        self._name = kwargs['name']
        self._price = kwargs['price']
        self._score = kwargs['score']

    def get_name(self):
        return self._name

    def get_price(self):
        return self._price

    def get_score(self):
        return self._score


class DataSource:

    def __init__(self, **kwargs):
        self._url = kwargs['url']
        self._pub_date = kwargs['pub_date']

    def get_url(self):
        return self._url

    def get_pub_date(self):
        """Publication date.
        """
        return self._pub_date


class Resolution:

    def __init__(self, **kwargs):
        self._width = kwargs['width']
        self._height = kwargs['height']

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height


class BadQualityError(Exception):

    def __init__(self, *args, **kwargs):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
        super(BadQualityError, self).__init__(self, *args, **kwargs)


class Quality:

    _NAME_MAP = {
        1: 'low',
        2: 'medium',
        3: 'high',
        4: 'very high',
        5: 'ultra',
        }

    def __init__(self, **kwargs):
        level = kwargs['level']
        if level not in self._NAME_MAP:
            raise BadQualityError('invalid quality level {0}'.format(level))
        self._level = level

    def get_level(self):
        """Quality level.
        """
        # FIXME: Remove this accessor. A quality does not have a level; a
        # quality IS the level. It should behave like an integer, be
        # comparable, et cetera.
        return self._level

    def __str__(self):
        return self._NAME_MAP[self._level]

    def __int__(self):
        return self._level

    def __lt__(self, other):
        return self._level < other._level

    def __le__(self, other):
        return self._level <= other._level

    def __eq__(self, other):
        return self.__class__ is other.__class__ and self._level == other._level

    def __ne__(self, other):
        return not self == other

    def __gt__(self, other):
        return self._level > other._level

    def __ge__(self, other):
        return self._level >= other._level

    def __hash__(self):
        return hash(self._level)


class Application:

    def __init__(self, **kwargs):
        self._name = kwargs['name']
        self._quality = kwargs['quality']
        self._resolution = kwargs['resolution']

    def get_name(self):
        return self._name

    def get_quality(self):
        return self._quality

    def get_resolution(self):
        return self._resolution


class FpsStudy:

    class DataPoint:

        def __init__(self, **kwargs):
            self._cpu = kwargs['cpu']
            self._gpu = kwargs['gpu']
            self._low_fps = kwargs['low_fps']
            self._avg_fps = kwargs['avg_fps']

        def get_cpu(self):
            return self._cpu

        def get_gpu(self):
            return self._gpu

        def get_low_fps(self):
            return self._low_fps

        def get_avg_fps(self):
            """Average FPS.
            """
            return self._avg_fps

    def __init__(self, **kwargs):
        self._source = kwargs['source']
        self._application = kwargs['application']
        self._data = list(kwargs['data'])
