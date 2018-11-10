import pytz


import csv
import datetime
import io
import sys


def input_dt(time_string, fmt, tz):
    if time_string is None:
        return None
    naive_dt = datetime.datetime.strptime(time_string, fmt) # Parse naive_dt
    aware_dt = tz.localize(naive_dt) # Assume naive_dt is in timezone tz
    return aware_dt.astimezone(pytz.utc) # Convert to utc timezone


def output_dt(aware_dt, fmt, tz):
    if aware_dt is None:
        return None
    out_dt = aware_dt.astimezone(tz)
    normal_dt = tz.normalize(out_dt)
    return normal_dt.strftime(fmt)


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

    def as_dict(self):
        d = {}
        d['amount'] = self._amount
        if self._date is not None:
            d['date'] = self._date
        if self._url is not None:
            d['url'] = self._url
        return d


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

    def as_dict(self):
        d = {}
        d['mt_score'] = self._mt_score
        d['st_score'] = self._st_score
        if self._date is not None:
            d['date'] = self._date
        return d


class Cpu:

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

    def as_dict(self):
        d = {}
        d['name'] = self._name
        d['score'] = self._score.as_dict()
        if self._price is not None:
            d['price'] = self._price.as_dict()
        return d


class GpuScore:

    def __init__(self, **kwargs):
        self._g3d_score = kwargs['g3d_score']
        self._date = kwargs.get('date')

    def get_g3d_score(self):
        return self._g3d_score

    def get_date(self):
        return self._date

    def as_dict(self):
        d = {}
        d['g3d_score'] = self._g3d_score
        if self._date is not None:
            d['date'] = self._date
        return d


class Gpu:

    def __init__(self, **kwargs):
        self._name = kwargs['name']
        self._score = kwargs['score']
        self._price = kwargs.get('price')

    def get_name(self):
        return self._name

    def get_score(self):
        return self._score

    def get_price(self):
        return self._price

    def as_dict(self):
        d = {}
        d['name'] = self._name
        d['score'] = self._score.as_dict()
        if self._price is not None:
            d['price'] = self._price.as_dict()
        return d


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

    def __iter__(self):
        return iter(self._data)

    def get_source(self):
        return self._source

    def get_application(self):
        return self._application


class CpuCsvReader:

    _FLD_PRICE_AMOUNT = 'price'
    _FLD_PRICE_DATE = 'price_date'
    _FLD_PRICE_URL = 'price_url'

    _FLD_SCORE_MT = 'mt_score'
    _FLD_SCORE_ST = 'st_score'
    _FLD_SCORE_DATE = 'score_date'

    _FLD_NAME = 'name'

    def __init__(self, infile):
        self._reader = csv.DictReader(infile)

    def __iter__(self):
        return self

    def __next__(self):
        d = next(self._reader)

        price = None
        if self._get(d, self._FLD_PRICE_AMOUNT) is not None:
            price = Price(
                self._filter(d[self._FLD_PRICE_AMOUNT]),
                date=input_dt(self._get(d, self._FLD_PRICE_DATE), '%m/%d/%y', pytz.utc),
                url=self._get(d, self._FLD_PRICE_URL),
                )

        score = CpuScore(
            mt_score=self._filter(d[self._FLD_SCORE_MT]),
            st_score=self._filter(d[self._FLD_SCORE_ST]),
            date=input_dt(self._get(d, self._FLD_SCORE_DATE), '%m/%d/%y', pytz.utc),
            )

        return Cpu(
            name=self._filter(d[self._FLD_NAME]),
            score=score,
            price=price,
            )

    def _get(self, d, key):
        value = d.get(key)
        return self._filter(value)

    def _filter(self, value):
        if value == '':
            return None
        return value

    @classmethod
    def dump_file(cls, csv_filename):
        """Read and output all Cpu's in csv_filename.

        For manual testing purposes.
        """
        with io.open(csv_filename) as infile:
            for cpu in cls(infile):
                sys.stdout.write(repr(cpu.as_dict()))
                sys.stdout.write('\n')


class GpuCsvReader:

    _FLD_PRICE_AMOUNT = 'price'
    _FLD_PRICE_DATE = 'price_date'
    _FLD_PRICE_URL = 'price_url'

    _FLD_G3D_SCORE = 'g3d_score'
    _FLD_SCORE_DATE = 'score_date'

    _FLD_NAME = 'name'

    def __init__(self, infile):
        self._reader = csv.DictReader(infile)

    def __iter__(self):
        return self

    def __next__(self):
        d = next(self._reader)

        price = None
        if self._get(d, self._FLD_PRICE_AMOUNT) is not None:
            price = Price(
                self._filter(d[self._FLD_PRICE_AMOUNT]),
                date=input_dt(self._get(d, self._FLD_PRICE_DATE), '%m/%d/%y', pytz.utc),
                url=self._get(d, self._FLD_PRICE_URL),
                )

        score = GpuScore(
            g3d_score=self._filter(d[self._FLD_G3D_SCORE]),
            date=input_dt(self._get(d, self._FLD_SCORE_DATE), '%m/%d/%y', pytz.utc),
            )

        return Gpu(
            name=self._filter(d[self._FLD_NAME]),
            score=score,
            price=price,
            )

    def _get(self, d, key):
        value = d.get(key)
        return self._filter(value)

    def _filter(self, value):
        if value == '':
            return None
        return value

    @classmethod
    def dump_file(cls, csv_filename):
        """Read and output all Gpu's in csv_filename.

        For manual testing purposes.
        """
        with io.open(csv_filename) as infile:
            for gpu in cls(infile):
                sys.stdout.write(repr(gpu.as_dict()))
                sys.stdout.write('\n')
