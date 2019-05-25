''' Represent files as templates and recursively format templates according to some dictionary of values '''

HEADER_EXT = '.hpp'
SOURCE_EXT = '.cpp'

INTERNAL_ATTRS = {'fpath', 'includes', 'ext', 'template', '_repstr'}

class AGE_FILE:
    def __init__(self, fpath, ext, template):
        self.fpath = fpath + ext
        self.includes = []
        self.ext = ext
        self.template = template

    def __str__(self):
        if self._repstr is None:
            raise 'file is not processed'
        includes = '\n'.join(['#include {path}'.format(path = include)
            for include in self.includes])
        if len(self.includes) > 0:
            includes = includes + '\n\n'
        return includes + self._repstr

    def process(self, values):
        items = self.__dict__.items()
        fmt = {}
        for key, value in items:
            if key in INTERNAL_ATTRS:
                continue
            arg, func = value
            lookup = values
            akeys = arg.split('.')
            for akey in akeys[:-1]:
                if akey in lookup:
                    lookup = lookup[akey]
                else:
                    break
            if akeys[-1] in lookup:
                entry = lookup[akeys[-1]]
            else:
                entry = None
            fmt[key] = func(entry)
        self._repstr = self.template.format(**fmt)
        return self

def sortkey(dic):
    arr = list(dic.keys())
    arr.sort()
    return arr
