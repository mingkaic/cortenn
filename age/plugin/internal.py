import age.templates.api_tmpl as api
import age.templates.codes_tmpl as codes
import age.templates.data_tmpl as data
import age.templates.grader_tmpl as grader
import age.templates.opera_tmpl as opera

from gen.plugin_base2 import PluginBase
from gen.file_rep import FileRep

_plugin_id = 'INTERNAL'

class InternalPlugin:

    def plugin_id(self):
        return _plugin_id

    def process(self, generated_files, arguments):
        directory = {
            api.header.fpath:
                FileRep(api.header.process(arguments),
                user_includes=['"ade/ade.hpp"'],
                internal_refs=[]),
            api.source.fpath:
                FileRep(api.source.process(arguments),
                user_includes=[],
                internal_refs=[
                    codes.header.fpath,
                    api.header.fpath
                ]),
            codes.header.fpath:
                FileRep(codes.header.process(arguments),
                user_includes=['<string>'],
                internal_refs=[]),
            codes.source.fpath:
                FileRep(codes.source.process(arguments),
                user_includes=[
                    '<unordered_map>',
                    '"logs/logs.hpp"',
                ],
                internal_refs=[
                    codes.header.fpath,
                ]),
            data.header.fpath:
                FileRep(data.header.process(arguments),
                user_includes=[],
                internal_refs=[
                    codes.header.fpath,
                ]),
            grader.header.fpath:
                FileRep(grader.header.process(arguments),
                user_includes=['"ade/ade.hpp"'],
                internal_refs=[codes.header.fpath]),
            grader.source.fpath:
                FileRep(grader.source.process(arguments),
                user_includes=[],
                internal_refs=[
                    api.header.fpath,
                    grader.header.fpath,
                ]),
            opera.header.fpath:
                FileRep(opera.header.process(arguments),
                user_includes=['"ade/functor.hpp"'],
                internal_refs=[
                    codes.header.fpath,
                ]),
        }

        for filename in directory:
            dfile = directory[filename]

            if 'includes' in arguments and\
                filename in arguments['includes']:
                dfile.includes += arguments['includes'][filename]

        return directory

PluginBase.register(InternalPlugin)
