''' Internal plugin process '''

import os

import age.templates.api_tmpl as api
import age.templates.codes_tmpl as codes
import age.templates.grader_tmpl as grader
# import age.templates.opera_tmpl as opera
import custom_opera_tmpl as opera

def process(directory, relpath, fields):

    api_hdr_path = os.path.join(relpath, api.header.fpath)
    codes_hdr_path = os.path.join(relpath, codes.header.fpath)
    grader_hdr_path = os.path.join(relpath, grader.header.fpath)
    opera_hdr_path = os.path.join(relpath, opera.header.fpath)

    # manitory headers
    api.header.includes = [
        '"bwd/grader.hpp"'
    ]
    api.source.includes = [
        '"' + codes_hdr_path + '"',
        '"' + api_hdr_path + '"',
    ]

    codes.header.includes = [
        '<string>'
    ]
    codes.source.includes = [
        '<unordered_map>',
        '"logs/logs.hpp"',
        '"' + codes_hdr_path + '"',
    ]

    grader.header.includes = [
        '"bwd/grader.hpp"',
        '"' + codes_hdr_path + '"',
    ]
    grader.source.includes = [
        '"' + codes_hdr_path + '"',
        '"' + api_hdr_path + '"',
        '"' + grader_hdr_path + '"',
    ]

    opera.header.includes = [
        '"ade/functor.hpp"',
        '"' + codes_hdr_path + '"',
    ]

    directory = {
        'api_hpp': api.header,
        'api_src': api.source,
        'codes_hpp': codes.header,
        'codes_src': codes.source,
        'grader_hpp': grader.header,
        'grader_src': grader.source,
        'opera_hpp': opera.header,
    }

    for akey in directory:
        afile = directory[akey]
        if 'includes' in fields and afile.fpath in fields['includes']:
            afile.includes += fields['includes'][afile.fpath]
        afile.process(fields)

    return directory
