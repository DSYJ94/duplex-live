"""Opt-in paired Unicode quotation removal for an intelligibility control.

Only balanced curly double quotes are removed; letters, digits, apostrophes,
straight inch marks and unmatched quotation characters remain unchanged.
"""
import re


def for_speech(text,policy='verbatim'):
    if policy=='verbatim':return text
    if policy!='unquote-paired':raise ValueError('Unknown speech surface policy')
    return re.sub('“([^“”]*)”',lambda match:match.group(1),text)
