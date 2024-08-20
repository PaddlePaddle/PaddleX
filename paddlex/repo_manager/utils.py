# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import json
import subprocess
import contextlib

from parsley import makeGrammar


def _check_call(*args, **kwargs):
    return subprocess.check_call(*args, **kwargs)


def _check_output(*args, **kwargs):
    return subprocess.check_output(*args, **kwargs)


def check_installation_using_pip(pkg):
    """ check_installation_using_pip """
    out = _check_output(['pip', 'list', '--format', 'json'])
    out = out.rstrip()
    lst = json.loads(out)
    return any(ele['name'] == pkg for ele in lst)


def uninstall_package_using_pip(pkg):
    """ uninstall_package_using_pip """
    return _check_call([sys.executable, '-m', 'pip', 'uninstall', '-y', pkg])


def install_packages_using_pip(pkgs,
                               editable=False,
                               req_files=None,
                               cons_files=None,
                               no_deps=False,
                               pip_flags=None):
    """ install_packages_using_pip """
    args = [sys.executable, '-m', 'pip', 'install']
    if editable:
        args.append('-e')
    if req_files is not None:
        for req_file in req_files:
            args.append('-r')
            args.append(req_file)
    if cons_files is not None:
        for cons_file in cons_files:
            args.append('-c')
            args.append(cons_file)
    if isinstance(pkgs, str):
        pkgs = [pkgs]
    args.extend(pkgs)
    if pip_flags is not None:
        args.extend(pip_flags)
    return _check_call(args)


def install_deps_using_pip():
    """ install requirements """
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    deps_path = os.path.join(current_file_path, 'requirements.txt')
    args = [sys.executable, '-m', 'pip', 'install', '-r', deps_path]
    return _check_call(args)


def clone_repo_using_git(url, branch=None):
    """ clone_repo_using_git """
    args = ['git', 'clone', '--depth', '1']
    if isinstance(url, str):
        url = [url]
    args.extend(url)
    if branch is not None:
        args.extend(['-b', branch])
    return _check_call(args)


def fetch_repo_using_git(branch, url, depth=1):
    """ fetch_repo_using_git """
    args = ['git', 'fetch', url, branch, '--depth', str(depth)]
    _check_call(args)


def reset_repo_using_git(pointer, hard=True):
    """ reset_repo_using_git """
    args = ['git', 'reset', '--hard', pointer]
    return _check_call(args)


def remove_repo_using_rm(name):
    """ remove_repo_using_rm """
    return _check_call(['rm', '-rf', name])


def build_wheel_using_pip(pkg, dst_dir='./', with_deps=False, pip_flags=None):
    """ build_wheel_using_pip """
    args = [sys.executable, '-m', 'pip', 'wheel', '--wheel-dir', dst_dir]
    if not with_deps:
        args.append('--no-deps')
    if pip_flags is not None:
        args.extend(pip_flags)
    args.append(pkg)

    return _check_call(args)


@contextlib.contextmanager
def mute():
    """ mute """
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            yield


@contextlib.contextmanager
def switch_working_dir(new_wd):
    """ switch_working_dir """
    cwd = os.getcwd()
    os.chdir(new_wd)
    try:
        yield
    finally:
        os.chdir(cwd)


def _build_dep_spec_pep508_grammar():
    # Refer to https://peps.python.org/pep-0508/
    grammar = """
        wsp           = ' ' | '\t'
        version_cmp   = wsp* <'<=' | '<' | '!=' | '==' | '>=' | '>' | '~=' | '==='>
        version       = wsp* <(letterOrDigit | '-' | '_' | '.' | '*' | '+' | '!')+>
        version_one   = version_cmp:op version:v wsp* -> (op, v)
        version_many  = version_one:v1 (wsp* ',' version_one)*:v2 -> [v1] + v2
        versionspec   = ('(' version_many:v ')' ->v) | version_many
        urlspec       = '@' wsp* <uri_reference>
        marker_op     = version_cmp | (wsp* 'in') | (wsp* 'not' wsp+ 'in')
        python_str_c  = (wsp | letter | digit | '(' | ')' | '.' | '{' | '}' |
                        '-' | '_' | '*' | '#' | ':' | ';' | ',' | '/' | '?' |
                        '[' | ']' | '!' | '~' | '`' | '@' | '$' | '%' | '^' |
                        '&' | '=' | '+' | '|' | '<' | '>' )
        dquote        = '"'
        squote        = '\\''
        comment       = '#' <anything*>:s end -> s
        python_str    = (squote <(python_str_c | dquote)*>:s squote |
                        dquote <(python_str_c | squote)*>:s dquote) -> s
        env_var       = ('python_version' | 'python_full_version' |
                        'os_name' | 'sys_platform' | 'platform_release' |
                        'platform_system' | 'platform_version' |
                        'platform_machine' | 'platform_python_implementation' |
                        'implementation_name' | 'implementation_version' |
                        'extra' # ONLY when defined by a containing layer
                        )
        marker_var    = wsp* (env_var | python_str)
        marker_expr   = marker_var:l marker_op:o marker_var:r -> (o, l, r)
                    | wsp* '(' marker:m wsp* ')' -> m
        marker_and    = marker_expr:l wsp* 'and' marker_expr:r -> ('and', l, r)
                    | marker_expr:m -> m
        marker_or     = marker_and:l wsp* 'or' marker_and:r -> ('or', l, r)
                        | marker_and:m -> m
        marker        = marker_or
        quoted_marker = ';' wsp* marker
        identifier_end = letterOrDigit | (('-' | '_' | '.' )* letterOrDigit)
        identifier    = <letterOrDigit identifier_end* >
        name          = identifier
        extras_list   = identifier:i (wsp* ',' wsp* identifier)*:ids -> [i] + ids
        extras        = '[' wsp* extras_list?:e wsp* ']' -> e
        name_req      = (name:n wsp* extras?:e wsp* versionspec?:v wsp* quoted_marker?:m
                        -> (n, e or [], v or [], m))
        url_req       = (name:n wsp* extras?:e wsp* urlspec:v (wsp+ | end) quoted_marker?:m
                        -> (n, e or [], v, m))
        specification = wsp* (url_req | name_req):s wsp* comment? -> s
        # The result is a tuple - name, list-of-extras,
        # list-of-version-constraints-or-a-url, marker-ast or None


        uri_reference = <uri | relative_ref>
        uri           = scheme ':' hier_part ('?' query )? ('#' fragment)?
        hier_part     = ('//' authority path_abempty) | path_absolute | path_rootless | path_empty
        absolute_uri  = scheme ':' hier_part ('?' query )?
        relative_ref  = relative_part ('?' query )? ('#' fragment )?
        relative_part = '//' authority path_abempty | path_absolute | path_noscheme | path_empty
        scheme        = letter (letter | digit | '+' | '-' | '.')*
        authority     = (userinfo '@' )? host (':' port )?
        userinfo      = (unreserved | pct_encoded | sub_delims | ':')*
        host          = ip_literal | ipv4_address | reg_name
        port          = digit*
        ip_literal    = '[' (ipv6_address | ipvfuture) ']'
        ipvfuture     = 'v' hexdig+ '.' (unreserved | sub_delims | ':')+
        ipv6_address   = (
                        (h16 ':'){6} ls32
                        | '::' (h16 ':'){5} ls32
                        | (h16 )?  '::' (h16 ':'){4} ls32
                        | ((h16 ':')? h16 )? '::' (h16 ':'){3} ls32
                        | ((h16 ':'){0,2} h16 )? '::' (h16 ':'){2} ls32
                        | ((h16 ':'){0,3} h16 )? '::' h16 ':' ls32
                        | ((h16 ':'){0,4} h16 )? '::' ls32
                        | ((h16 ':'){0,5} h16 )? '::' h16
                        | ((h16 ':'){0,6} h16 )? '::' )
        h16           = hexdig{1,4}
        ls32          = (h16 ':' h16) | ipv4_address
        ipv4_address   = dec_octet '.' dec_octet '.' dec_octet '.' dec_octet
        nz            = ~'0' digit
        dec_octet     = (
                        digit # 0-9
                        | nz digit # 10-99
                        | '1' digit{2} # 100-199
                        | '2' ('0' | '1' | '2' | '3' | '4') digit # 200-249
                        | '25' ('0' | '1' | '2' | '3' | '4' | '5') )# %250-255
        reg_name = (unreserved | pct_encoded | sub_delims)*
        path = (
                path_abempty # begins with '/' or is empty
                | path_absolute # begins with '/' but not '//'
                | path_noscheme # begins with a non-colon segment
                | path_rootless # begins with a segment
                | path_empty ) # zero characters
        path_abempty  = ('/' segment)*
        path_absolute = '/' (segment_nz ('/' segment)* )?
        path_noscheme = segment_nz_nc ('/' segment)*
        path_rootless = segment_nz ('/' segment)*
        path_empty    = pchar{0}
        segment       = pchar*
        segment_nz    = pchar+
        segment_nz_nc = (unreserved | pct_encoded | sub_delims | '@')+
                        # non-zero-length segment without any colon ':'
        pchar         = unreserved | pct_encoded | sub_delims | ':' | '@'
        query         = (pchar | '/' | '?')*
        fragment      = (pchar | '/' | '?')*
        pct_encoded   = '%' hexdig
        unreserved    = letter | digit | '-' | '.' | '_' | '~'
        reserved      = gen_delims | sub_delims
        gen_delims    = ':' | '/' | '?' | '#' | '(' | ')?' | '@'
        sub_delims    = '!' | '$' | '&' | '\\'' | '(' | ')' | '*' | '+' | ',' | ';' | '='
        hexdig        = digit | 'a' | 'A' | 'b' | 'B' | 'c' | 'C' | 'd' | 'D' | 'e' | 'E' | 'f' | 'F'
    """

    compiled = makeGrammar(grammar, {})
    return compiled


_pep508_grammar = None


def to_dep_spec_pep508(s):
    """ to_dep_spec_pep508 """
    global _pep508_grammar
    if _pep508_grammar is None:
        _pep508_grammar = _build_dep_spec_pep508_grammar()
    parsed = _pep508_grammar(s)
    return parsed.specification()


def env_marker_ast2expr(marker_ast):
    """ env_marker_ast2expr """
    MARKER_VARS = (
        'python_version',
        'python_full_version',
        'os_name',
        'sys_platform',
        'platform_release',
        'platform_system',
        'platform_version',
        'platform_machine',
        'platform_python_implementation',
        'implementation_name',
        'implementation_version',
        'extra'  # ONLY when defined by a containing layer
    )
    o, l, r = marker_ast
    if isinstance(l, tuple):
        l = env_marker_ast2expr(l)
    else:
        assert isinstance(l, str)
        if l not in MARKER_VARS:
            l = repr(l)
    if isinstance(r, tuple):
        r = env_marker_ast2expr(r)
    else:
        assert isinstance(r, str)
        if r not in MARKER_VARS:
            r = repr(r)
    return f"{l} {o} {r}"
