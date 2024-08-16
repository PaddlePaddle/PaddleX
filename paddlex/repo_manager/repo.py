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
import os.path as osp
import importlib
import tempfile
import shutil

from ..utils import logging
from ..utils.download import download_and_extract
from .meta import get_repo_meta, REPO_DOWNLOAD_BASE
from .utils import (install_packages_using_pip, fetch_repo_using_git,
                    reset_repo_using_git, uninstall_package_using_pip,
                    remove_repo_using_rm, check_installation_using_pip,
                    build_wheel_using_pip, mute, switch_working_dir,
                    to_dep_spec_pep508, env_marker_ast2expr)

__all__ = ['build_repo_instance', 'build_repo_group_installer']


def build_repo_instance(repo_name, *args, **kwargs):
    """ build_repo_instance """
    # XXX: Hard-code type
    repo_cls = PPRepository
    repo_instance = repo_cls(repo_name, *args, **kwargs)
    return repo_instance


def build_repo_group_installer(*repos):
    """ build_repo_group_installer """
    return RepositoryGroupInstaller(list(repos))


def build_repo_group_getter(*repos):
    """ build_repo_group_getter """
    return RepositoryGroupGetter(list(repos))


class PPRepository(object):
    """
    Installation, initialization, and PDX module import handler for a
    PaddlePaddle repository.
    """

    def __init__(self, name, repo_parent_dir, pdx_collection_mod):
        super().__init__()
        self.name = name
        self.repo_parent_dir = repo_parent_dir
        self.root_dir = osp.join(repo_parent_dir, self.name)

        self.meta = get_repo_meta(self.name)
        self.git_path = self.meta['git_path']
        self.pkg_name = self.meta['pkg_name']
        self.lib_name = self.meta['lib_name']
        self.pdx_mod_name = pdx_collection_mod.__name__ + '.' + self.meta[
            'pdx_pkg_name']
        self.main_req_file = self.meta.get('main_req_file', 'requirements.txt')

    def initialize(self):
        """ initialize """
        if not self.check_installation(quick_check=True):
            return False
        if 'path_env' in self.meta:
            # Set env var
            os.environ[self.meta['path_env']] = osp.abspath(self.root_dir)
        # NOTE: By calling `self.get_pdx()` we actually loads the repo PDX package
        # and do all registration.
        self.get_pdx()
        return True

    def check_installation(self, quick_check=False):
        """ check_installation """
        if quick_check:
            lib = self._get_lib(load=False)
            return lib is not None
        else:
            # TODO: Also check if correct dependencies are installed.
            return check_installation_using_pip(self.pkg_name)

    def check_repo_exiting(self, quick_check=False):
        """ check_repo_exiting """
        return os.path.exists(os.path.join(self.root_dir, '.git'))

    def install(self, *args, **kwargs):
        """ install """
        return RepositoryGroupInstaller([self]).install(*args, **kwargs)

    def uninstall(self, *args, **kwargs):
        """ uninstall """
        return RepositoryGroupInstaller([self]).uninstall(*args, **kwargs)

    def install_deps(self, *args, **kwargs):
        """ install_deps """
        return RepositoryGroupInstaller([self]).install_deps(*args, **kwargs)

    def install_package(self, no_deps=False, clean=True):
        """ install_package """
        editable = self.meta.get('editable', True)
        extra_editable = self.meta.get('extra_editable', None)
        if editable:
            logging.warning(
                f"{self.pkg_name} will be installed in editable mode.")
        with switch_working_dir(self.root_dir):
            try:
                install_packages_using_pip(
                    ['.'], editable=editable, no_deps=no_deps)
            finally:
                if clean:
                    # Clean build artifacts
                    tmp_build_dir = os.path.join(self.root_dir, 'build')
                    if os.path.exists(tmp_build_dir):
                        shutil.rmtree(tmp_build_dir)
        if extra_editable:
            with switch_working_dir(
                    os.path.join(self.root_dir, extra_editable)):
                try:
                    install_packages_using_pip(
                        ['.'], editable=True, no_deps=no_deps)
                finally:
                    if clean:
                        # Clean build artifacts
                        tmp_build_dir = os.path.join(self.root_dir, 'build')
                        if os.path.exists(tmp_build_dir):
                            shutil.rmtree(tmp_build_dir)

    def uninstall_package(self):
        """ uninstall_package """
        uninstall_package_using_pip(self.pkg_name)

    def download(self):
        """ download from remote """
        download_url = f'{REPO_DOWNLOAD_BASE}{self.name}.tar'
        os.makedirs(self.repo_parent_dir, exist_ok=True)
        download_and_extract(download_url, self.repo_parent_dir, self.name)
        # reset_repo_using_git('FETCH_HEAD')

    def remove(self):
        """ remove """
        with switch_working_dir(self.repo_parent_dir):
            remove_repo_using_rm(self.name)

    def update(self, platform=None):
        """ update """
        branch = self.meta.get('branch', None)
        git_url = f'https://{platform}{self.git_path}'
        with switch_working_dir(self.root_dir):
            try:
                fetch_repo_using_git(branch=branch, url=git_url)
                reset_repo_using_git('FETCH_HEAD')
            except Exception as e:
                logging.warning(
                    f"Update {self.name} from {git_url} failed, check your network connection. Error:\n{e}"
                )

    def wheel(self, dst_dir):
        """ wheel """
        with tempfile.TemporaryDirectory() as td:
            tmp_repo_dir = osp.join(td, self.name)
            tmp_dst_dir = osp.join(td, 'dist')
            shutil.copytree(self.root_dir, tmp_repo_dir, symlinks=False)

            # NOTE: Installation of the repo relies on `self.main_req_file` in root directory
            # Thus, we overwrite the content of it.
            main_req_file_path = osp.join(tmp_repo_dir, self.main_req_file)
            deps_str = self.get_deps()
            with open(main_req_file_path, 'w', encoding='utf-8') as f:
                f.write(deps_str)
            install_packages_using_pip([], req_files=[main_req_file_path])
            with switch_working_dir(tmp_repo_dir):
                build_wheel_using_pip('.', tmp_dst_dir)
            shutil.copytree(tmp_dst_dir, dst_dir)

    def _get_lib(self, load=True):
        """ _get_lib """
        import importlib.util
        importlib.invalidate_caches()
        if load:
            try:
                with mute():
                    return importlib.import_module(self.lib_name)
            except ImportError:
                return None
        else:
            spec = importlib.util.find_spec(self.lib_name)
            if spec is not None and not osp.exists(spec.origin):
                return None
            else:
                return spec

    def get_pdx(self):
        """ get_pdx """
        return importlib.import_module(self.pdx_mod_name)

    def get_deps(self):
        """ get_deps """
        # Merge requirement files
        req_list = [self.main_req_file]
        req_list.extend(self.meta.get('extra_req_files', []))
        deps = []
        for req in req_list:
            with open(osp.join(self.root_dir, req), 'r', encoding='utf-8') as f:
                deps.append(f.read())
        for dep in self.meta.get('pdx_pkg_deps', []):
            deps.append(dep)
        deps = '\n'.join(deps)
        return deps

    def get_version(self):
        """ get_version """
        version_file = osp.join(self.root_dir, '.pdx_gen.version')
        with open(version_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        sta_ver = lines[0].rstrip()
        commit = lines[1].rstrip()
        ret = [sta_ver, commit]
        # TODO: Get dynamic version in a subprocess.
        ret.append(None)
        return ret

    def __str__(self):
        return f"({self.name}, {id(self)})"


class RepositoryGroupInstaller(object):
    """ RepositoryGroupInstaller """

    def __init__(self, repos):
        super().__init__()
        self.repos = repos

    def install(self, force_reinstall=False, no_deps=False, constraints=None):
        """ install """
        # Rollback on failure is not yet supported. A failed installation
        # could leave a broken environment.
        if force_reinstall:
            self.uninstall()
        ins_flags = []
        repos = self._sort_repos(self.repos, check_missing=True)
        for repo in repos:
            if force_reinstall or not repo.check_installation():
                ins_flags.append(True)
            else:
                ins_flags.append(False)
        if not no_deps:
            # We collect the dependencies and install them all at once
            # such that we can make use of the pip resolver.
            self.install_deps(constraints=constraints)
        # XXX: For historical reasons the repo packages are sequentially
        # installed, and we have no failure rollbacks. Meanwhile, installation
        # failure of one repo package aborts the entire installation process.
        for ins_flag, repo in zip(ins_flags, repos):
            if ins_flag:
                repo.install_package(no_deps=True)

    def uninstall(self):
        """ uninstall """
        repos = self._sort_repos(self.repos, check_missing=False)
        repos = repos[::-1]
        for repo in repos:
            if repo.check_installation():
                # NOTE: Dependencies are not uninstalled.
                repo.uninstall_package()

    def get_deps(self):
        """ get_deps """
        deps_list = []
        repos = self._sort_repos(self.repos, check_missing=True)
        for repo in repos:
            deps = repo.get_deps()
            deps = self._normalize_deps(
                deps, headline=f"# {repo.name} dependencies")
            deps_list.append(deps)
        # Add an extra new line to separate dependencies of different repos.
        return '\n\n'.join(deps_list)

    def install_deps(self, constraints):
        """ install_deps """
        deps_str = self.get_deps()
        with tempfile.TemporaryDirectory() as td:
            req_file = os.path.join(td, 'requirements.txt')
            with open(req_file, 'w', encoding='utf-8') as fr:
                fr.write(deps_str)
            if constraints is not None:
                cons_file = os.path.join(td, 'constraints.txt')
                with open(cons_file, 'w', encoding='utf-8') as fc:
                    fc.write(constraints)
                cons_files = [cons_file]
            else:
                cons_files = []
            install_packages_using_pip(
                [], req_files=[req_file], cons_files=cons_files)

    def _sort_repos(self, repos, check_missing=False):
        # We sort the repos to ensure that the dependencies precede the
        # dependant in the list.
        name_meta_pairs = []
        for repo in repos:
            name_meta_pairs.append((repo.name, repo.meta))

        unique_pairs = []
        hashset = set()
        for name, meta in name_meta_pairs:
            if name in hashset:
                continue
            else:
                unique_pairs.append((name, meta))
                hashset.add(name)

        sorted_repos = []
        missing_names = []
        name2repo = {repo.name: repo for repo in repos}
        for name, meta in unique_pairs:
            if name in name2repo:
                repo = name2repo[name]
                sorted_repos.append(repo)
            else:
                missing_names.append(name)
        if check_missing and len(missing_names) > 0:
            be = 'is' if len(missing_names) == 1 else 'are'
            raise RuntimeError(
                f"{missing_names} {be} required in the installation.")
        else:
            assert len(sorted_repos) == len(self.repos)
        return sorted_repos

    def _normalize_deps(self, deps, headline=None):
        repo_pkgs = set(repo.pkg_name for repo in self.repos)
        normed_lines = []
        if headline is not None:
            normed_lines.append(headline)
        for line in deps.splitlines():
            line_s = line.strip()
            if len(line_s) == 0 or line_s.startswith('#'):
                continue
            # If `line` is not a comment, it must be a requirement specifier.
            # Other forms may cause a parse error.
            n, e, v, m = to_dep_spec_pep508(line_s)
            if isinstance(v, str):
                raise RuntimeError(
                    "Currently, URL based lookup is not supported.")
            if n in repo_pkgs:
                # Skip repo packages
                continue
            else:
                line_n = [n]
                fe = f"[{','.join(e)}]" if e else ''
                if fe:
                    line_n.append(fe)
                fv = []
                for tup in v:
                    fv.append(' '.join(tup))
                fv = ', '.join(fv) if fv else ''
                if fv:
                    line_n.append(fv)
                if m is not None:
                    fm = f"; {env_marker_ast2expr(m)}"
                    line_n.append(fm)
                line_n = ' '.join(line_n)
                normed_lines.append(line_n)

        return '\n'.join(normed_lines)


class RepositoryGroupGetter(object):
    """ RepositoryGroupGetter """

    def __init__(self, repos):
        super().__init__()
        self.repos = repos

    def get(self, force=False, platform=None):
        """ clone """
        if force:
            self.remove()
        for repo in self.repos:
            repo.download()
            repo.update(platform=platform)

    def remove(self):
        """ remove """
        for repo in self.repos:
            repo.remove()
