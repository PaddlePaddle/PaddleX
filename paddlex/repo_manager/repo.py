# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import importlib
import os
import os.path as osp
import shutil
import tempfile

from packaging.requirements import Requirement

from ..utils import logging
from ..utils.download import download_and_extract
from ..utils.file_interface import custom_open
from ..utils.install import (
    install_packages,
    install_packages_from_requirements_file,
    uninstall_packages,
)
from .meta import REPO_DIST_NAMES, REPO_DOWNLOAD_BASE, get_repo_meta
from .utils import (
    fetch_repo_using_git,
    install_external_deps,
    remove_repo_using_rm,
    reset_repo_using_git,
    switch_working_dir,
)

__all__ = ["build_repo_instance", "build_repo_group_installer"]


def build_repo_instance(repo_name, *args, **kwargs):
    """build_repo_instance"""
    # XXX: Hard-code type
    repo_cls = PPRepository
    repo_instance = repo_cls(repo_name, *args, **kwargs)
    return repo_instance


def build_repo_group_installer(*repos):
    """build_repo_group_installer"""
    return RepositoryGroupInstaller(list(repos))


def build_repo_group_getter(*repos):
    """build_repo_group_getter"""
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
        self.git_path = self.meta["git_path"]
        self.dist_name = self.meta.get("dist_name", None)
        self.import_name = self.meta.get("import_name", None)
        self.pdx_mod_name = (
            pdx_collection_mod.__name__ + "." + self.meta["pdx_pkg_name"]
        )
        self.main_req_file = self.meta.get("main_req_file", "requirements.txt")

    def initialize(self):
        """initialize"""
        if not self.check_installation():
            return False
        if "path_env" in self.meta:
            # Set env var
            os.environ[self.meta["path_env"]] = osp.abspath(self.root_dir)
        # NOTE: By calling `self.get_pdx()` we actually loads the repo PDX package
        # and do all registration.
        self.get_pdx()
        return True

    def check_installation(self):
        """check_installation"""
        return osp.exists(osp.join(self.root_dir, ".installed"))

    def replace_repo_deps(self, deps_to_replace, src_requirements):
        """replace_repo_deps"""
        with custom_open(src_requirements, "r") as file:
            lines = file.readlines()
        existing_deps = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            dep_to_replace = next((dep for dep in deps_to_replace if dep in line), None)
            if dep_to_replace:
                if deps_to_replace[dep_to_replace] == "None":
                    continue
                else:
                    existing_deps.append(
                        f"{dep_to_replace}=={deps_to_replace[dep_to_replace]}"
                    )
            else:
                existing_deps.append(line)
        with open(src_requirements, "w") as file:
            file.writelines([l + "\n" for l in existing_deps])

    def check_repo_exiting(self):
        """check_repo_exiting"""
        return osp.exists(osp.join(self.root_dir, ".git"))

    def install_packages(self, clean=True):
        """install_packages"""
        if self.meta["install_pkg"]:
            editable = self.meta.get("editable", True)
            if editable:
                logging.warning(
                    f"{self.import_name} will be installed in editable mode."
                )
            with switch_working_dir(self.root_dir):
                try:
                    pip_install_opts = ["--no-deps"]
                    if editable:
                        reqs = ["-e ."]
                    else:
                        reqs = ["."]
                    install_packages(reqs, pip_install_opts=pip_install_opts)
                    install_external_deps(self.name, self.root_dir)
                finally:
                    if clean:
                        # Clean build artifacts
                        tmp_build_dir = "build"
                        if osp.exists(tmp_build_dir):
                            shutil.rmtree(tmp_build_dir)
        for e in self.meta.get("extra_pkgs", []):
            if isinstance(e, tuple):
                with switch_working_dir(osp.join(self.root_dir, e[0])):
                    pip_install_opts = ["--no-deps"]
                    if e[3]:
                        reqs = ["-e ."]
                    else:
                        reqs = ["."]
                    try:
                        install_packages(reqs, pip_install_opts=pip_install_opts)
                    finally:
                        if clean:
                            tmp_build_dir = "build"
                            if osp.exists(tmp_build_dir):
                                shutil.rmtree(tmp_build_dir)

    def uninstall_packages(self):
        """uninstall_packages"""
        pkgs = []
        if self.meta["install_pkg"]:
            pkgs.append(self.dist_name)
        for e in self.meta.get("extra_pkgs", []):
            if isinstance(e, tuple):
                pkgs.append(e[1])
        uninstall_packages(pkgs)

    def mark_installed(self):
        with open(osp.join(self.root_dir, ".installed"), "wb"):
            pass

    def mark_uninstalled(self):
        os.unlink(osp.join(self.root_dir, ".installed"))

    def download(self):
        """download from remote"""
        download_url = f"{REPO_DOWNLOAD_BASE}{self.name}.tar"
        os.makedirs(self.repo_parent_dir, exist_ok=True)
        download_and_extract(download_url, self.repo_parent_dir, self.name)
        # reset_repo_using_git('FETCH_HEAD')

    def remove(self):
        """remove"""
        with switch_working_dir(self.repo_parent_dir):
            remove_repo_using_rm(self.name)

    def update(self, platform=None):
        """update"""
        branch = self.meta.get("branch", None)
        git_url = f"https://{platform}{self.git_path}"
        with switch_working_dir(self.root_dir):
            try:
                fetch_repo_using_git(branch=branch, url=git_url)
                reset_repo_using_git("FETCH_HEAD")
            except Exception as e:
                logging.warning(
                    f"Update {self.name} from {git_url} failed, check your network connection. Error:\n{e}"
                )

    def get_pdx(self):
        """get_pdx"""
        return importlib.import_module(self.pdx_mod_name)

    def get_deps(self, deps_to_replace=None):
        """get_deps"""
        # Merge requirement files
        req_list = [self.main_req_file]
        for e in self.meta.get("extra_pkgs", []):
            if isinstance(e, tuple):
                e = e[2] or osp.join(e[0], "requirements.txt")
            req_list.append(e)
        if deps_to_replace is not None:
            deps_dict = {}
            for dep in deps_to_replace:
                part, version = dep.split("=")
                repo_name, dep_name = part.split(".")
                deps_dict[repo_name] = {dep_name: version}
            src_requirements = osp.join(self.root_dir, "requirements.txt")
            if self.name in deps_dict:
                self.replace_repo_deps(deps_dict[self.name], src_requirements)
        deps = []
        for req in req_list:
            with open(osp.join(self.root_dir, req), "r", encoding="utf-8") as f:
                deps.append(f.read())
        for dep in self.meta.get("pdx_pkg_deps", []):
            deps.append(dep)
        deps = "\n".join(deps)
        return deps

    def get_version(self):
        """get_version"""
        version_file = osp.join(self.root_dir, ".pdx_gen.version")
        with open(version_file, "r", encoding="utf-8") as f:
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
    """RepositoryGroupInstaller"""

    def __init__(self, repos):
        super().__init__()
        self.repos = repos

    def install(
        self,
        force_reinstall=False,
        no_deps=False,
        constraints=None,
        deps_to_replace=None,
    ):
        """install"""
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
            self.install_deps(constraints=constraints, deps_to_replace=deps_to_replace)
        # XXX: For historical reasons the repo packages are sequentially
        # installed, and we have no failure rollbacks. Meanwhile, installation
        # failure of one repo package aborts the entire installation process.
        for ins_flag, repo in zip(ins_flags, repos):
            if ins_flag:
                repo.install_packages()
                repo.mark_installed()

    def uninstall(self):
        """uninstall"""
        repos = self._sort_repos(self.repos, check_missing=False)
        repos = repos[::-1]
        for repo in repos:
            if repo.check_installation():
                # NOTE: Dependencies are not uninstalled.
                repo.uninstall_packages()
                repo.mark_uninstalled()

    def get_deps(self, deps_to_replace=None):
        """get_deps"""
        deps_list = []
        repos = self._sort_repos(self.repos, check_missing=True)
        for repo in repos:
            deps = repo.get_deps(deps_to_replace=deps_to_replace)
            deps = self._normalize_deps(deps, headline=f"# {repo.name} dependencies")
            deps_list.append(deps)
        # Add an extra new line to separate dependencies of different repos.
        return "\n\n".join(deps_list)

    def install_deps(self, constraints, deps_to_replace=None):
        """install_deps"""
        deps_str = self.get_deps(deps_to_replace=deps_to_replace)
        with tempfile.TemporaryDirectory() as td:
            req_file = osp.join(td, "requirements.txt")
            with open(req_file, "w", encoding="utf-8") as fr:
                fr.write(deps_str)
            cons_file = osp.join(td, "constraints.txt")
            with open(cons_file, "w", encoding="utf-8") as fc:
                if constraints is not None:
                    fc.write(constraints)
                # HACK: Avoid installing OpenCV variants unexpectedly
                fc.write("opencv-python == 0.0.0\n")
                fc.write("opencv-python-headless == 0.0.0\n")
                fc.write("opencv-contrib-python-headless == 0.0.0\n")
            pip_install_opts = []
            pip_install_opts.append("-c")
            pip_install_opts.append(cons_file)
            install_packages_from_requirements_file(
                req_file, pip_install_opts=pip_install_opts
            )

    def _sort_repos(self, repos, check_missing=False):
        # We sort the repos to ensure that the dependencies precede the
        # dependent in the list.
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
            be = "is" if len(missing_names) == 1 else "are"
            raise RuntimeError(f"{missing_names} {be} required in the installation.")
        else:
            assert len(sorted_repos) == len(self.repos)
        return sorted_repos

    def _normalize_deps(self, deps, headline=None):
        lines = []
        if headline is not None:
            lines.append(headline)
        for line in deps.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            pos = line_s.find("#")
            if pos == 0:
                continue
            elif pos > 0:
                line_s = line_s[:pos]
            # If `line` is not an empty line or a comment, it must be a requirement specifier.
            # Other forms may cause a parse error.
            req = Requirement(line_s)
            if req.name in REPO_DIST_NAMES:
                # Skip repo packages
                continue
            elif req.name.replace("_", "-") in (
                "opencv-python",
                "opencv-contrib-python",
                "opencv-python-headless",
                "opencv-contrib-python-headless",
            ):
                # FIXME: The original version specifiers are ignored. It would be better to check them here.
                # The resolver will get the version info from the constraints file.
                line_s = "opencv-contrib-python"
            elif req.name == "albumentations":
                # HACK
                line_s = "albumentations @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/patched_packages/albumentations-1.4.10%2Bpdx-py3-none-any.whl"
                line_s += "\nalbucore @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/patched_packages/albucore-0.0.13%2Bpdx-py3-none-any.whl"
            elif req.name.replace("_", "-") == "nuscenes-devkit":
                # HACK
                line_s = "nuscenes-devkit @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/patched_packages/nuscenes_devkit-1.1.11%2Bpdx-py3-none-any.whl"
            elif req.name == "imgaug":
                # HACK
                line_s = "imgaug @ https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/patched_packages/imgaug-0.4.0%2Bpdx-py2.py3-none-any.whl"
            lines.append(line_s)

        return "\n".join(lines)


class RepositoryGroupGetter(object):
    """RepositoryGroupGetter"""

    def __init__(self, repos):
        super().__init__()
        self.repos = repos

    def get(self, force=False, platform=None):
        """clone"""
        if force:
            self.remove()
        for repo in self.repos:
            repo.download()
            repo.update(platform=platform)

    def remove(self):
        """remove"""
        for repo in self.repos:
            repo.remove()
