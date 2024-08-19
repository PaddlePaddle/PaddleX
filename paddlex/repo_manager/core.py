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
from collections import OrderedDict

from ..utils import logging
from .utils import install_deps_using_pip
from .meta import get_all_repo_names, get_repo_meta
from .repo import build_repo_instance, build_repo_group_cloner, build_repo_group_installer

__all__ = [
    'set_parent_dirs', 'setup', 'wheel', 'is_initialized', 'initialize',
    'get_versions'
]


def _parse_repo_deps(repos):
    ret = []
    for repo_name in repos:
        repo_meta = get_repo_meta(repo_name)
        ret.extend(_parse_repo_deps(repo_meta.get('requires', [])))
        ret.append(repo_name)
    return ret


class _GlobalContext(object):
    REPO_PARENT_DIR = None
    PDX_COLLECTION_MOD = None
    REPOS = None

    @classmethod
    def set_parent_dirs(cls, repo_parent_dir, pdx_collection_mod):
        """ set_parent_dirs """
        cls.REPO_PARENT_DIR = repo_parent_dir
        cls.PDX_COLLECTION_MOD = pdx_collection_mod

    @classmethod
    def build_repo_instance(cls, repo_name):
        """ build_repo_instance """
        return build_repo_instance(repo_name, cls.REPO_PARENT_DIR,
                                   cls.PDX_COLLECTION_MOD)

    @classmethod
    def is_initialized(cls):
        """ is_initialized """
        return cls.REPOS is not None

    @classmethod
    def initialize(cls):
        """ initialize """
        cls.REPOS = []

    @classmethod
    def add_repo(cls, repo):
        """ add_repo """
        if not cls.is_initialized():
            cls.initialize()
        cls.REPOS.append(repo)

    @classmethod
    def add_repos(cls, repos):
        """ add_repos """
        if len(repos) == 0 and not cls.is_initialized():
            cls.initialize()
        for repo in repos:
            cls.add_repo(repo)


set_parent_dirs = _GlobalContext.set_parent_dirs
is_initialized = _GlobalContext.is_initialized


def setup(repo_names,
          reinstall=None,
          no_deps=False,
          constraints=None,
          platform=None,
          update_repos=False,
          use_local_repos=False):
    """ setup """
    repo_names = list(set(_parse_repo_deps(repo_names)))

    repos = []
    for repo_name in repo_names:
        repo = _GlobalContext.build_repo_instance(repo_name)
        repos.append(repo)

    repos_to_clone = []
    for repo in repos:
        repo_name = repo.name
        if repo.check_repo_exiting():
            if use_local_repos:
                reinstall = True
                logging.warning(
                    f"We will use the existing repo of {repo.name}.")
                continue
            logging.warning(f"Existing of {repo.name} repo.")
            if reinstall is None:
                if sys.stdin.isatty():
                    logging.warning("Should we remove it (y/n)?")
                try:
                    remove_existing = input()
                except EOFError:
                    logging.warning(
                        "Unable to read from stdin. Please set `reinstall` to \
                        True or False to apply a global setting for reclone repos."
                    )
                    raise
                remove_existing = remove_existing.lower() in ('y', 'yes')
            else:
                remove_existing = reinstall
            if remove_existing:
                repo.remove()
                repos_to_clone.append(repo)
            else:
                logging.warning(
                    f"We will use the existing repo of {repo.name}.")
        else:
            repos_to_clone.append(repo)
    repos_to_install = []
    for repo in repos:
        repo_name = repo.name
        if repo.check_installation():
            logging.warning(f"Existing installation of {repo.name} detected.")
            if reinstall is None and not update_repos:
                if sys.stdin.isatty():
                    logging.warning("Should we uninstall it (y/n)?")
                try:
                    uninstall_existing = input()
                except EOFError:
                    logging.warning(
                        "Unable to read from stdin. Please set `reinstall` to \
                        True or False to apply a global setting for reinstalling repos."
                    )
                    raise
                uninstall_existing = uninstall_existing.lower() in ('y', 'yes')
            else:
                if reinstall or update_repos:
                    uninstall_existing = True
            if uninstall_existing:
                repo.uninstall()
                repos_to_install.append(repo)
            else:
                logging.warning(
                    f"We will use the existing installation of {repo.name}.")
        else:
            repos_to_install.append(repo)
    cloner = build_repo_group_cloner(*repos_to_clone)
    installer = build_repo_group_installer(*repos_to_install)

    logging.info("Now cloning the repos...")
    cloner.clone(force_reclone=False, platform=platform)
    logging.info("All repos are existing.")

    if not no_deps:
        logging.info("Dependencies are listed below:")
        logging.info(installer.get_deps())

    logging.info("Now installing the packages...")
    install_deps_using_pip()
    if update_repos:
        installer.update()
        logging.info("All repos are updated.")
    installer.install(
        force_reinstall=False, no_deps=no_deps, constraints=constraints)
    logging.info("All packages are installed.")


def wheel(repo_names, dst_dir='./', fail_fast=False):
    """ wheel """
    for repo_name in repo_names:
        repo = _GlobalContext.build_repo_instance(repo_name)
        logging.info(f"Now building Wheel for {repo_name}...")
        try:
            tgt_dir = os.path.join(dst_dir, repo.pkg_name)
            if os.path.exists(tgt_dir):
                raise FileExistsError(f"{tgt_dir} already exists.")
            repo.wheel(tgt_dir)
        except Exception as e:
            logging.warning(
                f"Failed to build wheel for {repo_name}. We encountered the following error:\n  {str(e)}\n"
            )
            if fail_fast:
                raise
        else:
            logging.info(f"Wheel for {repo_name} is built.\n")


def initialize(repo_names=None):
    """ initialize """
    if _GlobalContext.is_initialized():
        raise RuntimeError(
            "PDX has already been initialized. Reinitialization is not supported."
        )

    if repo_names is None:
        try_all = True
        repo_names = get_all_repo_names()
    else:
        try_all = False

    repos = []
    for repo_name in repo_names:
        logging.debug(f"Now initializing {repo_name}...")
        repo = _GlobalContext.build_repo_instance(repo_name)
        flag = repo.initialize()
        if flag:
            logging.debug(f"{repo_name} is initialized.")
            repos.append(repo)
        else:
            if try_all:
                logging.debug(
                    f"Failed to initialize {repo_name}. Please make sure {repo_name} is properly installed."
                )
            else:
                pass

    _GlobalContext.add_repos(repos)


def get_versions(repo_names=None):
    """ get_versions """
    if repo_names is None:
        repo_names = get_all_repo_names()

    name2versions = OrderedDict()
    for repo_name in repo_names:
        repo = _GlobalContext.build_repo_instance(repo_name)
        versions = repo.get_version()
        name2versions[repo_name] = versions
    return name2versions
