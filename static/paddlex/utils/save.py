# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import six
import os
import errno
import warnings
import six

import numpy as np

import paddle
from paddle.fluid import layers
from paddle.fluid import core
from paddle.fluid import unique_name
from paddle.fluid.executor import global_scope
from paddle.fluid.compiler import CompiledProgram
from paddle.fluid.framework import Program, Parameter, default_main_program, default_startup_program, Variable, \
    program_guard

__all__ = ["save_mask_inference_model"]


def _get_valid_program(main_program):
    if main_program is None:
        main_program = default_main_program()
    elif isinstance(main_program, CompiledProgram):
        main_program = main_program._program
        if main_program is None:
            raise TypeError("program should be as Program type or None")
        warnings.warn(
            "The input is a CompiledProgram, this is not recommended.")
    if not isinstance(main_program, Program):
        raise TypeError("program should be as Program type or None")
    return main_program


def prepend_feed_ops(inference_program,
                     feed_target_names,
                     feed_holder_name='feed'):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True)

    for i, name in enumerate(feed_target_names):
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i})


def append_fetch_ops(inference_program,
                     fetch_target_names,
                     fetch_holder_name='fetch'):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True)

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i})


def _clone_var_in_block_(block, var):
    assert isinstance(var, Variable)
    if var.desc.type() == core.VarDesc.VarType.LOD_TENSOR:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=True)
    else:
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            persistable=True)


def save_vars(executor,
              dirname,
              main_program=None,
              vars=None,
              predicate=None,
              filename=None):
    """
    This API saves specific variables in the `Program` to files.

    There are two ways to specify the variables to be saved: set variables in
    a list and assign it to the `vars`, or use the `predicate` function to select
    variables that make `predicate(variable) == True`. The first way has a higher priority.

    The `dirname` is used to specify the folder where to save variables.
    If you prefer to save variables in separate files in the `dirname` floder,
    do not set `filename`. If you prefer to save all variables in a single file,
    use `filename` to specify it.

    Args:
        executor(Executor): The executor to run for saving variables.
        dirname(str, optional): The folder where to save variables.
                            When you need to save the parameter to the memory, set it to None.
        main_program(Program, optional): The program whose variables will be saved.
                                    If it is None, the default main program will
                                    be used automatically.
                                    Default: None
        vars(list[Variable], optional): The list contains all variables to be saved.
                                        Default: None
        predicate(function, optional): The function selects the variables that make
                                       `predicate(variable) == True`.
                                       Default: None
        filename(str, optional): If you prefer to save all variables in a single file,
                                 use `filename` to specify it. Otherwise, let `filename` be None.
                                 Default: None

    Returns:
        str: When saving parameters to a file, returns None.
             When saving parameters to memory, returns a binary string containing parameters.

    Raises:
        TypeError: If `main_program` is not an instance of Program nor None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            main_prog = fluid.Program()
            startup_prog = fluid.Program()
            with fluid.program_guard(main_prog, startup_prog):
                data = fluid.layers.data(name="img", shape=[64, 784], append_batch_size=False)
                w = fluid.layers.create_parameter(shape=[784, 200], dtype='float32', name='fc_w')
                b = fluid.layers.create_parameter(shape=[200], dtype='float32', name='fc_b')
                hidden_w = fluid.layers.matmul(x=data, y=w)
                hidden_b = fluid.layers.elementwise_add(hidden_w, b)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(startup_prog)

            # The first usage: use `vars` to set the saved variables.
            var_list = [w, b]
            path = "./my_paddle_vars"
            fluid.io.save_vars(executor=exe, dirname=path, vars=var_list,
                            filename="vars_file")
            # w and b will be save in a file named "var_file".

            # The second usage: use `predicate` to select the saved variable.
            def name_has_fc(var):
                res = "fc" in var.name
                return res
            param_path = "./my_paddle_model"
            fluid.io.save_vars(executor=exe, dirname=param_path, main_program=main_prog, vars=None, predicate = name_has_fc)
            # all variables whose names contain "fc " are saved.
    """
    save_to_memory = False
    if dirname is None and filename is None:
        save_to_memory = True

    main_program = _get_valid_program(main_program)

    if vars is None:
        return save_vars(
            executor,
            main_program=main_program,
            dirname=dirname,
            vars=list(filter(predicate, main_program.list_vars())),
            filename=filename)
    else:
        params_var_name = unique_name.generate("saved_params")
        # give warning when there is no var in model
        if len(list(vars)) == 0:
            warnings.warn(
                "no variable in your model, please ensure there are any variables in your model to save"
            )
            return None

        save_program = Program()
        save_block = save_program.global_block()

        save_var_map = {}
        for each_var in vars:
            # NOTE: don't save the variable which type is RAW
            if each_var.type == core.VarDesc.VarType.RAW:
                continue
            new_var = _clone_var_in_block_(save_block, each_var)
            if filename is None and save_to_memory is False:
                save_file_path = os.path.join(
                    os.path.normpath(dirname), new_var.name)
                save_block.append_op(
                    type='save',
                    inputs={'X': [new_var]},
                    outputs={},
                    attrs={'file_path': os.path.normpath(save_file_path)})
            else:
                save_var_map[new_var.name] = new_var

        if filename is not None or save_to_memory:
            save_var_list = []
            for name in sorted(save_var_map.keys()):
                save_var_list.append(save_var_map[name])

            save_path = str()
            if save_to_memory is False:
                save_path = os.path.join(os.path.normpath(dirname), filename)

            saved_params = save_block.create_var(
                type=core.VarDesc.VarType.RAW, name=params_var_name)
            saved_params.desc.set_persistable(True)
            save_block.append_op(
                type='save_combine',
                inputs={'X': save_var_list},
                outputs={'Y': saved_params},
                attrs={
                    'file_path': save_path,
                    'save_to_memory': save_to_memory
                })

        #NOTE(zhiqiu): save op will add variable kLookupTablePath in save_program.desc,
        # which leads to diff on save_program and its desc. Call _sync_with_cpp
        # to keep consistency.
        save_program._sync_with_cpp()
        executor.run(save_program)
        if save_to_memory:
            return global_scope().find_var(params_var_name).get_bytes()


def _save_distributed_persistables(executor, dirname, main_program):
    """
    save_persistables for distributed training.
    the method will do things listed below:
    1.save part of persistable variables on trainer.
    2.receive "remote prefetch variables" from parameter servers and merge them.
    3.save "distributed lookup table" on parameter servers.
    4.receive "optimizer variables" from parameter servers and merge them.

    Args:
        executor(Executor): The executor to run for saving parameters.
        dirname(str): The saving directory path.
        main_program(Program): The program whose parameters will be
                            saved. the main_program must be the trainer_program
                            get after transpiler.

    Returns:
        None

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            exe = fluid.Executor(fluid.CPUPlace())
            param_path = "./my_paddle_model"
            t = distribute_transpiler.DistributeTranspiler()
            t.transpile(...)
            train_program = t.get_trainer_program()
            _save_distributed_persistables(executor=exe, dirname=param_path, main_program=train_program)
    """

    def __save_remote_params(executor, dirname, remote_params_map):
        """
        recive params on pserver through rpc.
        if the params are be sliced, will concat them to one, then save it.
        """
        if not remote_params_map:
            return

        prog = Program()
        block = prog.global_block()

        # recv optimize vars from pserver
        for name, remote_params in remote_params_map.items():
            origin = remote_params[0].origin
            is_slice = remote_params[0].is_slice

            slices = [None] * len(remote_params)
            slice_varnames = [None] * len(remote_params)
            remote_varnames = [None] * len(remote_params)
            endpoints = [None] * len(remote_params)

            for idx, optimizer in enumerate(remote_params):
                block_id = optimizer.block_id
                slice = optimizer.slice
                endpoint = optimizer.endpoint

                index = block_id if is_slice else idx
                slices[index] = slice
                slice_varnames[index] = "{}.slice.{}".format(slice.name, idx)
                remote_varnames[index] = slice.name
                endpoints[index] = endpoint

            slice_shapes = []
            for slice in slices:
                tmp = [str(dim) for dim in slice.shape]
                slice_shapes.append(",".join(tmp))

            block.append_op(
                type='recv_save',
                attrs={
                    "trainer_id": 0,
                    "shape": origin.shape,
                    "slice_shapes": slice_shapes,
                    "slice_varnames": slice_varnames,
                    "remote_varnames": remote_varnames,
                    "endpoints": endpoints,
                    "file_path": os.path.join(dirname, origin.name)
                })

        executor.run(prog)


def is_persistable(var):
    """
    Check whether the given variable is persistable.

    Args:
        var(Variable): The variable to be checked.

    Returns:
        bool: True if the given `var` is persistable
        False if not.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            param = fluid.default_main_program().global_block().var('fc.b')
            res = fluid.io.is_persistable(param)
    """
    if var.desc.type() == core.VarDesc.VarType.FEED_MINIBATCH or \
            var.desc.type() == core.VarDesc.VarType.FETCH_LIST or \
            var.desc.type() == core.VarDesc.VarType.READER:
        return False
    return var.persistable


def save_persistables(executor, dirname, main_program=None, filename=None):
    """
    This operator saves all persistable variables from :code:`main_program` to
    the folder :code:`dirname` or file :code:`filename`. You can refer to
    :ref:`api_guide_model_save_reader_en` for more details. And then
    saves these persistables variables to the folder :code:`dirname` or file
    :code:`filename`.

    The :code:`dirname` is used to specify the folder where persistable variables
    are going to be saved. If you would like to save variables in separate
    files, set :code:`filename` None; if you would like to save all variables in a
    single file, use :code:`filename` to specify the file name.

    Args:
        executor(Executor): The executor to run for saving persistable variables.
                            You can refer to :ref:`api_guide_executor_en` for
                            more details.
        dirname(str, optional): The saving directory path.
                            When you need to save the parameter to the memory, set it to None.
        main_program(Program, optional): The program whose persistbale variables will
                                         be saved. You can refer to
                                         :ref:`api_guide_Program_en` for more details.
                                         If it is None, the default main program will
                                         be used.
                                         Default: None.
        filename(str, optional): The file to save all variables. If you prefer to
                                 save variables in different files, set it to None.
                                 Default: None.

    Returns:
        str: When saving parameters to a file, returns None.
             When saving parameters to memory, returns a binary string containing parameters.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            dir_path = "./my_paddle_model"
            file_name = "persistables"
            image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())

            predict = fluid.layers.fc(input=image, size=10, act='softmax')
            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)
            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            fluid.io.save_persistables(executor=exe, dirname=dir_path, filename=file_name)
            # The persistables variables weights and bias in the fc layer of the network
            # are going to be saved in the same file named "persistables" in the path
            # "./my_paddle_model"
    """
    if main_program and main_program._is_distributed:
        return _save_distributed_persistables(
            executor, dirname=dirname, main_program=main_program)
    else:
        return save_vars(
            executor,
            dirname=dirname,
            main_program=main_program,
            vars=None,
            predicate=is_persistable,
            filename=filename)


def save_mask_inference_model(dirname,
                              feeded_var_names,
                              target_vars,
                              executor,
                              main_program=None,
                              model_filename=None,
                              params_filename=None,
                              export_for_deployment=True,
                              program_only=False):
    """
    Prune the given `main_program` to build a new program especially for inference,
    and then save it and all related parameters to given `dirname` .
    If you just want to save parameters of your trained model, please use the
    :ref:`api_fluid_io_save_params` . You can refer to :ref:`api_guide_model_save_reader_en`
    for more details.

    Note:
        The :code:`dirname` is used to specify the folder where inference model
        structure and parameters are going to be saved. If you would like to save params of
        Program in separate files, set `params_filename` None; if you would like to save all
        params of Program in a single file, use `params_filename` to specify the file name.

    Args:
        dirname(str): The directory path to save the inference model.
        feeded_var_names(list[str]): list of string. Names of variables that need to be feeded
                                     data during inference.
        target_vars(list[Variable]): list of Variable. Variables from which we can get
                                     inference results.
        executor(Executor): The executor that saves the inference model. You can refer
                            to :ref:`api_guide_executor_en` for more details.
        main_program(Program, optional): The original program, which will be pruned to
                                         build the inference model. If is setted None,
                                         the global default :code:`_main_program_` will be used.
                                         Default: None.
        model_filename(str, optional): The name of file to save the inference program
                                       itself. If is setted None, a default filename
                                       :code:`__model__` will be used.
        params_filename(str, optional): The name of file to save all related parameters.
                                        If it is setted None, parameters will be saved
                                        in separate files .
        export_for_deployment(bool): If True, programs are modified to only support
                                     direct inference deployment. Otherwise,
                                     more information will be stored for flexible
                                     optimization and re-training. Currently, only
                                     True is supported.
                                     Default: True.
        program_only(bool, optional): If True, It will save inference program only, and do not
                                      save params of Program.
                                      Default: False.

    Returns:
        The fetch variables' name list

     Return Type:
        list

    Raises:
        ValueError: If `feed_var_names` is not a list of basestring, an exception is thrown.
        ValueError: If `target_vars` is not a list of Variable, an exception is thrown.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid

            path = "./infer_model"

            # User defined network, here a softmax regresssion example
            image = fluid.data(name='img', shape=[None, 28, 28], dtype='float32')
            label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            feeder = fluid.DataFeeder(feed_list=[image, label], place=fluid.CPUPlace())
            predict = fluid.layers.fc(input=image, size=10, act='softmax')

            loss = fluid.layers.cross_entropy(input=predict, label=label)
            avg_loss = fluid.layers.mean(loss)

            exe = fluid.Executor(fluid.CPUPlace())
            exe.run(fluid.default_startup_program())

            # Feed data and train process

            # Save inference model. Note we don't save label and loss in this example
            fluid.io.save_inference_model(dirname=path,
                                          feeded_var_names=['img'],
                                          target_vars=[predict],
                                          executor=exe)

            # In this example, the save_inference_mode inference will prune the default
            # main program according to the network's input node (img) and output node(predict).
            # The pruned inference program is going to be saved in the "./infer_model/__model__"
            # and parameters are going to be saved in separate files under folder
            # "./infer_model".

    """
    if isinstance(feeded_var_names, six.string_types):
        feeded_var_names = [feeded_var_names]
    elif export_for_deployment:
        if len(feeded_var_names) > 0:
            # TODO(paddle-dev): polish these code blocks
            if not (bool(feeded_var_names) and all(
                    isinstance(name, six.string_types)
                    for name in feeded_var_names)):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    elif export_for_deployment:
        if not (bool(target_vars) and
                all(isinstance(var, Variable) for var in target_vars)):
            raise ValueError("'target_vars' should be a list of Variable.")

    main_program = _get_valid_program(main_program)

    # remind user to set auc_states to zeros if the program contains auc op
    all_ops = main_program.global_block().ops
    for op in all_ops:
        if op.type == 'auc':
            warnings.warn(
                "please ensure that you have set the auc states to zeros before saving inference model"
            )
            break

    # fix the bug that the activation op's output as target will be pruned.
    # will affect the inference performance.
    # TODO(Superjomn) add an IR pass to remove 1-scale op.
    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            if isinstance(var, Variable):
                var = layers.scale(
                    var, 1., name="save_infer_model/scale_{}".format(i))
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]

    # when a pserver and a trainer running on the same machine, mkdir may conflict
    save_dirname = dirname
    try:
        save_dirname = os.path.normpath(dirname)
        os.makedirs(save_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if model_filename is not None:
        model_basename = os.path.basename(model_filename)
    else:
        model_basename = "__model__"
    model_basename = os.path.join(save_dirname, model_basename)

    # When export_for_deployment is true, we modify the program online so that
    # it can only be loaded for inference directly. If it's false, the whole
    # original program and related meta are saved so that future usage can be
    # more flexible.

    origin_program = main_program.clone()

    if export_for_deployment:
        main_program = main_program.clone()
        global_block = main_program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)

        main_program.desc.flush()

        main_program = main_program._prune_with_input(
            feeded_var_names=feeded_var_names, targets=target_vars)
        main_program = main_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [v.name for v in target_vars]

        prepend_feed_ops(main_program, feeded_var_names)
        append_fetch_ops(main_program, fetch_var_names)

        main_program.desc._set_version()
        paddle.fluid.core.save_op_compatible_info(main_program.desc)
        with open(model_basename, "wb") as f:
            f.write(main_program.desc.serialize_to_string())
    else:
        # TODO(panyx0718): Save more information so that it can also be used
        # for training and more flexible post-processing.
        with open(model_basename + ".main_program", "wb") as f:
            f.write(main_program.desc.serialize_to_string())

    if program_only:
        warnings.warn(
            "save_inference_model specified the param `program_only` to True, It will not save params of Program."
        )
        return target_var_name_list

    main_program._copy_dist_param_info_from(origin_program)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    save_persistables(executor, save_dirname, main_program, params_filename)
    return target_var_name_list
