# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import re
import contextlib
import numpy as np
import torch
import warnings
import dnnlib
import re
import requests

#----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()

def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor

#----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None): # pylint: disable=redefined-builtin
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)

#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')

#----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().

def profiled_function(fn):
    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)
    decorator.__name__ = fn.__name__
    return decorator

#----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.

class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1

#----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

def copy_params_and_buffers(src_module, dst_module, require_all=False, freeze_imported_weights=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    dst_tensors = dict(named_params_and_buffers(dst_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all), f'Missing source tensor: {name}'
        if name in src_tensors:
            tensor.requires_grad_(False)
            tensor.copy_(src_tensors[name].detach()).requires_grad_(src_tensors[name].requires_grad)

def copy_params_and_buffers_lora_expansion(src_module, dst_module, require_all=False, freeze_imported_weights=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    dst_tensors = dict(named_params_and_buffers(dst_module))

    for name, src_tensor in named_params_and_buffers(src_module):
        # renaming from original EG3D generator names
        new_name_alt = name.replace('superresolution', 'superresolution.superresolution_module')
        
        new_name = name.replace('synthesis', 'synthesis.synthesis')
        new_name = re.sub(r"b\d+", lambda match: match.group(0) + '.synthesis_block', new_name, count=1)
        new_name = re.sub(r"block\d+", lambda match: match.group(0) + '.synthesis_block', new_name, count=1)
        new_name = new_name.replace('conv1', 'conv1.synthesis_layer')
        new_name = new_name.replace('conv0', 'conv0.synthesis_layer')
        new_name = new_name.replace('torgb', 'torgb.to_rgb_layer')
        new_name = new_name.replace('affine', 'affine.fully_connected_layer')
        #if new_name not in dst_tensors and name not in dst_tensors:
        #    print(f'{name} in source module but not in destination')
        assert (new_name in dst_tensors) or (name in dst_tensors) or (new_name_alt in dst_tensors), f'{name} AKA {new_name} in source module but not in destination'

    for name, dest_tensor in named_params_and_buffers(dst_module):
        new_name_alt = name.replace('superresolution.superresolution_module', 'superresolution')

        new_name = name.replace('synthesis.synthesis', 'synthesis')
        new_name = re.sub(r"b\d+.synthesis_block", lambda match: match.group(0).replace('.synthesis_block', ''), new_name, count=1)
        new_name = re.sub(r"block\d+.synthesis_block", lambda match: match.group(0).replace('.synthesis_block', ''), new_name, count=1)
        new_name = new_name.replace('conv1.synthesis_layer', 'conv1')
        new_name = new_name.replace('conv0.synthesis_layer', 'conv0')
        new_name = new_name.replace('torgb.to_rgb_layer', 'torgb')
        new_name = new_name.replace('affine.fully_connected_layer', 'affine')
        if name in src_tensors:
            dest_tensor.requires_grad_(False)
            dest_tensor.copy_(src_tensors[name].detach()).requires_grad_(src_tensors[name].requires_grad)
            if freeze_imported_weights:
                dest_tensor.requires_grad_(False)
        elif new_name in src_tensors:
            dest_tensor.requires_grad_(False)
            dest_tensor.copy_(src_tensors[new_name].detach()).requires_grad_(src_tensors[new_name].requires_grad)
            if freeze_imported_weights:
                dest_tensor.requires_grad_(False)
        elif new_name_alt in src_tensors:
            dest_tensor.requires_grad_(False)
            dest_tensor.copy_(src_tensors[new_name_alt].detach()).requires_grad_(src_tensors[new_name_alt].requires_grad)
            if freeze_imported_weights:
                dest_tensor.requires_grad_(False)
        else:
            assert ('lora' in name), f'Dest module has param that does not exist in source module: {name}'

def debug_params(src_module, dst_module, require_all=False, freeze_imported_weights=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    dst_tensors = dict(named_params_and_buffers(dst_module))
    print('Checking which dst tensors are not in source')
    for dest_tensor in dst_tensors.keys():
        new_name = dest_tensor.replace('synthesis.synthesis', 'synthesis')
        new_name = re.sub(r"b\d+.synthesis_block", lambda match: match.group(0).replace('.synthesis_block', ''), new_name, count=1)
        new_name = re.sub(r"block\d+.synthesis_block", lambda match: match.group(0).replace('.synthesis_block', ''), new_name, count=1)
        new_name = new_name.replace('conv1.synthesis_layer', 'conv1')
        new_name = new_name.replace('conv0.synthesis_layer', 'conv0')
        new_name = new_name.replace('torgb.to_rgb_layer', 'torgb')
        new_name = new_name.replace('affine.fully_connected_layer', 'affine')
        if new_name not in src_tensors.keys() and dest_tensor not in src_tensors.keys():
            if 'lora' not in dest_tensor:
                print(f'Missing tensor from source: {dest_tensor}')

    print('Checking that all src tensors are in destination')
    #print(src_tensors.keys())
    for src_tensor in src_tensors.keys():
        # renaming from original EG3D generator names
        new_name = src_tensor.replace('synthesis', 'synthesis.synthesis')
        new_name = re.sub(r"b\d+", lambda match: match.group(0) + '.synthesis_block', new_name, count=1)
        new_name = re.sub(r"block\d+", lambda match: match.group(0) + '.synthesis_block', new_name, count=1)
        new_name = new_name.replace('conv1', 'conv1.synthesis_layer')
        new_name = new_name.replace('conv0', 'conv0.synthesis_layer')
        new_name = new_name.replace('torgb', 'torgb.to_rgb_layer')
        new_name = new_name.replace('affine', 'affine.fully_connected_layer')
        if src_tensor not in dst_tensors.keys() and new_name not in dst_tensors.keys():
            print(f'Missing tensor from destination: {src_tensor}')
        #if src_tensor in dst_tensors.keys():
        #    print(f'Found tensor in destination: {src_tensor}')
    """
    print('printing dst_tensors')
    for dest_tensor in dst_tensors.keys():
        if 'superresolution' in dest_tensor:
            print(dest_tensor)
    
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all), f'Missing source tensor: {name}'
        if name in src_tensors:
            tensor.requires_grad_(False)
            tensor.copy_(src_tensors[name].detach()).requires_grad_(src_tensors[name].requires_grad)
        elif name.replace('synthesis.', 'synthesis.synthesis.').replace('superresolution.', 'superresolution.superresolution_module.') in src_tensors:
            new_name = name.replace('synthesis.', 'synthesis.synthesis.').replace('superresolution.', 'superresolution.superresolution_module.')
            tensor.requires_grad_(False)
            tensor.copy_(src_tensors[new_name].detach()).requires_grad_(src_tensors[new_name].requires_grad)
        else:
            if 'synthesis.b8' in name:
                print(f'Missing tensor from source: {name}')
    for name in src_tensors.keys():
        if name not in dst_tensors.keys():
            new_name = name.replace('synthesis.synthesis.', 'synthesis.').replace('superresolution.superresolution_module.', 'superresolution.')
            if new_name not in dst_tensors.keys():
                if 'synthesis.b8' in name:
                    print(f'Missing tensor from destination: {name}')
        #assert (name in dst_tensors.keys()), f'{name} in source module but not in destination'
    """
#----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.

@contextlib.contextmanager
def ddp_sync(module, sync):
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield

#----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.

def check_ddp_consistency(module, ignore_regex=None):
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + '.' + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname

#----------------------------------------------------------------------------
# Print summary table of module hierarchy.

def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(dnnlib.EasyDict(mod=mod, outputs=outputs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, 'Parameters', 'Buffers', 'Output shape', 'Datatype']]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e.outputs]
        rows += [[
            name + (':0' if len(e.outputs) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print('  '.join(cell + ' ' * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs

#----------------------------------------------------------------------------

def notify(message):
    chat_id = '6712696502'
    TOKEN = '6643471688:AAH_8A5SrUe9eI-nAs90No_CI1T8H2KYqQE'
    user_id = '6712696502'
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"
    requests.get(url).json() # this sends the message