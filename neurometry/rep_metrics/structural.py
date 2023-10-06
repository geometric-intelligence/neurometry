import numpy as np
import cupy as cp
import pandas as pd
import torch
import inspect
import itertools
from copy import copy

# Function Helpers --------------------------------------------------------


def get_fn_kwargs(fn, kwargs):
    if kwargs is None:
        return {}

    try:
        inspect.signature(fn)
    except ValueError:
        return kwargs

    return {k: v for k, v in kwargs.items() if k in inspect.signature(fn).parameters}


def apply_fn(fn, x, **kwargs):
    return fn(x, **get_fn_kwargs(fn, kwargs))


def parse_fns(fns, kwargs=None, **other_kwargs):
    if fns is None:
        fns = []
    elif callable(fns):
        fns = [fns]

    if isinstance(fns, dict) and isinstance(kwargs, dict):
        raise ValueError(
            "Pass either a fns dict or a fns list and kwargs list or dict, not both."
        )

    if isinstance(fns, dict):
        if len(other_kwargs) == 0:
            return fns
        if len(other_kwargs) >= 1:
            return {
                fn: get_fn_kwargs(fn, {**fn_kwargs, **other_kwargs})
                for fn, fn_kwargs in fns.items()
            }

    if kwargs is None:
        kwargs = {}

    # Initialize the result
    output_fns = {}

    # If kwargs is a dict, apply it to all functions
    if isinstance(kwargs, dict):
        kwargs = [kwargs] * len(fns)

    # Iterate over the functions and their corresponding kwargs
    for fn_item, kwarg_item in zip(fns, kwargs):
        # Handle case where fn_item is a function
        if callable(fn_item):
            output_fns[fn_item] = get_fn_kwargs(fn_item, {**kwarg_item, **other_kwargs})
        # Handle case where fn_item is a dict with function and kwargs
        elif isinstance(fn_item, dict):
            for fn, fn_kwargs in fn_item.items():
                output_fns[fn] = get_fn_kwargs(fn, {**fn_kwargs, **other_kwargs})

    return output_fns


# Data Structures --------------------------------------------------------


def modify_keys(key_data, key_mods, exact_match=False):
    if not isinstance(key_data, (list, dict, pd.DataFrame)):
        raise ValueError("key_data must be list, dict, or pd.DataFrame")

    if isinstance(key_data, list):
        keys = key_data
    if isinstance(key_data, dict):
        keys = list(key_data.keys())
    if isinstance(key_data, pd.DataFrame):
        keys = key_data.columns

    modified_keys = {}
    for key in keys:
        key_mod = copy(key)
        for key_query, mod in key_mods.items():
            if key_query == key and exact_match:
                key_mod = mod
            if key_query in key and not exact_match:
                if mod not in key:
                    key_mod = key_mod.replace(key_query, mod)

            modified_keys[key] = key_mod

    if isinstance(key_data, list):
        return list(modified_keys.values())
    if isinstance(key_data, dict):
        return {modified_keys[key]: key_data[key] for key in key_data}
    if isinstance(key_data, pd.DataFrame):
        return key_data.rename(columns={v: k for k, v in modified_keys.items()})


def make_filter(query, value, logic="==", value_fn=None):
    value = value if value_fn is None else value_fn(value)

    if logic == "==":
        return query == value
    elif logic == "!=":
        return query != value
    elif logic == "is":
        return query is value
    elif logic == "is not":
        return query is not value
    elif logic == "in":
        return query in value
    elif logic == "not in":
        return query not in value
    elif logic == "<":
        return query < value
    elif logic == "<=":
        return query <= value
    elif logic == ">":
        return query > value
    elif logic == ">=":
        return query >= value
    elif logic == "contains":
        return value in query
    elif logic == "matches":
        return re.fullmatch(value, query) is not None

    raise ValueError(f"Unknown logic: {logic}")


def make_pandas_query(query, value, logic="==", value_fn=None):
    value = value if value_fn is None else value_fn(value)

    if logic in ["in", "not in"]:
        logic = "==" if logic == "in" else "!="

    if logic in ["==", "!=", "<", "<=", ">", ">="]:
        return f"{query} {logic} {value}"
    elif logic == "contains":
        return f"{query}.str.contains.{value}"
    elif logic == "matches":
        return f"{query}.str.fullmatch({value})"

    raise ValueError(f"Unknown pandas logic: {logic}")


def apply_filter(data, value, query=None, logic="==", value_fn=None):
    filter_args = (value, logic, value_fn)

    if query is None:
        if isinstance(data, list):
            return [item for item in data if make_filter(item, *filter_args)]

        if isinstance(data, Mapping):
            return {
                key: val for key, val in data.items() if make_filter(val, *filter_args)
            }

        if isinstance(data, pd.DataFrame):
            raise ValueError(
                "filtering a pd.DataFrame requires that the query arg is not None"
            )

    if isinstance(data, list):
        return [item for item in data if make_filter(item[query], *filter_args)]

    elif isinstance(data, dict):
        return {
            key: val
            for key, val in data.items()
            if make_filter(val[query], *filter_args)
        }

    elif isinstance(data, pd.DataFrame):
        return data[data[query].apply(lambda x: make_filter(x, *filter_args))]

    raise ValueError("data type not recognized")


def parse_uid_keys(uid_keys=None, splits=None):
    if uid_keys is None:
        return None

    def split_uid_keys(uid_keys, splits):
        return {split: uid_keys for split in splits}

    if isinstance(uid_keys, str):
        uid_keys = [uid_keys]

    elif isinstance(uid_keys, list):
        if len(uid_keys) == 0:
            return None
        if splits is not None:
            return split_uid_keys(uid_keys, splits)

    elif isinstance(uid_keys, dict):
        uid_keys = {
            entry: parse_uid_keys(keys, None) for entry, keys in uid_keys.items()
        }

        return uid_keys


def iterative_subset(df, index, cat_list):
    out_dict = {}
    df = df[[index] + cat_list]
    for row in df.itertuples():
        current = out_dict
        for i, col in enumerate(cat_list):
            cat = getattr(row, col)
            if cat not in current:
                if i + 1 < len(cat_list):
                    current[cat] = {}
                else:
                    current[cat] = []
            current = current[cat]
            if i + 1 == len(cat_list):
                current += [getattr(row, index)]

    return out_dict


def convert_subsets_to_index(subset_dict, index_dtype=None):
    new_subset_dict = {}

    def convert_subset_to_index(subset_dict, new_subset_dict):
        for key, value in subset_dict.items():
            if isinstance(value, dict):
                new_subset_dict[key] = {}
                convert_subset_to_index(subset_dict[key], new_subset_dict[key])
            if isinstance(value, list):
                new_subset_dict[key] = pd.Index(value, dtype=index_dtype)

    convert_subset_to_index(subset_dict, new_subset_dict)

    return new_subset_dicts


def drop_constant_columns(df, include=["bool"], keep_cols=None):
    cols_to_check = df.columns[df.isnull().all()].tolist()
    cols_to_check += df.columns[(df == "").all()].tolist()
    cols_to_check += df.columns[df.isna().all()].tolist()
    cols_to_check += df.select_dtypes(include=include).columns.tolist()

    if keep_cols is not None:
        cols_to_check = [col for col in cols_to_check if col not in keep_cols]

    nunique = df[cols_to_check].applymap(str).nunique()
    cols_to_drop = nunique[nunique == 1].index

    return df.drop(cols_to_drop, axis=1, inplace=False)


# Model Structures --------------------------------------------------------


def get_model_device(model):
    return next(model.parameters()).device


def get_nesting_depth(module, depth=0):
    if not list(module.children()):
        return depth
    else:
        return max(
            get_max_module_depth(child, depth + 1) for child in module.children()
        )


# Tensors + CUDA ------------------------------------------------------------


def remove_from_cuda(*args):
    if isinstance(args[0], list):
        args = tuple(args[0])
    return tuple(arg.cpu() if isinstance(arg, torch.Tensor) else arg for arg in args)


def numbers_to_tensor(numbers):
    if isinstance(numbers, torch.Tensor):
        return numbers
    if isinstance(numbers, (np.number, np.ndarray)):
        return torch.Tensor(numbers)
    if isinstance(numbers, list):
        return torch.Tensor(numbers)
    if isinstance(numbers, dict):
        return torch.Tensor(list(numbers.values()))
    if isinstance(numbers, DataLoader):
        return torch.stack(torch.Tensor(batch) for batch in numbers).squeeze()


def tensor_to_numbers(tensor, numpy=False):
    if tensor.numel() == 1:
        numbers = tensor.item()
    if tensor.numel() > 1:
        if len(tensor.shape) == 1:
            numbers = tensor.tolist()
        if len(lensor.shape) >= 2:
            numpy = True
    return np.array(numbers) if numpy else numbers


def convert_tensor_backend(data, backend="torch"):
    if backend == "numpy":
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return cp.asnumpy(data)  # assumes cupy
    elif backend == "cupy":
        if isinstance(data, cp.ndarray):
            return data
        elif isinstance(data, np.ndarray):
            return cp.asarray(data)
        return cp.fromDlpack(torch.utils.dlpack.to_dlpack(data))
    elif backend == "torch":
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        return torch.utils.dlpack.from_dlpack(data.toDlpack())
    raise ValueError("data type and backend must be one of (numpy, cupy, torch)")


def convert_to_tensor(*args, dtype=None, device=None, copy=False):
    def convert_item(arg):
        def process_tensor(tensor):
            tensor = tensor.clone() if copy else tensor
            if dtype is not None:
                tensor = tensor.to(dtype)
            if device is not None:
                tensor = tensor.to(device)
            return tensor

        if isinstance(arg, torch.Tensor):
            return process_tensor(arg)
        elif isinstance(arg, (np.ndarray, cp.ndarray)):
            arg = convert_tensor_backend(arg, "torch")
            return process_tensor(arg)
        elif isinstance(arg, list):
            return [convert_item(item) for item in arg]
        elif isinstance(arg, dict):
            return {key: convert_item(val) for key, val in arg.items()}
        return arg

    outputs = [convert_item(arg) for arg in args]
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


# Conversion Classes ------------------------------------------------------------


class TorchSKLearn:
    def __init__(
        self,
        device=None,
        dtype=None,
        clone_all_inputs=True,
        replace_acronyms=True,
        sklearn_style_class_names=True,
    ):
        self.device = device
        self.dtype = dtype
        self.copy = clone_all_inputs

        self.replace_acronyms = replace_acronyms
        self.sklearn_style_names = sklearn_style_class_names
        self._names_at_initialization = self.__dict__.keys()

    def parse_input_data(self, X):
        if isinstance(X, torch.Tensor):
            if self.copy:
                X = X.clone()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if self.dtype:
            X = X.to(self.dtype)
        if self.device:
            X = X.to(self.device)
        return X  # parsed_input_data

    def to(self, device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__setattr__(attr, value.to(device))

    def cuda(self):
        self.to("cuda")

    def cpu(self):
        self.to("cpu")

    def remove_from_cuda(self, *args):
        self.to("cpu")
        if len(args) == 1:
            args = (args,)
        args_out = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                args_out.append(arg.to("cpu"))
        return args_out

    def _replace_acronyms(self, acronym, replacement):
        for key in list(self.__dict__.keys()):
            if acronym in key:
                new_key = key.replace(acronym, replacement)
                setattr(self, new_key, getattr(self, key))
                delattr(self, key)

    def _make_sklearn_style_names(self):
        for key in list(self.__dict__.keys()):
            if key not in list(self._names_at_initialization):
                new_key = key + "_"
                setattr(self, new_key, getattr(self, key))
                delattr(self, key)


### Downloads ----------------------------------------------------


def extract_tar(tar_file, dest_path, subfolder, delete=False):
    tar = tarfile.open(tar_file)
    if not subfolder:
        tar.extractall(dest_path)
    if subfolder:

        def members(tar):
            l = len(subfolder + "/")
            for member in tar.getmembers():
                if member.path.startswith(subfolder + "/"):
                    member.path = member.path[l:]
                    yield member

        tar.extractall(dest_path, members=members(tar))

    tar.close()
    os.remove(tar_file)
