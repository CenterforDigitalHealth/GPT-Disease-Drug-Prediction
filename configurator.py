import sys
from ast import literal_eval

# DDP-related args injected by torchrun that should be skipped
_ddp_skip_keys = {'local-rank', 'local_rank'}
_key_aliases = {
    'max_iter': 'max_iters',
    'eval_iter': 'eval_iters',
    'warmup_iter': 'warmup_iters',
    'lr_decay_iter': 'lr_decay_iters',
}

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        if not arg.startswith('--'):
            continue
        key, val = arg.split('=', 1)
        key = key[2:]
        if key in _key_aliases:
            alias = _key_aliases[key]
            print(f"Overriding: {key} -> {alias} (alias)")
            key = alias
        # Skip DDP-related args (e.g. --local-rank=0 from torchrun)
        if key in _ddp_skip_keys:
            continue
        if key in globals():
            try:
                # attempt to eval it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok (allow practical numeric coercions)
            current_val = globals()[key]
            current_type = type(current_val)
            if current_type is float and isinstance(attempt, (int, float)) and not isinstance(attempt, bool):
                attempt = float(attempt)
            elif current_type is int and isinstance(attempt, float) and attempt.is_integer():
                attempt = int(attempt)
            elif current_type is int and isinstance(attempt, bool):
                raise AssertionError
            assert type(attempt) == current_type
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
