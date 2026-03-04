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
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
