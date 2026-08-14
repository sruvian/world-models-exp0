from .utils import STATE_CHANNELS

CONFIG_KEYS = {
    "gravity": "gravity",
    "length":  "length",
    "drive_omega": "drive_omega",
    "drive_amp":   "drive_amp",
    # Lorenz: "sigma": "sigma", "rho": "rho", "beta": "beta"
}

DERIVED_NONINTERVENABLE = {"g_over_l", "sqrt_l_over_g", "sqrt_g_over_l"}


def intervention_for(variable, env_name):

    channels = STATE_CHANNELS.get(env_name, {})
    if variable in channels:
        return {"type": "state", "channel": channels[variable]}
    if variable in CONFIG_KEYS:
        return {"type": "config", "config_key": CONFIG_KEYS[variable], "channel": None}
    if variable in DERIVED_NONINTERVENABLE:
        return {"type": "none"}
    # unknown target for this env
    return {"type": "none"}


def _varies_in_regime(variable, regime, hold_param=None):

    if regime == "combined":
        return True
    if regime == "hold":
        if variable == hold_param:
            return False
        return True
    if regime == "single":
        return variable not in CONFIG_KEYS and variable not in DERIVED_NONINTERVENABLE
    return True


def probeable_vars(probe_targets, env_name, regime, hold_param=None):
    channels = STATE_CHANNELS.get(env_name, {})
    out = []
    for v in probe_targets:
        producible = (v in channels) or (v in CONFIG_KEYS) or (v in DERIVED_NONINTERVENABLE)
        if producible and _varies_in_regime(v, regime, hold_param):
            out.append(v)
    return out


def checkable_vars(probe_targets, env_name, regime, hold_param=None):
    return [v for v in probeable_vars(probe_targets, env_name, regime, hold_param)
            if intervention_for(v, env_name)["type"] != "none"]


def make_metadata(method, variable, layer, model_name, latent_dim, regime, seed, env_name):
    iv = intervention_for(variable, env_name)
    return {
        "method": method,
        "variable": variable,
        "intervention_type": iv["type"],
        "channel": iv.get("channel"),
        "config_key": iv.get("config_key"),
        "layer": layer,
        "model_name": model_name,
        "latent_dim": int(latent_dim),
        "regime": regime,
        "seed": int(seed),
        "env": env_name,
    }