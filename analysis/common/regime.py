def probeable_vars(regime: str, is_cartpole: bool = False) -> list[str]:
    """Variables that VARY in this regime (hence probeable/checkable)."""
    base = ["cos_theta", "sin_theta", "theta_dot"]
    if is_cartpole:
        base += ["x", "x_dot"]
    if regime in ("holdl", "combined"):
        base.append("gravity")
    if regime in ("holdg", "combined"):
        base.append("length")
    if regime in ("holdg", "holdl", "combined"):
        base += ["g_over_l", "sqrt_l_over_g"]
    return base

INTERVENTION = {
    "cos_theta":  {"type": "state", "channel": 0},
    "sin_theta":  {"type": "state", "channel": 1},
    "theta_dot":     {"type": "state",  "channel": 2},
    "x":             {"type": "state",  "channel": 3},
    "x_dot":         {"type": "state",  "channel": 4},
    "gravity":       {"type": "config", "config_key": "gravity", "channel": None,},
    "length":        {"type": "config", "config_key": "length", "channel": None,},
    "g_over_l":      {"type": "none"},
    "sqrt_l_over_g": {"type": "none"},
}


def make_metadata(method, variable, layer, model_name, latent_dim, regime, seed):
    iv = INTERVENTION[variable]
    return {
        "method": method,
        "variable": variable,
        "intervention_type": iv["type"],
        "channel": iv.get("channel"),
        "config_key": iv.get("config_key"),\
        "layer": layer,
        "model_name": model_name,
        "latent_dim": int(latent_dim),
        "regime": regime,
        "seed": int(seed),
    }

def checkable_vars(regime: str, is_cartpole: bool = False) -> list[str]:
    return [v for v in probeable_vars(regime, is_cartpole)
            if INTERVENTION[v]["type"] != "none"]