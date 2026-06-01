from pathlib import Path

def parse_model(path: Path) -> dict:
    name = path.stem 
    parts = name.split("_")

    result = {
        "flag": False,      
        "g": 0.0,
        "l": 0.0,
        "latent": 0,
        "latent_B": 0,       
        "k": 0,
        "steps": 0,
        "config": "",
        "env": "",            
        "protocol": None,     
        "baseline": False,
        "impulse": "impulse_policy" in str(path),  
        "N": 0,               
    }

    
    if parts[1].startswith("Protocol"):
        result["protocol"] = parts[1][-1] 
        
        idx = 2
        if parts[idx] == "baseline":
            result["baseline"] = True
            idx += 1

        # g and l
        result["g"] = float(parts[idx][1:])
        result["l"] = float(parts[idx+1][1:])
        result["config"] = f"g{result['g']}_l{result['l']}"
        idx += 2

        # N, k, latentA, latentB, steps
        for part in parts[idx:]:
            if part.startswith("N"):
                result["N"] = int(part[1:])
            elif part.startswith("k"):
                result["k"] = int(part[1:])
            elif part.startswith("latentA"):
                result["latent"] = int(part[7:])
            elif part.startswith("latentB"):
                result["latent_B"] = int(part[7:])
            elif part.startswith("steps"):
                result["steps"] = int(part[5:])

        return result

    
    result["env"] = parts[1]  

    components = parts[2:]

    if "combined" in components:
        result["flag"] = True
        result["config"] = "combined"
        for part in components:
            if part.startswith("k"):
                try:
                    result["k"] = int(part[1:])
                except ValueError:
                    pass
            elif part.startswith("latent"):
                try:
                    result["latent"] = int(part[6:])
                except ValueError:
                    pass
            elif part.startswith("steps"):
                try:
                    result["steps"] = int(part[5:])
                except ValueError:
                    pass
    else:
        result["g"] = float(components[0][1:])
        result["l"] = float(components[1][1:])
        result["config"] = f"g{result['g']}_l{result['l']}"
        for part in components[2:]:
            if part.startswith("k"):
                try:
                    result["k"] = int(part[1:])
                except ValueError:
                    pass
            elif part.startswith("latent"):
                try:
                    result["latent"] = int(part[6:])
                except ValueError:
                    pass
            elif part.startswith("steps"):
                try:
                    result["steps"] = int(part[5:])
                except ValueError:
                    pass
                
    path_str = str(path)
    if "cartpole" in path_str.lower():
        result["env"] = "CartPoleSim"
    else:
        result["env"] = "PendulumSim"

    return result