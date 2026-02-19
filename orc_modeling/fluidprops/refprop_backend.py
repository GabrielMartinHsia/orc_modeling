from dataclasses import dataclass

@dataclass
class RefpropBackend:
    fluid_id: str

    def __post_init__(self):
        raise NotImplementedError("REFPROP backend not wired yet.")
