from assist_sim import load_combined


class OpenExoEnvironment:
    def __init__(self):
        self.model, self.data = load_combined(
            "myolegs22",
            "OpenSourceLeg_A_L1",
        )

    def get_model_info(self):
        return {
            "nq": self.model.nq,
            "nv": self.model.nv,
            "nu": self.model.nu,
            "nsensordata": self.model.nsensordata,
        }