import argparse
class Config:
    def __init__(
        self,
        model,
        target,
        stability,
        mode,
        data,
        test_size=0.3,
        split_seed=42,
        model_seed=42,
        save_dir="saved_models",
    ):
        # Normalize inputs
        self.model = model.upper()
        self.target = target.upper()
        self.stability = stability.lower()
        self.mode = mode.lower()
        self.data = data.lower()


        # Optional params
        self.test_size = test_size
        self.split_seed = split_seed
        self.model_seed = model_seed        
        self.save_dir = save_dir
        # self.path = path
               

        # Derived name (VERY useful later)
        self.experiment_name = f"{self.model}_{self.target}_{self.stability}_{self.data}"

        # Validate everything
        self._validate()

    def _validate(self):
        assert self.model in ["RF", "ANN", "XGB"], \
            f"Invalid model: {self.model}"

        assert self.target in ["H", "TAU","L"], \
            f"Invalid target: {self.target}"

        assert self.stability in ["stable", "unstable", "all"], \
            f"Invalid stability: {self.stability}"

        assert self.mode in ["tune", "train", "predict"], \
            f"Invalid mode: {self.mode}"

    def __repr__(self):
        return (
        f"Config(model={self.model}, target={self.target}, data={self.data} "
        f"stability={self.stability}, mode={self.mode}, "
        f"test_size={self.test_size}," 
        f"split_seed={self.split_seed}, model_seed={self.model_seed})")



def get_config_from_cli():
    """
    Parse arguments from command line
    Example:
    python main.py --model RF --target H --stability unstable --mode train
    """
    parser = argparse.ArgumentParser(description="ML Pipeline Config")

    parser.add_argument("--model", required=True, help="RF / ANN / XGB")
    parser.add_argument("--target", required=True, help="H / Tau / L")
    parser.add_argument("--stability", required=True, help="stable / unstable / all")
    parser.add_argument("--mode", required=True, help="tune / train / predict")

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="saved_models")

    args = parser.parse_args()

    return Config(
        model=args.model,
        target=args.target,
        stability=args.stability,
        mode=args.mode,
        test_size=args.test_size,
        split_seed=args.split_seed,
        model_seed=args.model_seed,
        save_dir=args.save_dir,
    )


def get_config(**kwargs):
    """
    Create config directly (for Jupyter / scripts)

    Example:
    config = get_config(
        model="RF",
        target="H",
        stability="unstable",
        mode="train"
    )
    """
    return Config(**kwargs)
