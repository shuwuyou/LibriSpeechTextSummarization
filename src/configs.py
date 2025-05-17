from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).parents[1].resolve()
CONFIG_PATH = REPO_ROOT.joinpath("config", "configs.yaml")
CONFIGS = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8").read())

DATA_PATH = REPO_ROOT.joinpath(CONFIGS["DATA"]["PATH"])
BOOKS_TXT = REPO_ROOT.joinpath(CONFIGS["DATA"]["BOOKS"])
CHAPTERS_TXT = REPO_ROOT.joinpath(CONFIGS["DATA"]["CHAPTERS"])
SPEAKERS_TXT = REPO_ROOT.joinpath(CONFIGS["DATA"]["SPEAKER"])