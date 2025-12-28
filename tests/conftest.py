import sys
import pathlib
import warnings

# Silence a DeprecationWarning emitted by tqdm (datetime.utcfromtimestamp) which is
# triggered inside the tqdm library on some Python versions. This is a harmless
# dependency deprecation warning; filter messages that mention `utcfromtimestamp`.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*utcfromtimestamp.*",
)

# Add the project's `src` directory to sys.path so tests can import local modules
HERE = pathlib.Path(__file__).resolve().parents[1]
SRC = HERE / "src"
sys.path.insert(0, str(SRC))
