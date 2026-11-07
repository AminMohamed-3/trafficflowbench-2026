"""Public release files used by online queue and ODME predictors."""
from .paths import release_root, panels
from .submit import sha256

def additional_sources(root):
    release=release_root(root)
    paths={release/"config/corridors.json"}
    for panel in panels(root):
        paths.update((release/"corridors"/panel/"network").glob("*.csv"))
        for task in ["task2","task4"]:
            for split in ["train","validation","private"]:
                folder=release/task/panel/split
                paths.update(folder.glob("*.csv"))
                paths.update(folder.glob("*.parquet"))
    return {str(path.relative_to(release)).replace("\\","/"):sha256(path) for path in sorted(paths) if path.is_file()}
