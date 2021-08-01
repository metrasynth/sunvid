# SunVid

SunVid creates mp4 files from [SunVox](https://warmplace.ru/soft/sunvox/) projects.

## Prerequisites

Prerequisites:

- Python 3.9
- Poetry
- ImageMagick

## Install steps

1. `git clone https://github.com/metrasynth/sunvid`
2. `cd sunvid`
3. `poetry install`
4. `poetry run python -m sunvid --version`

# Running SunVid

The main command is `render`. Get help with its options using:

    poetry run python -m sunvid render --help

Example usage:

    poetry run python -m sunvid render --overwrite sunvid-test-project.sunvox
