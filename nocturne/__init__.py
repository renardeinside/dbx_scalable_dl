import dunamai as _dunamai

__version__ = _dunamai.get_version(
    "nocturne", third_choice=_dunamai.Version.from_git
).serialize()
