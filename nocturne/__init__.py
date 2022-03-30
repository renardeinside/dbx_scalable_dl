import dunamai as _dunamai

__version__ = _dunamai.get_version(
    "dbx_scalable_dl", third_choice=_dunamai.Version.from_git
).serialize()
