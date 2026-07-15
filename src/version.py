"""single source of truth for the app version.

the self-updater compares this against the newest github release tag (see ui/updater.py).
BUMP THIS before tagging a new release, and keep it equal to the release tag minus the leading "v"
(tag v0.1.0-alpha -> "0.1.0-alpha")
"""

__version__ = "0.3.0-beta"

# the repo the updater checks for new releases. owner/name form used by the github api.
REPO = "b-pardi/DBD-BP-Priority-Auto-Spender"
