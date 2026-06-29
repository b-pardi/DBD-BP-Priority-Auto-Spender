"""desktop ui for the dbd bloodweb auto-spender.

top-level package, a sibling of src/. the ui only does ui (rendering, interaction, reading/writing
config); all game logic, detection, capture, and the spend loop stay in src/ and are wired in, never
reimplemented here. run with `python -m ui`.
"""
