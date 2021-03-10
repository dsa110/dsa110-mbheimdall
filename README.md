# Multibeam FRB heimdall search code for DSA-110

# Requirements

- cuda ?
- libboost ?

# Installation

Installation done with cmake with standard steps (see `INSTALLATION` for details):
- `configure`
- `make`
- `make install`

Special arguments, paths, envs...?

# Development

The master branch holds code that runs in the real-time system. The development branch holds code that is being tested. All other branches ("feature" or "named" branches) hold code that is in development. No changes will be made to master/development other than through `git merge/rebase` commands.

# Operations

- mbheimdall runs on corr nodes 17-20.
- etcd keys to run it, trigger buffers...