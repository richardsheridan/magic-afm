#!/bin/sh
# Runs the Magic AFM command line interface frozen inside magic_afm_gui.app, which
# sits next to this script; arguments pass through. The macOS release ships this
# script as magic_afm_cli. Symlinks to it are followed, so it can be linked onto PATH.
script=$0
while [ -L "$script" ]; do
    target=$(readlink "$script")
    case $target in
        /*) script=$target ;;
        *) script=$(dirname "$script")/$target ;;
    esac
done
exec "$(dirname "$script")/magic_afm_gui.app/Contents/MacOS/magic_afm_cli" "$@"
