#!/bin/sh
# Run the local (untracked) git hook rules for one stage.
#
# Usage: local_hooks.sh <stage> [args...]
#
# The rules live in $MAFM_LOCAL_HOOKS/<stage>.d/, else in
# $(git rev-parse --git-common-dir)/local-hooks/<stage>.d/, which worktrees share.
# Every executable there runs in name order with the remaining arguments. Exits 1
# if any rule failed, 0 otherwise (including when the stage has no rules).

# Copyright (C) Richard J. Sheridan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

stage=$1
shift
root=${MAFM_LOCAL_HOOKS:-$(git rev-parse --git-common-dir)/local-hooks}
dir=$root/$stage.d
[ -d "$dir" ] || exit 0
status=0
for f in "$dir"/*; do
  [ -f "$f" ] && [ -x "$f" ] || continue
  if ! "$f" "$@"; then
    echo "local rule failed: ${f##*/}" >&2
    status=1
  fi
done
exit $status
